"""
Mamba/SSM 模块 — 用于 Mamba-Transformer 双专家路由
"""
import torch
import torch.nn as nn
import math


class MambaBlock(nn.Module):
    """
    简化版 Mamba SSM 模块
    使用选择性状态空间模型 (S6) 的核心思想
    当 mamba-ssm 包不可用时使用此简化实现
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(expand * d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D 卷积 (局部依赖)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # SSM 参数投影
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # SSM 矩阵
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_inner + 1).float().repeat(d_state, 1).T))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L, D]
        Returns: [B, L, D]
        """
        B, L, D = x.shape

        # 输入投影 + 分成两部分 (用于门控)
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # 各 [B, L, d_inner]

        # 1D 卷积 (局部依赖)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)

        # SiLU 激活
        x_conv = torch.nn.functional.silu(x_conv)

        # 简化 SSM: 使用线性递归近似
        x_ssm = self._selective_scan(x_conv)

        # 门控
        y = x_ssm * torch.nn.functional.silu(z)

        # 输出投影
        output = self.out_proj(y)
        return self.dropout(output)

    def _selective_scan(self, x):
        """
        简化的选择性扫描 — 使用指数移动平均近似 SSM
        x: [B, L, d_inner]
        """
        B, L, D = x.shape

        # 投影得到 SSM 参数
        ssm_params = self.x_proj(x)  # [B, L, 2*d_state + 1]
        dt_raw = ssm_params[:, :, -1:]     # [B, L, 1]

        dt = self.dt_proj(dt_raw)  # [B, L, d_inner]
        dt = torch.nn.functional.softplus(dt)  # [B, L, d_inner]

        # 简化: 用输入依赖的指数移动平均近似 SSM
        # decay: [B, L, d_inner] — per-timestep, per-channel decay rate
        A_diag = -torch.exp(self.A_log[:, 0])  # [d_inner] — take first state column as diagonal
        decay = torch.exp(A_diag.unsqueeze(0).unsqueeze(0) * dt)  # [B, L, d_inner]

        # 递归计算
        h = torch.zeros(B, D, device=x.device)
        ys = []
        for t in range(L):
            d = decay[:, t, :]  # [B, d_inner]
            h = d * h + x[:, t, :] * (1 - d)
            ys.append(h)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaLayer(nn.Module):
    """多层 Mamba"""

    def __init__(self, d_model, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)
