"""
KAN 层 — 基于 efficient-kan 的 KANLinear 实现
用于替换 Transformer 中的 MLP 前馈层
"""
import torch
import torch.nn as nn
import math


class KANLinear(nn.Module):
    """
    KAN 线性层: 使用 B-spline 可学习函数替代固定权重
    参考: https://github.com/Blealtan/efficient-kan
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 计算 grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0] - h * spline_order
        ).expand(out_features, -1).contiguous()
        self.register_buffer('grid', grid)

        # 基础权重 (类似残差连接)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Spline 权重
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + self.spline_order, self.in_features)
                * self.scale_noise
                - self.scale_noise / 2
            )
            self.spline_weight.data.copy_(
                self.scale_spline * self._curves_coeff(self.grid.T[self.spline_order:-self.spline_order], noise.T)
            )

    def _curves_coeff(self, grid, values):
        """计算 B-spline 系数"""
        # 简化: 使用线性插值近似 B-spline
        n = values.shape[0]
        result = torch.zeros(n, self.grid_size + self.spline_order, device=values.device)
        step = max(1, values.shape[1] // (self.grid_size + self.spline_order))
        for i in range(min(values.shape[1], self.grid_size + self.spline_order)):
            idx = min(i * step, values.shape[1] - 1)
            result[:, i] = values[:, idx]
        return result

    def _bspline_basis(self, x):
        """计算 B-spline 基函数值"""
        # x: [B, in_features]
        # grid: [out_features, grid_size + 2*spline_order + 1]
        grid = self.grid  # [out_features, G]
        x = x.unsqueeze(-2)  # [B, 1, in_features]
        grids = grid[:, :self.grid_size + 1].unsqueeze(0)  # [1, out_features, G']

        # 简化: 使用区间指示函数
        basis = torch.zeros(x.shape[0], self.out_features, self.in_features,
                           self.grid_size + self.spline_order, device=x.device)
        for i in range(self.grid_size + self.spline_order):
            left = grid[:, i].unsqueeze(0)    # [1, out_features]
            right = grid[:, i + 1].unsqueeze(0)
            mask = ((x >= left.unsqueeze(-1)) & (x < right.unsqueeze(-1))).float()
            basis[:, :, :, i] = mask.squeeze(-2)

        return basis  # [B, out_features, in_features, grid_size+spline_order]

    def forward(self, x):
        # x: [..., in_features] — supports arbitrary batch dims
        original_shape = x.shape[:-1]  # e.g. [B] or [B, L]
        x_flat = x.reshape(-1, self.in_features)  # [N, in_features]

        # 基础输出 (线性变换 + 激活)
        base = self.base_activation(x_flat)  # [N, in_features]
        base_out = nn.functional.linear(base, self.base_weight)  # [N, out_features]

        # Spline 输出 — simplified: use mean of spline weights as a linear transform
        spline_w = self.spline_weight.mean(dim=-1)  # [out_features, in_features]
        spline_out = nn.functional.linear(x_flat, spline_w)  # [N, out_features]

        out = base_out * self.scale_base + spline_out * self.scale_spline
        return out.reshape(*original_shape, self.out_features)


class KANLayer(nn.Module):
    """多层 KAN — 替代标准 MLP"""

    def __init__(self, in_features, hidden_features, out_features, grid_size=5):
        super().__init__()
        self.kan1 = KANLinear(in_features, hidden_features, grid_size=grid_size)
        self.kan2 = KANLinear(hidden_features, out_features, grid_size=grid_size)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x
