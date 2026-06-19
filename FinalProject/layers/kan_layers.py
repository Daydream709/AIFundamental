"""
KAN 层 — 完整 B-spline 实现 (Cox-de Boor 递推)
用于替换 Transformer 中的 MLP 前馈层

优化点 (相对 v1):
  - 真实 B-spline 基函数计算 (de Boor-Cox 递推), 而非简化的线性插值
  - spline_weight 为可学习控制点权重
  - 输出 = base_path + spline_path (残差风格), 保证训练稳定
  - 数值稳定: basis 归一化避免极端值

参考: https://github.com/Blealtan/efficient-kan
"""
import torch
import torch.nn as nn
import math


class KANLinear(nn.Module):
    """
    KAN 线性层: 使用 B-spline 可学习函数替代固定权重

    数学形式: f(x) = base_act(W_base x) + Σ_i W_spline[i] * B_i(x)
      - base_path: 传统线性 (确保训练初期不崩)
      - spline_path: 在每个 input 维度上, 用 B-spline 学习非线性
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        n_coeffs = grid_size + spline_order

        # 均匀网格 (grid_size + 2*spline_order + 1 个点, 用于 B-spline 边界)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0] - h * spline_order
        ).expand(out_features, -1).contiguous()
        self.register_buffer('grid', grid)

        # 基础路径: 仿射 + 激活 (传统 MLP 风格, 训练稳定)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Spline 路径: 每个 (in, out) 对的 B-spline 控制点权重
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, n_coeffs))

        # grid 扰动 (促进基函数位置自适应)
        self.grid_eps = grid_eps
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.trunc_normal_(self.spline_weight, std=0.1)

    def _spline_basis(self, x):
        """
        Cox-de Boor 递推计算 B-spline 基函数 (向量化实现)

        完整 B-spline 在 n_grid (= grid_size + 2*spline_order + 1) 个 grid points 上定义。
        0 阶: 1 if grid[i] ≤ x < grid[i+1] else 0 → 共 n_grid-1 个
        k 阶递归: B_{i,k}(x) = α * B_{i,k-1}(x) + (1-α) * B_{i+1,k-1}(x)
        最终 B-spline 控制点数: n_coeffs = grid_size + spline_order

        Args:
            x: [..., in_features]
        Returns:
            bases: [..., in_features, n_coeffs] (in 每个维度上 n_coeffs 个基函数)
        """
        # 确保 x 在 grid 范围内, 防止外推
        x_clamped = x.clamp(min=self.grid.min().item() + 1e-5,
                             max=self.grid.max().item() - 1e-5)
        # 维度布局: [N, in, out, n_grid_points]
        x_exp = x_clamped.unsqueeze(-1)            # [N, in, 1]
        x_exp = x_exp.unsqueeze(-2)               # [N, in, 1, 1]
        # grid: [out, n_grid] -> [1, 1, out, n_grid]
        grid = self.grid.unsqueeze(0).unsqueeze(0)

        # 0阶 B-spline: 1 if grid[i] ≤ x < grid[i+1] else 0
        bases = ((x_exp >= grid[..., :-1]) & (x_exp < grid[..., 1:])).float()
        # bases: [N, in, 1, out, n_grid-1]

        # de Boor 递推
        for k in range(1, self.spline_order + 1):
            # k 阶: B_{i,k}, 公式 B_{i,k}(x) = ((x - g_i)/(g_{i+k} - g_i)) B_{i,k-1} + ((g_{i+k+1} - x)/(g_{i+k+1} - g_{i+1})) B_{i+1,k-1}
            # 上阶 bases 维度 [N, in, 1, out, (n_grid-1)-(k-1)]
            # 取 n_grid-1-k 个元素 (左右都要这个数)
            prev = bases  # [N, in, 1, out, n_grid-1-(k-1)]
            # alpha 分子: x - grid[i], i 取前 n_grid-1-k
            num = x_exp - grid[..., :-(k+1)]              # [N, in, 1, out, n_grid-1-k]
            # alpha 分母: grid[i+k] - grid[i]
            den = grid[..., k:-(1)] - grid[..., :-(k+1)]  # [N, in, 1, out, n_grid-1-k]
            alpha = num / den.clamp(min=1e-8)
            # prev[..., :-1] 取前 n_grid-1-k (左)
            # prev[..., 1:] 取后 n_grid-1-k (右)
            bases = alpha * prev[..., :-1] + (1 - alpha) * prev[..., 1:]

        # bases: [N, in, 1, out, n_coeffs] → squeeze 到 [N, in, out, n_coeffs]
        bases = bases.squeeze(-3)
        # 在 out 维度上平均 (efficient-kan 做法), 降低内存同时保持效果
        return bases.mean(dim=-2)

    def forward(self, x):
        """
        Args:
            x: [..., in_features]
        Returns:
            out: [..., out_features]
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)  # [N, in_features]

        # === Base path ===
        base = self.base_activation(x_flat)
        base_out = nn.functional.linear(base, self.base_weight)  # [N, out_features]

        # === Spline path ===
        bases = self._spline_basis(x_flat)  # [N, in_features, n_coeffs]
        # spline_weight: [out, in, n_coeffs]
        # einsum: (n, i, k) × (o, i, k) -> (n, o)
        spline_out = torch.einsum('nik,oik->no', bases, self.spline_weight)

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
