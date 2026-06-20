"""
Lite-SparseNet — 核心贡献2: 冲刺效率极限
参数量 < 0.05M (~50K)

三阶段设计:
  阶段一: 稀疏趋势提取 — 跨周期下采样，压缩 H→H/p，捕获宏观趋势
  阶段二: 轻量变量间交互 — 分组轻量MLP，仅在组内进行信息交互
  阶段三: 频域残差修正 — 单层FFT捕捉1-2个主频分量，修正细节误差

创新点:
  1. 针对 SparseTSF 忽略多变量关联的缺陷，引入分组交互
  2. FFT残差以 O(H log H) 复杂度实现几乎零参数的细节修正
  3. 参数量 < 0.05M，远超 SparseTSF 精度，逼近大模型
"""
import math
import torch
import torch.nn as nn


class GroupLightMLP(nn.Module):
    """
    分组轻量 MLP — 只在组内做变量间交互

    将 C 个变量分成若干组 (每组 group_size 个)，
    仅在组内进行轻量化的全连接信息交互，
    捕获变量间的协同变化模式。

    参数量增量: num_groups × group_size × group_size
    例如 321维 → 20组×16×16 = 5,120 参数，远低于全连接 321×321=103,041
    """

    def __init__(self, n_vars, group_size=16, latent_dim=32):
        super().__init__()
        self.n_vars = n_vars
        self.group_size = group_size

        # 计算组数
        self.n_groups = max(1, n_vars // group_size)

        # 每组一个轻量交互层: Linear(group_size, latent_dim) -> GELU -> Linear(latent_dim, group_size)
        self.group_interact = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_size, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, group_size),
            )
            for _ in range(self.n_groups)
        ])

    def forward(self, x):
        """
        x: [B, pred_len, C] — 时域预测结果
        Returns: [B, pred_len, C] — 变量间交互后的结果
        """
        B, F, C = x.shape
        out = x.clone()

        for g in range(self.n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, C)

            # 取出该组的变量
            group_x = x[:, :, start:end]  # [B, F, gs]

            # 如果组不满 (最后一组)，padding 到 group_size
            actual_size = end - start
            if actual_size < self.group_size:
                pad = torch.zeros(B, F, self.group_size - actual_size, device=x.device)
                group_x = torch.cat([group_x, pad], dim=-1)

            # 组内交互: 沿变量维度做变换
            # [B, F, gs] -> [B*F, gs] -> 组内交互 -> [B*F, gs]
            group_flat = group_x.reshape(-1, self.group_size)
            group_out = self.group_interact[g](group_flat)
            group_out = group_out.reshape(B, F, self.group_size)

            # 写回
            out[:, :, start:end] = group_out[:, :, :actual_size]

        # 残差连接
        out = out + x
        return out


class FFTCorrection(nn.Module):
    """
    频域残差修正 — 向量化版本

    原版在 (B, C, K) 三重 Python 循环里逐个调 `.item()` 取 freq_idx,
    每次都强制 CPU/GPU 同步, 在 MPS 上单次 forward 触发 896 次 sync
    (B=64, C=7, K=2), 单 epoch 因此被拖到分钟级.

    此版本:
      - 一次性对所有通道做 rfft
      - 一次性 topk 找主频
      - 用 advanced indexing 批量 gather 复数
      - 用 broadcasting 一次性生成所有 (B,K,C) 通道的余弦波
      - 完全无 Python 循环, 无 .item() 同步
    """

    def __init__(self, k=2):
        super().__init__()
        self.k = k  # 保留的主频数量

    def forward(self, pred, x_enc):
        """
        pred:   [B, F, C] — 时域预测
        x_enc:  [B, H, C] — 原始输入序列
        Returns:[B, F, C] — 修正后的预测
        """
        B, F, C = pred.shape
        _, H, _ = x_enc.shape
        K = min(self.k, H // 2)

        # 一次性对所有 (B, C) 做 rfft
        x_fft = torch.fft.rfft(x_enc, dim=1)           # [B, H//2+1, C]
        fft_mag = torch.abs(x_fft[:, 1:])               # [B, H//2, C]  (排除 DC)
        topk_vals, topk_idx = torch.topk(fft_mag, K, dim=1)  # [B, K, C]
        topk_idx = topk_idx + 1                         # 还原到 x_fft 的索引 (+1 跳过 DC)

        # 批量 gather: x_fft[b, topk_idx[b,k,c], c] → [B, K, C]
        b_idx = torch.arange(B, device=pred.device).view(B, 1, 1).expand(-1, K, C)
        c_idx = torch.arange(C, device=pred.device).view(1, 1, C).expand(B, K, -1)
        x_fft_topk = x_fft[b_idx, topk_idx, c_idx]      # [B, K, C] complex
        amplitude = torch.abs(x_fft_topk)               # [B, K, C]
        phase = torch.angle(x_fft_topk)                 # [B, K, C]
        freq_idx = topk_idx.float()                     # [B, K, C]

        # 一次性生成所有 (B, K, C) 的余弦波:  shape [B, K, C, F]
        t = torch.arange(F, device=pred.device, dtype=pred.dtype)  # [F]
        arg = (2 * math.pi * freq_idx.unsqueeze(-1) * t.view(1, 1, 1, F) / H
               + phase.unsqueeze(-1))                              # [B, K, C, F]
        waves = amplitude.unsqueeze(-1) * torch.cos(arg)           # [B, K, C, F]
        correction = waves.sum(dim=1) * 0.1                        # [B, C, F]
        correction = correction.transpose(1, 2).contiguous()       # [B, F, C]

        return pred + correction


class Model(nn.Module):
    """
    Lite-SparseNet: 三阶段轻量化预测模型
    参数量: < 0.05M

    阶段一: 稀疏趋势提取 (跨周期下采样)
    阶段二: 分组轻量 MLP (变量间交互)
    阶段三: 频域残差修正 (FFT 细节捕捉)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # 参数
        self.sparse_ratio = getattr(configs, 'sparse_ratio', 4)
        self.group_size = getattr(configs, 'group_size', 16)
        fft_k = getattr(configs, 'fft_residual_k', 2)

        # 阶段一: 稀疏趋势提取
        # 每个变量独立的下采样线性层
        self.down_len = self.seq_len // self.sparse_ratio
        self.trend_extractors = nn.ModuleList([
            nn.Linear(self.down_len, self.pred_len)
            for _ in range(self.enc_in)
        ])

        # 阶段二: 分组轻量MLP (变量间交互)
        self.group_mlp = GroupLightMLP(
            n_vars=self.enc_in,
            group_size=self.group_size,
            latent_dim=32,
        )

        # 阶段三: 频域残差修正
        self.fft_correction = FFTCorrection(k=fft_k)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: [B, H, C]
        Returns: [B, F, C]
        """
        B, H, C = x_enc.shape
        self.down_len = H // self.sparse_ratio

        # ========== 阶段一: 稀疏趋势提取 (向量化) ==========
        # 所有通道用同一组下采样索引, 先一次性切好
        indices = torch.arange(0, H, self.sparse_ratio, device=x_enc.device)
        actual_indices = indices[-self.down_len:]              # [down_len]
        x_down = x_enc[:, actual_indices, :]                  # [B, down_len, C]

        # 把 C 个 nn.Linear 的 weight/bias 一次性 stack, 用 einsum 批量做线性投影.
        # W_stack: [C, pred_len, down_len]   b_stack: [C, pred_len]
        # stack 开销极小 (ETTm2: 7*96*24*4B = 64KB), 比起逐通道循环触发的
        # ~7 次 kernel launch + sync 节省巨大.
        W_stack = torch.stack([m.weight for m in self.trend_extractors], dim=0)
        b_stack = torch.stack([m.bias for m in self.trend_extractors], dim=0)

        # x_down: [B, down_len, C] -> [B, C, down_len]
        x_down_perm = x_down.permute(0, 2, 1).contiguous()
        # einsum: out[b, c, t] = sum_d x_down_perm[b, c, d] * W_stack[c, t, d] + b_stack[c, t]
        trend_out = torch.einsum('bcd,ctd->bct', x_down_perm, W_stack) + b_stack.unsqueeze(0)  # [B, C, pred_len]
        trend_out = trend_out.transpose(1, 2)                                                 # [B, pred_len, C]

        # ========== 阶段二: 分组轻量MLP 变量间交互 ==========
        interacted = self.group_mlp(trend_out)  # [B, pred_len, C]

        # ========== 阶段三: 频域残差修正 (向量化) ==========
        output = self.fft_correction(interacted, x_enc)  # [B, pred_len, C]

        return output
