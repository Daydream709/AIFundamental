"""
频域分解模块 — FFT 自适应频率分解
将时序分解为趋势 + 季节 + 残差
"""
import torch
import torch.nn as nn
import numpy as np


class AdaptiveFreqDecomp(nn.Module):
    """
    自适应频域分解
    使用 FFT 将信号分解为:
    - x_trend: 低频趋势 (前 top_k 个低频分量)
    - x_seasonal: 中频季节分量 (top_k 个主频率分量)
    - x_residual: 高频残差
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        """
        x: [B, L, C]
        Returns: x_trend, x_seasonal, x_residual (各 [B, L, C])
        """
        B, L, C = x.shape

        # FFT 变换
        x_fft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, C]
        freq_magnitude = torch.abs(x_fft)  # [B, L//2+1, C]

        # 趋势: 最低频分量 (DC + 前2个)
        n_freq = x_fft.shape[1]
        trend_mask = torch.zeros_like(freq_magnitude)
        trend_mask[:, :3, :] = 1.0  # DC + 前2个频率
        x_trend_fft = x_fft * trend_mask
        x_trend = torch.fft.irfft(x_trend_fft, n=L, dim=1)

        # 季节: 幅度最大的 top_k 个频率
        seasonal_mask = torch.zeros_like(freq_magnitude)
        # 排除 DC 和最低频
        mag = freq_magnitude.clone()
        mag[:, :3, :] = 0
        # 找到 top_k 频率
        mag_flat = mag.mean(dim=-1)  # [B, n_freq]
        topk_vals, topk_idx = torch.topk(mag_flat, min(self.top_k, n_freq - 3), dim=1)

        for b in range(B):
            seasonal_mask[b, topk_idx[b], :] = 1.0

        x_seasonal_fft = x_fft * seasonal_mask
        x_seasonal = torch.fft.irfft(x_seasonal_fft, n=L, dim=1)

        # 残差 = 原始 - 趋势 - 季节
        x_residual = x - x_trend - x_seasonal

        return x_trend, x_seasonal, x_residual


class FreqRouter(nn.Module):
    """
    频域分析路由器 — 用于 Mamba-Transformer 双专家
    提取输入的频域特征，输出各专家的权重
    """

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len // 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # 2个专家
        )

    def forward(self, x):
        """
        x: [B, L, C]
        Returns: weights [B, 2] (softmax)
        """
        # 计算频谱
        x_mean = x.mean(dim=-1)  # [B, L]
        fft_mag = torch.abs(torch.fft.rfft(x_mean, dim=1))  # [B, L//2+1]
        weights = torch.softmax(self.fc(fft_mag), dim=-1)  # [B, 2]
        return weights
