"""
元学习仲裁器 — 根据输入统计特征动态选择最优模型
"""
import torch
import torch.nn as nn
import numpy as np


class MetaArbitrator(nn.Module):
    """
    模型仲裁集成系统
    提取输入序列的统计特征，通过MLP路由器为各模型分配权重
    """

    def __init__(self, n_models, d_hidden=64):
        super().__init__()
        self.n_models = n_models
        # 输入特征: 谱熵, 趋势强度, 周期性, 方差, 自相关 = 5维
        self.feature_dim = 5
        self.router = nn.Sequential(
            nn.Linear(self.feature_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_models),
        )

    def extract_features(self, x):
        """
        从输入序列提取统计特征

        Args:
            x: [B, L, C] 输入序列

        Returns:
            features: [B, 5] (谱熵, 趋势强度, 周期性, 方差, 自相关)
        """
        B, L, C = x.shape

        # 对每个变量计算特征后取平均
        features = []
        for c in range(C):
            x_c = x[:, :, c]  # [B, L]

            # 1. 谱熵 (频率分布的均匀性)
            fft_mag = torch.abs(torch.fft.rfft(x_c, dim=1))
            fft_prob = fft_mag / (fft_mag.sum(dim=1, keepdim=True) + 1e-8)
            spectral_entropy = -(fft_prob * torch.log(fft_prob + 1e-8)).sum(dim=1)

            # 2. 趋势强度 (线性拟合的 R²)
            t = torch.arange(L, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
            t_mean = t.mean(dim=1, keepdim=True)
            x_mean = x_c.mean(dim=1, keepdim=True)
            slope = ((t - t_mean) * (x_c - x_mean)).sum(dim=1) / ((t - t_mean) ** 2).sum(dim=1)
            trend = slope.unsqueeze(1) * (t - t_mean) + x_mean
            ss_res = ((x_c - trend) ** 2).sum(dim=1)
            ss_tot = ((x_c - x_mean) ** 2).sum(dim=1)
            trend_strength = 1 - ss_res / (ss_tot + 1e-8)

            # 3. 周期性 (FFT 主频率的相对强度)
            fft_sorted = torch.sort(fft_mag, dim=1, descending=True)[0]
            periodicity = (fft_sorted[:, 0] / (fft_mag.sum(dim=1) + 1e-8))

            # 4. 方差
            variance = x_c.var(dim=1)

            # 5. 滞后1自相关
            x_shifted = x_c[:, 1:]
            x_orig = x_c[:, :-1]
            autocorr = (x_shifted * x_orig).mean(dim=1) / (x_c.var(dim=1) + 1e-8)

            feat = torch.stack([spectral_entropy, trend_strength, periodicity,
                              variance, autocorr], dim=-1)  # [B, 5]
            features.append(feat)

        # 对所有变量取平均
        features = torch.stack(features, dim=0).mean(dim=0)  # [B, 5]
        return features

    def forward(self, x_enc, model_predictions):
        """
        Args:
            x_enc: [B, H, C] 输入序列
            model_predictions: list of [B, F, C] 各模型预测结果

        Returns:
            ensemble_pred: [B, F, C] 加权集成预测
            weights: [B, n_models] 各模型权重
        """
        features = self.extract_features(x_enc)
        weights = torch.softmax(self.router(features), dim=-1)  # [B, n_models]

        # 加权组合
        stacked = torch.stack(model_predictions, dim=1)  # [B, n_models, F, C]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, n_models, 1, 1]
        ensemble_pred = (stacked * weights_expanded).sum(dim=1)  # [B, F, C]

        return ensemble_pred, weights
