"""
自适应门控融合网络 — 多模态特征融合
"""
import torch
import torch.nn as nn


class GatingFusion(nn.Module):
    """
    自适应门控融合
    学习每种模态的重要性权重，动态加权融合
    """

    def __init__(self, d_model, n_modalities=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_modalities),
            nn.Sigmoid(),
        )

    def forward(self, *features):
        """
        features: list of [B, D] tensors (可能有些是 None)
        Returns: fused [B, D]
        """
        # 处理缺失模态: 用零向量替代
        processed = []
        for feat in features:
            if feat is None:
                processed.append(torch.zeros_like(features[0]))
            else:
                processed.append(feat)

        # 拼接
        concat = torch.cat(processed, dim=-1)  # [B, D*n_modalities]
        weights = self.gate(concat)  # [B, n_modalities]

        # 加权融合
        fused = sum(w.unsqueeze(-1) * f for w, f in zip(weights.unbind(-1), processed))
        return fused, weights
