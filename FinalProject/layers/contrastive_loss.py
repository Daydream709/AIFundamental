"""
对比损失 — InfoNCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比损失
    让同一时刻的不同模态表示靠近，不同时刻的表示远离
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2):
        """
        feat1: [B, D]  (时序特征)
        feat2: [B, D]  (文本/图像特征)
        Returns: scalar loss
        """
        # L2 归一化
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        # 相似度矩阵
        B = feat1.shape[0]
        sim_matrix = torch.matmul(feat1, feat2.T) / self.temperature  # [B, B]

        # 对角线为正样本，其余为负样本
        labels = torch.arange(B, device=feat1.device)

        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        return loss / 2
