"""
共形预测区间估计 — 输出具有理论保证的置信区间
"""
import numpy as np
import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """
    分位数回归头 — 同时输出多个分位数的预测
    """

    def __init__(self, d_model, pred_len, c_out, quantiles=None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        self.quantiles = quantiles
        self.heads = nn.ModuleList([
            nn.Linear(d_model, pred_len * c_out)
            for _ in quantiles
        ])
        self.pred_len = pred_len
        self.c_out = c_out

    def forward(self, x):
        """
        x: [B, d_model]
        Returns: list of [B, pred_len, c_out]
        """
        outputs = []
        for head in self.heads:
            out = head(x).reshape(-1, self.pred_len, self.c_out)
            outputs.append(out)
        return outputs


def quantile_loss(preds, target, quantiles):
    """
    分位数损失 (Pinball Loss)
    preds: list of [B, F, C]
    target: [B, F, C]
    quantiles: list of float
    """
    losses = 0
    for q, pred in zip(quantiles, preds):
        errors = target - pred
        losses += torch.max(q * errors, (q - 1) * errors).mean()
    return losses / len(quantiles)


class ConformalPredictor:
    """
    共形预测 — 后处理包装器
    在任意训练好的模型上添加具有理论保证的置信区间
    """

    def __init__(self, quantiles=None):
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        self.quantiles = quantiles
        self.calibration_scores = None

    def calibrate(self, preds_list, true_values):
        """
        在校准集上计算非一致性分数

        Args:
            preds_list: list of [N, F, C] — 各分位数预测
            true_values: [N, F, C] — 真实值
        """
        pred_low = preds_list[0]   # 0.05 分位数
        pred_high = preds_list[-1]  # 0.95 分位数

        # 非一致性分数: 预测区间之外的程度
        scores = np.maximum(
            pred_low - true_values,
            true_values - pred_high
        )
        self.calibration_scores = scores

    def predict_with_intervals(self, preds_list, alpha=0.05):
        """
        调整预测区间，确保 (1-alpha) 的覆盖率

        Args:
            preds_list: list of [N, F, C]
            alpha: 显著性水平 (0.05 = 95% 置信区间)

        Returns:
            pred_mean, lower_bound, upper_bound (各 [N, F, C])
        """
        pred_low = preds_list[0]
        pred_mean = preds_list[len(preds_list) // 2]
        pred_high = preds_list[-1]

        if self.calibration_scores is not None:
            # 共形分位数
            n = len(self.calibration_scores.flatten())
            q_hat = np.quantile(
                self.calibration_scores.flatten(),
                min(1.0, (n + 1) * (1 - alpha) / n)
            )
            # 调整区间
            lower = pred_low - q_hat
            upper = pred_high + q_hat
        else:
            lower = pred_low
            upper = pred_high

        return pred_mean, lower, upper
