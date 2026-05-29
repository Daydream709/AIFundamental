"""
评价指标 — MSE, MAE, RMSE, MAPE, SMAPE
"""
import numpy as np
import torch


def metric(pred, true):
    """计算所有5个指标，返回 (mse, mae, rmse, mape, smape)"""
    pred = np.array(pred)
    true = np.array(true)
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)

    # MAPE: 避免除零
    mask = true != 0
    if mask.sum() == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100

    # SMAPE
    denominator = (np.abs(true) + np.abs(pred))
    mask2 = denominator != 0
    if mask2.sum() == 0:
        smape = 0.0
    else:
        smape = np.mean(2.0 * np.abs(true[mask2] - pred[mask2]) / denominator[mask2]) * 100

    return mse, mae, rmse, mape, smape


# PyTorch 版本 — 用于训练中的损失计算
def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)


def mae_loss(pred, true):
    return torch.mean(torch.abs(pred - true))


def gaussian_nll_loss(pred_mean, pred_logvar, true):
    """高斯负对数似然损失 — 用于概率预测"""
    var = torch.exp(pred_logvar)
    return torch.mean(0.5 * (torch.log(var) + (true - pred_mean) ** 2 / var))
