"""
Chronos2 Zero-Shot 推理实验
"""
import os
import sys
import numpy as np
import torch
from exp.exp_basic import ExpBasic
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.result_logger import ResultLogger


class ExpZeroShot(ExpBasic):
    """Chronos2 Zero-Shot 推理"""

    def __init__(self, config):
        self.config = config
        self.logger = ResultLogger()
        # Chronos2 不需要标准训练流程
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        from models.Chronos2 import Model
        self.model = Model(config).to(self.device)

    def test(self):
        """直接在测试集上推理"""
        test_loader = data_provider(self.config, 'test')
        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(self.device) for b in batch]
                x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
                true = x_y[:, -self.config.pred_len:, :]

                pred = self.model(x_enc, x_mark_enc, None, None)

                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mse, mae, rmse, mape, smape = metric(preds, trues)

        print(f'\n  Zero-Shot Results (Chronos2 on {self.config.data}, H={self.config.seq_len}, F={self.config.pred_len}):')
        print(f'    MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}')

        self.logger.log(
            model='Chronos2', dataset=self.config.data,
            seq_len=self.config.seq_len, pred_len=self.config.pred_len,
            mse=mse, mae=mae, rmse=rmse, mape=mape, smape=smape,
            loss_type='Zero-Shot',
        )
        self.logger.save('zero_shot_results.csv')

        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape}
