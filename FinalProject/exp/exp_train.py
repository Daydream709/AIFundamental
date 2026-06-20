"""
统一训练/验证/测试循环 — FinalProject v2.0
支持: 混合精度(AMP)、早停(EarlyStopping)、概率输出(GaussianNLL)
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from exp.exp_basic import ExpBasic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
from utils.metrics import metric
from utils.result_logger import ResultLogger
from utils.efficiency import (
    count_parameters, measure_flops, measure_inference_time, measure_gpu_memory,
)


class ExpTrain(ExpBasic):
    """完整的训练/验证/测试流程 — v2.0"""

    def __init__(self, config):
        super().__init__(config)
        # MPS: GradScaler 和 autocast 不可用，关闭混合精度
        self._is_mps = self.device.type == 'mps'
        if self._is_mps or not torch.cuda.is_available():
            self.scaler = GradScaler(device='cpu', enabled=False)
            config.use_amp = False
        else:
            self.scaler = GradScaler(device='cuda', enabled=config.use_amp)
        self.logger = ResultLogger()
        self.use_probabilistic = getattr(config, 'use_probabilistic', False)

    def _get_data(self, flag):
        return data_provider(self.config, flag)

    def _forward_pass(self, model, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        统一前向传播 — 处理标准输出和概率输出

        Returns:
            pred: [B, F, C] (均值)
            logvar: [B, F, C] 或 None
        """
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if isinstance(out, tuple):
            # 概率输出: (mean, logvar)
            mean, logvar = out
            return mean, logvar
        else:
            return out, None

    def train(self):
        """完整训练流程"""
        train_loader = self._get_data('train')
        val_loader = self._get_data('val')

        optimizer = self._get_optimizer()
        criterion = self._get_criterion()

        model_name = f"{self.config.model}_{self.config.data}_{self.config.seq_len}_{self.config.pred_len}"
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            verbose=True,
            save_path=self.config.checkpoints,
        )
        early_stopping.save_path = self.config.checkpoints

        for epoch in range(self.config.train_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)

            val_loss = self._val_epoch(val_loader, criterion)

            print(f'Epoch {epoch+1}/{self.config.train_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

            early_stopping(val_loss, self.model, model_name)
            if early_stopping.early_stop:
                print(f'  Early stopping at epoch {epoch+1}')
                break

        # 加载最优模型
        best_path = os.path.join(self.config.checkpoints, f'{model_name}_checkpoint.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        return self.test()

    def _train_epoch(self, data_loader, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in data_loader:
            batch = [b.to(self.device) for b in batch]
            x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]

            x_dec = torch.zeros_like(x_y[:, -self.config.pred_len:, :])
            x_mark_dec = x_mark_y[:, -self.config.pred_len:, :]
            true = x_y[:, -self.config.pred_len:, :]

            optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.config.use_amp):
                pred, logvar = self._forward_pass(
                    self.model, x_enc, x_mark_enc, x_dec, x_mark_dec
                )

                if logvar is not None and self.config.loss == 'GaussianNLL':
                    # Gaussian NLL: loss = 0.5 * (logvar + (true-mean)^2 / exp(logvar))
                    var = torch.exp(logvar) + 1e-8
                    loss = 0.5 * (logvar + (true - pred) ** 2 / var).mean()
                else:
                    loss = criterion(pred, true)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) for b in batch]
                x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]

                x_dec = torch.zeros_like(x_y[:, -self.config.pred_len:, :])
                x_mark_dec = x_mark_y[:, -self.config.pred_len:, :]
                true = x_y[:, -self.config.pred_len:, :]

                with autocast(self.device.type, enabled=self.config.use_amp):
                    pred, logvar = self._forward_pass(
                        self.model, x_enc, x_mark_enc, x_dec, x_mark_dec
                    )

                    if logvar is not None and self.config.loss == 'GaussianNLL':
                        var = torch.exp(logvar) + 1e-8
                        loss = 0.5 * (logvar + (true - pred) ** 2 / var).mean()
                    else:
                        loss = criterion(pred, true)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def test(self):
        """在测试集上评估 — 收集精度 + 效率指标"""
        test_loader = self._get_data('test')
        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(self.device) for b in batch]
                x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]

                x_dec = torch.zeros_like(x_y[:, -self.config.pred_len:, :])
                x_mark_dec = x_mark_y[:, -self.config.pred_len:, :]
                true = x_y[:, -self.config.pred_len:, :]

                pred, _ = self._forward_pass(
                    self.model, x_enc, x_mark_enc, x_dec, x_mark_dec
                )

                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mse, mae, rmse, mape, smape = metric(preds, trues)

        # 收集效率指标 (参数量 / FLOPs / 推理时间 / GPU 显存)
        n_params, _ = count_parameters(self.model)
        device_str = ('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
        input_shape = (self.config.batch_size, self.config.seq_len, self.config.enc_in)
        flops_g = measure_flops(self.model, input_shape, device=device_str,
                                 freq=self.config.freq)
        infer_time_ms = measure_inference_time(self.model, input_shape, device=device_str, n_runs=50,
                                                freq=self.config.freq)
        gpu_mem_mb = measure_gpu_memory(self.model, input_shape, device=device_str,
                                         freq=self.config.freq)

        # 记录结果 (含效率指标)
        self.logger.log(
            model=self.config.model,
            dataset=self.config.data,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            mse=mse, mae=mae, rmse=rmse, mape=mape, smape=smape,
            params_m=n_params,
            flops_g=flops_g,
            infer_time_ms=infer_time_ms,
            gpu_mem_mb=gpu_mem_mb,
            loss_type=self.config.loss,
        )

        print(f'\n  Test Results ({self.config.model} on {self.config.data}, '
              f'H={self.config.seq_len}, F={self.config.pred_len}):')
        print(f'    MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, '
              f'MAPE={mape:.4f}%, SMAPE={smape:.4f}%')
        print(f'    Params={n_params:.3f}M, FLOPs={flops_g:.3f}G, '
              f'InferTime={infer_time_ms:.2f}ms, GPUMem={gpu_mem_mb:.1f}MB')

        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape,
                'params_m': n_params, 'flops_g': flops_g,
                'infer_time_ms': infer_time_ms, 'gpu_mem_mb': gpu_mem_mb}

    def save_results(self):
        """保存所有实验结果到 CSV"""
        self.logger.save()
