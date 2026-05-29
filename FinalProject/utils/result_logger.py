"""
结果记录器 — 自动保存实验结果到 CSV
"""
import os
import pandas as pd
from datetime import datetime


class ResultLogger:
    def __init__(self, save_dir: str = 'results/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.records = []

    def log(self, model: str, dataset: str, seq_len: int, pred_len: int,
            mse: float, mae: float, rmse: float, mape: float, smape: float,
            params_m: float = 0, flops_g: float = 0, infer_time_ms: float = 0,
            gpu_mem_mb: float = 0, loss_type: str = 'MSE', extra: str = ''):
        """记录一条实验结果"""
        self.records.append({
            'model': model,
            'dataset': dataset,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'MSE': round(mse, 6),
            'MAE': round(mae, 6),
            'RMSE': round(rmse, 6),
            'MAPE': round(mape, 4),
            'SMAPE': round(smape, 4),
            'Params(M)': round(params_m, 3),
            'FLOPs(G)': round(flops_g, 3),
            'InferTime(ms)': round(infer_time_ms, 2),
            'GPUMem(MB)': round(gpu_mem_mb, 1),
            'loss_type': loss_type,
            'extra': extra,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    def save(self, filename: str = 'main_results.csv'):
        """保存到 CSV 文件"""
        path = os.path.join(self.save_dir, filename)
        df = pd.DataFrame(self.records)
        if os.path.exists(path):
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"Results saved to {path} ({len(df)} records)")

    def save_ablation(self, filename: str = 'ablation_results.csv'):
        """保存消融实验结果"""
        path = os.path.join(self.save_dir, filename)
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"Ablation results saved to {path} ({len(df)} records)")
