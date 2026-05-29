"""
每个数据集的配置覆盖 — 在 BaseConfig 基础上覆盖数据集特定参数

最终保留 6 个数据集:
  纯时序 (3): ETTm2, Weather, Electricity
  多模态 (3): Energy, Environment, Health — 来自 Time-MMD (NeurIPS 2024)
"""
from configs.base_config import BaseConfig
from dataclasses import dataclass


def get_dataset_config(dataset_name: str, seq_len: int = 96, pred_len: int = 96) -> BaseConfig:
    """根据数据集名称返回对应的配置"""
    configs = {
        # ===== 纯时序数据集 =====
        'ETTm2': BaseConfig(
            data='ETTm2', data_path='ETTm2.csv',
            features='M', target='OT', freq='h',
            enc_in=7, dec_in=7, c_out=7,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
        ),
        'Weather': BaseConfig(
            data='Weather', data_path='Weather.csv',
            features='M', target='OT', freq='h',
            enc_in=21, dec_in=21, c_out=21,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
        ),
        'Electricity': BaseConfig(
            data='Electricity', data_path='Electricity.csv',
            features='M', target='OT', freq='h',
            enc_in=321, dec_in=321, c_out=321,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
            batch_size=16, d_model=256, d_ff=1024,
        ),

        # ===== 多模态数据集 (Time-MMD, NeurIPS 2024) =====
        'Energy': BaseConfig(
            data='Energy', data_path='Energy.csv',
            features='M', target='OT', freq='w',
            enc_in=9, dec_in=9, c_out=9,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
            use_text=True,
        ),
        'Environment': BaseConfig(
            data='Environment', data_path='Environment.csv',
            features='M', target='OT', freq='d',
            enc_in=6, dec_in=6, c_out=6,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
            use_text=True,
        ),
        'Health': BaseConfig(
            data='Health', data_path='Health.csv',
            features='M', target='OT', freq='w',
            enc_in=7, dec_in=7, c_out=7,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
            use_text=True,
        ),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    return configs[dataset_name]
