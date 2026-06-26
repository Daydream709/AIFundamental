"""
每个数据集的配置覆盖 — 按照 FinalProject v2.0 计划

最终保留 4 个数据集:
  纯时序 (3): ETTm2, Weather, Electricity
  多模态 (1): Environment — 包含文本(环境报告+搜索摘要)
"""
from configs.base_config import BaseConfig


def get_dataset_config(dataset_name: str, seq_len: int = 96, pred_len: int = 96) -> BaseConfig:
    """根据数据集名称返回对应的配置"""
    configs = {
        # ===== 纯时序数据集 (3) =====
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
            batch_size=16,  # 显存限制: 321变量需要较小batch
        ),

        # ===== 多模态数据集 (1) =====
        'Environment': BaseConfig(
            data='Environment', data_path='Environment.csv',
            features='M', target='OT', freq='d',
            enc_in=6, dec_in=6, c_out=6,
            seq_len=seq_len, pred_len=pred_len, label_len=seq_len // 2,
            use_text=True,
            text_dim=128,  # v2.1: 实际 cache 文件是 128 维 (sentence-transformers MiniLM)
            # 文本模态说明:
            # - report: 环境报告 (宏观政策/年度总结, ~156条)
            # - search: 相关搜索摘要 (公众关注度, ~2,272条)
        ),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    return configs[dataset_name]
