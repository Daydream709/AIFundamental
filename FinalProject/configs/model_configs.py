"""
每个模型的预设超参数配置 — FinalProject v2.0

设计原则:
  1. 每个模型使用论文推荐/官方实现验证过的最优配置
  2. 不同数据集使用不同的配置（模型对不同数据特性敏感）
  3. 每个模型提供 2-3 组配置（默认 / 大数据集 / 小数据集）
  4. 基于 thuml/Time-Series-Library 官方脚本 + 原论文推荐值

参考来源:
  - DLinear (AAAI 2023): 无架构超参，仅 moving_avg=25
  - PatchTST (ICLR 2023): 官方脚本跨数据集差异大
  - TimesNet (ICLR 2023): 官方脚本 d_model 16-256 随数据集变化
  - Mamba (2023): 官方脚本所有数据集统一 d_model=128
  - SparseTSF (arXiv 2024): 极轻量, period_len 是唯一关键参数
  - KAN-iTransformer (自研): 参考 iTransformer+PatchTST 规模
  - Lite-SparseNet (自研): 参数量 < 0.05M 约束下最大化精度
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelPreset:
    """单个模型的预设配置"""
    # 通用架构参数
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.1
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 64
    train_epochs: int = 100
    patience: int = 10
    # 模型特定参数
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> dict:
        result = {
            'd_model': self.d_model, 'n_heads': self.n_heads,
            'e_layers': self.e_layers, 'd_ff': self.d_ff,
            'dropout': self.dropout, 'learning_rate': self.learning_rate,
            'batch_size': self.batch_size, 'train_epochs': self.train_epochs,
            'patience': self.patience,
        }
        result.update(self.extra)
        return result


# ============================================================================
# 模型预设配置表
# ============================================================================

MODEL_PRESETS: Dict[str, Dict[str, ModelPreset]] = {

    # ========================================================================
    # DLinear — 极简线性基线
    # 论文: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
    # 参数量: ~0.02M
    # 无架构超参，所有数据集统一配置
    # ========================================================================
    'DLinear': {
        'default': ModelPreset(
            d_model=512,       # 不使用但保持兼容
            n_heads=8,         # 不使用
            e_layers=1,        # DLinear 不需要多层
            d_ff=2048,         # 不使用
            dropout=0.1,       # 不使用
            learning_rate=1e-4,
            batch_size=32,
            train_epochs=50,   # 简单模型收敛快
            patience=5,
            extra={'moving_avg': 25},
        ),
    },

    # ========================================================================
    # PatchTST — Transformer + Patching 代表
    # 论文: A Time Series is Worth 64 Words (ICLR 2023)
    # 参数量: ~6.9M
    # 官方脚本: d_model=512 固定, e_layers 和 n_heads 随数据集变化
    # ========================================================================
    'PatchTST': {
        # 低维数据集 (ETTm2: 7变量)
        # 来源: scripts/long_term_forecast/ETT_script/PatchTST_ETTm2.sh (pred_len=96)
        'low_dim': ModelPreset(
            d_model=512, n_heads=16, e_layers=3, d_ff=2048,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'patch_len': 16, 'stride': 8},
        ),
        # 中维数据集 (Weather: 21变量)
        # 来源: scripts/long_term_forecast/Weather_script/PatchTST.sh (pred_len=96)
        'mid_dim': ModelPreset(
            d_model=512, n_heads=4, e_layers=2, d_ff=2048,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'patch_len': 16, 'stride': 8},
        ),
        # 高维数据集 (Electricity: 321变量)
        # 来源: scripts/long_term_forecast/ECL_script/PatchTST.sh
        'high_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=2, d_ff=2048,
            dropout=0.1, learning_rate=1e-4, batch_size=16,  # 显存限制
            train_epochs=50, patience=10,
            extra={'patch_len': 16, 'stride': 8},
        ),
        # 多模态 (Environment: 6变量+文本) — 参考低维配置
        'multimodal': ModelPreset(
            d_model=512, n_heads=8, e_layers=2, d_ff=2048,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'patch_len': 16, 'stride': 8},
        ),
    },

    # ========================================================================
    # TimesNet — CNN 代表
    # 论文: TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)
    # 参数量: ~5.2M (实际因 d_model 变化)
    # 官方脚本: d_model 16-256, d_ff=32-512, 与数据集维度强相关
    # ========================================================================
    'TimesNet': {
        # 低维 (ETTm2: 7变量) — 小 d_model 高效设计
        'low_dim': ModelPreset(
            d_model=32, n_heads=8, e_layers=2, d_ff=32,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'top_k': 5, 'num_kernels': 6},
        ),
        # 中维 (Weather: 21变量)
        # 来源: scripts/long_term_forecast/Weather_script/TimesNet.sh
        'mid_dim': ModelPreset(
            d_model=32, n_heads=8, e_layers=2, d_ff=32,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'top_k': 5, 'num_kernels': 6},
        ),
        # 高维 (Electricity: 321变量) — 官方 d_model=256
        'high_dim': ModelPreset(
            d_model=256, n_heads=8, e_layers=2, d_ff=512,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'top_k': 5, 'num_kernels': 6},
        ),
        # 多模态 (Environment: 6变量) — 低维配置
        'multimodal': ModelPreset(
            d_model=32, n_heads=8, e_layers=2, d_ff=32,
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'top_k': 5, 'num_kernels': 6},
        ),
    },

    # ========================================================================
    # Mamba — SSM 代表
    # 论文: Mamba: Linear-Time Sequence Modeling (2023)
    # 参数量: ~15.9M (MambaSimple 纯PyTorch版)
    # 官方脚本: 所有数据集统一 d_model=128, d_state=16, expand=2
    # ========================================================================
    'Mamba': {
        'default': ModelPreset(
            d_model=128, n_heads=8, e_layers=2, d_ff=16,  # d_ff → d_state
            dropout=0.1, learning_rate=1e-4, batch_size=32,
            train_epochs=50, patience=10,
            extra={'expand': 2, 'd_conv': 4, 'd_state': 16},
        ),
    },

    # ========================================================================
    # SparseTSF — 极轻量级稀疏预测
    # 论文: SparseTSF: Modeling Long-term Time Series Forecasting with 1K Params
    # 参数量: ~0.003M (~3K)
    # 核心参数: period_len (下采样周期)
    # ========================================================================
    'SparseTSF': {
        'default': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,  # 不使用
            dropout=0.0, learning_rate=1e-3,  # 极简模型可用更高学习率
            batch_size=64, train_epochs=50, patience=5,
            extra={'period_len': 12},
        ),
    },

    # ========================================================================
    # KAN-iTransformer — 自研高性能模型
    # 核心贡献1: KAN + CFD + 概率输出 + RevIN + 模型仲裁
    # 参数量: ~120M
    # 设计参考: iTransformer (d_model=512) + PatchTST (e_layers=3)
    # ========================================================================
    'KANiTransformer': {
        # 低维 (ETTm2: 7变量) — 调优结果: e_layers=1, lr=5e-5
        'low_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.1, learning_rate=5e-5, batch_size=32,
            train_epochs=50, patience=10,
            extra={
                'kan_grid_size': 5, 'use_cfd': True, 'use_revin': True,
                'use_probabilistic': True, 'top_k': 5,
            },
        ),
        # 中维 (Weather: 21变量)
        'mid_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.1, learning_rate=5e-5, batch_size=32,
            train_epochs=50, patience=10,
            extra={
                'kan_grid_size': 5, 'use_cfd': True, 'use_revin': True,
                'use_probabilistic': True, 'top_k': 5,
            },
        ),
        # 高维 (Electricity: 321变量) — 大模型+高维需要降低 batch_size
        'high_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=1024,
            dropout=0.1, learning_rate=5e-5, batch_size=16,
            train_epochs=50, patience=10,
            extra={
                'kan_grid_size': 5, 'use_cfd': True, 'use_revin': True,
                'use_probabilistic': True, 'top_k': 5,
            },
        ),
        # 多模态 (Environment: 6变量+文本) — 启用概率输出
        'multimodal': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.1, learning_rate=5e-5, batch_size=32,
            train_epochs=50, patience=10,
            extra={
                'kan_grid_size': 5, 'use_cfd': True, 'use_revin': True,
                'use_probabilistic': True, 'top_k': 5,
            },
        ),
    },

    # ========================================================================
    # Lite-SparseNet — 自研轻量化模型
    # 核心贡献2: 稀疏采样 + 分组MLP + FFT残差
    # 参数量: ~0.018M (< 0.05M ✓)
    # 核心参数: sparse_ratio, group_size, fft_residual_k
    # ========================================================================
    'LiteSparseNet': {
        # 低维 (ETTm2: 7变量) — 组数少，group_size 自动适应
        'low_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,  # 不使用
            dropout=0.05, learning_rate=1e-3,  # 轻量模型用更高学习率
            batch_size=64, train_epochs=50, patience=10,
            extra={'sparse_ratio': 2, 'group_size': 4, 'fft_residual_k': 2},
        ),
        # 中维 (Weather: 21变量) — group_size=8
        'mid_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.05, learning_rate=1e-3,
            batch_size=64, train_epochs=50, patience=10,
            extra={'sparse_ratio': 2, 'group_size': 8, 'fft_residual_k': 2},
        ),
        # 高维 (Electricity: 321变量) — 大group_size 捕获变量交互
        'high_dim': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.05, learning_rate=1e-3,
            batch_size=64, train_epochs=50, patience=10,
            extra={'sparse_ratio': 2, 'group_size': 16, 'fft_residual_k': 3},
        ),
        # 多模态 (Environment: 6变量)
        'multimodal': ModelPreset(
            d_model=512, n_heads=8, e_layers=1, d_ff=2048,
            dropout=0.05, learning_rate=1e-3,
            batch_size=64, train_epochs=50, patience=10,
            extra={'sparse_ratio': 2, 'group_size': 4, 'fft_residual_k': 2},
        ),
    },
}


# ============================================================================
# 数据集 → 配置档位映射
# ============================================================================

DATASET_CONFIG_TIER = {
    'ETTm2':       'low_dim',     # 7变量, 低维
    'Weather':     'mid_dim',     # 21变量, 中维
    'Electricity': 'high_dim',    # 321变量, 高维
    'Environment': 'multimodal',  # 6变量+文本, 多模态
}


def get_model_config(model_name: str, dataset_name: str) -> dict:
    """
    获取模型在指定数据集上的最优预设配置

    Args:
        model_name: 模型名称 (DLinear, PatchTST, TimesNet, Mamba,
                   SparseTSF, KANiTransformer, LiteSparseNet)
        dataset_name: 数据集名称 (ETTm2, Weather, Electricity, Environment)

    Returns:
        dict: 超参数字典，可直接用于更新 BaseConfig

    Example:
        >>> cfg = get_model_config('TimesNet', 'Electricity')
        >>> # cfg = {'d_model': 256, 'd_ff': 512, ...}
    """
    tier = DATASET_CONFIG_TIER.get(dataset_name, 'default')

    presets = MODEL_PRESETS.get(model_name, {})
    if tier in presets:
        preset = presets[tier]
    elif 'default' in presets:
        preset = presets['default']
    else:
        raise ValueError(
            f"No preset found for model '{model_name}' with dataset '{dataset_name}' "
            f"(tier='{tier}'). Available tiers: {list(presets.keys())}"
        )

    return preset.to_dict()


def print_model_config(model_name: str, dataset_name: str):
    """打印模型在指定数据集上的配置（调试用）"""
    config = get_model_config(model_name, dataset_name)
    print(f"\n{'='*50}")
    print(f"  {model_name} on {dataset_name}")
    print(f"{'='*50}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return config


if __name__ == '__main__':
    # 测试: 打印所有模型的配置
    for model in ['DLinear', 'PatchTST', 'TimesNet', 'Mamba',
                  'SparseTSF', 'KANiTransformer', 'LiteSparseNet']:
        for dataset in ['ETTm2', 'Weather', 'Electricity', 'Environment']:
            try:
                print_model_config(model, dataset)
            except ValueError as e:
                print(f"  {model} on {dataset}: SKIP ({e})")
