"""
消融实验自动化运行器 — 6组消融实验

纯时序数据集: ETTm2, Weather, Electricity
多模态数据集: Energy, Environment, Health (Time-MMD)
"""
import os
import sys
import pandas as pd
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(1, os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary'))

from configs.dataset_configs import get_dataset_config
from exp.exp_train import ExpTrain
from utils.tools import fix_seed


def run_ablation(config):
    """运行所有6组消融实验"""
    results = []

    # 消融1: 文本模态影响 (在 Environment 上 — 有最丰富的文本数据)
    print("\n=== Ablation 1: Text Modality (Environment) ===")
    for use_text, use_contrastive in [(False, False), (True, False), (True, True)]:
        cfg = get_dataset_config('Environment', seq_len=24, pred_len=24)
        cfg.model = 'MultimodalFusion'
        cfg.use_text = use_text
        cfg.use_image = False
        cfg.use_contrastive = use_contrastive
        cfg.train_epochs = 30
        cfg.checkpoints = './checkpoints/'

        label = f"text={use_text}_contrastive={use_contrastive}"
        print(f"  Running: {label}")
        try:
            exp = ExpTrain(cfg)
            res = exp.train()
            results.append({'ablation': 'text_modality', 'setting': label, **res})
        except Exception as e:
            print(f"  Error: {e}")

    # 消融2: 图像模态影响
    print("\n=== Ablation 2: Image Modality (Energy) ===")
    for use_image in [False, True]:
        cfg = get_dataset_config('Energy', seq_len=24, pred_len=24)
        cfg.model = 'MultimodalFusion'
        cfg.use_text = True
        cfg.use_image = use_image
        cfg.train_epochs = 30
        cfg.checkpoints = './checkpoints/'

        label = f"image={use_image}"
        print(f"  Running: {label}")
        try:
            exp = ExpTrain(cfg)
            res = exp.train()
            results.append({'ablation': 'image_modality', 'setting': label, **res})
        except Exception as e:
            print(f"  Error: {e}")

    # 消融3: 架构增强 (纯时序对比)
    print("\n=== Ablation 3: Architecture Enhancement (ETTm2) ===")
    for model_name in ['iTransformer', 'KANiTransformer', 'MambaTransformerDual']:
        cfg = get_dataset_config('ETTm2', seq_len=96, pred_len=96)
        cfg.model = model_name
        cfg.train_epochs = 30
        cfg.checkpoints = './checkpoints/'

        print(f"  Running: {model_name}")
        try:
            exp = ExpTrain(cfg)
            res = exp.train()
            results.append({'ablation': 'architecture', 'setting': model_name, **res})
        except Exception as e:
            print(f"  Error: {e}")

    # 消融4: 损失函数
    print("\n=== Ablation 4: Loss Function (ETTm2) ===")
    for loss_type in ['MSE', 'MAE']:
        cfg = get_dataset_config('ETTm2', seq_len=96, pred_len=96)
        cfg.model = 'iTransformer'
        cfg.loss = loss_type
        cfg.train_epochs = 30
        cfg.checkpoints = './checkpoints/'

        print(f"  Running: loss={loss_type}")
        try:
            exp = ExpTrain(cfg)
            res = exp.train()
            results.append({'ablation': 'loss_function', 'setting': loss_type, **res})
        except Exception as e:
            print(f"  Error: {e}")

    # 消融5: 频域分解 (Weather 上比较 iTransformer vs KANiTransformer)
    print("\n=== Ablation 5: Frequency Decomposition (Weather) ===")
    for model_name in ['iTransformer', 'KANiTransformer']:
        cfg = get_dataset_config('Weather', seq_len=96, pred_len=96)
        cfg.model = model_name
        cfg.train_epochs = 30
        cfg.checkpoints = './checkpoints/'

        label = f"{model_name}"
        print(f"  Running: {label}")
        try:
            exp = ExpTrain(cfg)
            res = exp.train()
            results.append({'ablation': 'freq_decomp', 'setting': label, **res})
        except Exception as e:
            print(f"  Error: {e}")

    # 消融6: 集成策略
    print("\n=== Ablation 6: Ensemble (requires pre-trained models) ===")
    print("  Skipping — requires all models to be trained first.")
    results.append({
        'ablation': 'ensemble',
        'setting': 'pending',
        'status': 'requires_pretrained',
    })

    # 保存
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'results/ablation_{timestamp}.csv', index=False)
    print(f"\nAblation results saved. ({len(results)} experiments)")
    return df


if __name__ == '__main__':
    run_ablation(None)
