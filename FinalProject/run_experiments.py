"""
批量实验运行器 — FinalProject v2.0 三条实验主线

主线一: 全架构对比实验 (广度)
  模型: DLinear, PatchTST, TimesNet, Mamba
  数据: ETTm2, Weather, Electricity
  预测长度: {96, 192, 336, 720}

主线二: 自研模型深度评测 (深度)
  模型: KAN-iTransformer, Lite-SparseNet, SparseTSF, DLinear
  数据: ETTm2, Weather, Electricity, Environment
  预测长度: {96, 192, 336, 720}

主线三: 多模态有效性消融 (聚焦)
  模型: PatchTST, KAN-iTransformer, Lite-SparseNet
  数据: Environment only
  预测长度: {96, 192}
  文本模态: 5组消融设置

用法:
    python run_experiments.py --line 1              # 运行主线一
    python run_experiments.py --line 2              # 运行主线二
    python run_experiments.py --line 3              # 运行主线三
    python run_experiments.py --line all            # 运行全部
"""
import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(1, os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary'))

from configs.dataset_configs import get_dataset_config
from exp.exp_train import ExpTrain
from utils.tools import fix_seed


# ============================================================
# 实验网格定义
# ============================================================

# 主线一: 全架构对比 — 4个基线模型 × 3个纯时序数据集
LINE1_MODELS = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba']
LINE1_DATASETS = ['ETTm2', 'Weather', 'Electricity']
LINE1_PRED_LENS = [96, 192, 336, 720]

# 主线二: 自研模型深度评测 — 4个模型 × 4个数据集
LINE2_MODELS = ['KANiTransformer', 'LiteSparseNet', 'SparseTSF', 'DLinear']
LINE2_DATASETS = ['ETTm2', 'Weather', 'Electricity', 'Environment']
LINE2_PRED_LENS = [96, 192, 336, 720]

# 主线三: 多模态消融 — 3个模型 × Environment × 5组文本设置
LINE3_MODELS = ['PatchTST', 'KANiTransformer', 'LiteSparseNet']
LINE3_DATASETS = ['Environment']
LINE3_PRED_LENS = [96, 192]
LINE3_TEXT_MODES = [
    ('baseline', {}),                           # 组1: 纯数值
    ('report', {'use_text': True, 'text_mode': 'report_only'}),   # 组2: +报告
    ('search', {'use_text': True, 'text_mode': 'search_only'}),   # 组3: +搜索
    ('both_concat', {'use_text': True, 'text_mode': 'concat'}),   # 组4: 简单拼接
    ('both_gating', {'use_text': True, 'text_mode': 'gating'}),   # 组5: 门控融合
]


def run_single_experiment(model, dataset, seq_len, pred_len, extra_config=None,
                          epochs=100, gpu=0):
    """运行单个实验"""
    label = f"{model} | {dataset} | H={seq_len} F={pred_len}"
    if extra_config:
        label += f" | {extra_config.get('label', '')}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    try:
        config = get_dataset_config(dataset, seq_len=seq_len, pred_len=pred_len)
        config.model = model
        config.train_epochs = epochs
        config.gpu = gpu
        config.checkpoints = './checkpoints/'
        os.makedirs(config.checkpoints, exist_ok=True)

        # 应用额外配置（如文本模态设置）
        if extra_config:
            for key, value in extra_config.items():
                if hasattr(config, key) and key != 'label':
                    setattr(config, key, value)

        fix_seed(config.seed)
        exp = ExpTrain(config)
        results = exp.train()
        results['status'] = 'success'
        results['label'] = label
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e), 'label': label}


def run_line1(epochs=100, gpu=0, seq_len=96):
    """
    主线一: 全架构对比实验
    MLP vs Transformer vs CNN vs SSM
    """
    print("\n" + "=" * 70)
    print("  主线一: 全架构对比实验")
    print("  模型: DLinear, PatchTST, TimesNet, Mamba")
    print("  数据: ETTm2, Weather, Electricity")
    print("=" * 70)

    models, datasets, pred_lens = LINE1_MODELS, LINE1_DATASETS, LINE1_PRED_LENS

    total = len(models) * len(datasets) * len(pred_lens)
    all_results = []
    current = 0

    for model in models:
        for dataset in datasets:
            for pred_len in pred_lens:
                current += 1
                print(f"\n>>> [{current}/{total}]", end=' ')

                result = run_single_experiment(
                    model, dataset, seq_len, pred_len, epochs=epochs, gpu=gpu
                )
                result.update({
                    'model': model, 'dataset': dataset,
                    'seq_len': seq_len, 'pred_len': pred_len,
                    'line': 1,
                })
                all_results.append(result)

    return _save_results(all_results, 'line1')


def run_line2(epochs=100, gpu=0, seq_len=96):
    """
    主线二: 自研模型深度评测
    KAN-iTransformer vs Lite-SparseNet vs SparseTSF vs DLinear
    """
    print("\n" + "=" * 70)
    print("  主线二: 自研模型深度评测")
    print("  模型: KANiTransformer, LiteSparseNet, SparseTSF, DLinear")
    print("  数据: ETTm2, Weather, Electricity, Environment")
    print("=" * 70)

    models, datasets, pred_lens = LINE2_MODELS, LINE2_DATASETS, LINE2_PRED_LENS

    total = len(models) * len(datasets) * len(pred_lens)
    all_results = []
    current = 0

    for model in models:
        for dataset in datasets:
            for pred_len in pred_lens:
                current += 1
                print(f"\n>>> [{current}/{total}]", end=' ')

                # Environment 上 KANiTransformer 开启概率输出
                extra = {}
                if model == 'KANiTransformer' and dataset == 'Environment':
                    extra = {'use_probabilistic': True}

                result = run_single_experiment(
                    model, dataset, seq_len, pred_len,
                    extra_config=extra if extra else None,
                    epochs=epochs, gpu=gpu,
                )
                result.update({
                    'model': model, 'dataset': dataset,
                    'seq_len': seq_len, 'pred_len': pred_len,
                    'line': 2,
                })
                all_results.append(result)

    return _save_results(all_results, 'line2')


def run_line3(epochs=100, gpu=0, seq_len=96):
    """
    主线三: 多模态有效性消融
    3个模型 × Environment × 5组文本设置
    """
    print("\n" + "=" * 70)
    print("  主线三: 多模态有效性消融")
    print("  模型: PatchTST, KANiTransformer, LiteSparseNet")
    print("  数据: Environment (含文本模态)")
    print("  文本设置: 纯数值 / +报告 / +搜索 / 简单拼接 / 门控融合")
    print("=" * 70)

    models = LINE3_MODELS
    all_results = []
    current = 0
    total = len(models) * len(LINE3_TEXT_MODES) * len(LINE3_PRED_LENS)

    for model in models:
        for text_label, text_config in LINE3_TEXT_MODES:
            for pred_len in LINE3_PRED_LENS:
                current += 1
                print(f"\n>>> [{current}/{total}]", end=' ')

                extra = {'label': text_label, **text_config}
                result = run_single_experiment(
                    model, 'Environment', seq_len, pred_len,
                    extra_config=extra,
                    epochs=epochs, gpu=gpu,
                )
                result.update({
                    'model': model, 'dataset': 'Environment',
                    'seq_len': seq_len, 'pred_len': pred_len,
                    'text_mode': text_label, 'line': 3,
                })
                all_results.append(result)

    return _save_results(all_results, 'line3')


def _save_results(all_results, name):
    """保存实验结果"""
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    filepath = f'results/{name}_{timestamp}.csv'
    df.to_csv(filepath, index=False)

    # 统计
    n_success = sum(1 for r in all_results if r.get('status') == 'success')
    print(f"\n{'='*60}")
    print(f"  {name}: {n_success}/{len(all_results)} 成功")
    print(f"  结果保存至: {filepath}")
    print(f"{'='*60}")
    return df


# ============================================================
# 快捷入口
# ============================================================

def run_all_lines(epochs=100, gpu=0, seq_len=96):
    """运行全部三条实验主线"""
    df1 = run_line1(epochs, gpu, seq_len)
    df2 = run_line2(epochs, gpu, seq_len)
    df3 = run_line3(epochs, gpu, seq_len)
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KAN-iTransformer v2.0 Batch Experiments')
    parser.add_argument('--line', type=str, default='1',
                       choices=['1', '2', '3', 'all'],
                       help='Which experiment line to run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=96)
    args = parser.parse_args()

    if args.line == '1':
        run_line1(epochs=args.epochs, gpu=args.gpu, seq_len=args.seq_len)
    elif args.line == '2':
        run_line2(epochs=args.epochs, gpu=args.gpu, seq_len=args.seq_len)
    elif args.line == '3':
        run_line3(epochs=args.epochs, gpu=args.gpu, seq_len=args.seq_len)
    elif args.line == 'all':
        run_all_lines(epochs=args.epochs, gpu=args.gpu, seq_len=args.seq_len)
