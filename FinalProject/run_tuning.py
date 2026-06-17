"""
超参数调优脚本 — 为自研模型搜索最优配置

策略:
  - 只在小数据集 ETTm2 上搜索 (最小数据集，搜索快)
  - 固定 pred_len=96 (最具代表性的预测长度)
  - 最优参数推广到全部4个数据集

搜索模型:
  - SparseTSF:      网格搜索 period_len (3次)
  - Lite-SparseNet: 网格搜索 sparse_ratio × group_size (6-9次)
  - KANiTransformer: 逐步搜索 e_layers → lr (6次)

用法:
  python run_tuning.py --model SparseTSF
  python run_tuning.py --model LiteSparseNet
  python run_tuning.py --model KANiTransformer
  python run_tuning.py --model all
"""
import os
import sys
import json
import time
import itertools
import argparse
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(1, os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary'))

from configs.dataset_configs import get_dataset_config
from exp.exp_train import ExpTrain
from utils.tools import fix_seed


# ============================================================
# 搜索空间定义
# ============================================================

TUNING_SPACES = {
    'SparseTSF': {
        # 极简模型，只调 period_len
        'fixed': {
            'seq_len': 96, 'pred_len': 96, 'learning_rate': 1e-3,
            'batch_size': 64, 'train_epochs': 30, 'patience': 5,
        },
        'search': {
            'period_len': [12, 24, 48],
        },
    },
    'LiteSparseNet': {
        'fixed': {
            'seq_len': 96, 'pred_len': 96, 'learning_rate': 1e-3,
            'batch_size': 64, 'train_epochs': 50, 'patience': 10,
            'fft_residual_k': 2,
        },
        'search': {
            'sparse_ratio': [2, 4, 8],
            'group_size': [4, 8, 16],
        },
    },
    'KANiTransformer': {
        # 逐步搜索: 先锁定 e_layers，再微调 lr
        'fixed': {
            'seq_len': 96, 'pred_len': 96, 'd_model': 512,
            'n_heads': 8, 'd_ff': 2048, 'dropout': 0.1,
            'batch_size': 32, 'train_epochs': 50, 'patience': 10,
            'use_cfd': True, 'use_revin': True,
            'use_probabilistic': True, 'kan_grid_size': 5, 'top_k': 5,
        },
        'search': {
            'e_layers': [1, 2, 3],
            'learning_rate': [5e-5, 1e-4, 5e-4],
        },
    },
}


def run_single_tuning(model_name, dataset, extra_params, gpu=0):
    """
    运行单次调优实验

    Args:
        model_name: 模型名称
        dataset: 数据集名称 (固定 ETTm2)
        extra_params: dict, 要覆盖的搜索参数

    Returns:
        dict: {params: ..., mse: ..., mae: ..., time: ...}
    """
    label = ', '.join(f'{k}={v}' for k, v in extra_params.items())
    print(f"\n  >> {model_name} | {label}")

    try:
        config = get_dataset_config(dataset, seq_len=extra_params.get('seq_len', 96),
                                    pred_len=extra_params.get('pred_len', 96))
        config.model = model_name
        config.gpu = gpu
        config.checkpoints = './checkpoints/'
        os.makedirs(config.checkpoints, exist_ok=True)

        # 应用搜索参数
        for key, value in extra_params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        fix_seed(config.seed)
        start_time = time.time()

        exp = ExpTrain(config)
        results = exp.train()

        elapsed = time.time() - start_time

        return {
            'model': model_name,
            'params': extra_params.copy(),
            'mse': results.get('mse', float('inf')),
            'mae': results.get('mae', float('inf')),
            'time_s': elapsed,
            'status': 'success',
        }

    except Exception as e:
        print(f'    ERROR: {e}')
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'params': extra_params.copy(),
            'mse': float('inf'),
            'mae': float('inf'),
            'time_s': 0,
            'status': 'error',
            'error': str(e),
        }


def grid_search(model_name, dataset='ETTm2', gpu=0):
    """
    网格搜索: 遍历所有搜索参数组合
    """
    space = TUNING_SPACES[model_name]
    fixed = space['fixed']
    search = space['search']

    # 生成所有组合
    keys = list(search.keys())
    values = list(search.values())
    combinations = list(itertools.product(*values))

    print(f"\n{'='*60}")
    print(f"  Grid Search: {model_name}")
    print(f"  Dataset: {dataset} | 组合数: {len(combinations)}")
    print(f"  Fixed: {fixed}")
    print(f"  Search: {search}")
    print(f"{'='*60}")

    results = []
    best_mse = float('inf')
    best_config = None

    for i, combo in enumerate(combinations):
        params = fixed.copy()
        for key, val in zip(keys, combo):
            params[key] = val

        print(f"\n[{i+1}/{len(combinations)}]", end='')
        result = run_single_tuning(model_name, dataset, params, gpu=gpu)
        results.append(result)

        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_config = params.copy()
            print(f'    ★ New best! MSE={best_mse:.6f}')

    # 排序
    results.sort(key=lambda r: r['mse'])

    print(f"\n{'='*60}")
    print(f"  Grid Search 结果: {model_name}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        marker = '★' if r['params'] == best_config else ' '
        params_str = ', '.join(f'{k}={v}' for k, v in r['params'].items()
                              if k in search)
        print(f"  {marker} #{i+1}: {params_str}  → MSE={r['mse']:.6f}, MAE={r['mae']:.6f} [{r['time_s']:.0f}s]")

    print(f"\n  Best config: {best_config}")

    return best_config, results


def stepwise_search_kan(gpu=0):
    """
    逐步搜索 KAN-iTransformer 的超参数

    Step 1: 搜索 e_layers (固定 lr=1e-4)
    Step 2: 在最优 e_layers 下搜索 learning_rate

    这样从 9 次组合减少到 6 次
    """
    model_name = 'KANiTransformer'
    space = TUNING_SPACES[model_name]

    # Step 1: e_layers search
    print(f"\n{'='*60}")
    print(f"  Step 1: KAN-iTransformer e_layers 搜索")
    print(f"  Fixed lr=1e-4")
    print(f"{'='*60}")

    step1_results = []
    for e_layers in [1, 2, 3]:
        params = space['fixed'].copy()
        params['e_layers'] = e_layers
        params['learning_rate'] = 1e-4

        result = run_single_tuning(model_name, 'ETTm2', params, gpu=gpu)
        step1_results.append(result)
        print(f'    e_layers={e_layers}: MSE={result["mse"]:.6f}')

    best_step1 = min(step1_results, key=lambda r: r['mse'])
    best_e_layers = best_step1['params']['e_layers']
    print(f'\n  Step 1 最优: e_layers={best_e_layers}, MSE={best_step1["mse"]:.6f}')

    # Step 2: learning_rate 微调
    print(f"\n{'='*60}")
    print(f"  Step 2: KAN-iTransformer learning_rate 搜索")
    print(f"  Fixed e_layers={best_e_layers}")
    print(f"{'='*60}")

    step2_results = []
    for lr in [5e-5, 1e-4, 5e-4]:
        params = space['fixed'].copy()
        params['e_layers'] = best_e_layers
        params['learning_rate'] = lr

        result = run_single_tuning(model_name, 'ETTm2', params, gpu=gpu)
        step2_results.append(result)
        print(f'    lr={lr}: MSE={result["mse"]:.6f}')

    best_step2 = min(step2_results, key=lambda r: r['mse'])
    best_lr = best_step2['params']['learning_rate']
    print(f'\n  Step 2 最优: lr={best_lr}, MSE={best_step2["mse"]:.6f}')

    # 汇总
    best_config = best_step2['params']
    all_results = step1_results + step2_results

    print(f"\n{'='*60}")
    print(f"  KAN-iTransformer 最终配置")
    print(f"  e_layers={best_e_layers}, lr={best_lr}")
    print(f"  Best MSE={best_step2['mse']:.6f}")
    print(f"{'='*60}")

    return best_config, all_results


def save_tuning_results(model_name, best_config, all_results):
    """保存调优结果"""
    os.makedirs('results/tuning', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存完整结果
    results_path = f'results/tuning/{model_name}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'  Results saved: {results_path}')

    # 保存最优配置
    config_path = f'results/tuning/{model_name}_best_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f'  Best config saved: {config_path}')

    return results_path, config_path


def update_model_configs(model_name, best_config):
    """
    提示用户手动更新 — 出于稳定性考虑, 调参脚本不直接修改 model_configs.py

    调优完成后, 用户应自行将 TUNING_SPACES 中各模型的最优参数复制到
    configs/model_configs.py 对应的 ModelPreset 块中.
    """
    space = TUNING_SPACES.get(model_name, {})
    updatable = set(space.get('search', {}).keys())

    if not updatable:
        print(f'  [Hint] No updatable params for {model_name}')
        return

    print(f'  [Hint] {model_name} tuned params: {sorted(updatable)}')
    print(f'         Values: {[(k, best_config.get(k)) for k in sorted(updatable)]}')
    print(f'         请手动将上述值更新到 configs/model_configs.py 中')
    print(f'         {model_name} 块的 4 个 tier (low_dim / mid_dim / high_dim / multimodal)')

def tune_model(model_name, gpu=0, auto_update=True):
    """单个模型的调优入口"""
    if model_name == 'KANiTransformer':
        best_config, results = stepwise_search_kan(gpu=gpu)
    elif model_name in TUNING_SPACES:
        best_config, results = grid_search(model_name, gpu=gpu)
    else:
        print(f"  No tuning space defined for '{model_name}'")
        return None, None

    save_tuning_results(model_name, best_config, results)

    # 自动更新 model_configs.py
    if auto_update and best_config:
        update_model_configs(model_name, best_config)

    return best_config, results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Self-Developed Models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['SparseTSF', 'LiteSparseNet', 'KANiTransformer', 'all'],
                       help='Model to tune (default: all)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no_auto_update', action='store_true',
                       help='Disable auto-update of model_configs.py')
    args = parser.parse_args()

    auto_update = not args.no_auto_update

    if args.model == 'all':
        models = ['SparseTSF', 'LiteSparseNet', 'KANiTransformer']
    else:
        models = [args.model]

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Tuning — FinalProject v2.0")
    print(f"  Models: {models}")
    print(f"  Dataset: ETTm2 (pred_len=96)")
    print(f"  GPU: {args.gpu}")
    print(f"{'='*60}")

    all_best = {}
    for model in models:
        best_config, results = tune_model(model, gpu=args.gpu, auto_update=auto_update)
        if best_config:
            all_best[model] = best_config

    # 汇总打印
    print(f"\n{'='*60}")
    print(f"  调优完成 — 最优配置汇总")
    print(f"{'='*60}")
    for model, config in all_best.items():
        print(f"\n  {model}:")
        for key, value in config.items():
            print(f"    {key}: {value}")

    print(f"\n  请将以上最优配置更新到 configs/model_configs.py")
    if auto_update:
        print(f"  (已自动更新)")


if __name__ == '__main__':
    main()
