"""
批量实验运行器 — 8模型 × 6数据集 × 多预测长度

纯时序数据集: ETTm2, Weather, Electricity
多模态数据集: Energy, Environment, Health (Time-MMD, NeurIPS 2024)
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

# 实验网格
MODELS = ['DLinear', 'PatchTST', 'iTransformer', 'TimeMixer', 'TimeKAN',
          'KANiTransformer', 'MambaTransformerDual', 'MultimodalFusion']
DATASETS_PURE = ['ETTm2', 'Weather', 'Electricity']
DATASETS_MULTIMODAL = ['Energy', 'Environment', 'Health']
DATASETS = DATASETS_PURE + DATASETS_MULTIMODAL

# 多模态数据集使用较短的 seq_len（周频/日频，数据量较少）
SEQ_LENS = [96, 192]
PRED_LENS = [96, 192, 336, 720]

# 多模态数据集的预测长度（数据量小，不宜太长）
MM_SEQ_LENS = [24, 48]
MM_PRED_LENS = [12, 24, 48]


def run_single_experiment(model, dataset, seq_len, pred_len, epochs=100, gpu=0):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  {model} | {dataset} | H={seq_len} F={pred_len}")
    print(f"{'='*60}")

    try:
        config = get_dataset_config(dataset, seq_len=seq_len, pred_len=pred_len)
        config.model = model
        config.train_epochs = epochs
        config.gpu = gpu
        config.checkpoints = './checkpoints/'
        os.makedirs(config.checkpoints, exist_ok=True)

        if model == 'MultimodalFusion':
            config.use_text = True
            config.use_image = True
            config.use_contrastive = True

        fix_seed(config.seed)

        if model == 'Chronos2':
            from exp.exp_zero_shot import ExpZeroShot
            exp = ExpZeroShot(config)
            results = exp.test()
        else:
            exp = ExpTrain(config)
            results = exp.train()

        results['status'] = 'success'
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def run_all(models=None, datasets=None, epochs=100, gpu=0):
    """运行所有实验组合"""
    models = models or MODELS
    datasets = datasets or DATASETS

    all_results = []
    total = 0
    for model in models:
        for dataset in datasets:
            if dataset in DATASETS_MULTIMODAL:
                total += len(MM_SEQ_LENS) * len(MM_PRED_LENS)
            else:
                total += len(SEQ_LENS) * len(PRED_LENS)

    current = 0
    for model in models:
        for dataset in datasets:
            if dataset in DATASETS_MULTIMODAL:
                seq_lens, pred_lens = MM_SEQ_LENS, MM_PRED_LENS
            else:
                seq_lens, pred_lens = SEQ_LENS, PRED_LENS

            for seq_len in seq_lens:
                for pred_len in pred_lens:
                    current += 1
                    print(f"\n>>> [{current}/{total}] {model} | {dataset} | H={seq_len} F={pred_len}")

                    result = run_single_experiment(model, dataset, seq_len, pred_len, epochs, gpu)
                    result.update({
                        'model': model,
                        'dataset': dataset,
                        'seq_len': seq_len,
                        'pred_len': pred_len,
                    })
                    all_results.append(result)

    # 保存汇总结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'results/all_experiments_{timestamp}.csv', index=False)
    print(f"\nAll {total} experiments completed. Results saved.")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    run_all(
        models=args.models, datasets=args.datasets,
        epochs=args.epochs, gpu=args.gpu,
    )
