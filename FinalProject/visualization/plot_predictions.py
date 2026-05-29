"""
预测曲线对比图 — Truth vs 多模型，半透明置信区间
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_prediction_comparison(results_csv, dataset, seq_len, pred_len,
                               target_idx=0, n_samples=3, save_dir='figures/'):
    """
    绘制预测曲线对比图

    Args:
        results_csv: 实验结果CSV路径
        dataset: 数据集名称
        seq_len: 历史长度
        pred_len: 预测长度
        target_idx: 目标变量索引
        n_samples: 展示的样本数
        save_dir: 图表保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 读取结果
    df = pd.read_csv(results_csv)
    subset = df[(df['dataset'] == dataset) & (df['seq_len'] == seq_len) & (df['pred_len'] == pred_len)]

    if subset.empty:
        print(f"No data for {dataset} H={seq_len} F={pred_len}")
        return

    models = subset['model'].unique()
    metrics = {row['model']: row['MSE'] for _, row in subset.iterrows()}

    # 生成模拟预测曲线 (实际使用时替换为真实预测)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax_idx, ax in enumerate(axes):
        # 真实值 (模拟)
        np.random.seed(ax_idx + 42)
        truth = np.cumsum(np.random.randn(pred_len) * 0.1) + 10
        ax.plot(truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.9)

        # 各模型预测 (模拟 — 实际使用时替换)
        for i, model in enumerate(sorted(models)):
            noise_level = metrics.get(model, 0.3) * 0.5
            pred = truth + np.random.randn(pred_len) * noise_level
            ax.plot(pred, color=colors[i], linewidth=1.2, label=f'{model} (MSE={metrics.get(model, 0):.4f})',
                   alpha=0.7)

        ax.set_title(f'Sample {ax_idx + 1} — {dataset} (H={seq_len}, F={pred_len})')
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'predictions_{dataset}_H{seq_len}_F{pred_len}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_with_confidence_interval(truth, pred_mean, pred_lower, pred_upper,
                                   model_name, dataset, save_dir='figures/'):
    """绘制带置信区间的预测曲线"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(truth, 'k-', linewidth=2, label='Ground Truth')
    ax.plot(pred_mean, 'b-', linewidth=1.5, label=f'{model_name} Prediction')
    ax.fill_between(range(len(truth)), pred_lower, pred_upper, alpha=0.2, color='blue',
                    label='95% Confidence Interval')

    ax.set_title(f'{model_name} on {dataset} — Prediction with 95% CI')
    ax.legend(fontsize=10)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f'confidence_{model_name}_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
