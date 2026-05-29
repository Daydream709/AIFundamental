"""
热力图 — 模型×数据集×预测长度性能矩阵
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def plot_performance_heatmap(results_csv, metric='MSE', save_dir='figures/'):
    """绘制模型×(数据集+预测长度)的热力图"""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(results_csv)
    if df.empty:
        print("No results to plot.")
        return

    # 创建数据集+预测长度的组合标签
    df['setting'] = df['dataset'] + '\nF=' + df['pred_len'].astype(str)

    # 透视表
    pivot = df.pivot_table(index='model', columns='setting', values=metric, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 2), max(6, len(pivot.index) * 0.8)))

    if metric in ['MSE', 'MAE', 'RMSE']:
        cmap = 'RdYlGn_r'  # 越低越好
    else:
        cmap = 'RdYlGn'    # 越高越好

    sns.heatmap(pivot, annot=True, fmt='.4f', cmap=cmap, ax=ax,
                linewidths=0.5, cbar_kws={'label': metric})

    ax.set_title(f'{metric} Performance Heatmap', fontsize=14)
    ax.set_xlabel('Dataset + Prediction Length')
    ax.set_ylabel('Model')

    plt.tight_layout()
    path = os.path.join(save_dir, f'heatmap_{metric}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
