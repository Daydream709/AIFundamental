"""
效率对比图 — 参数量 / FLOPs / 速度 柱状图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_efficiency_comparison(results_csv, save_dir='figures/'):
    """绘制模型效率对比柱状图"""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(results_csv)
    if df.empty:
        print("No results to plot.")
        return

    # 每个模型取平均
    avg = df.groupby('model').agg({
        'Params(M)': 'mean',
        'InferTime(ms)': 'mean',
        'MSE': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = avg['model'].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # 参数量
    axes[0].barh(models, avg['Params(M)'], color=colors)
    axes[0].set_xlabel('Parameters (M)')
    axes[0].set_title('Model Size')
    axes[0].invert_yaxis()

    # 推理时间
    axes[1].barh(models, avg['InferTime(ms)'], color=colors)
    axes[1].set_xlabel('Inference Time (ms)')
    axes[1].set_title('Inference Speed')
    axes[1].invert_yaxis()

    # 精度
    axes[2].barh(models, avg['MSE'], color=colors)
    axes[2].set_xlabel('MSE')
    axes[2].set_title('Prediction Accuracy')
    axes[2].invert_yaxis()

    plt.suptitle('Model Efficiency Comparison', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'efficiency_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
