"""
消融实验图表 — 分组柱状图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_ablation_results(ablation_csv, save_dir='figures/'):
    """绘制消融实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(ablation_csv)
    if df.empty:
        print("No ablation results to plot.")
        return

    groups = df['ablation'].unique()
    n_groups = len(groups)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, 10))

    for idx, group in enumerate(groups):
        ax = axes[idx]
        group_df = df[df['ablation'] == group]

        settings = group_df['setting'].values
        mse_vals = group_df['mse'].values if 'mse' in group_df.columns else group_df.get('MSE', 0).values

        bars = ax.bar(range(len(settings)), mse_vals, color=colors[:len(settings)])
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(settings, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('MSE')
        ax.set_title(f'Ablation: {group}', fontsize=11)

        # 标注数值
        for bar, val in zip(bars, mse_vals):
            if isinstance(val, (int, float)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        ax.grid(axis='y', alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Ablation Study Results', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'ablation_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
