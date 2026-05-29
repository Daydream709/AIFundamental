"""
注意力/Mamba状态可视化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def plot_attention_heatmap(attn_weights, model_name, layer_idx=0, head_idx=0,
                           save_dir='figures/'):
    """
    绘制注意力权重热力图

    Args:
        attn_weights: [n_heads, L, L] 或 [L, L]
    """
    os.makedirs(save_dir, exist_ok=True)

    if attn_weights.ndim == 3:
        attn = attn_weights[head_idx]
    else:
        attn = attn_weights

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(attn, cmap='YlOrRd', ax=ax, square=True,
                cbar_kws={'label': 'Attention Weight'})
    ax.set_title(f'{model_name} Attention (Layer {layer_idx}, Head {head_idx})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    path = os.path.join(save_dir, f'attention_{model_name}_L{layer_idx}_H{head_idx}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mamba_states(hidden_states, model_name, save_dir='figures/'):
    """
    绘制 Mamba 隐状态随时间变化的可视化

    Args:
        hidden_states: [L, d_model]
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    # 只画前 20 个维度
    n_show = min(20, hidden_states.shape[1])
    for i in range(n_show):
        ax.plot(hidden_states[:, i], alpha=0.6, linewidth=0.8)

    ax.set_title(f'{model_name} — Mamba Hidden States Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Activation')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f'mamba_states_{model_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
