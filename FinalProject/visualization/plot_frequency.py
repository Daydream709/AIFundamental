"""
频域分解可视化 — 原始+趋势+季节+残差
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_frequency_decomposition(series, trend, seasonal, residual,
                                  model_name='KAN-iTransformer', save_dir='figures/'):
    """
    绘制频域分解结果

    Args:
        series: 原始时序 [L]
        trend: 趋势分量 [L]
        seasonal: 季节分量 [L]
        residual: 残差分量 [L]
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(series, 'k-', linewidth=1.2)
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trend, 'b-', linewidth=1.5)
    axes[1].set_title('Trend (Low Frequency)')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(seasonal, 'g-', linewidth=1)
    axes[2].set_title('Seasonal (Mid Frequency)')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(residual, 'r-', linewidth=0.8, alpha=0.7)
    axes[3].set_title('Residual (High Frequency)')
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('Value')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} — Adaptive Frequency Decomposition', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f'freq_decomp_{model_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_fft_spectrum(series, title='FFT Spectrum', save_dir='figures/'):
    """绘制FFT频谱图"""
    os.makedirs(save_dir, exist_ok=True)

    fft_vals = np.fft.rfft(series)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(series))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(freqs[1:], fft_mag[1:], width=0.002, color='steelblue', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f'fft_{title.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
