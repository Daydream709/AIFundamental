"""
Generate all figures for experiment-report.md.

Output: docs/figures/*.png
"""
import sys, os
sys.path.insert(0, '/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

# 强制注册 macOS 中文字体（用 fontManager 直接 addfont）
for fpath in [
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
]:
    if os.path.exists(fpath):
        try:
            font_manager.fontManager.addfont(fpath)
        except Exception as e:
            print(f'  warn: addfont {fpath} failed: {e}')

# Style - 优先 Times New Roman (英文) + Hiragino/STHeiti (中文)
plt.rcParams.update({
    'font.family': ['Times New Roman', 'Hiragino Sans GB', 'STHeiti', 'STSong', 'Arial Unicode MS', 'sans-serif'],
    'font.sans-serif': ['Times New Roman', 'Hiragino Sans GB', 'STHeiti', 'STSong', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Colors
C_PRIMARY = '#2E5C8A'
C_ACCENT = '#E67E22'
C_GREEN = '#27AE60'
C_RED = '#C0392B'
C_PURPLE = '#8E44AD'
C_GRAY = '#7F8C8D'
MODEL_COLORS = {
    'DLinear': '#2E5C8A',
    'PatchTST': '#E67E22',
    'TimesNet': '#27AE60',
    'Mamba': '#C0392B',
    'KANiTransformer': '#8E44AD',
    'LiteSparseNet': '#16A085',
    'SparseTSF': '#D35400',
}

OUT = '/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/docs/figures'
os.makedirs(OUT, exist_ok=True)

# Load data
l1 = pd.read_csv('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/results/line1_latest.csv')
l2 = pd.read_csv('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/results/line2_latest.csv')
eff = pd.read_csv('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/results/efficiency/flops_params_summary_v3.csv')
l3sparse = pd.read_csv('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/results/line3_sparsetsf_latest.csv')
all_data = pd.concat([l1, l2], ignore_index=True)


# ============================================================
# 图 1 (5.1 开头): 主实验热力图 - 4 模型 × 3 数据集 × 4 pred_len
# ============================================================
def fig1_main_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={'wspace': 0.35})
    datasets = ['ETTm2', 'Weather', 'Electricity']
    for ax, ds in zip(axes, datasets):
        # 4 模型 × 4 pred_len
        models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba']
        pred_lens = [96, 192, 336, 720]
        matrix = np.zeros((len(models), len(pred_lens)))
        for i, m in enumerate(models):
            for j, pl in enumerate(pred_lens):
                row = all_data[(all_data['model']==m) & (all_data['dataset']==ds) & (all_data['pred_len']==pl)]
                matrix[i, j] = row['MSE'].values[0] if len(row) > 0 else np.nan

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pred_lens)))
        ax.set_xticklabels([f'F={pl}' for pl in pred_lens])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(f'{ds} (C={7 if ds=="ETTm2" else 21 if ds=="Weather" else 321})', pad=12)

        # Add value labels
        for i in range(len(models)):
            for j in range(len(pred_lens)):
                val = matrix[i, j]
                color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05, label='MSE')

    fig.suptitle('基线模型主实验结果 (4 模型 × 3 数据集 × 4 pred_len)', fontsize=14, fontweight='bold', y=1.04)
    plt.savefig(f'{OUT}/fig1_baseline_heatmap.png')
    plt.close()
    print('✓ fig1')


# ============================================================
# 图 2 (5.1.4): 退化率折线图
# ============================================================
def fig2_degradation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    datasets = ['ETTm2', 'Weather', 'Electricity']
    models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']
    pred_lens = [96, 192, 336, 720]

    for ax, ds in zip(axes, datasets):
        for m in models:
            mses = []
            for pl in pred_lens:
                row = all_data[(all_data['model']==m) & (all_data['dataset']==ds) & (all_data['pred_len']==pl)]
                mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
            if not np.isnan(mses[0]):
                # 归一化到 F=96 = 1.0
                normalized = [m / mses[0] for m in mses]
                ax.plot(pred_lens, normalized, 'o-', label=m, color=MODEL_COLORS.get(m, C_GRAY), linewidth=1.5, markersize=5)

        ax.set_xlabel('pred_len')
        ax.set_ylabel('MSE (归一化到 F=96)')
        ax.set_title(f'{ds} - 退化曲线 (值越高=退化越严重)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)

    fig.suptitle('F=96→720 的退化率对比 (7 模型 × 3 数据集)', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{OUT}/fig2_degradation.png')
    plt.close()
    print('✓ fig2')


# ============================================================
# 图 3 (5.1.4): 跨数据集一致性 (CV) 柱状图
# ============================================================
def fig3_cv():
    fig, ax = plt.subplots(figsize=(10, 5))
    models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']
    cvs = []
    means = []
    for m in models:
        mses = []
        for ds in ['ETTm2', 'Weather', 'Electricity']:
            row = all_data[(all_data['model']==m) & (all_data['dataset']==ds) & (all_data['pred_len']==96)]
            if len(row) > 0:
                mses.append(row['MSE'].values[0])
        mses = np.array(mses)
        cv = mses.std() / mses.mean() * 100
        cvs.append(cv)
        means.append(mses.mean())

    # Sort by CV
    sorted_idx = np.argsort(cvs)
    models_sorted = [models[i] for i in sorted_idx]
    cvs_sorted = [cvs[i] for i in sorted_idx]
    colors_sorted = [MODEL_COLORS[m] for m in models_sorted]

    bars = ax.barh(range(len(models)), cvs_sorted, color=colors_sorted)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models_sorted)
    ax.set_xlabel('CV (%) = std/mean × 100, 越小越稳定')
    ax.set_title('跨数据集一致性 (F=96, 3 个数据集上的 MSE 变异系数)')

    # Add value labels
    for i, (bar, cv) in enumerate(zip(bars, cvs_sorted)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cv:.1f}%', va='center', fontsize=9)

    # Highlight best
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(2)
    ax.text(0.02, 0.98, '← 最稳定', transform=ax.transAxes, fontsize=9, color='gold', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_cv_consistency.png')
    plt.close()
    print('✓ fig3')


# ============================================================
# 图 4 (5.1.4): 架构×数据集适配矩阵
# ============================================================
def fig4_arch_dataset():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    architectures = ['Linear', 'Channel-Indep\nTransformer', '2D-CNN', 'SSM', 'KAN+iTransformer']
    datasets = ['ETTm2\n(低维强周期)', 'Weather\n(中维弱相关)', 'Electricity\n(高维多周期)']

    # 简化版: 用文字标注适配性（不是数字）
    matrix_text = [
        ['✓✓✓ DLinear', '✓ DLinear', '✓ DLinear'],
        ['✓ PatchTST', '✓✓✓ PatchTST', '✓✓✓ PatchTST'],
        ['✓ TimesNet', '✓✓ TimesNet', '✓ TimesNet'],
        ['△ Mamba', '✗ Mamba', '✓✓ Mamba'],
        ['△ KAN-iTF', '✓✓✓ KAN-iTF', '✓✓✓ KAN-iTF'],
    ]

    # 颜色编码
    cell_colors = np.zeros((5, 3))
    mapping = {'✓✓✓': 3, '✓✓': 2, '✓': 1, '△': 0, '✗': -1}
    for i, row in enumerate(matrix_text):
        for j, cell in enumerate(row):
            for sym, score in mapping.items():
                if sym in cell:
                    cell_colors[i, j] = score
                    break

    im = ax.imshow(cell_colors, cmap='RdYlGn', vmin=-1, vmax=3, aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(5))
    ax.set_yticklabels(architectures)

    for i in range(5):
        for j in range(3):
            ax.text(j, i, matrix_text[i][j], ha='center', va='center', fontsize=9)

    ax.set_title('架构 × 数据集 适配矩阵 (✓✓✓=最佳, ✓✓=良好, ✓=可用, △=勉强, ✗=差)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_arch_dataset.png')
    plt.close()
    print('✓ fig4')


# ============================================================
# 图 5 (5.2.4): 创新模型 vs 最佳基线 分组柱状图
# ============================================================
def fig5_innovative_vs_baseline():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    best_baseline = {'ETTm2': 0.0993, 'Weather': 0.1602, 'Electricity': 0.1618}
    models = ['KANiTransformer', 'LiteSparseNet', 'SparseTSF']
    datasets = ['ETTm2', 'Weather', 'Electricity']

    x = np.arange(len(datasets))
    width = 0.25
    for i, m in enumerate(models):
        gaps = []
        for ds in datasets:
            row = all_data[(all_data['model']==m) & (all_data['dataset']==ds) & (all_data['pred_len']==96)]
            mse = row['MSE'].values[0] if len(row) > 0 else np.nan
            gap = (mse / best_baseline[ds] - 1) * 100
            gaps.append(gap)
        bars = ax.bar(x + i*width, gaps, width, label=m, color=MODEL_COLORS[m])
        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if gap > 0 else -3),
                    f'{gap:+.1f}%', ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{ds}\n(best={best_baseline[ds]:.4f})' for ds in datasets])
    ax.set_ylabel('相对最佳基线的 MSE 差距 (%)')
    ax.set_title('创新模型 vs 最佳基线 (F=96, 负值=超越基线)')
    ax.legend(loc='upper right')

    # Add 0 line annotation
    ax.text(0.02, 0.05, '↓ 0% 基准线', transform=ax.transAxes, fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_innovative_vs_baseline.png')
    plt.close()
    print('✓ fig5')


# ============================================================
# 图 6 (5.3.1): 参数量对比 (4 数据集)
# ============================================================
def fig6_params_comparison():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'SparseTSF', 'KANiTransformer', 'LiteSparseNet']
    datasets = ['ETTm2', 'Weather', 'Electricity', 'Environment']

    x = np.arange(len(models))
    width = 0.2

    for i, ds in enumerate(datasets):
        params = []
        for m in models:
            row = eff[(eff['model']==m) & (eff['dataset']==ds)]
            params.append(row['params_M'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i*width, params, width, label=ds)

    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('参数量 (M)')
    ax.set_yscale('log')
    ax.set_title('7 模型 × 4 数据集 的参数量对比 (对数坐标)')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_params_comparison.png')
    plt.close()
    print('✓ fig6')


# ============================================================
# 图 7 (5.3.3): Pareto 散点图
# ============================================================
def fig7_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    # 数据 (Infer, MSE) 从 line1/line2 CSV
    models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']
    for ax, ds in zip(axes, ['ETTm2', 'Weather']):
        x, y, sizes, labels = [], [], [], []
        for m in models:
            row = all_data[(all_data['model']==m) & (all_data['dataset']==ds) & (all_data['pred_len']==96)]
            if len(row) > 0:
                x.append(row['InferTime(ms)'].values[0])
                y.append(row['MSE'].values[0])
                # 气泡大小 = Params (M) * 50
                p_row = eff[(eff['model']==m) & (eff['dataset']==ds)]
                p = p_row['params_M'].values[0] if len(p_row) > 0 else 1
                sizes.append(p * 30 + 30)
                labels.append(m)

        for xi, yi, s, l in zip(x, y, sizes, labels):
            ax.scatter(xi, yi, s=s, alpha=0.6, color=MODEL_COLORS.get(l, C_GRAY), edgecolors='black', linewidth=0.5)
            ax.annotate(l, (xi, yi), xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('推理时间 (ms)')
        ax.set_ylabel('MSE (F=96)')
        ax.set_title(f'{ds} - 推理速度 vs 精度 (气泡=参数量)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Pareto 前沿: 不同数据集的最优效率-精度权衡点', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_pareto.png')
    plt.close()
    print('✓ fig7')


# ============================================================
# 图 8 (5.4.2): 多模态结果柱状图
# ============================================================
def fig8_multimodal():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    modes = ['baseline', 'report', 'search', 'both_concat']
    mode_labels = ['baseline\n(纯时序)', 'report\n(单日报告)', 'search\n(周范围搜索)', 'both_concat\n(拼接)']
    colors_modes = [C_GRAY, C_PRIMARY, C_ACCENT, C_PURPLE]

    pred_lens = [96, 192]
    x = np.arange(len(modes))
    width = 0.35

    for i, pl in enumerate(pred_lens):
        mses = []
        for mode in modes:
            row = l3sparse[(l3sparse['text_mode']==mode) & (l3sparse['pred_len']==pl)]
            mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + i*width, mses, width, label=f'pred_len={pl}', alpha=0.85)
        for bar, mse in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{mse:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel('MSE')
    ax.set_title('多模态融合: 4 种 text_mode × 2 pred_len')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add baseline improvement annotation
    bl_96 = mses[0] if mses else 0
    search_96 = mses[2] if len(mses) > 2 else 0
    if bl_96 > 0 and search_96 > 0:
        improvement = (1 - search_96/bl_96) * 100
        ax.text(0.98, 0.95, f'search 改善 (F=96): {improvement:.1f}%\nsearch 改善 (F=192): {(1-mses[2]/mses[0])*100:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig8_multimodal.png')
    plt.close()
    print('✓ fig8')


# ============================================================
# 图 9 (5.5): 消融实验柱状图
# ============================================================
def fig9_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={'wspace': 0.25})

    # KAN 消融
    ax = axes[0]
    configs = ['A0\n完整', 'A1\nw/o 频域', 'A2\nw/o 概率', 'A3\nw/o RevIN']
    datasets = ['ETTm2', 'Electricity', 'Environment']
    data_kan = {
        'A0\n完整':       [0.1033, 0.1427, 0.3786],
        'A1\nw/o 频域':  [0.1066, 0.1456, 0.3634],
        'A2\nw/o 概率':  [0.1050, 0.1424, 0.3938],
        'A3\nw/o RevIN': [0.1073, 0.1533, 0.3825],
    }
    x = np.arange(len(datasets))
    width = 0.20
    colors_ab = [C_GREEN, C_PRIMARY, C_ACCENT, C_RED]
    for i, (cfg, mses) in enumerate(data_kan.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, mses, width, label=cfg, color=colors_ab[i])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE')
    ax.set_title('KAN-iTransformer 消融 (5 配置 × 3 数据集)', pad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)

    # Lite 消融
    ax = axes[1]
    configs_lite = ['B0\n完整 (latent=4)', 'B1\n窄瓶颈 (latent=1)', 'B2\n关闭残差 (latent=0)']
    data_lite = {
        'B0\n完整 (latent=4)': [0.1129, 0.2368, 0.3721],
        'B1\n窄瓶颈 (latent=1)': [0.1152, 0.2344, 0.3635],
        'B2\n关闭残差 (latent=0)': [0.1136, 0.2348, 0.3743],
    }
    x = np.arange(len(datasets))
    width = 0.25
    colors_lite = [C_GREEN, C_PRIMARY, C_RED]
    for i, (cfg, mses) in enumerate(data_lite.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, mses, width, label=cfg, color=colors_lite[i])
        for bar, mse in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{mse:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE')
    ax.set_title('LiteSparseNet 消融 (3 配置 × 3 数据集)', pad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('消融实验: 关键组件贡献度', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig9_ablation.png')
    plt.close()
    print('✓ fig9')


# Run all
if __name__ == '__main__':
    fig1_main_heatmap()
    fig2_degradation()
    fig3_cv()
    fig4_arch_dataset()
    fig5_innovative_vs_baseline()
    fig6_params_comparison()
    fig7_pareto()
    fig8_multimodal()
    fig9_ablation()
    print(f"\n✓ 9 张图全部生成在 {OUT}/")
