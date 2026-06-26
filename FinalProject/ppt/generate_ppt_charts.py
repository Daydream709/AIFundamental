"""
Generate all bar charts needed for the PPT (14 slides).
Style: IKB Klein blue accent + Carbon role tokens (high-contrast, clean).
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# CJK font registration
for fp in [
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
]:
    if os.path.exists(fp):
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass

plt.rcParams.update({
    'font.family': ['Times New Roman', 'Hiragino Sans GB', 'STHeiti', 'STSong', 'Arial Unicode MS', 'sans-serif'],
    'font.sans-serif': ['Times New Roman', 'Hiragino Sans GB', 'STHeiti', 'STSong', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': None,
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# IKB accent palette (matches PPT)
IKB = '#002FA7'
INK = '#0a0a0a'
GREY_1 = '#f0f0ee'
GREY_2 = '#d4d4d2'
GREY_3 = '#737373'
ACCENT_BRIGHT = '#5B7BFF'

MODEL_COLORS = {
    'DLinear': '#2E5C8A',
    'PatchTST': '#E67E22',
    'TimesNet': '#27AE60',
    'Mamba': '#C0392B',
    'KANiTransformer': IKB,
    'LiteSparseNet': '#16A085',
    'SparseTSF': '#D35400',
}

OUT = '/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/ppt/images'
os.makedirs(OUT, exist_ok=True)

# Load data
ROOT = '/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject'
l1 = pd.read_csv(f'{ROOT}/results/line1_latest.csv')
l2 = pd.read_csv(f'{ROOT}/results/line2_latest.csv')
l3sparse = pd.read_csv(f'{ROOT}/results/line3_sparsetsf_latest.csv')
abl_kan = pd.read_csv(f'{ROOT}/results/ablation_kan_latest.csv')
abl_lite = pd.read_csv(f'{ROOT}/results/ablation_lite_latest.csv')
eff = pd.read_csv(f'{ROOT}/results/efficiency/flops_params_summary_v3.csv')


def style_axes(ax, has_legend=True):
    ax.spines['left'].set_color(GREY_3)
    ax.spines['bottom'].set_color(GREY_3)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors=GREY_3, which='both')
    ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    if has_legend:
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)


# ============================================================
# P4: Line 1 — ETTm2 only, 4 模型 × 4 pred_len (single big chart)
# Data 双源: line1 优先, 缺失时回落 line2
# ============================================================
def p4_line1_heatmap():
    ds = 'ETTm2'
    models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba']
    pred_lens = [96, 192, 336, 720]

    fig, ax = plt.subplots(figsize=(14, 6.5))

    x = np.arange(len(pred_lens))
    width = 0.18
    for i, m in enumerate(models):
        mses = []
        for pl in pred_lens:
            # 双源: 基线在 line1, 同样可能在 line2
            row = l1[(l1['model'] == m) & (l1['dataset'] == ds) & (l1['pred_len'] == pl)]
            if len(row) == 0:
                row = l2[(l2['model'] == m) & (l2['dataset'] == ds) & (l2['pred_len'] == pl)]
            mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + (i - 1.5) * width, mses, width, label=m,
                      color=MODEL_COLORS[m], edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, mses):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f'F = {pl}' for pl in pred_lens], fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(f'ETTm2 数据集  ·  4 模型 × 4 预测长度  (7 变量 · 15min)',
                 color=INK, pad=14, fontsize=14, fontweight='500')
    ax.spines['left'].set_color(GREY_3)
    ax.spines['bottom'].set_color(GREY_3)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors=GREY_3, labelsize=11)
    ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=12)
    plt.tight_layout()
    fig.savefig(f'{OUT}/p4_line1_data.png')
    plt.close()
    print('✓ p4_line1_data')


# ============================================================
# (p5 was a chart for Line 1 degradation; removed in v2 — P5
# is now pure-text architecture comparison, see index.html)
# ============================================================


# ============================================================
# P7: KAN-iTransformer vs 4 baselines — Weather only, F=96
# Data 双源: KAN 在 line2, 4 baselines 在 line1
# ============================================================
def p7_kan_vs_baseline():
    ds = 'Weather'
    pred_len = 96
    compare_models = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer']

    # 双源读每个模型的 MSE
    mses = []
    for m in compare_models:
        # 先在 line1 找 (DLinear/PatchTST/TimesNet/Mamba)
        row = l1[(l1['model'] == m) & (l1['dataset'] == ds) & (l1['pred_len'] == pred_len)]
        # 没有再在 line2 找 (KANiTransformer)
        if len(row) == 0:
            row = l2[(l2['model'] == m) & (l2['dataset'] == ds) & (l2['pred_len'] == pred_len)]
        mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(13, 6.0))
    colors = [MODEL_COLORS[m] for m in compare_models]
    bars = ax.bar(compare_models, mses, color=colors, edgecolor='white', linewidth=0.5,
                  width=0.6)
    for b, v in zip(bars, mses):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.015,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=12, color=INK, fontweight='500')

    ax.set_ylabel('MSE  (F=96)', fontsize=12)
    ax.set_title(f'Weather 数据集  ·  5 模型对比  (21 vars · 10min)',
                 color=INK, pad=14, fontsize=14, fontweight='500')
    ax.spines['left'].set_color(GREY_3)
    ax.spines['bottom'].set_color(GREY_3)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors=GREY_3, labelsize=11)
    ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max([v for v in mses if not np.isnan(v)]) * 1.18)
    plt.tight_layout()
    fig.savefig(f'{OUT}/p7_kan_vs_baseline.png')
    plt.close()
    print('✓ p7_kan_vs_baseline')


# ============================================================
# P8: LiteSparseNet 3 指标对比 (MSE / Params / InferTime)
# vs SparseTSF / DLinear / TimesNet  (ETTm2 · F=96)
# Data 双源: line1 (DLinear/TimesNet)  +  line2 (LiteSparseNet/SparseTSF)
# ============================================================
def p8_lite_efficiency():
    models = ['SparseTSF', 'DLinear', 'TimesNet', 'LiteSparseNet']
    ds = 'ETTm2'
    pl = 96

    def lookup(model, col):
        """双源读: line1 优先, 缺失回落 line2"""
        r = l1[(l1['model'] == model) & (l1['dataset'] == ds) & (l1['pred_len'] == pl)]
        if len(r) == 0:
            r = l2[(l2['model'] == model) & (l2['dataset'] == ds) & (l2['pred_len'] == pl)]
        return r[col].values[0] if len(r) > 0 else np.nan

    # MSE
    mses = [lookup(m, 'MSE') for m in models]

    # Params: line1 优先, 否则从 v3 efficiency 拿
    def lookup_params(model):
        r = l1[(l1['model'] == model) & (l1['dataset'] == ds) & (l1['pred_len'] == pl)]
        if len(r) > 0 and not np.isnan(r['Params(M)'].values[0]):
            return r['Params(M)'].values[0]
        r = l2[(l2['model'] == model) & (l2['dataset'] == ds) & (l2['pred_len'] == pl)]
        if len(r) > 0 and not np.isnan(r['Params(M)'].values[0]):
            return r['Params(M)'].values[0]
        r = eff[(eff['model'] == model) & (eff['dataset'] == ds)]
        return r['params_M'].values[0] if len(r) > 0 else np.nan

    params = [lookup_params(m) for m in models]

    # InferTime
    times = [lookup(m, 'InferTime(ms)') for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={'wspace': 0.32})

    colors = [MODEL_COLORS[m] for m in models]

    # MSE (log scale for clarity)
    bars = axes[0].bar(models, mses, color=colors, edgecolor='white', linewidth=0.5)
    for b, v in zip(bars, mses):
        if not np.isnan(v):
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                          f'{v:.4f}', ha='center', va='bottom', fontsize=10, color=INK, fontweight='500')
    axes[0].set_ylabel('MSE  (↓ 越好)', fontsize=12)
    axes[0].set_title('精度  ·  MSE', color=INK, pad=10, fontsize=13, fontweight='500')
    valid = [v for v in mses if v > 0]
    if valid:
        axes[0].set_ylim(min(valid) * 0.7, max(valid) * 1.30)

    # Params (log)
    bars = axes[1].bar(models, params, color=colors, edgecolor='white', linewidth=0.5)
    for b, v in zip(bars, params):
        if not np.isnan(v) and v > 0:
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                          f'{v:.4f}', ha='center', va='bottom', fontsize=10, color=INK, fontweight='500')
    axes[1].set_ylabel('参数量  ·  M  (↓ 越轻)', fontsize=12)
    axes[1].set_title('效率  ·  Params', color=INK, pad=10, fontsize=13, fontweight='500')
    valid = [v for v in params if v > 0]
    if valid:
        axes[1].set_ylim(min(valid) * 0.5, max(valid) * 2)

    # Time
    bars = axes[2].bar(models, times, color=colors, edgecolor='white', linewidth=0.5)
    for b, v in zip(bars, times):
        if not np.isnan(v):
            axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                          f'{v:.1f}', ha='center', va='bottom', fontsize=10, color=INK, fontweight='500')
    axes[2].set_ylabel('推理时间  ·  ms  (↓ 越快)', fontsize=12)
    axes[2].set_title('效率  ·  InferTime', color=INK, pad=10, fontsize=13, fontweight='500')

    for ax in axes:
        ax.spines['left'].set_color(GREY_3)
        ax.spines['bottom'].set_color(GREY_3)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(colors=GREY_3, axis='x', labelsize=11, rotation=15)
        ax.tick_params(colors=GREY_3, axis='y', labelsize=10)
        ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle('LiteSparseNet  vs  SparseTSF / DLinear / TimesNet  (ETTm2 · F=96)',
                 fontsize=14, fontweight='500', color=INK, y=1.04)
    fig.savefig(f'{OUT}/p8_lite_efficiency.png')
    plt.close()
    print('✓ p8_lite_efficiency')


# ============================================================
# P9: 双消融 (KAN 4 + Lite 3)
# ============================================================
def p9_dual_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'wspace': 0.30})

    # KAN: 4 settings × 3 datasets (CFD 已删, v2.1.2 之后只测 3 模块 + A0 baseline)
    # 注意: CSV 里 setting 字段是 'A0 - 完整', 带中横线
    ax = axes[0]
    kan_settings = ['A0 - 完整', 'A1 - w/o KAN', 'A2 - w/o 概率输出', 'A3 - w/o RevIN']
    datasets = ['ETTm2', 'Electricity', 'Environment']
    x = np.arange(len(datasets))
    width = 0.18
    colors_kan = [IKB, '#5B7BFF', '#7F8C8D', '#E67E22']
    for i, cfg in enumerate(kan_settings):
        mses = []
        for ds in datasets:
            row = abl_kan[(abl_kan['setting'] == cfg) & (abl_kan['dataset'] == ds)]
            mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + (i - 1.5) * width, mses, width, label=cfg,
                      color=colors_kan[i], edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, mses):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7, color=GREY_3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE')
    ax.set_title('KAN-iTransformer 消融  (3 模块 × 3 数据集)', color=INK, pad=10)
    ax.spines['left'].set_color(GREY_3)
    ax.spines['bottom'].set_color(GREY_3)
    ax.tick_params(colors=GREY_3)
    ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)

    # Lite: 3 settings × 3 datasets
    ax = axes[1]
    lite_settings = abl_lite['setting'].unique().tolist()
    x = np.arange(len(datasets))
    width = 0.25
    colors_lite = [IKB, '#5B7BFF', '#E67E22']
    for i, cfg in enumerate(lite_settings):
        mses = []
        for ds in datasets:
            row = abl_lite[(abl_lite['setting'] == cfg) & (abl_lite['dataset'] == ds)]
            mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + (i - 1) * width, mses, width, label=cfg,
                      color=colors_lite[i], edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, mses):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7, color=GREY_3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE')
    ax.set_title('LiteSparseNet 消融  (3 阶段 × 3 数据集)', color=INK, pad=10)
    ax.spines['left'].set_color(GREY_3)
    ax.spines['bottom'].set_color(GREY_3)
    ax.tick_params(colors=GREY_3)
    ax.yaxis.grid(True, color=GREY_2, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

    fig.suptitle('消融实验  ·  关键模块贡献度',
                 fontsize=14, fontweight='500', color=INK, y=1.04)
    plt.savefig(f'{OUT}/p9_dual_ablation.png')
    plt.close()
    print('✓ p9_dual_ablation')


# ============================================================
# P11: 多模态 4 text_mode × 2 pred_len
# ============================================================
def p11_multimodal():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    modes = ['baseline', 'report', 'search', 'both_concat']
    mode_labels = ['baseline\n(纯时序)', 'report\n(单日报告)', 'search\n(周范围搜索)', 'both_concat\n(拼接)']
    pred_lens = [96, 192]
    x = np.arange(len(modes))
    width = 0.32
    colors_pl = [IKB, ACCENT_BRIGHT]

    for i, pl in enumerate(pred_lens):
        mses = []
        for mode in modes:
            row = l3sparse[(l3sparse['text_mode'] == mode) & (l3sparse['pred_len'] == pl)]
            mses.append(row['MSE'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + (i - 0.5) * width, mses, width, label=f'pred_len = {pl}',
                      color=colors_pl[i], edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, mses):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=10)
    ax.set_ylabel('MSE')
    ax.set_title('SparseTSF + 4 种文本模态  (Environment 数据集)', color=INK, pad=10)
    style_axes(ax, has_legend=False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=11)

    # 注释关键发现
    bl_96 = l3sparse[(l3sparse['text_mode'] == 'baseline') & (l3sparse['pred_len'] == 96)]['MSE'].values[0]
    search_96 = l3sparse[(l3sparse['text_mode'] == 'search') & (l3sparse['pred_len'] == 96)]['MSE'].values[0]
    search_192 = l3sparse[(l3sparse['text_mode'] == 'search') & (l3sparse['pred_len'] == 192)]['MSE'].values[0]
    bl_192 = l3sparse[(l3sparse['text_mode'] == 'baseline') & (l3sparse['pred_len'] == 192)]['MSE'].values[0]
    ax.text(0.98, 0.96,
            f'search 模式 vs baseline:\n  F=96: {(1 - search_96/bl_96)*100:+.1f}%\n  F=192: {(1 - search_192/bl_192)*100:+.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=11, color=IKB,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F4FF', edgecolor=IKB, linewidth=1))

    plt.tight_layout()
    plt.savefig(f'{OUT}/p11_multimodal.png')
    plt.close()
    print('✓ p11_multimodal')


# Run all
if __name__ == '__main__':
    p4_line1_heatmap()
    p7_kan_vs_baseline()
    p8_lite_efficiency()
    p9_dual_ablation()
    p11_multimodal()
    print(f"\n✓ 6 张图全部生成在 {OUT}/")
