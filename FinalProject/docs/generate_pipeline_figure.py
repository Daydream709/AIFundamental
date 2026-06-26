"""
Generate detailed project pipeline diagram (Figure 1 style).
"""
import sys, os
sys.path.insert(0, '/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

# Register CJK fonts
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
    'savefig.bbox': 'tight',
    'font.size': 10,
})

COLORS = {
    'data':       '#E8F4FD',
    'data_brd':   '#2E5C8A',
    'baseline':   '#FFF3E0',
    'baseline_brd': '#E67E22',
    'innov':      '#E8F8E8',
    'innov_brd':  '#27AE60',
    'multimodal': '#FDEDE8',
    'multimodal_brd': '#C0392B',
    'ablation':   '#F4ECF7',
    'ablation_brd': '#8E44AD',
    'analysis':   '#FEF9E7',
    'analysis_brd': '#F39C12',
    'arrow':       '#34495E',
    'text':        '#2C3E50',
    'header':      '#34495E',
}

def box(ax, x, y, w, h, text='', fc='#FFF', ec='#000', fontsize=9, weight='normal'):
    p = Rectangle((float(x), float(y)), float(w), float(h),
                  linewidth=1.5, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(float(x) + float(w)/2, float(y) + float(h)/2, str(text),
            ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], weight=weight)


def section_label(ax, x, y, w, h, text='', fontsize=11, fc='#34495E'):
    p = Rectangle((float(x), float(y)), float(w), float(h),
                  linewidth=0, facecolor=fc)
    ax.add_patch(p)
    ax.text(float(x) + float(w)/2, float(y) + float(h)/2, str(text),
            ha='center', va='center', fontsize=fontsize,
            color='white', weight='bold')


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.2, style='->'):
    color = color or COLORS['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

fig = plt.figure(figsize=(20, 23))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 23)
ax.axis('off')

# Title
ax.text(10, 22.5, 'Time Series Forecasting Benchmark — Full Pipeline',
        ha='center', va='center', fontsize=18, weight='bold', color=COLORS['text'])
ax.text(10, 22.0, '4 Main Experimental Lines + Cross-cutting Analysis',
        ha='center', va='center', fontsize=12, style='italic', color='#555')

# ============ ① DATA SOURCES ============
section_label(ax, 0.5, 20.6, 19, 0.6, '① Data Sources (Raw, Heterogeneous)')

datasets = [
    ('ETTm2\n7 vars · 15min · 69,680 rows', 1.5, 19.3),
    ('Weather\n21 vars · 10min · 52,696 rows', 6.0, 19.3),
    ('Electricity\n321 vars · 1hr · 26,304 rows', 10.5, 19.3),
    ('Environment (multimodal)\n6 vars + text · 15,979 rows', 15.0, 19.3),
]
for txt, x, y in datasets:
    box(ax, x, y, 4.0, 1.1, text=txt, fc=COLORS['data'], ec=COLORS['data_brd'], fontsize=9)

# ② PREPROCESSING
section_label(ax, 0.5, 18.5, 19, 0.6, '② Preprocessing (Unified Pipeline)')
prep_items = [
    ('① Z-score\nstandardization', 1.0, 17.5),
    ('② Sliding-window\n(seq_len, label_len, pred_len)', 5.5, 17.5),
    ('③ Time features\n(month, day, weekday, hour)', 10.0, 17.5),
    ('④ Train/Val/Test\nsplit 6:2:2', 14.5, 17.5),
]
for txt, x, y in prep_items:
    box(ax, x, y, 4.0, 0.9, text=txt, fc=COLORS['data'], ec=COLORS['data_brd'], fontsize=8.5)

for _, x, _ in datasets:
    arrow(ax, x + 2.0, 19.3, 6.0, 18.4)

# ============ ③ LINE 1: BASELINES ============
y_l1 = 16.3
section_label(ax, 0.5, y_l1 + 0.5, 9.2, 0.6, '③ Line 1: Cross-Architecture Baseline Comparison', fontsize=10)
baselines = ['DLinear\n(MLP)', 'PatchTST\n(Transformer)', 'TimesNet\n(2D-CNN)', 'Mamba\n(SSM)']
for i, m in enumerate(baselines):
    box(ax, 0.8 + i * 2.2, y_l1 - 0.6, 2.0, 1.0, m, COLORS['baseline'], COLORS['baseline_brd'], fontsize=8.5)

b_ds = ['ETTm2', 'Weather', 'Electricity']
for i, d in enumerate(b_ds):
    box(ax, 0.8 + i * 2.85, y_l1 - 1.8, 2.5, 0.5, d, COLORS['analysis'], COLORS['analysis_brd'], fontsize=8.5)

box(ax, 0.8, y_l1 - 2.7, 7.5, 0.5,
    'AdamW · lr=1e-4 · batch=32 (Electricity:16) · epochs=100 · patience=10',
    '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 0.8, y_l1 - 3.5, 7.5, 0.5, 'pred_lens ∈ {96, 192, 336, 720}', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 0.8, y_l1 - 4.3, 7.5, 0.6,
    '48 runs → results/line1_latest.csv',
    '#D5F5E3', '#229954', fontsize=9, weight='bold')

# ============ ④ LINE 2: INNOVATIONS ============
y_l2 = 16.3
section_label(ax, 10.3, y_l2 + 0.5, 9.2, 0.6, '④ Line 2: Self-Developed Models Evaluation', fontsize=10)
innovations = [
    ('KAN-iTransformer\n(频域分解 + KAN层\n+ 概率输出 + RevIN)', 10.6, y_l2 - 0.6),
    ('LiteSparseNet\n(稀疏趋势 + 轻量MLP\n+ 残差修正, <0.05M)', 12.8, y_l2 - 0.6),
    ('SparseTSF\n(跨周期下采样\n0.001M params)', 15.0, y_l2 - 0.6),
]
for txt, x, y in innovations:
    box(ax, x, y, 2.0, 1.0, text=txt, fc=COLORS['innov'], ec=COLORS['innov_brd'], fontsize=8)

box(ax, 12.8, y_l2 - 1.8, 4.2, 0.5, 'vs 4 baselines (DLinear/PatchTST/TimesNet/Mamba)',
    COLORS['analysis'], COLORS['analysis_brd'], fontsize=8.5)
box(ax, 10.6, y_l2 - 2.7, 6.4, 0.5, '3 datasets: ETTm2, Weather, Electricity', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 10.6, y_l2 - 3.5, 6.4, 0.5, 'pred_lens ∈ {96, 192, 336, 720}', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 10.6, y_l2 - 4.3, 6.4, 0.6,
    '84 runs → results/line2_latest.csv',
    '#D5F5E3', '#229954', fontsize=9, weight='bold')

arrow(ax, 8.3, y_l2 - 1.0, 10.6, y_l2 - 1.0, color='#888', style='<-')

# ============ ⑤ LINE 3: MULTIMODAL ============
y_l3 = 10.5
section_label(ax, 0.5, y_l3 + 0.5, 19, 0.6, '⑤ Line 3: Multimodal Fusion (SparseTSF + TextEncoder)', fontsize=10)

box(ax, 1.5, y_l3 - 0.6, 4.5, 1.0,
    'Environment Text JSON\n81 reports + 2272 searches\n→ sentence-transformers\nall-MiniLM-L6-v2\n→ 128-dim embeddings',
    '#FCE4EC', '#C2185B', fontsize=8.5)

box(ax, 6.5, y_l3 - 0.6, 3.0, 1.0,
    'Window: s_end-1\ntake LAST step\n(text per window)',
    COLORS['data'], COLORS['data_brd'], fontsize=8)

box(ax, 10.0, y_l3 - 0.6, 5.0, 1.0,
    'SparseTSF (time-series branch)\n+ TextEncoder (Linear 128→64→6)\n+ Gated fusion: gate·text·0.1',
    COLORS['multimodal'], COLORS['multimodal_brd'], fontsize=8.5)

modes = [
    ('baseline\nuse_text=False', 15.3, y_l3 + 0.05),
    ('report\nreport_only', 15.3, y_l3 - 0.45),
    ('search\nsearch_only', 16.3, y_l3 + 0.05),
    ('both_concat\nconcat', 16.3, y_l3 - 0.45),
]
for text, x, y in modes:
    box(ax, x, y, 0.85, 0.4, text=text, fc=COLORS['multimodal'], ec=COLORS['multimodal_brd'], fontsize=7)

box(ax, 17.4, y_l3 - 0.3, 2.2, 0.6,
    '8 runs\nline3_sparsetsf\n_latest.csv',
    '#D5F5E3', '#229954', fontsize=8, weight='bold')

box(ax, 10.0, y_l3 - 1.7, 5.0, 0.5, 'Dataset: Environment, pred_lens ∈ {96, 192}',
    '#F8F9FA', '#6C757D', fontsize=8)

arrow(ax, 6.0, y_l3 - 0.1, 6.5, y_l3 - 0.1)
arrow(ax, 9.5, y_l3 - 0.1, 10.0, y_l3 - 0.1)
arrow(ax, 15.0, y_l3 - 0.1, 15.3, y_l3 - 0.3)
arrow(ax, 16.3, y_l3 - 0.3, 17.4, y_l3 - 0.3)

# ============ ⑥ LINE 4: ABLATION ============
y_l4 = 7.5
section_label(ax, 0.5, y_l4 + 0.5, 19, 0.6, '⑥ Line 4: Ablation Studies (Component Contribution Analysis)', fontsize=10)

section_label(ax, 0.5, y_l4 - 0.3, 9.2, 0.4, 'Line 4a: KAN-iTransformer (4 settings × 3 datasets)', fc=COLORS['ablation_brd'], fontsize=9)
section_label(ax, 10.3, y_l4 - 0.3, 9.2, 0.4, 'Line 4b: LiteSparseNet (3 settings × 3 datasets)', fc=COLORS['ablation_brd'], fontsize=9)

kan_settings = [
    ('A0 完整\n(full)', 0.8, y_l4 - 1.0),
    ('A1 w/o 频域分解\n(CFD off)', 3.0, y_l4 - 1.0),
    ('A2 w/o 概率输出\n(MSE only)', 5.2, y_l4 - 1.0),
    ('A3 w/o RevIN\n(simple norm)', 7.4, y_l4 - 1.0),
]
for text, x, y in kan_settings:
    box(ax, x, y, 1.5, 0.7, text=text, fc=COLORS['ablation'], ec=COLORS['ablation_brd'], fontsize=7)

lite_settings = [
    ('B0 完整\n(latent=4)', 10.6, y_l4 - 1.0),
    ('B1 窄瓶颈\n(latent=1)', 13.0, y_l4 - 1.0),
    ('B2 关闭残差\n(latent=0)', 15.4, y_l4 - 1.0),
]
for text, x, y in lite_settings:
    box(ax, x, y, 2.0, 0.7, text=text, fc=COLORS['ablation'], ec=COLORS['ablation_brd'], fontsize=7.5)

box(ax, 0.8, y_l4 - 2.0, 8.4, 0.5, '3 datasets: ETTm2, Electricity, Environment', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 10.6, y_l4 - 2.0, 6.4, 0.5, '3 datasets: ETTm2, Electricity, Environment', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 0.8, y_l4 - 2.8, 8.4, 0.5, 'pred_len=96 (default)', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 10.6, y_l4 - 2.8, 6.4, 0.5, 'pred_len=96 (default)', '#F8F9FA', '#6C757D', fontsize=8)
box(ax, 0.8, y_l4 - 3.7, 8.4, 0.6, '12 runs → ablation_kan_latest.csv', '#D5F5E3', '#229954', fontsize=8.5, weight='bold')
box(ax, 10.6, y_l4 - 3.7, 6.4, 0.6, '9 runs → ablation_lite_latest.csv', '#D5F5E3', '#229954', fontsize=8.5, weight='bold')

# ============ ⑦ ANALYSIS ============
section_label(ax, 0.5, 2.0, 19, 0.6, '⑦ Cross-cutting Analysis & Visualization', fontsize=11)
analyses = [
    ('Performance\ncomparison\nheatmap', 0.8, 0.4),
    ('Degradation rate\ncurves\n(96→720)', 4.0, 0.4),
    ('Cross-dataset\nstability (CV)\nanalysis', 7.2, 0.4),
    ('Architecture ×\nDataset\ncompatibility', 10.4, 0.4),
    ('Innovative vs\nbest baseline\ncomparison', 13.6, 0.4),
    ('Efficiency\ncomparison\n(params, FLOPs)', 16.8, 0.4),
]
for text, x, y in analyses:
    box(ax, x, y, 2.8, 1.4, text=text, fc=COLORS['analysis'], ec=COLORS['analysis_brd'], fontsize=8)

for x in [4.5, 13.5]:
    arrow(ax, x, 0.8, x, 2.0, color='#888', lw=1.5)

plt.savefig('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/docs/figures/fig0_pipeline.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✓ Figure 0 (pipeline) generated')
