"""
regenerate_all_figures.py — 一次性重新生成报告引用的所有 9 张图
基于 results/ 文件夹最新 CSV 数据 (2026-06-27)

使用：
  python regenerate_all_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import seaborn as sns

# 中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

SAVE_DIR = 'figures/'
os.makedirs(SAVE_DIR, exist_ok=True)

LINE1 = 'results/line1_latest.csv'
LINE2 = 'results/line2_latest.csv'
LINE3 = 'results/line3_latest.csv'
LINE3_SPARSE = 'results/line3_sparsetsf_latest.csv'
ABLA_KAN = 'results/ablation_kan_latest.csv'
ABLA_LITE = 'results/ablation_lite_latest.csv'
EFF = 'results/efficiency/flops_params_summary_v3.csv'

# 颜色配置
ARCH_COLORS = {
    'DLinear': '#1f77b4',
    'PatchTST': '#ff7f0e',
    'TimesNet': '#2ca02c',
    'Mamba': '#d62728',
    'SparseTSF': '#9467bd',
    'KANiTransformer': '#8c564b',
    'LiteSparseNet': '#e377c2',
}
DATASET_ORDER = ['ETTm2', 'Weather', 'Electricity']
PRED_ORDER = [96, 192, 336, 720]

# ============================================================================
# 图 0: 项目整体流程图 (4 主线结构)
# ============================================================================
def fig0_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # 数据源
    data_box = mpatches.FancyBboxPatch((0.3, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
                                         facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.55, 3.85, '4 个数据集', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.55, 3.45, 'ETTm2 / Weather\nElectricity / Environment', ha='center', fontsize=8.5)
    ax.text(1.55, 2.95, '(统一滑窗 + 标准化)', ha='center', fontsize=8, style='italic', color='gray')

    # 4 主线
    lines = [
        ('Line 1: 全架构对比', '4 基线 × 3 数据集 × 4 F\n(48 runs)', 3.2, 5.5, '#FFE0B2'),
        ('Line 2: 自研评测', '3 自研 × 3 数据集 × 4 F\n(36 runs)', 3.2, 3.7, '#FFCCBC'),
        ('Line 3: 多模态消融', '2 架构 × 7 模态 × 2 F\n(28 runs)', 3.2, 1.9, '#C8E6C9'),
        ('Line 4: 消融实验', 'KAN 4 + Lite 3 × 3 数据集\n(24 runs)', 3.2, 0.1, '#E1BEE7'),
    ]
    for name, detail, x, y, color in lines:
        box = mpatches.FancyBboxPatch((x, y), 2.8, 1.4, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 1.4, y + 1.1, name, ha='center', fontsize=10.5, fontweight='bold')
        ax.text(x + 1.4, y + 0.45, detail, ha='center', fontsize=8.5)
        # 箭头
        ax.annotate('', xy=(x, y + 0.7), xytext=(2.8, 3.55),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    # 模型阵容
    model_box = mpatches.FancyBboxPatch((7.5, 4.5), 2.5, 2.0, boxstyle="round,pad=0.1",
                                          facecolor='#FFF9C4', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(model_box)
    ax.text(8.75, 6.2, '7 个模型', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 5.7, '基线: DLinear, PatchTST,\nTimesNet, Mamba', ha='center', fontsize=8.5)
    ax.text(8.75, 5.05, '外部: SparseTSF', ha='center', fontsize=8.5)
    ax.text(8.75, 4.7, '⭐ 自研: KAN-iTF, LiteSparseNet', ha='center', fontsize=8.5, color='#D32F2F', fontweight='bold')

    # 评估指标
    metric_box = mpatches.FancyBboxPatch((7.5, 1.8), 2.5, 2.0, boxstyle="round,pad=0.1",
                                           facecolor='#F8BBD0', edgecolor='#C2185B', linewidth=2)
    ax.add_patch(metric_box)
    ax.text(8.75, 3.5, '5 项评估指标', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 3.0, '精度: MSE / MAE\nRMSE / MAPE / SMAPE', ha='center', fontsize=8.5)
    ax.text(8.75, 2.3, '效率: Params / FLOPs\nInfer / GPU Mem', ha='center', fontsize=8.5)
    ax.text(8.75, 1.95, '统计: Wilcoxon', ha='center', fontsize=8.5)

    # 跨切片分析
    analysis_box = mpatches.FancyBboxPatch((11.0, 2.8), 2.7, 2.0, boxstyle="round,pad=0.1",
                                             facecolor='#D1C4E9', edgecolor='#5E35B1', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(12.35, 4.5, '跨切片分析', ha='center', fontsize=11, fontweight='bold')
    ax.text(12.35, 3.9, '• 架构 × 数据集适配\n• 精度-效率 Pareto\n• 模块消融贡献度', ha='center', fontsize=8.5)
    ax.text(12.35, 2.95, '• 多模态根因分析', ha='center', fontsize=8.5)

    # 右侧连接箭头
    ax.annotate('', xy=(7.5, 5.5), xytext=(6.0, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(7.5, 4.5), xytext=(6.0, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(7.5, 2.8), xytext=(6.0, 2.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(11.0, 3.8), xytext=(10.0, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(11.0, 3.8), xytext=(10.0, 2.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.set_title('图 0: 项目整体流程图（4 条主线 + 数据 + 评估 + 分析）', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig0_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig0_pipeline.png')


# ============================================================================
# 图 1: 基线模型热力图
# ============================================================================
def fig1_baseline_heatmap():
    df = pd.read_csv(LINE1)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba'])]

    df['setting'] = df['dataset'] + '\nF=' + df['pred_len'].astype(str)
    pivot = df.pivot_table(index='model', columns='setting', values='MSE', aggfunc='mean')
    # 列排序
    col_order = [f'{d}\nF={f}' for d in DATASET_ORDER for f in PRED_ORDER]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                linewidths=0.6, cbar_kws={'label': 'MSE'}, ax=ax)
    ax.set_title('图 5-1: 基线模型 × 数据集 × 预测长度的 MSE 热力图\n(颜色越深 = MSE 越高 = 性能越差)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('数据集 / 预测长度')
    ax.set_ylabel('模型')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig1_baseline_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig1_baseline_heatmap.png')


# ============================================================================
# 图 2: 退化率折线图
# ============================================================================
def fig2_degradation():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF'])]
    df = df[df['dataset'].isin(DATASET_ORDER)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASET_ORDER):
        sub = df[df['dataset'] == ds]
        for model in ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']:
            m = sub[sub['model'] == model].sort_values('pred_len')
            if len(m) < 2:
                continue
            # 归一化到 F=96 = 1.0
            base = m['MSE'].iloc[0]
            if base == 0:
                continue
            norm = m['MSE'].values / base
            ax.plot(m['pred_len'], norm, 'o-', label=model,
                    color=ARCH_COLORS[model], linewidth=2, markersize=6)
        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.set_xlabel('预测长度 F')
        ax.set_xticks(PRED_ORDER)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('退化率 (F=96 归一化到 1.0)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    fig.suptitle('图 5-2: 7 模型在 3 数据集上的退化曲线 (值越高 = 退化越严重)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig2_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig2_degradation.png')


# ============================================================================
# 图 3: 跨数据集一致性柱状图 (CV)
# ============================================================================
def fig3_cv_consistency():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF'])]
    df = df[df['dataset'].isin(DATASET_ORDER)]

    # 算 CV: 每个模型在 3 数据集上 F=96 的 MSE 的 std/mean
    cv_data = []
    for model in ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']:
        mses = []
        for ds in DATASET_ORDER:
            row = df[(df['model'] == model) & (df['dataset'] == ds) & (df['pred_len'] == 96)]
            if len(row) > 0:
                mses.append(row['MSE'].iloc[0])
        if len(mses) == 3:
            cv = (np.std(mses) / np.mean(mses)) * 100
            cv_data.append({'model': model, 'cv': cv, 'mean_mse': np.mean(mses)})

    cv_df = pd.DataFrame(cv_data).sort_values('cv')

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ARCH_COLORS[m] for m in cv_df['model']]
    bars = ax.bar(cv_df['model'], cv_df['cv'], color=colors, edgecolor='black', linewidth=1.2)

    # 高亮最稳（绿色边框）和最不稳（红色边框）
    bars[0].set_edgecolor('#2E7D32')
    bars[0].set_linewidth(3)
    bars[-1].set_edgecolor('#C62828')
    bars[-1].set_linewidth(3)

    for bar, val in zip(bars, cv_df['cv']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('跨数据集变异系数 CV (%)', fontsize=11)
    ax.set_title('图 5-3: 7 模型在 3 数据集上的 CV (越小越稳定)\n绿色边框 = 最稳, 红色边框 = 最不稳',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig3_cv_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig3_cv_consistency.png')


# ============================================================================
# 图 4: 架构 × 数据集 适配矩阵
# ============================================================================
def fig4_arch_dataset():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF'])]
    df = df[df['dataset'].isin(DATASET_ORDER)]

    # 每个模型在每个数据集上 F=96 的相对排名（百分比）
    archs = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']

    # 按"架构族"分组
    arch_groups = {
        '线性\n(DLinear)': ['DLinear'],
        'Transformer\n(PatchTST)': ['PatchTST'],
        'CNN\n(TimesNet)': ['TimesNet'],
        'SSM\n(Mamba)': ['Mamba'],
        'KAN + iTransformer\n(自研 1)': ['KANiTransformer'],
        '稀疏+组MLP\n(自研 2)': ['LiteSparseNet'],
        '跨周期下采样\n(SparseTSF)': ['SparseTSF'],
    }

    # 算每组在 3 数据集上的相对排名
    rank_matrix = np.zeros((len(arch_groups), 3))
    for i, (group_name, members) in enumerate(arch_groups.items()):
        for j, ds in enumerate(DATASET_ORDER):
            sub = df[(df['dataset'] == ds) & (df['pred_len'] == 96) & (df['model'].isin(members))]
            if len(sub) > 0:
                # 在 7 个模型中的排名
                all_mses = df[(df['dataset'] == ds) & (df['pred_len'] == 96) & (df['model'].isin(archs))]['MSE'].values
                all_mses_sorted = np.sort(all_mses)
                my_mse = sub['MSE'].iloc[0]
                rank_pct = np.searchsorted(all_mses_sorted, my_mse) / (len(all_mses_sorted) - 1)
                rank_matrix[i, j] = rank_pct

    # 评级
    def score_to_label(s):
        if s <= 0.2:
            return '✓✓✓'
        elif s <= 0.4:
            return '✓✓'
        elif s <= 0.6:
            return '✓'
        elif s <= 0.8:
            return '△'
        else:
            return '✗'

    fig, ax = plt.subplots(figsize=(10, 6))
    cell_colors = []
    cell_text = []
    for i in range(len(arch_groups)):
        row_text = []
        row_colors = []
        for j in range(3):
            label = score_to_label(rank_matrix[i, j])
            row_text.append(label)
            # 颜色: ✓✓✓深绿, ✓✓浅绿, ✓黄绿, △黄, ✗红
            if label == '✓✓✓':
                row_colors.append('#1B5E20')
            elif label == '✓✓':
                row_colors.append('#66BB6A')
            elif label == '✓':
                row_colors.append('#C8E6C9')
            elif label == '△':
                row_colors.append('#FFE082')
            else:
                row_colors.append('#EF9A9A')
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_text, rowLabels=list(arch_groups.keys()),
                     colLabels=['低维\nETTm2 (C=7)', '中维\nWeather (C=21)', '高维\nElectricity (C=321)'],
                     cellColours=cell_colors, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    ax.axis('off')
    ax.set_title('图 5-4: 架构 × 数据集 适配矩阵\n✓✓✓ (前 20%) / ✓✓ (40%) / ✓ (60%) / △ (80%) / ✗ (后 20%)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig4_arch_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig4_arch_dataset.png')


# ============================================================================
# 图 5: 创新模型 vs 最佳基线
# ============================================================================
def fig5_innovative_vs_baseline():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF'])]
    df = df[df['dataset'].isin(DATASET_ORDER)]

    # 找每个数据集 F=96 上的最佳基线
    innovatives = ['KANiTransformer', 'LiteSparseNet', 'SparseTSF']
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(DATASET_ORDER))
    width = 0.25
    for i, model in enumerate(innovatives):
        diffs = []
        for ds in DATASET_ORDER:
            # 创新模型
            m1 = df[(df['model'] == model) & (df['dataset'] == ds) & (df['pred_len'] == 96)]
            # 最佳基线 (4 个 thuml 基线中最好的)
            m2 = df[(df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba'])) &
                    (df['dataset'] == ds) & (df['pred_len'] == 96)]
            if len(m1) > 0 and len(m2) > 0:
                best = m2['MSE'].min()
                diff = (m1['MSE'].iloc[0] - best) / best * 100
                diffs.append(diff)
            else:
                diffs.append(0)
        bars = ax.bar(x + (i - 1) * width, diffs, width, label=model,
                      color=ARCH_COLORS[model], edgecolor='black', linewidth=1)
        for bar, val in zip(bars, diffs):
            if val < 0:
                txt = f'{val:.1f}%'
                color = 'white' if val < -10 else 'black'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                        txt, ha='center', va='top', fontsize=9, fontweight='bold', color=color)
            else:
                txt = f'+{val:.1f}%' if val > 0 else '0%'
                # 截断过高 (如 SparseTSF +264%)
                if val > 50:
                    txt = f'+{val:.0f}%\n(截断)'
                    bar.set_height(50)
                ax.text(bar.get_x() + bar.get_width() / 2, min(bar.get_height() + 1, 52),
                        txt, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel('相对最佳基线的 MSE 差异 (%)', fontsize=11)
    ax.set_title('图 5-5: 创新模型 vs 最佳基线 (F=96)\n负值 = 超越基线, 正值 = 落后基线',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(-20, 60)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig5_innovative_vs_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig5_innovative_vs_baseline.png')


# ============================================================================
# 图 6: 7 模型参数量对比 (对数坐标)
# ============================================================================
def fig6_params_comparison():
    eff = pd.read_csv(EFF)
    models_order = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'SparseTSF', 'KANiTransformer', 'LiteSparseNet']

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(models_order))
    width = 0.2
    ds_colors = {'ETTm2': '#1f77b4', 'Weather': '#ff7f0e', 'Electricity': '#2ca02c', 'Environment': '#d62728'}

    for i, ds in enumerate(['ETTm2', 'Weather', 'Electricity', 'Environment']):
        params = []
        for m in models_order:
            row = eff[(eff['model'] == m) & (eff['dataset'] == ds)]
            params.append(row['params_M'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1.5) * width, params, width, label=ds,
               color=ds_colors[ds], edgecolor='black', linewidth=0.8)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20)
    ax.set_ylabel('参数量 (M, 对数)', fontsize=11)
    ax.set_title('图 5-6: 7 模型 × 4 数据集的参数量对比 (对数坐标)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig6_params_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig6_params_comparison.png')


# ============================================================================
# 图 7: Pareto 散点图 (推理时间 vs MSE)
# ============================================================================
def fig7_pareto():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF'])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, ds in zip(axes, ['ETTm2', 'Weather']):
        for model in ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'KANiTransformer', 'LiteSparseNet', 'SparseTSF']:
            row = df[(df['model'] == model) & (df['dataset'] == ds) & (df['pred_len'] == 96)]
            if len(row) == 0:
                continue
            x = row['InferTime(ms)'].iloc[0]
            y = row['MSE'].iloc[0]
            s = row['Params(M)'].iloc[0] * 30 + 30  # 气泡大小 = 参数量
            ax.scatter(x, y, s=s, color=ARCH_COLORS[model], alpha=0.7,
                       edgecolor='black', linewidth=1, label=model)
            ax.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlabel('推理时间 (ms, 对数)', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('图 5-7: 7 模型在 ETTm2 和 Weather 上的推理时间 vs MSE (F=96)\n气泡大小 ∝ 参数量',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig7_pareto.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig7_pareto.png')


# ============================================================================
# 图 8: 多模态柱状图 (7 模态 × 2 架构)
# ============================================================================
def fig8_multimodal():
    df = pd.read_csv(LINE3)
    df_96 = df[df['pred_len'] == 96]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    modalities = ['baseline', 'report', 'search', 'both_concat', 'both_gating', 'satellite', 'text+satellite']

    for ax, model in zip(axes, ['PatchTST', 'Mamba']):
        sub = df_96[df_96['model'] == model]
        mses = []
        for mod in modalities:
            row = sub[sub['text_mode'] == mod]
            mses.append(row['MSE'].iloc[0] if len(row) > 0 else 0)

        colors = ['#9E9E9E'] + ['#42A5F5'] * 6  # baseline 灰色, 其他蓝色
        bars = ax.bar(range(len(modalities)), mses, color=colors, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=7.5, rotation=90)

        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels(modalities, rotation=30, fontsize=9)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(f'{model} on Environment (F=96)\n7 模态 MSE 完全相同 → 多模态未生效',
                     fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(mses) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('图 5-8: 多模态实验结果 (基于 line3_latest.csv)\n7 种模态 MSE 完全相同 — 根因: 训练循环丢弃 batch[4:6]',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig8_multimodal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig8_multimodal.png')


# ============================================================================
# 图 9: 消融实验柱状图
# ============================================================================
def fig9_ablation():
    kan = pd.read_csv(ABLA_KAN)
    lite = pd.read_csv(ABLA_LITE)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # KAN 消融
    ax = axes[0]
    kan_settings = ['A0 - 完整', 'A1 - w/o KAN', 'A2 - w/o 概率输出', 'A3 - w/o RevIN']
    kan_colors = ['#4CAF50', '#FF7043', '#FFA726', '#EF5350']
    x = np.arange(len(kan_settings))
    width = 0.25
    for i, ds in enumerate(DATASET_ORDER):
        mses = []
        for s in kan_settings:
            row = kan[(kan['setting'] == s) & (kan['dataset'] == ds)]
            mses.append(row['MSE'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1) * width, mses, width, label=ds, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(kan_settings, rotation=15, fontsize=8)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title('KAN-iTransformer 4 模块消融', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Lite 消融
    ax = axes[1]
    lite_settings = ['B0 - 完整 (latent=4)', 'B1 - 窄瓶颈 (latent=1)', 'B2 - 关闭 (latent=0)']
    lite_colors = ['#4CAF50', '#42A5F5', '#EF5350']
    x = np.arange(len(lite_settings))
    for i, ds in enumerate(DATASET_ORDER):
        mses = []
        for s in lite_settings:
            row = lite[(lite['setting'] == s) & (lite['dataset'] == ds)]
            mses.append(row['MSE'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1) * width, mses, width, label=ds, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lite_settings, rotation=15, fontsize=8)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title('Lite-SparseNet 3 阶段消融', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('图 5-9: 消融实验柱状图 (基于 ablation_*_latest.csv)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig9_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig9_ablation.png')


# ============================================================================
# 图 10: Params / FLOPs 双子图 (来自表 5-8)
# ============================================================================
def fig10_params_flops():
    eff = pd.read_csv(EFF)
    models_order = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'SparseTSF', 'KANiTransformer', 'LiteSparseNet']
    ds_order = ['ETTm2', 'Weather', 'Electricity', 'Environment']
    ds_colors = {'ETTm2': '#1f77b4', 'Weather': '#ff7f0e', 'Electricity': '#2ca02c', 'Environment': '#d62728'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(models_order))
    width = 0.2

    # 左图: 参数量
    ax = axes[0]
    for i, ds in enumerate(ds_order):
        params = []
        for m in models_order:
            row = eff[(eff['model'] == m) & (eff['dataset'] == ds)]
            params.append(row['params_M'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1.5) * width, params, width, label=ds,
               color=ds_colors[ds], edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20, fontsize=9)
    ax.set_ylabel('参数量 Params (M, 对数)', fontsize=11)
    ax.set_title('参数量', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # 右图: FLOPs
    ax = axes[1]
    for i, ds in enumerate(ds_order):
        flops = []
        for m in models_order:
            row = eff[(eff['model'] == m) & (eff['dataset'] == ds)]
            flops.append(row['flops_G'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1.5) * width, flops, width, label=ds,
               color=ds_colors[ds], edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20, fontsize=9)
    ax.set_ylabel('浮点运算 FLOPs (G, 对数)', fontsize=11)
    ax.set_title('FLOPs', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    fig.suptitle('图 5-10: 7 模型 × 4 数据集的 Params 与 FLOPs (fvcore 测量, 配置来自 MODEL_PRESETS)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig10_params_flops.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig10_params_flops.png')


# ============================================================================
# 图 11: 推理时间 / 显存 双子图 (来自表 5-8b)
# ============================================================================
def fig11_infer_memory():
    df1 = pd.read_csv(LINE1)
    df2 = pd.read_csv(LINE2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['model'].isin(['DLinear', 'PatchTST', 'TimesNet', 'Mamba',
                              'SparseTSF', 'KANiTransformer', 'LiteSparseNet'])]
    df = df[df['dataset'].isin(DATASET_ORDER)]

    models_order = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'SparseTSF', 'KANiTransformer', 'LiteSparseNet']
    ds_colors = {'ETTm2': '#1f77b4', 'Weather': '#ff7f0e', 'Electricity': '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(models_order))
    width = 0.27

    # 左图: 推理时间 (对数)
    ax = axes[0]
    for i, ds in enumerate(DATASET_ORDER):
        infers = []
        for m in models_order:
            row = df[(df['model'] == m) & (df['dataset'] == ds) & (df['pred_len'] == 96)]
            infers.append(row['InferTime(ms)'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1) * width, infers, width, label=ds,
               color=ds_colors[ds], edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20, fontsize=9)
    ax.set_ylabel('推理时间 Infer (ms, 对数)', fontsize=11)
    ax.set_title('推理时间', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # 右图: 显存 (对数)
    ax = axes[1]
    for i, ds in enumerate(DATASET_ORDER):
        mems = []
        for m in models_order:
            row = df[(df['model'] == m) & (df['dataset'] == ds) & (df['pred_len'] == 96)]
            mems.append(row['GPUMem(MB)'].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + (i - 1) * width, mems, width, label=ds,
               color=ds_colors[ds], edgecolor='black', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20, fontsize=9)
    ax.set_ylabel('显存 GPU Memory (MB, 对数)', fontsize=11)
    ax.set_title('显存', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    fig.suptitle('图 5-11: 7 模型 × 3 数据集的推理时间与显存 (line1/line2 CSV 测量, F=96)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig11_infer_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ fig11_infer_memory.png')


if __name__ == '__main__':
    print('=' * 60)
    print('  重新生成 11 张图 (基于 results/ 最新 CSV)')
    print('=' * 60)
    fig0_pipeline()
    fig1_baseline_heatmap()
    fig2_degradation()
    fig3_cv_consistency()
    fig4_arch_dataset()
    fig5_innovative_vs_baseline()
    fig6_params_comparison()
    fig7_pareto()
    fig8_multimodal()
    fig9_ablation()
    fig10_params_flops()
    fig11_infer_memory()
    print('=' * 60)
    print(f'  全部 11 张图已保存到 {SAVE_DIR}')
    print('=' * 60)
