"""
Standalone: 生成表 5-8 (Params/FLOPs) 和表 5-8b (Infer/GPU) 的图
不依赖 pandas/seaborn，只用 numpy + matplotlib
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

SAVE_DIR = 'figures/'
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS_ORDER = ['DLinear', 'PatchTST', 'TimesNet', 'Mamba', 'SparseTSF', 'KANiTransformer', 'LiteSparseNet']
DS_ORDER_ALL = ['ETTm2', 'Weather', 'Electricity', 'Environment']
DS_ORDER_3 = ['ETTm2', 'Weather', 'Electricity']
DS_COLORS = {
    'ETTm2': '#1f77b4',
    'Weather': '#ff7f0e',
    'Electricity': '#2ca02c',
    'Environment': '#d62728',
}

# 读 fvcore 测的 Params/FLOPs
params_dict = {}  # {(model, dataset): params_M}
flops_dict = {}   # {(model, dataset): flops_G}
with open('results/efficiency/flops_params_summary_v3.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['model'], row['dataset'])
        params_dict[key] = float(row['params_M'])
        flops_dict[key] = float(row['flops_G'])

# 读 line1 + line2 的 Infer/GPUMem (取 F=96)
infer_dict = {}   # {(model, dataset): infer_ms}
mem_dict = {}     # {(model, dataset): gpu_mb}
for path in ('results/line1_latest.csv', 'results/line2_latest.csv'):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'success':
                continue
            if int(row['pred_len']) != 96:
                continue
            key = (row['model'], row['dataset'])
            if row['model'] not in MODELS_ORDER:
                continue
            if row['dataset'] not in DS_ORDER_3:
                continue
            try:
                infer_dict[key] = float(row['InferTime(ms)'])
                mem_dict[key] = float(row['GPUMem(MB)'])
            except (ValueError, KeyError):
                pass

# ============================================================
# 图 5-10: FLOPs 单图（来自表 5-8）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(MODELS_ORDER))
width = 0.2

for i, ds in enumerate(DS_ORDER_ALL):
    flops = [flops_dict.get((m, ds), 0) for m in MODELS_ORDER]
    ax.bar(x + (i - 1.5) * width, flops, width, label=ds,
           color=DS_COLORS[ds], edgecolor='black', linewidth=0.7)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(MODELS_ORDER, rotation=20, fontsize=9)
ax.set_ylabel('浮点运算 FLOPs (G, 对数)', fontsize=11)
ax.set_title('7 模型 × 4 数据集的 FLOPs', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')

fig.suptitle('图 5-10: 7 模型 × 4 数据集的 FLOPs (fvcore 测量, 来自 MODEL_PRESETS, 对数坐标)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig10_params_flops.png', dpi=150, bbox_inches='tight')
plt.close()
print('  saved fig10_params_flops.png')

# ============================================================
# 图 5-11: 推理时间 / 显存 双子图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(MODELS_ORDER))
width = 0.27

# 左: 推理时间
ax = axes[0]
for i, ds in enumerate(DS_ORDER_3):
    vals = [infer_dict.get((m, ds), 0) for m in MODELS_ORDER]
    ax.bar(x + (i - 1) * width, vals, width, label=ds,
           color=DS_COLORS[ds], edgecolor='black', linewidth=0.7)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(MODELS_ORDER, rotation=20, fontsize=9)
ax.set_ylabel('推理时间 Infer (ms, 对数)', fontsize=11)
ax.set_title('推理时间', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')

# 右: 显存
ax = axes[1]
for i, ds in enumerate(DS_ORDER_3):
    vals = [mem_dict.get((m, ds), 0) for m in MODELS_ORDER]
    ax.bar(x + (i - 1) * width, vals, width, label=ds,
           color=DS_COLORS[ds], edgecolor='black', linewidth=0.7)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(MODELS_ORDER, rotation=20, fontsize=9)
ax.set_ylabel('显存 GPU Memory (MB, 对数)', fontsize=11)
ax.set_title('显存', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')

fig.suptitle('图 5-11: 7 模型 × 3 数据集的推理时间与显存 (line1/line2 CSV 测量, F=96)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig11_infer_memory.png', dpi=150, bbox_inches='tight')
plt.close()
print('  saved fig11_infer_memory.png')

print('Done. 2 figures saved to', SAVE_DIR)
