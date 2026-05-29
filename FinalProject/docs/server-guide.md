# 服务器训练指南 — Linux + RTX 4090 24G

## 一、项目部署

### 1. 上传项目到服务器
```bash
# 方式1: rsync (推荐)
rsync -avz --exclude='venv' --exclude='__pycache__' --exclude='.git' \
  ./FinalProject/ user@server:/path/to/FinalProject/

# 方式2: 先打包再scp
tar czf FinalProject.tar.gz --exclude='venv' --exclude='__pycache__' --exclude='.git' FinalProject/
scp FinalProject.tar.gz user@server:/path/to/
```

### 2. 服务器环境搭建
```bash
cd /path/to/FinalProject
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> 注意: `mamba-ssm` 需要 CUDA 编译，如果安装失败，项目已内置纯PyTorch替代实现，不影响运行。

### 3. 下载真实数据集
```bash
# ETTm2
wget -O dataset/ETTm2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv

# Weather
wget -O dataset/Weather.csv https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/Weather.csv

# Electricity
wget -O dataset/Electricity.csv https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/Electricity.csv

# 构建 Time-MMD 多模态数据集 (Energy, Environment, Health)
python -c "from data_provider.preprocess_timemmd import preprocess_all; preprocess_all()"
```

---

## 二、训练命令

### 单模型单数据集训练
```bash
source venv/bin/activate

# 基线模型
python run.py --model DLinear --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model DLinear --data ETTm2 --seq_len 96 --pred_len 192 --epochs 100
python run.py --model PatchTST --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model iTransformer --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100

# SOTA模型
python run.py --model TimeMixer --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model TimeKAN --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100

# 创新模型
python run.py --model KANiTransformer --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model MambaTransformerDual --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model MultimodalFusion --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100

# Chronos2 Zero-Shot (不需要训练，直接推理)
python run.py --model Chronos2 --data ETTm2 --seq_len 96 --pred_len 96
```

### 批量训练：所有模型 × 所有数据集 × 所有预测长度
```bash
# 完整实验 (8模型 × 4数据集 × 2×4配置 = 约200+次训练)
python run_experiments.py --epochs 100 --gpu 0

# 只跑部分 (推荐先跑这个验证)
python run_experiments.py \
  --models DLinear PatchTST iTransformer TimeMixer \
  --datasets ETTm2 Weather \
  --seq_lens 96 \
  --pred_lens 96 192 \
  --epochs 50 --gpu 0
```

### 消融实验
```bash
python run_ablation.py
```

---

## 三、预计训练时间 (RTX 4090)

| 模型 | ETTm2 (单次) | 全量 (4数据集×4长度) |
|------|-------------|-------------------|
| DLinear | ~2 min | ~30 min |
| PatchTST | ~5 min | ~1.5 h |
| iTransformer | ~5 min | ~1.5 h |
| TimeMixer | ~3 min | ~45 min |
| TimeKAN | ~10 min | ~3 h |
| KANiTransformer | ~20 min | ~5 h |
| MambaTransformerDual | ~10 min | ~3 h |
| MultimodalFusion | ~8 min | ~2 h |
| Chronos2 (Zero-Shot) | ~5 min (推理) | ~30 min |

**全量实验总计约 17-20 小时。**

### 后台运行 (推荐)
```bash
# 使用 nohup
nohup python run_experiments.py --epochs 100 --gpu 0 > train.log 2>&1 &

# 或使用 tmux
tmux new -s train
python run_experiments.py --epochs 100 --gpu 0
# Ctrl+B, D 分离会话
# tmux attach -t train 重新连接
```

---

## 四、结果收集

训练完成后，结果自动保存在：
```
results/
├── main_results.csv          # 主实验结果 (所有模型×数据集×预测长度)
├── zero_shot_results.csv     # Chronos2 Zero-Shot 结果
├── ablation_*.csv            # 消融实验结果
└── all_experiments_*.csv     # 批量实验汇总

figures/                      # 可视化图表
├── predictions_*.png         # 预测曲线对比
├── radar_*.png               # 雷达图
├── heatmap_*.png             # 热力图
├── efficiency_*.png          # 效率对比
├── freq_decomp_*.png         # 频域分解
└── attention_*.png           # 注意力可视化

checkpoints/                  # 最优模型检查点
└── {model}_{dataset}_{seq_len}_{pred_len}_checkpoint.pth
```

### 生成可视化图表
```bash
# 训练完成后生成所有图表
python -c "
from visualization.plot_predictions import plot_prediction_comparison
from visualization.plot_radar import plot_radar_chart
from visualization.plot_heatmap import plot_performance_heatmap
from visualization.plot_efficiency import plot_efficiency_comparison

csv = 'results/main_results.csv'
for dataset in ['ETTm2', 'Weather', 'Electricity', 'Energy', 'Environment', 'Health']:
    for pred_len in [96, 192, 336, 720]:
        plot_prediction_comparison(csv, dataset, 96, pred_len)
        plot_radar_chart(csv, dataset, pred_len)

plot_performance_heatmap(csv, 'MSE')
plot_performance_heatmap(csv, 'MAE')
plot_efficiency_comparison(csv)
print('All figures generated!')
"
```

### 启动 Gradio Demo
```bash
python app/gradio_demo.py
# 访问 http://localhost:7860
# 如需外网访问: python app/gradio_demo.py + 修改 share=True
```

---

## 五、下载结果到本地
```bash
# 从服务器下载结果
scp -r user@server:/path/to/FinalProject/results/ ./results/
scp -r user@server:/path/to/FinalProject/figures/ ./figures/
scp -r user@server:/path/to/FinalProject/checkpoints/ ./checkpoints/
```
