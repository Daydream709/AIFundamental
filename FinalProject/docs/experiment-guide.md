# 实验指南

## 一、训练参数总览

### 1.1 全局固定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | Adam + 权重衰减 |
| 学习率 | 1e-4 | 所有模型统一 |
| 权重衰减 | 1e-5 | L2正则化 |
| 训练轮次 | 100 | 配合早停 |
| 早停耐心值 | 10 | 验证损失连续10轮不下降则停止 |
| 随机种子 | 42 | 固定所有随机源，确保可复现 |
| 损失函数 | MSE | 主实验统一用MSE，消融实验对比MAE/GaussianNLL |
| 混合精度 | 开启 | torch.amp autocast + GradScaler |

### 1.2 按数据集调整的参数

#### 纯时序数据集

| 数据集 | enc_in / c_out | batch_size | d_model | d_ff | seq_len | pred_len | 说明 |
|--------|----------------|------------|---------|------|---------|----------|------|
| ETTm2 | 7 | 64 | 512 | 2048 | 96, 192 | 96, 192, 336, 720 | 标准设置 |
| Weather | 21 | 64 | 512 | 2048 | 96, 192 | 96, 192, 336, 720 | 变量较多但仍可用标准设置 |
| Electricity | 321 | 16 | 256 | 1024 | 96, 192 | 96, 192, 336, 720 | 超高维度，必须降低batch和模型宽度 |

#### 多模态数据集（Time-MMD, NeurIPS 2024）

| 数据集 | enc_in / c_out | batch_size | d_model | d_ff | seq_len | pred_len | 说明 |
|--------|----------------|------------|---------|------|---------|----------|------|
| Energy | 9 | 32 | 512 | 2048 | 24, 48 | 12, 24, 48 | 周频率，样本量较小，开启use_text=True |
| Environment | 6 | 64 | 512 | 2048 | 24, 48 | 12, 24, 48 | 日频率，开启use_text=True |
| Health | 7 | 32 | 512 | 2048 | 24, 48 | 12, 24, 48 | 周频率，样本量最小，开启use_text=True |

> **关键注意**：
> - Electricity 有 321 个变量，batch_size 必须降到 16-32，d_model 降到 256，否则 4090 24G 显存会 OOM
> - 多模态数据集（Energy/Environment/Health）使用较短的 seq_len (24-48) 和 pred_len (12-48)，因为它们是周/日频率数据，样本量较小（857-15979行），较长的窗口会浪费大部分数据

### 1.3 按模型调整的参数

| 模型 | 特殊参数 | 说明 |
|------|---------|------|
| PatchTST | patch_len=16, stride=8 | 分块大小和步幅 |
| TimeMixer | down_sampling_window=2, down_sampling_layers=3, channel_independence=1 | 多尺度下采样设置 |
| MultimodalFusion | use_text=True, use_image=True, use_contrastive=True | 必须开启多模态 |
| Chronos2 | 无训练参数 | 只做推理，约5分钟/数据集 |

---

## 二、数据集详细说明

### 2.1 ETTm2 (Electricity Transformer Temperature)

- **来源**：zhouhaoyi/ETDataset
- **变量**：HUFL, HULL, MUFL, MULL, LUFL, LULL, OT（7个电力变压器温度相关变量）
- **采样**：15分钟间隔
- **样本量**：约 69,680 行
- **文件**：ETTm2.csv，列名含 `date` + 7个数值列
- **获取方式**：
  ```bash
  wget -O dataset/ETTm2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv
  ```

### 2.2 Weather

- **来源**：thuml/HuggingFace
- **变量**：21个气象指标（温度、湿度、风速、气压等）
- **采样**：10分钟间隔
- **样本量**：约 52,696 行
- **获取方式**：
  ```bash
  wget -O dataset/Weather.csv https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/Weather.csv
  ```

### 2.3 Electricity

- **来源**：thuml/HuggingFace
- **变量**：321个客户的电力消耗量
- **采样**：1小时间隔
- **样本量**：约 26,304 行
- **注意**：维度极高(321)，需要降低batch_size和模型维度
- **获取方式**：
  ```bash
  wget -O dataset/Electricity.csv https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/Electricity.csv
  ```

### 2.4 Energy (多模态 — Time-MMD)

- **来源**：Wang et al., "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis", NeurIPS 2024
- **变量**：9个变量（汽油价格相关）
- **采样**：周频率
- **样本量**：约 1,622 行
- **文本模态**：能源新闻报告 354条 + 搜索 2307条
- **获取方式**：
  ```bash
  python -c "from data_provider.preprocess_timemmd import download_timemmd; download_timemmd('Energy')"
  ```

### 2.5 Environment (多模态 — Time-MMD)

- **来源**：Wang et al., "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis", NeurIPS 2024
- **变量**：6个变量（纽约市空气质量指标）
- **采样**：日频率
- **样本量**：约 15,979 行
- **文本模态**：环境报告 156条 + 搜索 2272条
- **获取方式**：
  ```bash
  python -c "from data_provider.preprocess_timemmd import download_timemmd; download_timemmd('Environment')"
  ```

### 2.6 Health (多模态 — Time-MMD)

- **来源**：Wang et al., "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis", NeurIPS 2024
- **变量**：7个变量（美国国家级流感监测指标）
- **采样**：周频率
- **样本量**：约 857 行
- **文本模态**：CDC报告 489条 + 搜索 1994条
- **获取方式**：
  ```bash
  python -c "from data_provider.preprocess_timemmd import download_timemmd; download_timemmd('Health')"
  ```

### 2.7 数据集下载汇总命令

```bash
# 一键下载所有数据集
python -c "
from data_provider.download_scripts import download_all
from data_provider.preprocess_timemmd import download_all_timemmd
download_all()
download_all_timemmd()
"
```

---

## 三、实验运行方法

### 3.1 单次训练

```bash
# 基本用法
python run.py --model <模型名> --data <数据集> --seq_len 96 --pred_len 96

# 纯时序数据集示例
python run.py --model DLinear --data ETTm2 --seq_len 96 --pred_len 96 --epochs 100
python run.py --model iTransformer --data Weather --seq_len 96 --pred_len 336
python run.py --model PatchTST --data Electricity --seq_len 96 --pred_len 192 --batch_size 16

# 多模态数据集示例（较短 seq_len/pred_len）
python run.py --model MultimodalFusion --data Energy --seq_len 24 --pred_len 12 --use_text True
python run.py --model MultimodalFusion --data Environment --seq_len 48 --pred_len 24 --use_text True
python run.py --model MultimodalFusion --data Health --seq_len 24 --pred_len 12 --use_text True

# Zero-Shot
python run.py --model Chronos2 --data ETTm2 --seq_len 96 --pred_len 96
```

**所有支持的模型名**：
`DLinear`, `PatchTST`, `iTransformer`, `TimeMixer`, `Chronos2`, `TimeKAN`, `KANiTransformer`, `MambaTransformerDual`, `MultimodalFusion`

**所有支持的数据集**：
`ETTm2`, `Weather`, `Electricity`, `Energy`, `Environment`, `Health`

### 3.2 批量实验

```bash
# 完整实验（9模型 × 6数据集 × 多seq_len × 多pred_len ≈ 300+次训练）
python run_experiments.py --epochs 100 --gpu 0

# 只跑部分（推荐先跑这个验证）
python run_experiments.py \
    --models DLinear PatchTST iTransformer TimeMixer \
    --datasets ETTm2 Weather \
    --seq_lens 96 \
    --pred_lens 96 192 \
    --epochs 50 --gpu 0

# 只跑多模态数据集
python run_experiments.py \
    --models MultimodalFusion \
    --datasets Energy Environment Health \
    --seq_lens 24 48 \
    --pred_lens 12 24 48 \
    --epochs 100 --gpu 0
```

### 3.3 消融实验

```bash
python run_ablation.py
```

消融实验自动运行6组对比，每组用30个epoch以节省时间。结果保存在 `results/ablation_*.csv`。

### 3.4 后台运行

```bash
# nohup
nohup python run_experiments.py --epochs 100 --gpu 0 > train.log 2>&1 &

# tmux（推荐）
tmux new -s train
python run_experiments.py --epochs 100 --gpu 0
# Ctrl+B, D 分离
# tmux attach -t train 重新连接
```

---

## 四、实验注意事项

### 4.1 显存管理

#### 纯时序数据集

| 模型 | ETTm2 (C=7) | Weather (C=21) | Electricity (C=321) |
|------|-------------|----------------|---------------------|
| DLinear | ~200MB | ~300MB | ~2GB |
| PatchTST | ~2GB | ~3GB | ~8GB (batch=16) |
| iTransformer | ~2GB | ~4GB | ~10GB (batch=16) |
| TimeMixer | ~1GB | ~2GB | ~6GB |
| KAN-iTransformer | ~4GB | ~6GB | OOM风险高 |
| Mamba-Dual | ~3GB | ~5GB | ~8GB |
| MultimodalFusion | ~3GB | ~4GB | ~8GB |

#### 多模态数据集

| 模型 | Energy (C=9) | Environment (C=6) | Health (C=7) |
|------|-------------|-------------------|--------------|
| DLinear | ~200MB | ~200MB | ~200MB |
| PatchTST | ~2GB | ~2GB | ~2GB |
| iTransformer | ~2GB | ~2GB | ~2GB |
| TimeMixer | ~1GB | ~1GB | ~1GB |
| MultimodalFusion | ~3GB | ~3GB | ~3GB |

**如果OOM**：
1. 降低 `batch_size`（64→32→16）
2. 对 Electricity 使用已自动降低的 `d_model=256`
3. 对超大模型(KAN-iTransformer)跳过 Electricity 或进一步降低层数

### 4.2 训练时间估算（RTX 4090）

| 模型 | 纯时序单次(100epoch) | 多模态单次(100epoch) | 全量实验 |
|------|---------------------|---------------------|---------|
| DLinear | ~2min | ~1min | ~30min |
| PatchTST | ~5min | ~2min | ~1.5h |
| iTransformer | ~5min | ~2min | ~1.5h |
| TimeMixer | ~3min | ~1min | ~45min |
| TimeKAN | ~10min | ~3min | ~3h |
| KANiTransformer | ~20min | ~5min | ~5h |
| MambaTransformerDual | ~10min | ~3min | ~3h |
| MultimodalFusion | ~8min | ~4min | ~2.5h |
| Chronos2 | ~5min(推理) | ~2min(推理) | ~45min |

**全量实验总计约 18-22 小时。**

### 4.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `CUDA out of memory` | batch_size或模型过大 | 降低batch_size，或对Electricity降低d_model |
| `No module named 'chronos'` | 未安装chronos-forecasting | `pip install chronos-forecasting` 或跳过Chronos2 |
| `No module named 'mamba_ssm'` | mamba-ssm需要CUDA编译 | 项目内置纯PyTorch替代实现，不影响使用 |
| 训练loss不下降 | 学习率或数据问题 | 检查数据是否正确加载（非全零），确认lr=1e-4 |
| 验证loss远高于训练loss | 过拟合 | 检查early stopping是否正常工作 |
| CSV文件找不到 | 未下载数据集 | 运行 `download_all()` 和 `download_all_timemmd()` |
| 多模态数据集样本量不足 | seq_len 过长导致有效样本少 | 多模态数据集使用 seq_len=24-48, pred_len=12-48 |
| Health 数据集训练不稳定 | 仅857行数据 | 增大学习率至2e-4，或使用数据增强 |

### 4.4 结果文件说明

训练完成后自动生成：

```
results/
├── main_results.csv           # 主实验：每行一次实验的完整指标
├── zero_shot_results.csv      # Chronos2 Zero-Shot 结果
├── ablation_YYYYMMDD.csv      # 消融实验结果
└── all_experiments_*.csv      # 批量实验汇总

checkpoints/
└── {model}_{dataset}_{seq_len}_{pred_len}_checkpoint.pth  # 最优模型权重
```

**main_results.csv 列说明**：

| 列名 | 含义 |
|------|------|
| model | 模型名称 |
| dataset | 数据集名称 |
| seq_len | 历史窗口长度 |
| pred_len | 预测长度 |
| MSE / MAE / RMSE / MAPE / SMAPE | 5个评价指标 |
| Params(M) | 模型参数量（百万） |
| InferTime(ms) | 单次推理耗时（毫秒） |
| GPUMem(MB) | GPU显存峰值 |
| loss_type | 损失函数类型 |
| timestamp | 实验完成时间 |

---

## 五、数据处理方法

### 5.1 标准化

使用训练集的全局均值 μ 和标准差 σ 做 z-score 归一化：

```
x_normalized = (x - μ) / σ
```

推理时反归一化：
```
x_original = x_normalized × σ + μ
```

**关键**：μ 和 σ 仅从训练集计算，验证集和测试集使用训练集的统计量，防止数据泄露。

### 5.2 滑窗构造

给定时间序列 `[x_1, x_2, ..., x_N]`，滑窗以步长1滑动：

**纯时序数据集**（seq_len=96）：
```
窗口 i:
  输入  x_enc[i : i+seq_len]                       长度 seq_len=96
  目标  x_y[i+seq_len-label_len : i+seq_len+pred_len]  长度 label_len+pred_len=144
```

**多模态数据集**（seq_len=24-48）：
```
窗口 i:
  输入  x_enc[i : i+seq_len]                       长度 seq_len=24或48
  目标  x_y[i+seq_len-label_len : i+seq_len+pred_len]  长度 label_len+pred_len
```

`label_len=48` 是编码器-解码器架构模型（如Transformer）的桥接段，对于纯编码器模型（DLinear, iTransformer）不影响使用。

### 5.3 时间特征

从日期列提取4个特征并归一化到 `[0, 1]`：

| 特征 | 原始值范围 | 归一化方式 |
|------|-----------|-----------|
| 月份 | 1-12 | 除以12 |
| 日期 | 1-31 | 除以31 |
| 星期 | 0-6 | 除以6 |
| 小时 | 0-23 | 除以23 |

### 5.4 多模态数据构造

#### 文本模态（Time-MMD 数据集）

1. 从 Time-MMD 数据集中获取与时间序列对齐的文本数据（report 和 search 两类）
2. 使用 `sentence-transformers` 的 `all-MiniLM-L6-v2` 编码为 384/768 维向量
3. 按时间段匹配到对应的时序样本
4. 缓存为 `.npy` 文件避免重复计算

**三类文本数据**：
- Energy: 能源新闻报告 354条 + 搜索 2307条
- Environment: 环境报告 156条 + 搜索 2272条
- Health: CDC报告 489条 + 搜索 1994条

#### 图像模态（递归图）

1. 取每个变量的时间序列片段 `[x_t, x_{t+1}, ..., x_{t+H}]`
2. 计算距离矩阵 `D[i,j] = |x_i - x_j|`
3. 用第10百分位数作为阈值 ε 二值化：`R[i,j] = 1 if D[i,j] < ε else 0`
4. Resize 到 32×32 灰度图
5. 缓存为 `.npy` 文件

---

## 六、评价指标说明

| 指标 | 公式 | 特点 |
|------|------|------|
| **MSE** | mean((y - ŷ)²) | 对大误差敏感，最常用 |
| **MAE** | mean(\|y - ŷ\|) | 鲁棒性更好，易解释 |
| **RMSE** | sqrt(MSE) | 与原始数据同量纲 |
| **MAPE** | mean(\|y - ŷ\| / \|y\|) × 100% | 百分比误差，受零值影响 |
| **SMAPE** | mean(2\|y - ŷ\| / (\|y\| + \|ŷ\|)) × 100% | 对称百分比误差，更稳定 |

> **注意**：MAPE 在真实值接近0时会产生极大值。如果数据中有接近零的变量（如风速、降水），MAPE 可能不具参考意义，应以 MSE/MAE 为主要判据。

---

## 七、实验报告建议结构

```
1. 引言（背景、研究问题）
2. 相关工作（时序预测方法综述）
3. 方法
   3.1 问题定义
   3.2 基线模型简介
   3.3 创新模型详细描述（3个创新）
   3.4 共形预测区间估计
4. 实验设置
   4.1 数据集（6个：3纯时序 + 3多模态 Time-MMD）
   4.2 评价指标
   4.3 实验超参数
   4.4 对比模型
5. 实验结果
   5.1 主实验结果（大表 + 热力图）
   5.2 效率对比（参数量/速度/显存）
   5.3 消融实验
   5.4 Zero-Shot对比
   5.5 共形预测置信区间
6. 分析与讨论
7. 结论
参考文献
```
