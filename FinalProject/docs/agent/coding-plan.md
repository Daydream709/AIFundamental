# 多模态长周期时序预测项目 - 代码实现计划

## Context

数据科学课程 Project Two：多模态长周期时间序列预测。项目要求实现7+模型（3基线 + 2个2025 SOTA + 1个基础大模型 + 3个创新自研模型），在4个数据集上进行完整实验，包含消融实验、Zero-Shot对比、可视化系统和交互式Demo。

**用户选择**：模块化多文件结构 | TimeMixer++ + TimeKAN作为SOTA | Linux + RTX 4090 24G | EPA-Air自行构建

---

## 项目目录结构

```
FinalProject/
├── configs/
│   ├── base_config.py             # 统一超参数配置(dataclass)
│   └── dataset_configs.py         # 4个数据集各自的配置覆盖
├── data_provider/
│   ├── __init__.py
│   ├── data_factory.py            # 统一数据集加载调度器
│   ├── dataset_base.py            # 基础时序滑窗Dataset类
│   ├── dataset_etth.py            # ETTm2数据集
│   ├── dataset_weather.py         # Weather数据集
│   ├── dataset_electricity.py     # Electricity数据集
│   ├── dataset_epa_air.py         # EPA-Air多模态数据集
│   ├── multimodal_builder.py      # 文本嵌入 + 递归图生成
│   └── download_scripts.py        # 自动下载脚本
├── models/
│   ├── __init__.py                # 模型注册表
│   ├── DLinear.py                 # 基线1: 极简MLP
│   ├── PatchTST.py                # 基线2: 分块Transformer
│   ├── iTransformer.py            # 基线3: 倒置Transformer
│   ├── TimeMixer.py               # SOTA1: TimeMixer++ (ICLR 2025)
│   ├── TimeKAN.py                 # SOTA2: KAN网络用于时序
│   ├── Chronos2.py                # Zero-Shot: 亚马逊时序基础大模型
│   ├── kan_iTransformer.py        # 创新1: KAN增强iTransformer
│   ├── mamba_transformer_dual.py  # 创新2: Mamba-Transformer双专家路由
│   └── multimodal_fusion.py       # 创新3: 跨模态对比对齐融合
├── layers/
│   ├── __init__.py
│   ├── Transformer_EncDec.py      # Transformer编码器/解码器
│   ├── SelfAttention_Family.py    # 注意力机制族
│   ├── Embed.py                   # 嵌入层(时间/数据/补丁)
│   ├── Autoformer_EncDec.py       # series_decomp分解模块
│   ├── MambaBlock.py              # Mamba/SSM模块
│   ├── StandardNorm.py            # 归一化层
│   ├── kan_layers.py              # KANLinear + KAN层(efficient-kan)
│   ├── frequency_decomp.py        # FFT/DWT频域分解模块
│   ├── contrastive_loss.py        # InfoNCE对比损失
│   ├── gating_fusion.py           # 自适应门控融合网络
│   ├── conformal_prediction.py    # 共形预测区间估计
│   └── meta_arbitrator.py         # 元学习仲裁器
├── exp/
│   ├── __init__.py
│   ├── exp_basic.py               # 实验基类
│   ├── exp_train.py               # 统一训练/验证/测试循环
│   ├── exp_zero_shot.py           # Chronos2 Zero-Shot推理
│   └── exp_ablation.py            # 自动化消融实验运行器
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # MSE, MAE, RMSE, MAPE, SMAPE
│   ├── tools.py                   # 早停/学习率调度/随机种子
│   ├── efficiency.py              # 参数量/FLOPs/推理时间/GPU显存统计
│   ├── statistical_tests.py       # Wilcoxon符号秩检验
│   └── result_logger.py           # CSV结果记录/最优模型保存
├── visualization/
│   ├── __init__.py
│   ├── plot_predictions.py        # 预测曲线对比图
│   ├── plot_radar.py              # 五维雷达图
│   ├── plot_heatmap.py            # 模型×数据集×预测长度热力图
│   ├── plot_attention.py          # 注意力/Mamba状态可视化
│   ├── plot_frequency.py          # 频域分解可视化
│   ├── plot_efficiency.py         # 效率对比图
│   └── plot_ablation.py           # 消融实验图表
├── app/
│   └── gradio_demo.py             # Gradio交互式Demo
├── results/                       # 实验结果CSV输出目录
├── checkpoints/                   # 模型检查点保存目录
├── dataset/                       # 数据集存放目录
├── figures/                       # 可视化图表输出目录
├── run.py                         # 主入口(CLI)
├── run_experiments.py             # 批量实验运行器
├── run_ablation.py                # 消融实验运行器
├── requirements.txt
└── docs/                          # 已有的文档
```

---

## 实现阶段（按依赖顺序）

### 阶段1: 基础设施 + 配置 + 数据管道

**创建文件**：
1. `configs/base_config.py` — 统一超参数dataclass
2. `configs/dataset_configs.py` — 4个数据集配置
3. `utils/tools.py` — seed固定(42)、EarlyStopping
4. `utils/metrics.py` — 5个指标函数
5. `utils/result_logger.py` — CSV记录器
6. `data_provider/download_scripts.py` — 自动下载
7. `data_provider/dataset_base.py` — 基础滑窗Dataset
8. `data_provider/data_factory.py` — 统一加载调度

**关键设计**：
- 所有Dataset统一返回 `(x_enc, x_mark_enc, x_dec, x_mark_dec, [text_embed], [img_tensor])`
- x_enc: `[B, H, C]`, x_mark_enc: `[B, H, 4]` (月/日/周/时)
- 多模态部分(text_embed, img_tensor)可选为None

**数据集来源**：
- ETTm2/Weather/Electricity: 从GitHub/HuggingFace标准链接下载CSV
- EPA-Air: 自行构建 — 从EPA AQS下载空气质量数据 + 配对政策新闻文本

### 阶段2: 共享层模块

**创建文件**：
1. `layers/Autoformer_EncDec.py` — series_decomp (移动平均分解)
2. `layers/Embed.py` — DataEmbedding / DataEmbedding_inverted / PatchEmbedding
3. `layers/Transformer_EncDec.py` — EncoderLayer / Encoder / DecoderLayer / Decoder
4. `layers/SelfAttention_Family.py` — FullAttention / ProbAttention 等
5. `layers/StandardNorm.py` — Normalize层

**来源**：直接从 thuml/Time-Series-Library 移植，保持接口兼容。

### 阶段3: 3个基线模型

**创建文件**：
1. `models/DLinear.py` — 移植自thuml，依赖series_decomp
2. `models/PatchTST.py` — 移植自thuml，依赖PatchEmbedding
3. `models/iTransformer.py` — 移植自thuml，依赖DataEmbedding_inverted

**统一forward签名**：
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # 返回: [B, pred_len, C]
```

**归一化模式**：Non-stationary Transformer范式 — 输入时计算mean/std做归一化，输出时反归一化。

### 阶段4: 统一训练管线

**创建文件**：
1. `exp/exp_basic.py` — 实验基类(模型构建、优化器、损失)
2. `exp/exp_train.py` — 完整train/val/test循环

**特性**：
- 混合精度: `torch.cuda.amp.autocast()` + `GradScaler`
- AdamW, lr=1e-4, weight_decay=1e-5
- EarlyStopping patience=10
- 自动保存最优模型到 `checkpoints/`
- GPU显存不足时自动梯度累积

**验证**：DLinear在ETTm2上训练5个epoch，MSE应<0.5

### 阶段5: 2025 SOTA模型

**创建文件**：
1. `models/TimeMixer.py` — TimeMixer++，移植自thuml并增强
2. `models/TimeKAN.py` — KAN网络用于时序预测

**TimeMixer++关键**：多尺度分解 + DFT频率滤波 + 跨尺度混合
**TimeKAN关键**：KAN层替代MLP + 时序特征嵌入

**额外依赖**：
- `layers/kan_layers.py` — efficient-kan的KANLinear实现
- `layers/MambaBlock.py` — Mamba/SSM模块(为阶段6准备)

### 阶段6: Chronos2 Zero-Shot

**创建文件**：
1. `models/Chronos2.py` — 包装chronos-forecasting库
2. `exp/exp_zero_shot.py` — Zero-Shot推理循环

**实现**：使用`chronos-forecasting`的`BaseChronosPipeline`，逐变量预测后组装。无训练过程，直接推理。

### 阶段7: 3个创新模型（核心）

**创建文件**：

#### 创新1: KAN-iTransformer (`models/kan_iTransformer.py`)
- 基于iTransformer架构
- 将Encoder中的MLP前馈层替换为KAN层
- 加入自适应频域分解(FFT): x → x_trend + x_seasonal + x_residual
- 三分支各自经KAN-Transformer编码后拼接
- 依赖: `layers/kan_layers.py`, `layers/frequency_decomp.py`

#### 创新2: Mamba-Transformer双专家路由 (`models/mamba_transformer_dual.py`)
- 频域分析路由器: 对输入做FFT提取频域特征，输出2专家权重
- 长程专家: Mamba/SSM处理低频全局趋势(线性复杂度)
- 短程专家: 局部窗口Transformer处理高频局部波动
- 加权组合: output = w1*mamba_out + w2*transformer_out
- 依赖: `layers/MambaBlock.py`

#### 创新3: 跨模态对比对齐融合 (`models/multimodal_fusion.py`)
- TS编码器: Transformer → ts_feat [B, d_model]
- 文本编码器: Linear(text_dim, d_model) → text_feat
- 图像编码器: CNN(Conv2d) → img_feat
- InfoNCE对比损失: 同一时刻ts_feat和text_feat靠近
- 自适应门控: gate_weights = sigmoid(Linear) → 加权融合
- 依赖: `layers/contrastive_loss.py`, `layers/gating_fusion.py`

**多模态数据构建** (`data_provider/multimodal_builder.py`)：
- 文本嵌入: sentence-transformers预计算，缓存为.npy
- 递归图: 1D时序→2D灰度距离矩阵，resize到32×32

### 阶段8: 共形预测 + 仲裁集成

**创建文件**：
1. `layers/conformal_prediction.py` — 共形预测区间估计
   - 分位数回归头: 3个Linear层分别输出0.05/0.5/0.95分位数
   - 校准集计算非一致性分数
   - 调整区间实现精确95%覆盖率
2. `layers/meta_arbitrator.py` — 元学习仲裁器
   - 提取输入统计特征: 谱熵/趋势强度/周期性/方差/自相关
   - MLP路由器: [B, 5] → [B, n_models] softmax权重
   - 加权组合各模型预测

### 阶段9: 效率统计 + 统计检验

**创建文件**：
1. `utils/efficiency.py` — 参数量/FLOPs/推理时间/GPU显存
2. `utils/statistical_tests.py` — Wilcoxon符号秩检验

### 阶段10: 批量实验 + 消融

**创建文件**：
1. `run.py` — 主CLI入口
2. `run_experiments.py` — 批量: 8模型 × 4数据集 × 4预测长度
3. `run_ablation.py` — 6组消融实验自动化

**消融实验矩阵**：
| 组 | 变量 | 对比 |
|----|------|------|
| 1 | 文本模态 | 无文本 vs 有文本 vs 文本+对比对齐 |
| 2 | 图像模态 | 无图像 vs 有递归图 |
| 3 | 架构增强 | 纯Transformer vs +Mamba专家 vs +KAN层 |
| 4 | 损失函数 | MSE vs GaussianNLL vs 共形预测 |
| 5 | 频域分解 | 无频域分解 vs 有频域分解 |
| 6 | 集成策略 | 单模型最优 vs 仲裁集成 |

### 阶段11: 可视化 (≥15张图)

**创建文件**：`visualization/` 下7个模块
1. 预测曲线对比图 (Truth vs 8模型, 半透明置信区间)
2. 五维雷达图 (精度/速度/参数量/长程/短程)
3. 热力图 (模型×数据集×预测长度)
4. 注意力/Mamba状态热力图
5. 频域分解可视化 (原始+趋势+季节+残差)
6. 效率对比柱状图 (参数量/FLOPs/速度)
7. 消融实验分组柱状图 (6组)

### 阶段12: Gradio交互式Demo

**创建文件**：`app/gradio_demo.py`
- 左侧: 上传CSV / 选择数据集 / 选择预测长度
- 中间: 选择模型(多选) / 勾选置信区间
- 右侧: 实时预测曲线 + 指标表 + 推理耗时
- 底部: 架构示意图 + 参数统计
- Killer功能: "模型仲裁模式" — 自动推荐最优模型

---

## 关键技术决策

1. **模型来源**: DLinear/PatchTST/iTransformer/TimeMixer从thuml/Time-Series-Library移植，保持成熟实现
2. **KAN实现**: 使用efficient-kan (Blealtan/efficient-kan)的KANLinear，非pykan
3. **Mamba**: 使用mamba-ssm包，安装失败时回退到纯PyTorch简化实现
4. **EPA-Air构建**: 下载EPA AQS空气质量数据 + 爬取EPA新闻政策文本 + sentence-transformers预计算嵌入
5. **Electricity高维(321变量)**: 降低batch_size到16-32，使用channel independence
6. **混合精度**: 全程开启torch.cuda.amp，显存减半，4090上加速

## 依赖 (requirements.txt)

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
gradio>=4.0
fvcore
chronos-forecasting
transformers
sentence-transformers
mamba-ssm
opencv-python
scipy
tqdm
```

## 验证策略

1. 每个阶段完成后单独测试该阶段产出
2. 基线模型在ETTm2上5 epoch的MSE与已知benchmark对比
3. 创新模型与对应基线(iTransformer)对比，验证改进效果
4. 共形预测检查测试集上95%覆盖率
5. 仲裁集成不低于最优单模型性能
6. Gradio Demo能正确加载模型并展示预测结果
