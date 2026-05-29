# 项目结构详解

## 一、顶层目录

```
FinalProject/
├── configs/            # 超参数配置
├── data_provider/      # 数据加载与预处理
├── models/             # 模型定义
├── layers/             # 共享网络层
├── exp/                # 训练/测试实验流程
├── utils/              # 工具函数
├── visualization/      # 图表生成
├── app/                # Gradio Web Demo
├── third_party/        # 第三方库（thuml/Time-Series-Library）
├── dataset/            # 数据集存放（CSV文件）
├── checkpoints/        # 训练好的模型权重
├── results/            # 实验结果（CSV）
├── figures/            # 生成的可视化图表
├── docs/               # 项目文档
├── run.py              # 主入口（单模型单次训练）
├── run_experiments.py  # 批量实验运行器
├── run_ablation.py     # 消融实验运行器
└── requirements.txt    # Python依赖
```

---

## 二、configs/ — 超参数配置

| 文件 | 职责 |
|------|------|
| `base_config.py` | 定义 `BaseConfig` dataclass，包含所有超参数的默认值 |
| `dataset_configs.py` | 提供 `get_dataset_config()` 函数，返回每个数据集专属的配置覆盖 |

### 配置参数分类

**数据参数**：
- `data` / `data_path`：数据集名称和文件路径
- `seq_len`（H）：历史窗口长度（纯时序数据集默认96，多模态数据集默认24-48）
- `pred_len`（F）：预测长度（纯时序数据集默认96-720，多模态数据集默认12-48）
- `enc_in` / `c_out`：输入/输出变量数，随数据集自动设置

**模型参数**：
- `d_model=512`：Transformer隐藏维度
- `n_heads=8`：注意力头数
- `e_layers=2`：编码器层数
- `d_ff=2048`：前馈网络中间维度
- `dropout=0.1`：Dropout率

**训练参数**：
- `train_epochs=100`：最大训练轮次
- `batch_size=64`：批大小（Electricity自动降为16）
- `learning_rate=1e-4`：AdamW学习率
- `weight_decay=1e-5`：权重衰减
- `patience=10`：早停耐心值
- `use_amp=True`：混合精度训练开关

**多模态参数**（创新模型专用）：
- `use_text` / `use_image`：是否启用文本/图像模态
- `use_contrastive`：是否启用对比学习损失
- `text_dim=768`：文本嵌入维度
- `img_size=32`：递归图尺寸

### 使用方式

```python
from configs.dataset_configs import get_dataset_config
config = get_dataset_config('ETTm2', seq_len=96, pred_len=192)
# 自动设置 enc_in=7, c_out=7, data_path='ETTm2.csv' 等
```

---

## 三、data_provider/ — 数据管道

| 文件 | 职责 |
|------|------|
| `dataset_base.py` | `BaseDataset` 类，所有数据集的父类 |
| `data_factory.py` | `data_provider()` 工厂函数，根据config返回DataLoader |
| `download_scripts.py` | 自动下载纯时序数据集（ETTm2/Weather/Electricity） |
| `preprocess_timemmd.py` | 下载和预处理 Time-MMD 多模态数据集（Energy/Environment/Health） |
| `multimodal_builder.py` | 生成文本嵌入和递归图（多模态专用） |

### BaseDataset 核心逻辑

```
__read_data__():
    1. 读取CSV，分离数值列和日期列
    2. 按比例划分 train/val/test
    3. 用训练集统计量做 z-score 标准化
    4. 提取时间特征 [月, 日, 周几, 小时] → 归一化到 [0,1]

__getitem__(index):
    返回 (x_enc, x_y, x_mark_enc, x_mark_y, [text_embed], [img_tensor])
    - x_enc:     [seq_len, C]          历史输入
    - x_y:       [label_len+pred_len, C] 目标序列（含桥接段）
    - x_mark_enc: [seq_len, 4]          输入时间特征
    - x_mark_y:   [label_len+pred_len, 4] 目标时间特征
```

### 数据集汇总

**纯时序数据集（3个）**：

| 数据集 | 变量数 | 样本量 | 频率 | 来源 |
|--------|--------|--------|------|------|
| ETTm2 | 7 | ~69,680 | 15min | zhouhaoyi/ETDataset |
| Weather | 21 | ~52,696 | 10min | thuml/HuggingFace |
| Electricity | 321 | ~26,304 | 1h | thuml/HuggingFace |

**多模态数据集（3个）— Time-MMD (NeurIPS 2024)**：

| 数据集 | 变量数 | 样本量 | 频率 | 时序内容 | 文本模态 |
|--------|--------|--------|------|---------|---------|
| Energy | 9 | ~1,622 | 周频率 | 汽油价格 | 能源新闻报告 354条 + 搜索 2307条 |
| Environment | 6 | ~15,979 | 日频率 | 纽约市空气质量 | 环境报告 156条 + 搜索 2272条 |
| Health | 7 | ~857 | 周频率 | 美国流感监测 | CDC报告 489条 + 搜索 1994条 |

### 多模态数据

- **文本嵌入**：使用 `sentence-transformers` 预计算文本向量，缓存为 `.npy` 文件
- **递归图**：将1D时序转换为2D距离矩阵，阈值化后 resize 到 32×32 灰度图
- **Time-MMD 预处理**：`preprocess_timemmd.py` 负责：
  1. 从 Time-MMD 仓库下载数据
  2. 对齐时序和文本数据的时间戳
  3. 提取 report 和 search 两类文本数据
  4. 生成标准化的 CSV 和 JSON 文件

---

## 四、models/ — 模型定义

| 文件 | 类型 | 来源 |
|------|------|------|
| `DLinear.py` | 基线 | thuml 官方实现 |
| `PatchTST.py` | 基线 | thuml 官方实现 |
| `iTransformer.py` | 基线 | thuml 官方实现 |
| `TimeMixer.py` | SOTA | thuml 官方实现 |
| `Chronos2.py` | Zero-Shot | thuml 官方实现 |
| `TimeKAN.py` | 创新 | 自研，基于KAN层 |
| `kan_iTransformer.py` | 创新 | 自研，KAN+频域分解+倒置架构 |
| `mamba_transformer_dual.py` | 创新 | 自研，双专家路由 |
| `multimodal_fusion.py` | 创新 | 自研，三模态对比融合 |

### 统一接口

所有模型遵循相同的 forward 签名：

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    """
    x_enc:     [B, H, C] 输入时序
    x_mark_enc: [B, H, 4] 输入时间特征
    x_dec:     [B, F, C] 解码器输入（通常为零张量）
    x_mark_dec: [B, F, 4] 解码器时间特征
    Returns:   [B, F, C] 预测输出
    """
```

### 模型注册

`models/__init__.py` 维护 `MODEL_REGISTRY` 字典，将模型名称映射到对应的类。`exp/exp_basic.py` 通过此注册表动态加载模型。

---

## 五、layers/ — 共享网络层

| 文件 | 内容 |
|------|------|
| `Embed.py` | 桥接到 thuml 的嵌入层（DataEmbedding, DataEmbedding_inverted, PatchEmbedding 等） |
| `Transformer_EncDec.py` | 桥接到 thuml 的 EncoderLayer / Encoder / Decoder |
| `SelfAttention_Family.py` | 桥接到 thuml 的 FullAttention / AttentionLayer |
| `Autoformer_EncDec.py` | 桥接到 thuml 的 series_decomp（移动平均分解） |
| `StandardNorm.py` | 桥接到 thuml 的 Normalize 层 |
| `kan_layers.py` | **自研** — KANLinear / KANLayer（B-spline 可学习函数） |
| `MambaBlock.py` | **自研** — 简化版 Mamba/SSM 模块 |
| `frequency_decomp.py` | **自研** — FFT频域分解 + 频域路由器 |
| `contrastive_loss.py` | **自研** — InfoNCE 对比损失 |
| `gating_fusion.py` | **自研** — 自适应门控融合网络 |
| `conformal_prediction.py` | **自研** — 分位数回归头 + 共形预测 |
| `meta_arbitrator.py` | **自研** — 元学习模型仲裁器 |

### 与 thuml 的关系

- 标准层（Embed, Transformer, Attention, Norm）直接桥接 thuml 官方实现
- 自研层（KAN, Mamba, 频域分解, 对比损失, 门控融合, 共形预测, 仲裁器）放在项目 layers/ 下，同时拷贝到 `third_party/TimeSeriesLibrary/layers/` 以保证 import 路径一致

---

## 六、exp/ — 实验流程

| 文件 | 职责 |
|------|------|
| `exp_basic.py` | 基类：模型构建、设备选择、优化器/损失函数获取 |
| `exp_train.py` | `ExpTrain`：完整的 train → validate → test 流程 |
| `exp_zero_shot.py` | `ExpZeroShot`：Chronos2 直接推理（无需训练） |

### 训练流程

```
ExpTrain.train():
    1. 创建 train/val DataLoader
    2. 初始化 AdamW 优化器 + GradScaler(AMP)
    3. for epoch in range(epochs):
        a. _train_epoch(): 前向 → 损失 → 反向 → 更新（含AMP）
        b. _val_epoch(): 验证集评估
        c. EarlyStopping: 验证损失不下降则计数，达patience则停止
        d. 保存最优模型
    4. 加载最优模型
    5. test(): 在测试集上计算5个指标
    6. 记录结果到 CSV
```

---

## 七、utils/ — 工具函数

| 文件 | 功能 |
|------|------|
| `metrics.py` | MSE, MAE, RMSE, MAPE, SMAPE 指标计算 + 高斯NLL损失 |
| `tools.py` | `fix_seed(42)` 固定随机种子 + `EarlyStopping` 早停类 |
| `efficiency.py` | 参数量统计、推理时间测量、GPU显存测量 |
| `statistical_tests.py` | Wilcoxon 符号秩检验 + 模型对两两检验 |
| `result_logger.py` | `ResultLogger`：实验结果自动记录到CSV |

---

## 八、visualization/ — 可视化模块

| 文件 | 产出 |
|------|------|
| `plot_predictions.py` | 预测曲线对比图（Truth vs 多模型 + 置信区间） |
| `plot_radar.py` | 五维雷达图（精度/速度/参数量/长程/短程） |
| `plot_heatmap.py` | 模型×数据集×预测长度 MSE/MAE 热力图 |
| `plot_attention.py` | 注意力权重热力图 + Mamba隐状态可视化 |
| `plot_frequency.py` | 频域分解可视化 + FFT频谱图 |
| `plot_efficiency.py` | 参数量/推理速度/精度 三面板柱状图 |
| `plot_ablation.py` | 消融实验分组柱状图 |

---

## 九、app/ — Gradio Demo

`gradio_demo.py` 提供两个页面：

1. **Single Model Prediction**：选择数据集/模型/预测长度 → 展示预测曲线 + 指标
2. **Model Arbitration**：对比所有模型，自动推荐最优

启动方式：`python app/gradio_demo.py`，访问 `http://localhost:7860`

---

## 十、third_party/ — 第三方依赖

| 目录 | 内容 |
|------|------|
| `TimeSeriesLibrary/` | thuml/Time-Series-Library 完整仓库（shallow clone） |

此目录是项目的核心依赖：
- 提供 DLinear/PatchTST/iTransformer/TimeMixer/Chronos2 的官方实现
- 提供标准 layers（Embed, Transformer_EncDec, SelfAttention_Family, Autoformer_EncDec, StandardNorm）
- 提供 utils（masking, timefeatures, metrics 等）
