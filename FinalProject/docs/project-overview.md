# 项目概述

## 一、项目简介

本项目是**数据科学与工程课程 Project Two**——多模态长周期时间序列预测系统。

**核心任务**：给定历史窗口长度 H 的多维时间序列输入，预测未来长度 F 的时序走势，即实现 `H×C → F×C` 的映射。在此基础上，引入文本模态和图像模态进行多模态融合预测，并对比全监督训练模型与 Zero-Shot 基础大模型的性能差异。

**关键词**：长周期时序预测、Transformer、Mamba、KAN、多模态融合、Zero-Shot、共形预测

---

## 二、研究问题

> 在多变量长周期时间序列预测任务中，如何通过架构创新（KAN层、Mamba双专家路由、多模态融合）和概率预测方法（共形预测）提升预测精度和可靠性？基础大模型的 Zero-Shot 能力又与全监督训练模型存在多大差距？

---

## 三、模型阵容（9个）

### 3.1 基线模型（来自 thuml/Time-Series-Library 官方实现）

| 模型 | 论文 | 架构类型 | 参数量 | 角色 |
|------|------|---------|--------|------|
| **DLinear** | AAAI 2023 | MLP | ~0.02M | 极简强基线，性能试金石 |
| **PatchTST** | ICLR 2023 | Transformer+Patching | ~6.9M | 通道独立+分块编码经典方案 |
| **iTransformer** | AAAAI 2024 Best Paper | 倒置Transformer | ~6.4M | 长期预测传统SOTA，创新改造基础 |

### 3.2 2024-2025 SOTA 模型

| 模型 | 论文 | 架构类型 | 参数量 | 角色 |
|------|------|---------|--------|------|
| **TimeMixer** | ICLR 2024 | 多尺度混合 | ~4.3M | 多尺度分解+跨尺度混合最新方案 |
| **TimeKAN** | 自研 | KAN网络 | ~37.9M | 用可学习B-spline函数替代MLP |

### 3.3 时序基础大模型（Zero-Shot）

| 模型 | 来源 | 说明 | 角色 |
|------|------|------|------|
| **Chronos2** | Amazon Science | 预训练时序基础大模型，直接推理不需训练 | Zero-Shot vs 全监督对比 |

### 3.4 自研创新模型（核心贡献）

| 模型 | 创新点 | 参数量 | 基座 |
|------|--------|--------|------|
| **KAN-iTransformer** | KAN层替代MLP + 自适应频域三分支分解 | ~120.5M | iTransformer |
| **Mamba-Transformer Dual** | 频域路由器 + Mamba/Transformer双专家 | ~10.8M | Transformer + Mamba |
| **Multimodal Fusion** | 三模态编码 + InfoNCE对比对齐 + 门控融合 | ~9.2M | Transformer |

---

## 四、数据集

本项目使用 6 个数据集，其中 3 个纯时序数据集 + 3 个多模态数据集。

### 4.1 纯时序数据集（3个）

| 数据集 | 变量数 | 样本量 | 频率 | 来源 | 特点 |
|--------|--------|--------|------|------|------|
| **ETTm2** | 7 | ~69,680 | 15min | zhouhaoyi/ETDataset | 电力变压器温度，经典基准，中等维度 |
| **Weather** | 21 | ~52,696 | 10min | thuml/HuggingFace | 气象多变量，中高维度 |
| **Electricity** | 321 | ~26,304 | 1h | thuml/HuggingFace | 电力消耗，超高维度挑战 |

### 4.2 多模态数据集（3个）— 来自 Time-MMD (NeurIPS 2024)

> **引用**：Wang et al., "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis", NeurIPS 2024

| 数据集 | 变量数 | 样本量 | 频率 | 时序内容 | 文本模态 |
|--------|--------|--------|------|---------|---------|
| **Energy** | 9 | ~1,622 | 周频率 | 汽油价格 | 能源新闻报告 354条 + 搜索 2307条 |
| **Environment** | 6 | ~15,979 | 日频率 | 纽约市空气质量 | 环境报告 156条 + 搜索 2272条 |
| **Health** | 7 | ~857 | 周频率 | 美国流感监测（国家级） | CDC报告 489条 + 搜索 1994条 |

### 数据集划分

所有数据集按时间顺序划分为三部分：

| 划分 | 比例 | 用途 |
|------|------|------|
| 训练集 | 0% ~ 70% | 模型参数学习 |
| 验证集 | 70% ~ 85% | 超参数调优、早停判断 |
| 测试集 | 85% ~ 100% | 最终性能评估 |

### 数据预处理流程

```
原始CSV → 缺失值处理 → 标准化(训练集统计量) → 滑窗切分 → 时间特征提取
```

1. **标准化**：使用训练集的均值和标准差做 z-score 归一化，避免数据泄露
2. **滑窗切分**：以 `(seq_len + label_len + pred_len)` 为窗口长度滑动，每次移动1步
3. **时间特征**：提取 月/日/周几/小时 四个归一化特征 `[0, 1]`

---

## 五、实验设计

### 5.1 主实验

**9模型 × 6数据集 × 多预测长度** 的全组合对比实验。

评价指标：MSE, MAE, RMSE, MAPE, SMAPE

### 5.2 消融实验（6组）

| 组号 | 消融变量 | 对比设置 |
|------|---------|---------|
| 1 | 文本模态 | 无文本 vs 有文本 vs 文本+对比对齐 |
| 2 | 图像模态 | 无递归图 vs 有递归图 |
| 3 | 架构增强 | 纯Transformer vs +Mamba专家 vs +KAN层 |
| 4 | 损失函数 | MSE vs MAE vs GaussianNLL |
| 5 | 频域分解 | 无频域分解 vs 有频域分解 |
| 6 | 集成策略 | 单模型最优 vs 仲裁集成 |

### 5.3 Zero-Shot 实验

Chronos2 在 6 个数据集上直接推理（不训练），与全监督模型对比。

### 5.4 效率对比

| 指标 | 说明 |
|------|------|
| Parameters (M) | 模型可训练参数总量 |
| Inference Time (ms) | 单次前向推理耗时 |
| GPU Memory (MB) | 训练时显存峰值 |

### 5.5 统计检验

使用 **Wilcoxon 符号秩检验** 验证模型间性能差异的统计显著性（α=0.05）。

---

## 六、创新点摘要

### 创新1：KAN-Enhanced iTransformer

- **动机**：传统MLP前馈层表达能力有限，KAN的B-spline可学习函数具有更强的非线性拟合能力
- **方法**：将iTransformer每个Encoder层的FFN替换为KAN层；增加FFT自适应频域分解（趋势/季节/残差三分支）
- **预期效果**：更少参数达到更高精度

### 创新2：Mamba-Transformer 双专家路由

- **动机**：不同时序模式需要不同架构——长程平滑趋势适合Mamba，短程剧烈波动适合Transformer
- **方法**：FFT频域路由器自动分析输入频谱，动态分配两个专家的权重
- **预期效果**：在不同数据特性上自适应，综合性能优于任何单一架构

### 创新3：跨模态对比对齐融合

- **动机**：外部文本信息（如能源政策新闻、环境报告、CDC流感报告）与时序数值存在因果关系，但特征分布差异大
- **方法**：时序Transformer + 文本MLP编码器 + 图像CNN编码器 → InfoNCE对比损失对齐 → 门控融合
- **数据集来源**：使用 Time-MMD (NeurIPS 2024) 提供的多模态数据集（Energy、Environment、Health），涵盖能源、环境和健康三个领域
- **预期效果**：在多模态数据集上显著提升预测精度

### 创新4：共形预测区间估计

- **动机**：点预测无法量化不确定性，实际决策需要置信区间
- **方法**：分位数回归 + 校准集非一致性分数 → 理论保证的95%覆盖率

### 创新5：模型仲裁集成

- **动机**：没有单一模型在所有场景下最优
- **方法**：提取输入序列的5维统计特征（谱熵/趋势强度/周期性/方差/自相关），MLP路由器动态分配各模型权重

---

## 七、可视化与交付

### 静态图表（≥15张）

- 预测曲线对比图（每数据集每预测长度1张，共≥8张）
- 五维雷达图（精度/速度/参数量/长程/短程）
- 热力图（模型×数据集×预测长度性能矩阵）
- 注意力权重热力图
- 频域分解可视化（原始/趋势/季节/残差）
- 效率对比柱状图
- 消融实验分组图
- 共形预测置信区间图

### Gradio 交互式Demo

- 单模型预测 + 模型仲裁模式
- 支持上传CSV / 选择内置数据集
- 实时预测曲线 + 指标表 + 推理耗时

---

## 八、参考文献

1. Zeng et al., "Are Transformers Effective for Time Series Forecasting?" AAAI 2023
2. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" ICLR 2023
3. Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" IJCAI 2024
4. Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" ICLR 2024
5. Liu et al., "KAN: Kolmogorov-Arnold Networks" 2024
6. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" 2024
7. Ansari et al., "Chronos: Learning the Language of Time Series" 2024
8. Wang et al., "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis" NeurIPS 2024
