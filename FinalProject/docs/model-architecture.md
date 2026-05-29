# 模型架构与技术细节

## 一、基线模型（thuml 官方实现）

### 1.1 DLinear

**论文**：Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

**核心思想**：将时序分解为趋势和季节两个分量，分别用两个单层线性映射直接预测。质疑了 Transformer 在时序预测中的必要性。

**架构**：
```
输入 x [B, H, C]
  → series_decomp (移动平均, kernel=25)
    → 趋势 trend [B, H, C] → Linear(H, F) → 趋势预测 [B, F, C]
    → 季节 seasonal [B, H, C] → Linear(H, F) → 季节预测 [B, F, C]
  → 相加 → 输出
```

**参数量**：仅 `2 × H × F × C`（ETTm2 上约 0.02M）

**使用场景**：作为"试金石"——如果复杂模型连 DLinear 都打不过，说明该模型有问题。

### 1.2 PatchTST

**论文**：Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)

**核心思想**：
1. **通道独立**：每个变量单独建模，不做跨变量注意力
2. **Patching**：将连续时间步打包成长度为 patch_len=16 的 patch，每个 patch 作为一个 token

**架构**：
```
输入 x [B, H, C]
  → 对每个变量 c ∈ [1, C] 独立处理:
    → 切 patch (len=16, stride=8) → [B*C, num_patches, d_model]
    → Transformer Encoder (2层)
    → FlattenHead → Linear → [B*C, F]
  → 拼接所有变量 → [B, F, C]
```

**关键设计**：
- Patch 减少了序列长度，降低了注意力复杂度
- BatchNorm 作为 Encoder 的 norm（而非 LayerNorm）
- Non-stationary 归一化：减均值除标准差，推理时反归一化

### 1.3 iTransformer

**论文**：Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (AAAI 2024 Best Paper)

**核心思想**：将变量视为 token（而非时间步），注意力在变量之间计算。

**架构**：
```
输入 x [B, H, C]
  → 转置为 [B, C, H]（C 个变量，每个有 H 个时间步）
  → DataEmbedding_inverted: Linear(H, d_model) → [B, C, d_model]
  → Transformer Encoder (注意力在 C 个变量间计算)
  → Linear(d_model, F) → [B, C, F]
  → 转置回 [B, F, C]
```

**为什么"倒置"有效**：
- 变量间存在真实物理相关性（温度↔湿度），注意力学这种关系
- 变量数 C 通常远小于时间步 H，注意力计算更高效
- 避免了长序列注意力的二次复杂度

---

## 二、SOTA模型

### 2.1 TimeMixer

**论文**：Wang et al., "TimeMixer: Decomposable Multiscale Mixing" (ICLR 2024)

**核心思想**：时序具有多尺度特性（小时→天→周），在不同尺度上分别建模再跨尺度混合。

**关键模块**：

1. **多尺度生成**：通过平均池化逐级下采样
   ```
   尺度0: [B, H, C] (原始)
   尺度1: [B, H/2, C] (2倍下采样)
   尺度2: [B, H/4, C] (4倍下采样)
   ```

2. **PastDecomposableMixing**：
   - DFT 频域分解为 season + trend
   - Bottom-up season mixing（细→粗传递季节信息）
   - Top-down trend mixing（粗→细传递趋势信息）

3. **多预测器融合**：每个尺度独立预测，最终加权求和

**配置参数**：
- `down_sampling_window=2`：每级下采样倍率
- `down_sampling_layers=3`：下采样级数（共4个尺度）
- `channel_independence=1`：通道独立模式
- `decomp_method='moving_avg'`：序列分解方法

### 2.2 TimeKAN

**核心思想**：用 KAN (Kolmogorov-Arnold Network) 层替代传统 MLP 前馈网络。KAN 在网络的边上放置可学习的 B-spline 函数，而非固定的激活函数+线性权重。

**KAN vs MLP**：
```
MLP:  节点上有激活函数(ReLU/GELU)，边上有固定权重 W
KAN:  边上有可学习的 B-spline 函数 φ(x)，节点只做求和
```

**架构**：
```
输入 x [B, H, C]
  → 倒置 [B, C, H] → Linear(H, d_model) → [B, C, d_model]
  → 位置编码
  → ×N KANBlock:
    → LayerNorm
    → KANLayer(d_model, d_ff, d_model) (替代FFN)
    → 残差连接
  → Linear(d_model, F) → [B, C, F]
  → 转置 → [B, F, C]
```

**KANLinear 实现要点**：
- 使用 efficient-kan 的简化实现
- B-spline 用 grid + 基函数求和近似
- 支持 3D 输入 `[B, L, D]`，内部 flatten 到 2D 计算

---

## 三、创新模型

### 3.1 KAN-iTransformer

**创新点**：KAN层替代MLP + FFT自适应频域三分支分解

**架构**：
```
输入 x [B, H, C]
  → 归一化
  → FFT频域分解 → 趋势 + 季节 + 残差
  → 三个分支各自:
    → DataEmbedding_inverted (变量作为token)
    → KANEncoderLayer × N (KAN层替代FFN)
  → 自适应门控融合 (learned weights)
  → Linear(d_model, F)
  → 反归一化
```

**频域分解原理**：
```
对 x [B, H, C] 做 FFT → 频谱 [B, H/2+1, C]

趋势: 保留 DC + 前2个低频分量 → irfft → x_trend
季节: 取幅度最大的 top_k=5 个频率 → irfft → x_seasonal
残差: x - x_trend - x_seasonal → x_residual
```

**KANEncoderLayer**：
```
标准 EncoderLayer 但 FFN 替换为:
  LayerNorm → KANLayer(d_model, d_ff, d_model) → 残差
(注意力部分不变)
```

### 3.2 Mamba-Transformer 双专家路由

**创新点**：FFT频域路由器 + Mamba长程专家 + Transformer短程专家

**架构**：
```
输入 x [B, H, C]
  → 归一化 → DataEmbedding → [B, H, d_model]
  
  → FreqRouter(FFT分析):
    对 x 做 rfft → 取幅度谱 → MLP → softmax → w1, w2

  → Mamba专家: MambaLayer(d_model) → out_mamba [B, H, d_model]
  → Transformer专家: EncoderLayer → out_tf [B, H, d_model]

  → 加权融合: w1·out_mamba + w2·out_tf
  → 取最后时间步 → Linear → [B, F, C]
  → 反归一化
```

**Mamba专家**（简化实现）：
```
输入投影 → Conv1d(局部依赖) → x_proj(SSM参数) → 选择性扫描:
  for t in range(L):
    h = decay[t] * h + x[t] * (1 - decay[t])
  → 门控(z分支) → 输出投影
```

### 3.3 多模态融合

**创新点**：三模态编码 + InfoNCE对比对齐 + 自适应门控融合

**数据集来源**：使用 Time-MMD (Wang et al., NeurIPS 2024) 提供的多模态数据集，涵盖 Energy（能源价格+新闻）、Environment（空气质量+环境报告）、Health（流感监测+CDC报告）三个领域。

**架构**：
```
输入:
  x_enc [B, H, C]          # 时序
  text_embed [B, text_dim]  # 文本嵌入
  img_tensor [B, 1, 32, 32] # 递归图

编码:
  TS:   DataEmbedding + Transformer Encoder → ts_feat [B, d_model]
  Text: Linear(text_dim, d_model) → text_feat [B, d_model]
  Img:  Conv2d(1→16→32) + AdaptiveAvgPool + Linear → img_feat [B, d_model]

对比损失 (训练时):
  L_contrast = InfoNCE(ts_feat, text_feat) + InfoNCE(ts_feat, img_feat)
  InfoNCE: -log(exp(sim(a+, b+)) / Σ exp(sim(a, b_i)))

融合:
  gate = sigmoid(Linear(concat_dim, 3))
  fused = gate[0]*ts + gate[1]*text + gate[2]*img (缺失模态用零填充)

输出:
  fused → Linear → [B, F, C]
```

**InfoNCE 对比损失**：
- 对同一时刻的 (ts_feat, text_feat) 拉近（正样本对）
- 对不同时刻的表示推远（负样本对）
- 温度参数 τ=0.07 控制集中度

**多模态数据特点**：
- Time-MMD 数据集同时提供 report（正式报告）和 search（搜索相关）两类文本
- Energy: 能源新闻报告 354条 + 搜索 2307条
- Environment: 环境报告 156条 + 搜索 2272条
- Health: CDC报告 489条 + 搜索 1994条
- 由于多模态数据集频率为周/日级别，样本量相对较小（857-15979行），需要使用较短的 seq_len (24-48) 和 pred_len (12-48)

---

## 四、概率预测与共形预测

### 4.1 分位数回归

不输出单点预测值，而是同时输出多个分位数（0.05, 0.5, 0.95）：

```
QuantileHead:
  Linear(d_model, F×C) × 3  → q_0.05, q_0.50, q_0.95

Pinball Loss:
  L_q = max(q·(y - ŷ_q), (1-q)·(y - ŷ_q))
```

### 4.2 共形预测

在分位数回归基础上，利用校准集给出**理论保证**的置信区间：

```
1. 在校准集上计算非一致性分数:
   s_i = max(q_0.05(x_i) - y_i, y_i - q_0.95(x_i))

2. 计算 conformal quantile:
   q_hat = quantile(s_1,...,s_n, min(1, (n+1)(1-α)/n))

3. 调整预测区间:
   lower = q_0.05 - q_hat
   upper = q_0.95 + q_hat
```

**理论保证**：P(lower ≤ y ≤ upper) ≥ 1-α（当α=0.05时，覆盖率≥95%）

---

## 五、模型仲裁集成

### 5.1 特征提取

从输入序列提取5维统计特征：

| 特征 | 计算方法 | 含义 |
|------|---------|------|
| 谱熵 | FFT→概率分布→-Σp·log(p) | 频谱复杂度 |
| 趋势强度 | 线性拟合R² | 趋势明显程度 |
| 周期性 | FFT主频率相对强度 | 周期性明显程度 |
| 方差 | var(x) | 波动幅度 |
| 自相关 | lag-1 autocorrelation | 序列惯性 |

### 5.2 路由器

```
特征 [B, 5] → MLP(5→64→GELU→n_models) → Softmax → weights [B, n_models]

预测 = Σ(weights[i] × model_i_prediction)
```

### 5.3 训练流程

1. 先训练所有单独模型并保存最优权重
2. 冻结所有模型参数
3. 只训练路由器的 MLP 参数（使用 MSE 损失）
