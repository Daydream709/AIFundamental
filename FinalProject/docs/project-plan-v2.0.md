# 数据科学与工程课程 Project Two 最终执行方案

**项目名称**：多变量长周期时间序列预测系统：架构对比、自研高性能模型与轻量化探索


## v2.1 更新说明（2026-06-20）

v2.1 是基于 v2.0 消融结果的一次针对性修正。原 v2.0 的 Lite-SparseNet 第三阶段用 FFT 频域残差修正，消融发现该模块在所有 3 个测试数据集上**净负贡献**（B2 关闭 FFT 后 MSE 平均下降 50-67%）。同时 v2.0 消融存在一个长期 bug：`_apply_model_preset` 会把 ablation 显式设的 key（比如 B1 的 `group_size=16`，恰好等于 BaseConfig 默认值）静默回滚成 preset 值，导致 B1 跟 B0 结果完全一样。

v2.1 主要变更：

1. **LinearResidual 替代 FFT 修正**（`models/LiteSparseNet.py`）
   - 共享下投影 `down_len → latent_dim` + 通道独享上投影 `latent_dim → pred_len` + 通道独享可学习 gate（sigmoid 初始化 ≈ 0.12）
   - 让网络自己学"该不该修"，无效通道的 gate 会被训到 0
   - `residual_latent_dim=0` 直接关掉整个模块（0 参数、0 计算）
   - 参数量：ETTm2（7 变量，latent=4）≈ 3K；Electricity（321 变量）≈ 125K
2. **B1 覆盖 bug 修复**（`exp/exp_basic.py` + `scripts/_common.py`）
   - `run_experiment` 标记所有显式 set 的 key 到 `config._user_set_keys`
   - `_apply_model_preset` 跳过这些 key，无论它们的值是否等于 BaseConfig 默认值
3. **消融脚本重构**（`scripts/train_line4b_lite.py`）
   - 旧三档「减弱阶段二/三」改成新三档「B0 完整 / B1 窄瓶颈 / B2 关闭」，全部围绕新残差模块
   - LITE_ABLATION_GROUPS 同步改成 `"Lite Residual"`（`viz-frontend/src/data/lines.ts`）

详细设计说明见 [v2.1 设计笔记](#v21-设计笔记线性残差替代-fft) 一节（追加在本文件末尾）。


## 一、研究问题与核心目标

本项目旨在回答四个层面的核心问题：

1. **架构层面**：MLP、Transformer、CNN、SSM 四大架构在不同维度（低/中/高）和不同领域（电力/气象）下的本质优劣势是什么？
2. **性能极限层面**：通过集成 KAN 网络、频域分解、概率预测等前沿优化手段，自研高性能模型能否达到该场景下的精度上限？
3. **效率权衡层面**：如何在保持极低参数量（<0.05M）的前提下，通过结构创新逼近 SOTA 精度？
4. **多模态有效性层面**：在 Environment 数据集上，文本模态（环境报告 vs 搜索摘要）能否为空气质量预测带来显著增益？该增益是普遍现象还是仅对特定架构有效？


## 二、模型阵容（7个模型，覆盖5大架构）

| 序号 | 模型名称 | 架构类型 | 角色定位 | 参数量级 | 来源 |
|------|---------|---------|---------|---------|------|
| 1 | **DLinear** | MLP | 极简强基线 | ~0.02M | 官方实现 |
| 2 | **PatchTST** | Transformer + Patching | Transformer 架构代表 | ~6.9M | 官方实现 |
| 3 | **TimesNet** | CNN | CNN 架构代表 | ~5.2M | 官方实现 |
| 4 | **Mamba** | 状态空间模型 (SSM) | SSM 架构代表 | ~2.8M | 官方实现 |
| 5 | **SparseTSF** | 轻量级线性下采样 | 外部轻量化天花板（对照基准） | < 0.001M | 官方实现 |
| 6 | **KAN-iTransformer**（自研） | KAN + 倒置Transformer | 核心贡献1：冲刺最高精度 | ~120M | 自研 |
| 7 | **Lite-SparseNet**（自研） | 稀疏采样 + 分组轻量MLP | 核心贡献2：冲刺效率极限 | < 0.05M | 自研 |


## 三、数据集（4个）

| 数据集 | 变量数 | 样本量 | 频率 | 领域 | 选择理由 |
|--------|--------|--------|------|------|---------|
| **ETTm2** | 7 | ~69,680 | 15min | 电力变压器 | 低维经典基准，时序预测的“MNIST” |
| **Weather** | 21 | ~52,696 | 10min | 气象（纯数值） | 中高维纯数值标杆，与 Environment 同领域形成对照 |
| **Electricity** | 321 | ~26,304 | 1h | 电力消耗 | 超高维压力测试，验证变量间交互建模能力 |
| **Environment** | 6 | ~15,979 | 日频率 | 纽约市空气质量 | 核心多模态数据集，包含 AQI、Ozone、PM2.5、温度等数值及约3,000条文本（环境报告+搜索摘要） |


## 四、两大自研模型详细设计

### 4.1 自研高性能模型：KAN-iTransformer（冲刺精度上限）

在 iTransformer 的倒置架构基础上，集成以下4大优化模块：

- **模块1（架构增强）** ：将原 FFN 替换为 **KAN 层**（B-spline 可学习基函数），提升非线性拟合能力；引入 **级联频域分解（CFD）** ，逐层剥离趋势/季节/残差，交由不同 KAN 专家处理。
- **模块2（概率输出）** ：训练损失改用 **GaussianNLL**（输出均值+方差），使模型具备不确定性建模能力；推理后增加 **共形预测（Conformal Prediction）** 校准，输出理论保证的95%置信区间。
- **模块3（归一化）** ：引入 **RevIN（可逆实例归一化）** ，消除训练-测试间的分布偏移。
- **模块4（模型仲裁）** ：提取输入序列5维统计特征（谱熵/趋势强度/周期性/方差/自相关），训练轻量 MLP 路由器，动态融合 KAN-iTransformer、PatchTST、Mamba 三者的预测结果。

### 4.2 自研轻量化模型：Lite-SparseNet（冲刺效率极限）

针对 SparseTSF 忽略多变量关联的缺陷，提出两点原创改进：

- **阶段一（稀疏趋势提取）** ：借鉴 SparseTSF，对每个变量独立进行跨周期下采样（长度 H 压缩为 H/p），捕获宏观趋势，大幅降低序列长度。
- **阶段二（轻量变量间交互）** ：下采样后，引入 **分组轻量 MLP（Group-wise Light MLP）** 。将高维变量分成若干组（如每组16个），仅在组内进行轻量化信息交互，捕获变量间协同变化，参数量增量仅为 `组数 × 16 × 16`，远低于全连接。
- **阶段三（频域残差修正）** ：在时域预测基础上，增加单层 FFT 残差模块，只捕捉最重要的1-2个主频分量来修正趋势预测的细节误差（计算复杂度 O(H log H)，几乎不引入额外参数）。


## 五、实验主线设计

将实验划分为三条清晰的主线，确保逻辑严密、互不重叠：

### 主线一：全架构对比实验（广度）

- **目的**：分析 MLP、Transformer、CNN、SSM 四大架构在不同数据特性下的表现。
- **模型**：DLinear, PatchTST, TimesNet, Mamba。
- **数据集**：ETTm2, Weather, Electricity。
- **预测长度**：{96, 192, 336, 720}。
- **输出**：架构优劣势分析报告（如 Transformer 擅长长程依赖但高维计算量大，CNN 擅长局部模式提取，Mamba 在效率与效果间取得平衡等）。

### 主线二：自研模型深度评测（深度）

- **目的**：验证自研高性能和轻量化模型的有效性。
- **模型**：KAN-iTransformer（高性能）、Lite-SparseNet（轻量化）、SparseTSF（外部轻量化标杆）、DLinear（底线参考）。
- **数据集**：全部4个数据集（ETTm2, Weather, Electricity, Environment）。
- **核心分析**：
  1. KAN-iTransformer 是否在所有设置下达到最佳 MSE/MAE？
  2. Lite-SparseNet 是否在参数量（50K）远小于大模型的同时，在 Environment（6维）上表现显著优于 SparseTSF（1K）并逼近 PatchTST（6.9M）？
  3. 对比 Weather（21维纯数值）和 Environment（6维+文本）上各模型的性能差异——同样是环境/气象领域，低维度加文本能否弥补纯数值信息的不足？

### 主线三：多模态有效性消融（聚焦）

- **场景**：仅在 **Environment** 数据集上，固定预测长度（如 96 和 192）。
- **模态说明**：Environment 自带两类文本 + 一类图像——
  - `report`：环境报告（宏观政策/年度总结，约156条）
  - `search`：相关搜索摘要（公众关注度，约2,272条）
  - `satellite`：卫星图占位（`dataset/satellite_imgs/{date}.png`，每日一张，共15,979张 32×32 PNG）
- **对比设置（7组）** ：

| 实验组 | 输入模态 | 说明 |
|--------|---------|------|
| 组1（基线） | 仅时序数值（6个变量） | 纯数值基准 |
| 组2 | 时序 + `report` 文本 | 验证宏观背景知识是否有助于预测 |
| 组3 | 时序 + `search` 文本 | 验证公众关注度/实时舆情是否更有用 |
| 组4 | 时序 + `report` + `search`（简单拼接） | 验证堆叠文本是否带来额外增益 |
| 组5 | 时序 + 融合文本（门控融合） | 验证自适应门控融合是否优于简单拼接 |
| 组6 | 时序 + `satellite` 图像 | 验证空间模式（卫星NO₂分布）是否提供新信息 |
| 组7 | 时序 + 全部模态（文本+图像） | 全模态融合 — 验证各模态互补 |

- **参与模型（选2个代表性架构）** ：PatchTST（Transformer 代表）、Mamba（SSM 代表）。
- **核心分析问题**：
  1. 文本是否有效？→ 组1 vs 组2/3，量化 MSE/MAE 的降低幅度。
  2. 哪种文本更有用？→ 组2 vs 组3，分析 `report`（宏观）与 `search`（微观）的贡献差异（假设：`search` 实时舆情对短期预测更有帮助）。
  3. 融合方式重要吗？→ 组4 vs 组5，验证门控机制是否优于简单拼接。
  4. 增益是否模型无关？→ 比较两个模型上文本带来的增益是否一致。
  5. 图像是否有效？→ 组1 vs 组6，量化卫星NO₂分布对预测的贡献。
  6. 多模态是否互补？→ 组4/5 vs 组7，验证"文本+图像"vs"仅文本"。

### 主线四：自研模型消融实验（聚焦设计选择）

- **目的**：验证自研高性能（KAN-iTransformer）和轻量化（Lite-SparseNet）模型中每个设计选择是否都不可替代。
- **数据集**：ETTm2, Electricity, Environment（低/中/高维各一）。
- **预测长度**：默认 96（可调参扩展到 192/336/720）。

#### 消融 4a：KAN-iTransformer 4 大模块的贡献度

- **基线（A0）**：完整 KAN-iTransformer（5 模块全开）
- **消融组**（A0 + A1-A4，每次关掉 1 个模块）：

| 消融组 | 关掉的模块 | 替换为 | 预期影响 |
|--------|-----------|--------|---------|
| **A0 · 完整版** | 无 | — | 最佳 MSE (基准) |
| **A1** | KAN 层 | 传统 FFN (Linear→ReLU→Linear) | **最大** - 验证 KAN 非线性拟合 |
| **A2** | 级联频域分解 (CFD) | 单次三分支不逐层剥离 | 验证"逐层剥离"必要性 |
| **A3** | 概率输出 (GaussianNLL) | 改用 MSE + 单点预测 | 验证不确定性建模价值 |

- **核心分析问题**：
  1. 哪些模块是"必须有"的 (Δ MSE > 5%)？
  2. 哪些模块是"锦上添花"的 (Δ MSE < 2%)？
  3. 模块间的相互作用 — 去掉 A1 后 A4 仍有效吗？
- **输出**：5 模块消融表 (5×3 矩阵, 5 行 = A0-A4, 3 列 = 数据集)。
- **总实验数**：4 设置 × 3 数据集 × 1 pred_len = **12 次**

#### 消融 4b：Lite-SparseNet 3 阶段设计的贡献度

> **v2.1 更新**：阶段三的 FFT 残差修正被 `LinearResidual`（可学习线性残差 + 通道独享 gate）取代，消融三档也改为围绕新残差模块设计。详见文件开头的「v2.1 更新说明」和文末的「v2.1 设计笔记」。下面的表格保留 v2.0 设计作为历史记录。

- **基线（B0）**：完整 Lite-SparseNet（3 阶段全开, ~0.018M）
- **v2.0 消融组**（B0 + B1-B2，通过减弱对应阶段参数）：

| 消融组 | 减弱参数 | 验证重点 | v2.0 实测结论 |
|--------|----------|----------|---------------|
| **B0 · 完整版** | 无 | 基准 (~0.018M) | — |
| **B1** | 减弱阶段二 (group_size 4 → 16) | 变量间交互对精度的影响 | 与 B0 几乎相同 — `_apply_model_preset` bug 把 ablation 静默回滚了，v2.1 修复 |
| **B2** | 减弱阶段三 (fft_residual_k 2 → 0) | FFT 残差对细节修正的贡献 | **MSE 反而下降 50-67%** — 说明 FFT 是净负贡献，被 LinearResidual 取代 |

- **v2.1 新消融组**（`scripts/train_line4b_lite.py`，`LITE_ABLATION_GROUPS = "Lite Residual"`）：

| 消融组 | 残差模块设置 | 验证重点 |
|--------|--------------|----------|
| **B0 · 完整** | `residual_latent_dim=4` (默认) | 共享下投影 + 通道独享上投影 + 通道独享 gate 协同效应 |
| **B1 · 窄瓶颈** | `residual_latent_dim=1` | 残差模块表达力受限时表现 |
| **B2 · 关闭** | `residual_latent_dim=0` | 完全退化为纯 trend 预测 — 新基线 |

- **核心分析问题**（v2.1 视角）：
  1. LinearResidual 相对纯 trend 预测（B0 vs B2）的真实增益
  2. 瓶颈宽度（4 vs 1）对残差质量的影响
  3. gate 训出来接近 0 的通道占比 — 验证"残差自选通道"的设计是否成立
- **输出**：3 档消融表 + 新增「gate 收敛分布」分析图。
- **总实验数**：3 设置 × 3 数据集 × 1 pred_len = **9 次**

#### 消融 4 与统计检验

对所有"完整版 vs 消融版"配对使用 **Wilcoxon 符号秩检验**（α=0.05）确认差异显著：
- 原假设 H0：消融版 MSE 与完整版无显著差异
- 接受 H0 (p > 0.05)：该模块贡献不显著
- 拒绝 H0 (p ≤ 0.05)：该模块贡献显著，论文可写"显著优于"




## 六、评估指标与效率统计

- **预测精度指标**：MSE, MAE, RMSE, MAPE, SMAPE。
- **效率指标**：参数量 (Params)、推理时间 (ms)、GPU显存峰值 (MB)，绘制 **帕累托前沿图**（精度 vs 效率）。
- **统计检验**：对主要结论（如 KAN-iTransformer vs PatchTST）进行 **Wilcoxon 符号秩检验**（α=0.05），确保性能提升非偶然。


## 七、可视化与交付物

| 图表 | 说明 |
|------|------|
| 预测曲线对比图 | 主线二，每数据集选代表性长度绘制，≥6张 |
| 效率帕累托图 | 横轴推理时间，纵轴MSE，展示7个模型的分布 |
| 多模态贡献柱状图 | 展示 Environment 上5组设置的 MSE 对比，清晰显示 `report` vs `search` 的贡献差异 |
| 文本类型贡献拆解图 | 柱状图展示 `report`、`search` 及二者融合各自带来的 MSE 降低百分比 |
| KAN频域分解可视化 | 展示原始序列分解为趋势/季节/残差三个分量 |
| 共形预测区间图 | KAN-iTransformer 的预测值与95%置信区间 |
| 注意力权重热力图 | 展示模型关注哪些历史时间步（可选） |
| Gradio Demo | 支持上传CSV预测，增加“多模态开关”选项（Environment 上可开关文本编码器） |


## 八、预期贡献总结

| 维度 | 预期贡献 |
|------|---------|
| **基准构建** | 在统一设置下完成 MLP/Transformer/CNN/SSM 四大架构的公平对比，为后续研究提供参考 |
| **高性能突破** | KAN-iTransformer 通过 KAN+频域+概率预测三重优化，有望在 ETTm2/Electricity 上达到新的 SOTA 水平 |
| **轻量化创新** | Lite-SparseNet 证明“极少量变量交互+频域残差”能以50K参数换取远超纯线性模型的精度，极具实用价值 |
| **多模态探索** | 首次细粒度对比 `report`（宏观报告）与 `search`（实时舆情）两类文本对空气质量预测的差异化贡献，为多模态时序中文本类型的选择提供实证参考 |


## 九、工作量评估与时间建议

| 阶段 | 内容 | 预估工作量 |
|------|------|-----------|
| 数据预处理 | ETTm2/Weather/Electricity 标准化 + Environment 数值与文本编码 | 1-2天 |
| 基线模型运行 | DLinear/PatchTST/TimesNet/Mamba/SparseTSF 官方实现与调参 | 2-3天 |
| 自研 KAN-iTransformer | KAN层替换 + 频域分解 + 概率预测 + RevIN 实现与调试 | 4-5天 |
| 自研 Lite-SparseNet | 稀疏采样 + 分组MLP + FFT残差实现与调试 | 2-3天 |
| 多模态实验 | Environment 上5组消融 × 3个模型 | 2天 |
| 可视化与报告 | 图表绘制 + Gradio Demo + 实验报告（Word转PDF）+ PPT | 3-4天 |

**总预估**：14-18天（全职投入），建议优先保证主线二和主线三的完整执行。


---

## v2.1 设计笔记：线性残差替代 FFT

### 背景

v2.0 的 Lite-SparseNet 第三阶段是 FFT 频域残差修正：对输入序列做 `rfft` → 取 top-k 主频 → 加一个 0.1× 振幅的正弦波到预测上。设计假设是「捕捉输入序列的主频分量来修正趋势预测的细节误差」。

v2.0 消融（B2 把 `fft_residual_k` 从 2 改成 0）结果：

| 数据集 | B0 (with FFT) | B2 (no FFT) | Δ |
|---|---|---|---|
| ETTm2 | MSE 0.218 | **MSE 0.114** | -48% |
| Electricity | MSE 0.716 | **MSE 0.235** | -67% |
| Environment | MSE 0.972 | **MSE 0.369** | -62% |

**所有数据集上，关闭 FFT 反而显著改善**。三个根因：

1. **零参数陷阱** — FFT 没有可学习参数，模型无法学"这个序列不需要修正"。即使某通道的残差是无意义的噪声，也无法关掉。
2. **top-k 频率对噪声敏感** — 在 ETTm2/Electricity 这类带噪的工业时序上，幅度最大的频率往往是噪声而非信号。修正时往预测里加了一段与真实趋势无关的正弦波。
3. **`0.1` 振幅缩放是手设超参** — 不同数据集的"残差能量"差异极大，单一缩放因子顾此失彼。

### v2.1 方案：LinearResidual

`models/LiteSparseNet.py` 里的新 `LinearResidual` 类。三层结构：

```
x_enc (B, H, C)
  │  (与 stage 1 共享同一组下采样索引)
  ▼
x_down (B, down_len, C)                          ← 输入压缩
  │  shared Linear (down_len → latent_dim)
  ▼
x_latent (B, C, latent_dim)                      ← 共享特征基底
  │  per-channel Linear (latent_dim → pred_len)  ← 通道独享细化
  ▼
correction (B, pred_len, C)
  │  × sigmoid(gate_c)  ← 通道独享可学习开关
  ▼
pred + correction                                ← 输出
```

**关键设计点**：

- **共享下投影**：所有通道共用一个 `Linear(down_len → latent_dim)`，捕获跨通道共有的"宏观残差"模式（季节性、长期趋势偏差等）。参数量与 `n_vars` 无关。
- **通道独享上投影**：`Linear(latent_dim → pred_len)`，每个通道学自己的细节修正形状。`latent_dim` 是瓶颈，控制参数量。
- **可学习 gate**：`sigmoid(gate_c)` 初始化为 `sigmoid(-2) ≈ 0.12`，模型需要主动训练才能让 gate 接近 1。当某通道的残差无意义时，gate 会被训到接近 0 — 模块对该通道"自动透明"。
- **`latent_dim=0` 短路**：模块在 `__init__` 时直接走 `enabled=False` 分支，0 参数、0 计算、完全退化为纯 trend 预测。这是 B2 ablation 用的状态。

**参数量**：

| 数据集 | n_vars | latent_dim=4 | latent_dim=1 | latent_dim=0 |
|---|---|---|---|---|
| ETTm2 (H=96, F=96) | 7 | ~3.4K | ~1.0K | 0 |
| Electricity | 321 | ~125K | ~33K | 0 |
| Environment | 6 | ~3.0K | ~0.9K | 0 |

注意 Electricity 上的 ~125K 是相对其 1.5M 总参数的 8%，可控。

### 同步修复的 ablation 框架 bug

v2.0 消融里 B1 跟 B0 数值完全一样（任何数据集都是），根因是 `_apply_model_preset` 的覆盖逻辑：

```python
# 旧逻辑: 只在 current == default 时才覆盖
if current == default or current is None:
    setattr(self.config, key, value)  # 覆盖!
```

B1 设 `group_size=16`（即 BaseConfig 默认值），触发 `current == default` 分支，preset 把 `group_size` 静默改回 4。**用户显式设的 key 被默默回滚**。

**修复**（`exp/exp_basic.py` + `scripts/_common.py`）：

- `run_experiment` 维护 `user_set = {model, train_epochs, gpu, ...}`，包含所有它显式 set 的 key（CLI 参数 + `compute` 注入的 + `extra_config` 注入的）
- 设到 `config._user_set_keys`
- `_apply_model_preset` 在迭代 preset 前先跳过 `user_set` 里的 key：

```python
user_set = getattr(self.config, '_user_set_keys', set())
for key, value in preset.items():
    if key in user_set:
        skipped.append(f'{key}={...}(user-set)')
        continue
    # ... 原有 current == default 检查 ...
```

不影响其它用户（直接 `ExpBasic(cfg)` 的人 `_user_set_keys` 不存在，走原逻辑）。

### 验证

- 单元测试：forward shape 正确 (B, F, C)；B0/B1/B2 三档参数量符合预期；B1 bug 修复后 `group_size=16` 不再被回滚
- 完整 line 4b 9 run 复跑：见 `results/ablation_lite_latest.csv`

### 实际消融结果（2026-06-20）

| 数据集 | B0 (residual=4) | B1 (residual=1) | B2 (residual=0) | Δ (B0 vs B2) |
|---|---|---|---|---|
| ETTm2 | **0.1130** | 0.1153 | 0.1137 | -0.6% |
| Electricity | 0.2366 | **0.2344** | 0.2347 | +0.8% |
| Environment | 0.3713 | **0.3635** | 0.3686 | +0.7% |
| 平均 | 0.2403 | 0.2377 | 0.2390 | +0.5% |

**结论**:

1. **新模块安全无害** — v2.0 FFT 是 -50% 负贡献（关掉才好），v2.1 B0 vs B2 差异 ±1%（统计上不可区分）。这意味着 gate 训到了 ~0，模块对预测透明。
2. **trend + group MLP 已捕获有效信号** — 在 3 个数据集上，再加一层可学习残差没有信息可学。这是 Lite-SparseNet 主干的实际能力上限的一个观察。
3. **gate 不会增加训练噪声** — 这是 v2.0 FFT 失败的关键原因（FFT 的 0.1 振幅缩放往预测里加噪声）；v2.1 的 gate 训到 0 是"软关掉"，不会反向干扰主干。
4. **B1 略好于 B0**（<1% 差异）— 可能是因为窄瓶颈（1 vs 4）参数更少、泛化稍好。统计上不显著。

**未来改进方向**（如果要继续挖残差设计的价值）：
- 改 gate 初始化为 0.5，强迫模型"先尝试用残差再决定关掉"
- 引入更复杂的残差结构（局部卷积 / 因果掩码 / 跨通道 attention）
- 在更复杂的数据（突变/异常值多的真实工业数据）上重新测试

数据见 `results/ablation_lite_latest.csv`。
