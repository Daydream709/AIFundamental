# 多变量长周期时间序列预测系统 (FinalProject)

> 数据科学与工程课程 Project Two — 多变量时序预测，覆盖 **7 个模型**（4 个 thuml 基线 + 3 个自研）、**4 个数据集**、**5 条消融主线**。
>
> 默认分支：`mps-perf`（包含 MPS 性能优化 + LiteSparseNet v2.1 LinearResidual）。
> 稳定分支：`main`（不含 MPS 专项优化，CUDA 友好）。

---

## 目录

- [0. TL;DR — 给接手同事的 5 分钟速览](#0-tldr--给接手同事的-5-分钟速览)
- [1. 项目结构](#1-项目结构)
- [2. 环境搭建](#2-环境搭建)
- [3. 数据集](#3-数据集)
- [4. 模型阵容](#4-模型阵容)
- [5. 怎么跑实验](#5-怎么跑实验)
- [6. 输出格式与数据流向](#6-输出格式与数据流向)
- [7. 可视化](#7-可视化)
- [8. 自研模型改进指南](#8-自研模型改进指南)
- [9. 计算资源 (M5 vs 4090)](#9-计算资源-m5-vs-4090)
- [10. 分支策略](#10-分支策略)
- [11. 常见问题](#11-常见问题)
- [12. 进阶阅读](#12-进阶阅读)

---

## 0. TL;DR — 给接手同事的 5 分钟速览

```bash
# 1) 装环境
cd FinalProject
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) 数据集在 dataset/ 下, 已就绪, 不用下载
ls dataset/   # 应该有 ETTm2.csv / Weather.csv / Electricity.csv / Environment.csv

# 3) 跑一条消融主线, 全部 9 个 run
python scripts/train_line4b_lite.py        # 轻量化自研模型消融, ~15 分钟 (M5) / 更短 (4090)

# 4) 看结果
cat results/ablation_lite_latest.csv       # 当前最新结果
cat results/ablation_lite_*.csv           # 历史快照 (按时间戳)
```

更多主线：
- `python scripts/train_line1.py` — 4 架构 × 3 数据集 × 4 pred_len = 48 runs（耗时最长）
- `python scripts/train_line2.py` — 3 自研模型 × 3 数据集 × 4 pred_len = 36 runs
- `python scripts/train_line3.py` — 多模态融合消融（Environment 上跑）
- `python scripts/train_line4a_kan.py` — KANiTransformer 5 模块消融
- `python scripts/train_line4b_lite.py` — LiteSparseNet 残差消融

**自研模型改进入口**：
- `models/kan_iTransformer.py`（5 模块集成的高性能模型，~120M）
- `models/LiteSparseNet.py`（3 阶段轻量模型，<0.05M 在低维数据上）
- 详细改进指南见 [§8](#8-自研模型改进指南)

---

## 1. 项目结构

```
FinalProject/
├── README.md                       # 本文件
├── requirements.txt
├── .venv/                          # Python 虚拟环境 (已就绪)
│
├── configs/                        # 超参数配置
│   ├── base_config.py              #   统一参数 dataclass (BaseConfig)
│   ├── dataset_configs.py          #   4 个数据集的硬约束 (ETTm2/Weather/Electricity/Environment)
│   └── model_configs.py            #   7 个模型 × 4 档 (low/mid/high/multimodal) 预设
│
├── data_provider/                  # 数据加载
│   ├── dataset_base.py             #   基础滑窗 Dataset + 一次性 preload_to_device (MPS)
│   ├── data_factory.py             #   DataLoader 工厂 (CUDA / MPS / CPU 三路径)
│   ├── preloaded_dataset.py        #   全部预加载到 MPS 上的 InMemoryDataset
│   ├── multimodal_builder.py       #   文本嵌入 + 卫星图加载
│   ├── preprocess_timemmd.py       #   Time-MMD 多模态预处理脚本
│   ├── preprocess_satellite.py     #   卫星图生成
│   └── download_scripts.py         #   纯时序数据集自动下载
│
├── models/                         # 7 个模型定义
│   ├── DLinear.py / PatchTST.py / iTransformer.py / TimeMixer.py
│   │     # ↑ thuml 官方模型桥接 (4 字节的 stub, 实际从 third_party/TimeSeriesLibrary 导入)
│   ├── Chronos2.py                 #   Zero-Shot 基础大模型入口
│   ├── SparseTSF.py                # 外部轻量基线 (SparseTSF 论文复现)
│   ├── kan_iTransformer.py         # ⭐ 自研高性能模型 (5 模块集成)
│   └── LiteSparseNet.py            # ⭐ 自研轻量模型 (3 阶段, v2.1 含 LinearResidual)
│
├── layers/                         # 共享网络层
│   ├── kan_layers.py               #   KANLinear / KANLayer (B-spline 基函数)
│   ├── MambaBlock.py               #   Mamba / SSM 模块
│   ├── frequency_decomp.py         #   FFT 频域分解 (KANiTransformer 用)
│   ├── gating_fusion.py            #   多模态门控融合
│   ├── contrastive_loss.py         #   InfoNCE 对比损失
│   ├── conformal_prediction.py     #   共形预测区间估计
│   ├── meta_arbitrator.py          #   模型仲裁路由器
│   ├── satellite_encoder.py        #   卫星图 CNN 编码器
│   └── Embed.py / Transformer_EncDec.py / ...   # thuml 桥接层
│
├── exp/                            # 实验流程
│   ├── exp_basic.py                #   实验基类 (model 构建 + device)
│   ├── exp_train.py                #   训练/验证/测试循环 (含 masked pretrain / 概率输出 / FFT 向量化)
│   └── exp_zero_shot.py            #   Chronos2 Zero-Shot 推理
│
├── utils/
│   ├── tools.py                    #   随机种子 + 早停
│   ├── metrics.py                  #   MSE / MAE / RMSE / MAPE / SMAPE
│   ├── efficiency.py               #   参数量 / FLOPs / 推理时间 / 显存统计
│   ├── statistical_tests.py        #   Wilcoxon 符号秩检验
│   ├── result_logger.py            #   CSV 记录
│   └── masking.py                  #   掩码预训练辅助
│
├── scripts/                        # ⭐ 主要入口
│   ├── _common.py                  #   共享工具: detect_compute / run_experiment / save_*
│   ├── train_line1.py              #   主线一: 跨架构对比 (4 架构 × 3 数据集 × 4 pred_len)
│   ├── train_line2.py              #   主线二: 自研模型评测 (3 自研 + 复用 Line 1 基线)
│   ├── train_line3.py              #   主线三: 多模态消融 (Environment 7 组)
│   ├── train_line4a_kan.py         #   消融 4a: KAN-iTransformer 5 模块
│   ├── train_line4b_lite.py        #   消融 4b: LiteSparseNet 残差设计
│   └── sync_results.py              #   合并多个 line CSV 到 main_results.csv
│
├── dataset/                        # ⭐ CSV 数据 + 预生成的文本/卫星嵌入
│   ├── ETTm2.csv / Weather.csv / Electricity.csv / Environment.csv
│   ├── Energy.csv / Health.csv     # (备用, 暂未在 train_line 里启用)
│   ├── *_text_embeds.npy           # 预计算的 sentence-transformer 文本嵌入
│   ├── *_recurrence.npy            # 预生成的递归图 (32×32)
│   └── satellite_imgs/             # Environment 用的卫星图占位
│
├── results/                        # ⭐ 实验输出
│   ├── line{1,2,3}_{ts}.csv       # 各 line 的时间戳快照
│   ├── line{1,2,3}_latest.csv     # viz 直接读的"最新"文件
│   ├── ablation_{kan,lite}_{ts}.csv
│   ├── ablation_{kan,lite}_latest.csv
│   ├── efficiency/                 # 参数量/FLOPs/推理时间
│   ├── main_results.csv            # 跨 line 汇总 (sync_results.py 生成)
│   └── tuning/                     # 超参搜索结果
│
├── viz-frontend/                   # 可视化前端 (Next.js)
│   └── src/data/lines.ts           # 与 train_line*.py 的 LITE_ABLATION_GROUPS 等常量对齐
│
├── third_party/TimeSeriesLibrary/  # thuml 官方库 (fork)
├── checkpoints/                    # 训练过程保存的 .pth
├── docs/                           # 项目文档
│   ├── project-plan-v2.0.md        # 主计划 (含 v2.1 更新说明 + 设计笔记)
│   ├── model-architecture.md
│   ├── experiment-guide.md
│   ├── project-structure.md
│   └── server-guide.md
│
├── app/gradio_demo.py              # Gradio 交互式 Demo
├── run.py / run_experiments.py / run_ablation.py  # 旧入口 (已被 scripts/train_line*.py 替代)
└── download_nasa_earthdata.py      # 卫星图原始下载脚本
```

---

## 2. 环境搭建

### 2.1 依赖

- **Python**: 3.13（已用 `.venv` 配置好）
- **PyTorch**: 2.12+（含 MPS / CUDA 双后端）
- **核心包**: `pandas numpy scikit-learn matplotlib fvcore tqdm einops transformers sentence-transformers reformer-pytorch mamba-ssm Pillow gradio`

### 2.2 第一次启动

```bash
cd FinalProject

# 已存在 .venv/ 时
source .venv/bin/activate

# 第一次创建时
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
```

### 2.3 验证安装

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available())
from models.LiteSparseNet import Model
print('LiteSparseNet import OK')
"
```

预期输出（M5 Mac）：`MPS available: True`，`LiteSparseNet import OK`。

### 2.4 已知 venv 状态

当前 `.venv` 在 M5 Mac 上：
- Python 3.13.13
- torch 2.12.0（含 MPS 后端）
- 全部依赖已装好，可以直接 `source .venv/bin/activate` 用

如果在另一台机器上，需要：
```bash
pip install torch torchvision torchaudio   # 或 pytorch 官方对应 CUDA 版本
pip install -r requirements.txt
```

---

## 3. 数据集

### 3.1 现状（已就绪）

`dataset/` 下 4 个训练数据集已就位：

| 文件 | 变量 | 样本 | 频率 | 用途 |
|---|---|---|---|---|
| `ETTm2.csv` | 7 | 69,680 | 15 min | 低维经典基准 |
| `Weather.csv` | 21 | 52,696 | 10 min | 中维纯数值 |
| `Electricity.csv` | 321 | 26,304 | 1 h | 超高维压力测试 |
| `Environment.csv` | 6 | 15,979 | 1 day | 多模态（带文本/卫星图） |

外加 `Energy.csv` / `Health.csv`（Time-MMD 备用数据集），`scripts/train_line*.py` 暂未启用。

### 3.2 重新生成（如果需要）

```bash
# 1) 纯时序 CSV
python -c "from data_provider.download_scripts import download_all; download_all()"

# 2) 多模态预处理 (Environment 文本嵌入 + 卫星图)
python data_provider/preprocess_timemmd.py
# 卫星图占位由 preprocess_satellite.py 生成
```

预生成的辅助文件（已存在，不要删除）：
- `dataset/*_text_embeds.npy` — sentence-transformer 算好的文本向量（128 维）
- `dataset/*_recurrence.npy` — 32×32 递归图
- `dataset/satellite_imgs/{date}.png` — Environment 的卫星图占位

### 3.3 数据划分

按时间顺序切分（不要随机打乱）：

| split | 比例 | train_line 里的 flag |
|---|---|---|
| train | 0% – 70% | `'train'` |
| val | 70% – 85% | `'val'` |
| test | 85% – 100% | `'test'` |

划分在 `data_provider/dataset_base.py:__read_data__` 里通过 `border_ratios = [0.0, 0.7, 0.85, 1.0]` 控制。

---

## 4. 模型阵容

### 4.1 7 个模型

| 类别 | 模型 | 架构 | 来源 | 参数量 | 推荐运行主线 |
|---|---|---|---|---|---|
| 基线 | `DLinear` | MLP | thuml | 0.02M | Line 1 |
| 基线 | `PatchTST` | Transformer + Patching | thuml | 6.9M | Line 1 |
| 基线 | `TimesNet` | CNN (2D-variation) | thuml | 5.2M | Line 1 |
| 基线 | `Mamba` | SSM (MambaSimple) | thuml | 2.8M | Line 1 |
| 外部轻量 | `SparseTSF` | 稀疏线性（<1K params） | 论文复现 | <0.001M | Line 2 |
| ⭐ 自研高性能 | `KANiTransformer` | KAN + 倒置Transformer + 5 模块 | 自研 | ~120M | Line 2 / 4a |
| ⭐ 自研轻量 | `LiteSparseNet` | 稀疏采样 + 组MLP + 残差 | 自研 | <0.05M (低维) | Line 2 / 4b |

`iTransformer.py` / `TimeMixer.py` / `Chronos2.py` 是 thuml 桥接 stub（4 字节），但 `model_configs.py` 当前没启用它们 — 不要直接在 train_line*.py 里用，需要先补全 preset。

### 4.2 自研模型的设计要点

#### KANiTransformer（`models/kan_iTransformer.py`）

5 模块集成，每个可独立关闭（对应 Line 4a 的 5 档 ablation）：

1. **KAN 层** (`layers/kan_layers.py`) — B-spline 可学习基函数替代 iTransformer 的 FFN
2. **级联频域分解 (CFD)** (`layers/frequency_decomp.py`) — trend / seasonal / residual 三分支
3. **掩码预训练** (`exp/exp_train.py:pretrain_step`) — 15% 掩码率的自监督预训练
4. **概率输出** (GaussianNLL) — 输出 (mean, logvar)，支持不确定性
5. **模型仲裁** (`layers/meta_arbitrator.py`) — 基于 5 维统计特征动态选模型

详见 [§8.1 KANiTransformer 改进指南](#81-kanitransformer-改进指南)。

#### LiteSparseNet（`models/LiteSparseNet.py`）

3 阶段设计，v2.1 重构了第三阶段：

1. **稀疏趋势提取** — per-variable `Linear(down_len, pred_len)`，借鉴 SparseTSF 思想
2. **分组轻量 MLP** (`GroupLightMLP`) — 组内变量交互，参数量与变量数线性
3. **可学习残差** (`LinearResidual`, v2.1 新增) — 共享下投影 + 通道独享上投影 + 通道独享可学习 gate，**替代了 v2.0 的 FFT 频域修正**（v2.0 消融证明 FFT 是负贡献，详见 `docs/project-plan-v2.0.md`）

详见 [§8.2 LiteSparseNet 改进指南](#82-litesparsenet-改进指南)。

---

## 5. 怎么跑实验

所有主线都从 `scripts/train_line*.py` 入口。每条线独立输出到 `results/line{N}_*.csv` 或 `results/ablation_*_*.csv`。

### 5.1 通用调用格式

```bash
python scripts/train_line<N>_<topic>.py [options]

# 通用选项
--epochs N      # 覆盖训练轮数 (默认 30 for ablations, 100 for full)
--gpu 0         # GPU id (CUDA)
--pred_len 96   # 预测长度
```

跑前会自动检测计算资源并打印 banner：

```
────────────────────────────────────────────────────────────────
  💻 Compute: MPS (Apple M5)  [device_id=m5]
  ⚙️  AMP: OFF (FP32 only)
  📦 Batch size multiplier: ×0.5 (to fit memory)
  🔄 Default epochs: 30 (overridable via --epochs)
────────────────────────────────────────────────────────────────
```

### 5.2 主线一：跨架构对比（耗时最长）

```bash
python scripts/train_line1.py
```

- 4 架构 × 3 数据集 × 4 pred_len (96/192/336/720) = **48 runs**
- 纯时序数据，无多模态
- M5 大约 2-3 小时，4090 大约 30-40 分钟

输出：
- `results/line1_{ts}.csv` (历史快照)
- `results/line1_latest.csv` (viz 读这个)
- `results/efficiency/line1_latest.csv` (参数量/FLOPs/推理时间)

### 5.3 主线二：自研模型评测

```bash
python scripts/train_line2.py
```

- 3 自研模型 (KANiTransformer / LiteSparseNet / SparseTSF) × 3 数据集 × 4 pred_len = **36 runs**
- 注意：Line 1 已经跑过 4 个 thuml 基线，Line 2 不重复跑，由 `viz-frontend` 跨线合并
- M5 大约 2 小时（KANiTransformer 慢，LiteSparseNet 快）

### 5.4 主线三：多模态消融

```bash
python scripts/train_line3.py
```

- 2 架构 (PatchTST / Mamba) × 1 数据集 (Environment) × 2 pred_len × 7 模态 = **28 runs**
- 7 模态组合：baseline / report / search / both_concat / both_gating / satellite / text+satellite
- M5 大约 1.5 小时

### 5.5 消融 4a：KAN-iTransformer 5 模块

```bash
python scripts/train_line4a_kan.py
```

- 5 消融设置 (A0 完整 / A1 w/o KAN / A2 w/o CFD / A3 w/o 预训练 / A4 w/o 概率输出) × 3 数据集 × 1 pred_len = **15 runs**
- A0 / A1 / A4 必跑；A2/A3 跑过即可
- M5 大约 1.5 小时

### 5.6 消融 4b：LiteSparseNet 残差设计

```bash
python scripts/train_line4b_lite.py
```

- 3 设置 (B0 完整 / B1 窄瓶颈 / B2 关闭) × 3 数据集 × 1 pred_len = **9 runs**
- M5 大约 15 分钟（这一条最便宜，可以快速迭代）

### 5.7 单模型快速验证

```bash
# 跑单条 (LiteSparseNet on ETTm2, pred_len=96)
python scripts/train_line4b_lite.py --epochs 5
# 5 个 epoch + early stopping 大约 1-2 分钟

# 临时改设置 (例如关掉残差)
# 改 scripts/train_line4b_lite.py 里的 LITE_ABLATION_SETTINGS
```

### 5.8 实时进度监控

每个 run 完成后立刻 append 到 `results/line{N}_partial.csv`（或 `ablation_*_partial.csv`）：

```bash
# 训练中随时看进度
watch -n 5 'cat results/line1_partial.csv | column -t -s, | tail -10'
```

中途崩了也不丢已完成 run。

### 5.9 全表合并

所有 line 跑完后：

```bash
python scripts/sync_results.py
# 把 line1/2/3 + ablation 合并成 results/main_results.csv
```

---

## 6. 输出格式与数据流向

### 6.1 单 run 输出列

每个 CSV 行的列（来自 `exp/exp_train.py:test()`）：

| 列 | 类型 | 含义 |
|---|---|---|
| `MSE` | float | 主指标 |
| `MAE` | float | |
| `RMSE` | float | |
| `MAPE` | float | %，分母小会爆炸（参考用） |
| `SMAPE` | float | %，更稳 |
| `Params(M)` | float | 参数量（百万） |
| `FLOPs(G)` | float | GFLOPs（10 亿次浮点） |
| `InferTime(ms)` | float | 单次推理时间（50 次平均） |
| `GPUMem(MB)` | float | 推理时显存峰值（**MPS 上恒为 0**，已知缺陷） |
| `status` | str | `success` / `error` |
| `model` | str | 模型名 |
| `dataset` | str | 数据集名 |
| `seq_len` / `pred_len` | int | |
| `label` | str | ablation 设置标签（"B0 - 完整"等） |
| `ablation` | str | 大组名（"Lite Residual"等），viz 用 |
| `setting` | str | 具体设置 |
| `line` | int | 1/2/3，标记属于哪条主线 |

### 6.2 文件命名

| 模式 | 例子 | 用途 |
|---|---|---|
| `line{N}_{ts}.csv` | `line1_20260620_215657.csv` | 时间戳历史快照 |
| `line{N}_latest.csv` | `line1_latest.csv` | viz 永远读这个 |
| `line{N}_partial.csv` | `line1_partial.csv` | 实时进度，训练结束后被 rename |
| `ablation_{prefix}_{ts}.csv` | `ablation_lite_20260620_215657.csv` | 同上 for ablation |
| `efficiency/line{N}_latest.csv` | | 各 line 的效率指标（参数量/FLOPs 等） |

`ablation_lite_20260620_202507.csv` 是 **v2.0 FFT** 的基线，`ablation_lite_20260620_215657.csv` 是 **v2.1 LinearResidual** 的新结果 — 保留两个做对比。

### 6.3 MPS 性能优化的"魔术"在哪

- **`mps-perf` 分支额外做的事**（**对 M5 用户必看**）：
  1. `data_provider/preloaded_dataset.py` — 把每个 (train/val/test) 一次性 materialize 到 MPS unified memory，避免 per-batch CPU→MPS 拷贝
  2. `data_provider/dataset_base.py:preload_to_device()` — 用 `np.sliding_window_view` 一次性构造所有滑窗
  3. `models/LiteSparseNet.py:LinearResidual` — 替代原 FFT 修正（v2.0 的 FFT 在 MPS 上有 Python 三重循环 + `.item()` 同步，3+ 分钟/epoch；新版本 ~1.4ms/batch）
  4. `scripts/_common.py:detect_compute()` — 自动检测设备，CUDA 用原 DataLoader，MPS 走 preloaded 路径
- **CUDA 用户**（4090）这些都用不上，CUDA 走原来的 num_workers=8 + pin_memory=True 路径
- **MPS 用户**这些加起来把单 epoch 从 ~45s 降到 ~0.2s（200x 加速数据侧）

---

## 7. 可视化

`viz-frontend/` 是 Next.js 前端。重要对齐点：

- `viz-frontend/src/data/lines.ts` 里的 `LITE_ABLATION_GROUPS = ["Lite Residual"]` 必须和 `scripts/train_line4b_lite.py` 里的 `LITE_GROUP_NAME = "Lite Residual"` 一致
- 类似有 `KAN_ABLATION_GROUPS = ["KAN 5 Modules"]`

启动前端（不需要 GPU）：

```bash
cd viz-frontend
npm install
npm run dev
# 访问 http://localhost:3000
```

前端会自动从 `results/line{N}_latest.csv` 和 `results/ablation_*_latest.csv` 拉数据。

---

## 8. 自研模型改进指南

> 接手自研模型时，先理解设计动机 → 找出最弱模块 → 做最小改动验证。

### 8.1 KANiTransformer 改进指南

**入口**：`models/kan_iTransformer.py`（~300 行），5 模块分布在 `layers/` 下。

#### 当前瓶颈（从 v2.0 消融看）

- 5 模块全开时 ~120M 参数，推理慢
- 单一模块贡献难隔离（Line 4a 的结果在 `results/ablation_kan_*.csv`）
- 预训练 (模块 3) 的 mask 策略相对朴素

#### 改进方向（按性价比排序）

| 优先级 | 方向 | 在哪改 | 验证方法 |
|---|---|---|---|
| 🟢 高 | 跑完 Line 4a，看哪个模块 Δ MSE 最小 → 考虑删掉 | `results/ablation_kan_*.csv` | Line 4a |
| 🟢 高 | 简化 KAN 层 grid_size（默认 5 → 试 3 或 7）| `configs/model_configs.py` + `layers/kan_layers.py` | Line 2 |
| 🟡 中 | 模型仲裁 (模块 5) 加更多 meta-features（目前 5 维：谱熵/趋势/周期/方差/自相关）| `layers/meta_arbitrator.py` | Line 4a (A5 关仲裁) |
| 🟡 中 | 模块 4 概率输出：尝试 learned heteroscedastic 或 mixture of Gaussians | `exp/exp_train.py` | Line 2 (看 NLL/CRPS) |
| 🟡 中 | 模块 2 CFD 改为可学习层数（现在是固定 3 层） | `layers/frequency_decomp.py` | Line 4a (A2) |
| 🟠 低 | 模块 3 掩码预训练：尝试 contiguous / structured mask 策略 | `exp/exp_train.py:pretrain_step` | Line 2 |
| 🟠 低 | KAN 替代为 Chebyshev / Fourier 基函数 | `layers/kan_layers.py` | Line 2 |
| ⚪ 长线 | 替换为 sparse KAN（部分 grid 设为 0）| `layers/kan_layers.py` | Line 2 |

#### 推荐起步实验

```bash
# 1) 跑 Line 4a 看 5 模块各自贡献
python scripts/train_line4a_kan.py
cat results/ablation_kan_latest.csv | head -20

# 2) 在 model_configs.py 里把 KAN grid_size 从 5 改到 3
# 改 KANiTransformer 的所有 'kan_grid_size' 值, 然后跑 Line 2
# grep 'kan_grid_size' configs/model_configs.py
python scripts/train_line2.py --epochs 50
```

### 8.2 LiteSparseNet 改进指南

**入口**：`models/LiteSparseNet.py`（~330 行），3 阶段 + v2.1 `LinearResidual`。

#### 当前瓶颈（v2.1 消融的关键发现）

详见 `docs/project-plan-v2.0.md`「v2.1 设计笔记」节。核心结论：

| | B0 (residual=4) | B1 (residual=1) | B2 (residual=0) | 差异 |
|---|---|---|---|---|
| ETTm2 MSE | 0.113 | 0.115 | 0.114 | <1% |
| Electricity MSE | 0.237 | 0.234 | 0.235 | <1% |
| Environment MSE | 0.371 | 0.364 | 0.369 | <1% |

**LinearResidual 的 gate 训到了 ~0** — 模块对预测透明。这告诉我们：
- Trend 提取 + GroupMLP 已经捕获了有效信号
- 残差层没可学的东西（在这 3 个数据集上）
- 改进应该聚焦在 stage 1/2，或者重新设计 stage 3 的"诱因"

#### 改进方向（按性价比排序）

| 优先级 | 方向 | 在哪改 | 验证方法 |
|---|---|---|---|
| 🟢 高 | 改 `LinearResidual` 的 gate 初始化：从 -2 (sigmoid≈0.12) 改成 0 (sigmoid=0.5)，强迫模型"先尝试用残差再决定关掉" | `models/LiteSparseNet.py:_init_lazy_params` 第 `gate = nn.Parameter(torch.full((n,), -2.0))` | Line 4b |
| 🟢 高 | Stage 1 加 learnable skip connection（残差绕开 per-variable linear） | `models/LiteSparseNet.py:Model.forward` | Line 4b (新加 B 设置) |
| 🟡 中 | 替换 stage 2 GroupLightMLP 为 tiny Mamba block（selective state space，per-group） | `models/LiteSparseNet.py:GroupLightMLP` | Line 4b |
| 🟡 中 | Stage 1 sparse_ratio 改为 per-channel 可学习 | `models/LiteSparseNet.py:Model.forward` 的 `self.sparse_ratio` | Line 4b |
| 🟡 中 | GroupMLP 加 dynamic grouping（学习哪些变量应该同组）| `models/LiteSparseNet.py:GroupLightMLP` | Line 4b |
| 🟠 低 | Stage 3 改用 causal 1D conv（更结构化的"局部残差"）| 替换 `LinearResidual` | Line 4b |
| 🟠 低 | 加 LiteRevIN 通道独立归一化（项目已留 `use_lite_revin` 字段但未启用）| `models/LiteSparseNet.py:Model` | Line 2 |
| ⚪ 长线 | 蒸馏 LiteSparseNet 教更小的学生模型（<10K params）| 新增 `distill.py` | 参数量 vs MSE 对比图 |

#### 推荐起步实验

```bash
# 1) 跑 Line 4b 当前 3 档 (基线)
python scripts/train_line4b_lite.py
# 对比 results/ablation_lite_latest.csv (v2.1) vs ablation_lite_20260620_202507.csv (v2.0)

# 2) 改 gate 初始化: 把 models/LiteSparseNet.py:228 的
#    self.gate = nn.Parameter(torch.full((self.n_vars,), -2.0, ...))
#    改成
#    self.gate = nn.Parameter(torch.zeros(self.n_vars, ...))
#    然后跑 Line 4b, 对比新结果
python scripts/train_line4b_lite.py
```

**关键代码位置速查**：

- Stage 1 (sparse trend) — `models/LiteSparseNet.py:Model.forward` 中 `_seq_windows` 调用
- Stage 2 (group MLP) — `GroupLightMLP` 类（约 line 19-82）
- Stage 3 (LinearResidual) — `LinearResidual` 类（约 line 95-218），gate 初始化在 lazy init
- Per-channel 参数量计算 — `model_configs.py:lite_iTransformer` 节

---

## 9. 计算资源 (M5 vs 4090)

### 9.1 设备检测

`scripts/_common.py:detect_compute()` 在训练启动时自动检测并打印 banner：

| 设备 | 行为 |
|---|---|
| NVIDIA RTX 4090 (CUDA) | `use_amp=True`, `amp_dtype='bfloat16'`, 走标准 DataLoader |
| Apple M5 (MPS) | `use_amp=False`, `amp_dtype=None`, 走 preloaded 路径 |
| 无 GPU | `use_amp=False`, `batch_size_multiplier=0.25` |

### 9.2 训练参数一致性

按设计，**不同设备上 batch_size / epochs / lr 完全一致**，只有 AMP 开关不同。所以 M5 和 4090 上跑出来的结果**理论上可比**（虽然 M5 慢很多）。M5 上没 AMP 速度会慢 2x 左右。

### 9.3 推荐配置

| 资源 | 跑哪些主线 | 估算时间 |
|---|---|---|
| M5 Mac (24GB unified) | 1-2 个 line 优先（用 mps-perf 分支）| Line 1 约 2-3h，Line 4b 约 15min |
| RTX 4090 (24GB) | 全跑 | 全部加起来约 4-6 小时 |
| 双机协同 | M5 跑 4a/4b，4090 跑 1/2/3 | 各半天 |

---

## 10. 分支策略

| 分支 | 状态 | 包含内容 | 推荐给 |
|---|---|---|---|
| `main` | 稳定 | 不含 MPS 专项优化（preload / LinearResidual 等）| CUDA 用户、想发版的快照 |
| `mps-perf` | 最新 | 包含 `main` 全部 + MPS preload + v2.1 LinearResidual + B1 bug fix + 更新文档 | M5 Mac 用户、需要最新改进 |
| `mac-mps` | 旧 | 同步自 main 的旧版本 | — |

**给接手同事的建议**：
1. 如果是 M5 Mac → `git checkout mps-perf`
2. 如果是 4090 → `git checkout main`（mps-perf 的 MPS 优化在 CUDA 上不会启用，但代码更复杂）
3. 实验结果在不同分支上**不能直接对比**（mps-perf 的 LiteSparseNet 已经是 v2.1 LinearResidual，跟 main 的 v2.0 FFT 不一样）

合并计划：等 v2.1 验收完，merge `mps-perf` → `main` 作为 v2.1 正式版。

---

## 11. 常见问题

### Q1: 跑训练时报 `ModuleNotFoundError: No module named 'exp.exp_train'`

原因：`sys.path` 里 `third_party/TimeSeriesLibrary/exp` 优先于本仓库 `exp/`。**修复**：确保用最新版的 `_common.py:setup_path()`（本 README 假设的版本会把项目根放到 sys.path 第一个）：

```python
# _common.py:setup_path 应该长这样
project_root = str(PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 项目根在最前
tsl_root = str(PROJECT_ROOT / "third_party" / "TimeSeriesLibrary")
if tsl_root not in sys.path:
    sys.path.append(tsl_root)        # TSL 在最后
```

### Q2: 数据集 `.csv` 找不到

确认在 FinalProject/ 目录下运行，CSV 在 `dataset/` 子目录里（不是 `FinalProject/FinalProject/dataset/`）。

### Q3: `models/__init__.py` 报 KAN 错误

KAN 模型依赖 `layers/kan_layers.py`。`models/__init__.py` 在 import 时会触发 KAN 导入。如果只想测 LiteSparseNet 而不想触发整个 `models/` 的导入：

```python
import importlib.util
spec = importlib.util.spec_from_file_location('LSN', 'models/LiteSparseNet.py')
LSN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LSN)
```

### Q4: 训练中途崩了，已完成的 run 还在吗

**在**。每个 run 完成后立即 append 到 `line{N}_partial.csv`，崩了只是丢当前这个。重启训练会自动续跑（不会重新跑已完成的）。

### Q5: 怎么强制重跑某个 run

删掉 `line{N}_partial.csv` 里的对应行（按 model/dataset/setting 删），再重跑脚本。

### Q6: `GPUMem(MB)` 永远是 0

MPS 已知缺陷（`utils/efficiency.py:measure_gpu_memory` 在 MPS 上没实现）。4090 上会正常返回显存峰值。

### Q7: KAN 训练特别慢 / 显存爆炸

减小 batch_size（在 `configs/model_configs.py` 里 KANiTransformer 的 preset 改 `batch_size`），或关掉模块 1 (KAN) 看是不是 KAN 层在吃资源（对应 Line 4a 的 A1）。

### Q8: 想看训练中间过程

每个 epoch 结束打印 `Epoch N/M | Train Loss: ... | Val Loss: ...`，stdout 是 stdout 实时输出。如果跑了后台想看：`tail -f /tmp/claude-*/.../tasks/{task_id}.output`。

---

## 12. 进阶阅读

| 文档 | 内容 |
|---|---|
| [docs/project-plan-v2.0.md](docs/project-plan-v2.0.md) | **主计划**（含 v2.1 更新说明 + LinearResidual 设计笔记 + 实测结果）|
| [docs/model-architecture.md](docs/model-architecture.md) | 9 个模型的技术原理（部分已过时）|
| [docs/experiment-guide.md](docs/experiment-guide.md) | 实验参数、运行方法（旧版）|
| [docs/project-structure.md](docs/project-structure.md) | 项目结构（旧版）|
| [docs/server-guide.md](docs/server-guide.md) | 服务器部署 |

**关键 commit**（按时间倒序）：

```
8267721 results: v2.1 line 4b ablation (9/9) + v2.0 baseline preserved
ce3e6d4 docs: document v2.1 changes (LinearResidual + B1 bug fix + new ablation)
72e385d feat(mps): replace LiteSparseNet FFT correction with learnable LinearResidual
a56b6e2 perf(mps): preload all sliding windows to unified memory (200x data-side speedup)
6e915cb feat: device-aware AMP toggle + vectorize LiteSparseNet FFT correction
91507f3 chore: untrack .pyc / __pycache__ (already in .gitignore)
```

---

## 联系与支持

接手时遇到问题：
1. 先看 [docs/project-plan-v2.0.md](docs/project-plan-v2.0.md) 的 v2.1 设计笔记
2. 跑 `python -c "from _common import detect_compute; print(detect_compute())"` 确认环境
3. 跑 `python scripts/train_line4b_lite.py --epochs 5` 做端到端冒烟
4. 用 git log 看最近的 commit message 了解改动意图
