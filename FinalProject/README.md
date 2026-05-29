# 多模态长周期时间序列预测系统

> 数据科学与工程课程 Project Two — 多模态长周期时间序列预测，覆盖 9 个模型、6 个数据集、6 组消融实验、Zero-Shot 对比与交互式 Demo。

---

## 项目简介

本项目实现了一个完整的多模态长周期时间序列预测系统，核心任务是根据历史窗口长度 H 的多维时序数据预测未来长度 F 的走势。在标准时序数值输入之外，还引入文本模态（政策新闻/报告）和图像模态（递归图）进行多模态融合预测，并对比全监督训练模型与 Zero-Shot 基础大模型的性能差异。

**主要工作**：

- 复现 4 个经典/最新基线模型（DLinear、PatchTST、iTransformer、TimeMixer），均来自 [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) 官方实现
- 提出 3 个创新模型：KAN-iTransformer（KAN层+频域分解）、Mamba-Transformer 双专家路由、跨模态对比融合
- 实现共形预测区间估计和模型仲裁集成系统
- 在 6 个数据集（3 个纯时序 + 3 个多模态）上进行完整对比实验和 6 组消融实验
- 提供 Gradio 交互式 Demo

---

## 模型阵容

| 模型 | 类型 | 来源 | 参数量 |
|------|------|------|--------|
| DLinear | MLP 基线 | thuml | 0.02M |
| PatchTST | Transformer+Patching 基线 | thuml | 6.9M |
| iTransformer | 倒置 Transformer 基线 | thuml | 6.4M |
| TimeMixer | 多尺度混合 SOTA | thuml | 4.3M |
| Chronos2 | 时序基础大模型 Zero-Shot | thuml | — |
| TimeKAN | KAN 网络 | 自研 | 37.9M |
| KAN-iTransformer | KAN+频域三分支 | 自研 | 120.5M |
| Mamba-Transformer Dual | 双专家路由 | 自研 | 10.8M |
| Multimodal Fusion | 三模态对比融合 | 自研 | 9.2M |

---

## 数据集

### 纯时序数据集

| 数据集 | 变量数 | 样本数 | 特点 |
|--------|--------|--------|------|
| ETTm2 | 7 | 69680 | 电力变压器温度，经典基准 |
| Weather | 21 | 52696 | 气象多变量 |
| Electricity | 321 | 26304 | 电力消耗，超高维度 |

### 多模态数据集（来自 [Time-MMD, NeurIPS 2024]）

| 数据集 | 变量数 | 样本数 | 时序内容 | 文本内容 |
|--------|--------|--------|----------|----------|
| Energy | 9 | 1622 | 周频率能源指标 | 每周天然气价格新闻 |
| Environment | 6 | 15979 | 日频率空气质量指标 | 每日空气质量报告 |
| Health | 7 | 857 | 周频率流感监测数据 | CDC 流感监测报告 |

---

## 项目结构

```
FinalProject/
├── configs/                # 超参数配置
│   ├── base_config.py      #   统一参数 dataclass
│   └── dataset_configs.py  #   各数据集配置覆盖
├── data_provider/          # 数据加载与预处理
│   ├── dataset_base.py     #   基础滑窗 Dataset
│   ├── data_factory.py     #   DataLoader 工厂
│   ├── download_scripts.py #   自动下载脚本
│   ├── preprocess_timemmd.py # Time-MMD 多模态数据预处理
│   └── multimodal_builder.py # 文本嵌入 + 递归图生成
├── models/                 # 模型定义
│   ├── DLinear.py ... Chronos2.py  # thuml 官方模型（桥接导入）
│   ├── TimeKAN.py          #   自研：KAN 时序模型
│   ├── kan_iTransformer.py #   自研：KAN+频域分解
│   ├── mamba_transformer_dual.py  # 自研：双专家路由
│   └── multimodal_fusion.py       # 自研：多模态融合
├── layers/                 # 共享网络层
│   ├── Embed.py ... StandardNorm.py  # thuml 标准层桥接
│   ├── kan_layers.py       #   KANLinear / KANLayer
│   ├── MambaBlock.py       #   Mamba/SSM 模块
│   ├── frequency_decomp.py #   FFT 频域分解 + 路由器
│   ├── contrastive_loss.py #   InfoNCE 对比损失
│   ├── gating_fusion.py    #   自适应门控融合
│   ├── conformal_prediction.py  # 共形预测
│   └── meta_arbitrator.py  #   模型仲裁器
├── exp/                    # 实验流程
│   ├── exp_basic.py        #   实验基类
│   ├── exp_train.py        #   训练/验证/测试循环
│   └── exp_zero_shot.py    #   Chronos2 Zero-Shot 推理
├── utils/                  # 工具函数
│   ├── metrics.py          #   MSE/MAE/RMSE/MAPE/SMAPE
│   ├── tools.py            #   随机种子 + 早停
│   ├── efficiency.py       #   参数量/速度/显存统计
│   ├── statistical_tests.py #  Wilcoxon 检验
│   └── result_logger.py    #   CSV 结果记录
├── visualization/          # 可视化（7 个模块）
├── app/
│   └── gradio_demo.py      # Gradio 交互式 Demo
├── third_party/
│   └── TimeSeriesLibrary/  # thuml/Time-Series-Library
├── run.py                  # 主入口（单次训练）
├── run_experiments.py      # 批量实验运行器
├── run_ablation.py         # 消融实验运行器
├── requirements.txt        # Python 依赖
└── docs/                   # 项目文档
```

详细说明参见 [docs/project-structure.md](docs/project-structure.md)。

---

## 快速开始

### 1. 环境搭建

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 纯时序数据集（ETTm2、Weather、Electricity）
python -c "from data_provider.download_scripts import download_all; download_all()"

# 预处理 Time-MMD 多模态数据集（Energy、Environment、Health）
python data_provider/preprocess_timemmd.py
```

### 3. 单模型训练

```bash
# DLinear 在 ETTm2 上训练
python run.py --model DLinear --data ETTm2 --seq_len 96 --pred_len 96

# iTransformer 在 Weather 上训练，预测336步
python run.py --model iTransformer --data Weather --pred_len 336

# 多模态融合模型（Energy 数据集）
python run.py --model MultimodalFusion --data Energy --seq_len 96 --pred_len 96

# Chronos2 Zero-Shot（不需训练）
python run.py --model Chronos2 --data ETTm2 --seq_len 96 --pred_len 96
```

### 4. 批量实验

```bash
# 完整实验：8模型 × 6数据集 × 多预测长度
python run_experiments.py --epochs 100 --gpu 0

# 只跑部分（快速验证）
python run_experiments.py \
    --models DLinear PatchTST iTransformer \
    --datasets ETTm2 Weather \
    --pred_lens 96 192 \
    --epochs 50
```

### 5. 消融实验

```bash
python run_ablation.py
```

### 6. 可视化

```bash
python -c "
from visualization.plot_heatmap import plot_performance_heatmap
plot_performance_heatmap('results/main_results.csv', 'MSE')
"
```

### 7. Gradio Demo

```bash
python app/gradio_demo.py
# 访问 http://localhost:7860
```

---

## 实验设置

| 参数 | 值 |
|------|-----|
| 历史窗口 H | 96, 192 |
| 预测长度 F | 96, 192, 336, 720 |
| 批大小 | 64（Electricity 降为 16） |
| 学习率 | 1e-4 |
| 优化器 | AdamW（weight_decay=1e-5） |
| 训练轮次 | 100（早停 patience=10） |
| 混合精度 | 开启 |
| 随机种子 | 42 |
| 评价指标 | MSE, MAE, RMSE, MAPE, SMAPE |

详细参数说明和注意事项参见 [docs/experiment-guide.md](docs/experiment-guide.md)。

---

## 创新点

### 1. KAN-Enhanced iTransformer
将 iTransformer 的 MLP 前馈层替换为 KAN（B-spline 可学习函数），并引入 FFT 自适应频域分解（趋势/季节/残差三分支），增强非线性建模能力。

### 2. Mamba-Transformer 双专家路由
FFT 频域路由器动态分配 Mamba（长程趋势）和 Transformer（短程波动）两个专家的权重，自适应不同数据特性。

### 3. 跨模态对比对齐融合
基于 Time-MMD（NeurIPS 2024）多模态数据集，时序 Transformer + 文本 MLP + 图像 CNN 三模态编码，通过 InfoNCE 对比损失对齐表示空间，自适应门控网络动态融合。

### 4. 共形预测区间估计
分位数回归 + 校准集非一致性分数 → 理论保证的 95% 置信区间。

### 5. 模型仲裁集成
元学习路由器根据输入序列的统计特征（谱熵/趋势/周期性/方差/自相关）动态选择最优模型。

详见 [docs/model-architecture.md](docs/model-architecture.md)。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [project-overview.md](docs/project-overview.md) | 项目概述：研究问题、模型、数据集、创新点 |
| [project-structure.md](docs/project-structure.md) | 项目结构：每个目录和文件的职责 |
| [model-architecture.md](docs/model-architecture.md) | 模型架构：9 个模型的技术原理 |
| [experiment-guide.md](docs/experiment-guide.md) | 实验指南：参数、数据、运行方法、注意事项 |
| [server-guide.md](docs/server-guide.md) | 服务器部署和训练操作指南 |

---

## 依赖

```
torch>=2.0          numpy          pandas          scikit-learn
matplotlib          seaborn        gradio>=4.0     fvcore
chronos-forecasting transformers   sentence-transformers  einops
reformer-pytorch    mamba-ssm      opencv-python   scipy
tqdm                Pillow
```

完整列表见 [requirements.txt](requirements.txt)。

---

## 运行环境

- **开发**：Windows 11 / Python 3.12 / PyTorch 2.x
- **训练**：Linux + CUDA 12.x + RTX 4090 24G（推荐）
- **Demo**：任意支持 Gradio 的环境

---

## 致谢

- [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) — 基线模型和共享层的官方实现
- [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) — 高效 KAN 层实现参考
- [Time-MMD (NeurIPS 2024)](https://github.com/AdityaLab/Time-MMD) — 多模态时序数据集
- Amazon Chronos — 时序基础大模型

## 许可证

本项目仅用于课程学习和学术研究。
