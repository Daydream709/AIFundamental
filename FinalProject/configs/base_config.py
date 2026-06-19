"""
统一超参数配置 — FinalProject v2.0
所有模型、数据集、实验共享
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    # ==================== 任务 ====================
    task_name: str = 'long_term_forecast'
    is_training: int = 1
    model_id: str = 'test'
    model: str = 'DLinear'           # 模型名称 (exp_basic 用)

    # ==================== 数据 ====================
    data: str = 'ETTm2'
    root_path: str = './dataset/'
    data_path: str = 'ETTm2.csv'
    features: str = 'M'               # M=多变量多变量, S=单变量, MS=多变量单变量
    target: str = 'OT'
    freq: str = 'h'
    checkpoints: str = './checkpoints/'

    # ==================== 预测维度 ====================
    seq_len: int = 96                 # 历史窗口长度 H
    label_len: int = 48               # 编解码器桥接长度
    pred_len: int = 96                # 预测长度 F
    enc_in: int = 7                   # 输入变量数
    dec_in: int = 7
    c_out: int = 7                    # 输出变量数

    # ==================== 模型架构 (通用) ====================
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    factor: int = 3
    dropout: float = 0.1
    embed: str = 'timeF'
    activation: str = 'gelu'
    output_attention: bool = False
    num_class: int = 1

    # PatchTST 专用
    patch_len: int = 16
    stride: int = 8
    padding: Optional[int] = None

    # TimesNet 专用
    top_k: int = 5
    num_kernels: int = 6              # Inception block 的卷积核数

    # ==================== KAN-iTransformer 专用 ====================
    kan_grid_size: int = 5            # KAN B-spline grid大小
    use_cfd: bool = True              # 级联频域分解 (Cascaded Freq Decomp)
    use_wavelet: bool = False         # ★ 创新优化: 用 Wavelet 替代 FFT (适合非平稳信号)
    use_revin: bool = True            # 可逆实例归一化
    use_probabilistic: bool = True    # 概率输出 (GaussianNLL)
    use_masked_pretrain: bool = False # 掩码重建自监督预训练
    mask_ratio: float = 0.15          # 掩码率
    use_model_arbitration: bool = False  # 模型仲裁集成

    # ==================== Lite-SparseNet 专用 ====================
    sparse_ratio: int = 4             # 下采样因子 p (H -> H/p)
    group_size: int = 16              # 分组MLP的组大小
    fft_residual_k: int = 2           # FFT残差保留的主频数
    use_lite_revin: bool = True       # ★ 创新优化: 极简RevIN实例归一化
    use_shared_weight: bool = True    # ★ 创新优化: trend_extractor 共享权重 + 变量bias

    # ==================== SparseTSF 专用 ====================
    period_len: int = 24              # 稀疏采样的周期长度

    # Mamba/SSM 专用
    expand: int = 2                   # Mamba 内部扩展因子
    d_conv: int = 4                   # Mamba 卷积核大小
    d_state: int = 16                 # Mamba 状态维度

    # ==================== 训练 ====================
    train_epochs: int = 100
    batch_size: int = 64
    patience: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    loss: str = 'MSE'                  # MSE, MAE, GaussianNLL

    # ==================== 优化 ====================
    num_workers: int = 8
    use_gpu: bool = True
    gpu: int = 0
    use_multi_gpu: bool = False
    use_amp: bool = True               # 混合精度 (AMP)
    seed: int = 42

    # ==================== 多模态 ====================
    use_text: bool = False
    use_image: bool = False
    text_dim: int = 768
    img_size: int = 32
    use_contrastive: bool = False
    use_conformal: bool = False
    text_fusion_mode: str = 'concat'   # concat, gating (门控融合)

    # ==================== 共形预测 ====================
    quantiles: list = field(default_factory=lambda: [0.05, 0.5, 0.95])

    # ==================== 其他 ====================
    des: str = 'test'
    itr: int = 1
    train_only: bool = False

    def __post_init__(self):
        if self.padding is None:
            self.padding = self.stride
