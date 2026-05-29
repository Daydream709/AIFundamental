"""
统一超参数配置 — 所有模型、数据集、实验共享
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    # ==================== 任务 ====================
    task_name: str = 'long_term_forecast'
    is_training: int = 1
    model_id: str = 'test'

    # ==================== 数据 ====================
    data: str = 'ETTm2'
    root_path: str = './dataset/'
    data_path: str = 'ETTm2.csv'
    features: str = 'M'            # M=多变量预测多变量, S=单变量, MS=多变量预测单变量
    target: str = 'OT'
    freq: str = 'h'
    checkpoints: str = './checkpoints/'

    # ==================== 预测维度 ====================
    seq_len: int = 96              # 历史窗口长度 H
    label_len: int = 48            # 编解码器桥接长度
    pred_len: int = 96             # 预测长度 F
    enc_in: int = 7                # 输入变量数
    dec_in: int = 7
    c_out: int = 7                 # 输出变量数

    # ==================== 模型架构 ====================
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

    # TimeMixer 专用
    down_sampling_window: int = 2
    down_sampling_layers: int = 3
    down_sampling_method: str = 'avg'
    channel_independence: int = 1
    decomp_method: str = 'moving_avg'
    top_k: int = 5
    use_norm: bool = True

    # ==================== 训练 ====================
    train_epochs: int = 100
    batch_size: int = 64
    patience: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    loss: str = 'MSE'              # MSE, MAE, GaussianNLL

    # ==================== 优化 ====================
    num_workers: int = 8
    use_gpu: bool = True
    gpu: int = 0
    use_multi_gpu: bool = False
    use_amp: bool = True           # 混合精度
    seed: int = 42

    # ==================== 多模态(创新专用) ====================
    use_text: bool = False
    use_image: bool = False
    text_dim: int = 768
    img_size: int = 32
    use_contrastive: bool = False
    use_kan: bool = False
    use_mamba_expert: bool = False
    use_conformal: bool = False
    use_ensemble: bool = False

    # ==================== 共形预测 ====================
    quantiles: list = field(default_factory=lambda: [0.05, 0.5, 0.95])

    # ==================== 其他 ====================
    des: str = 'test'
    itr: int = 1
    train_only: bool = False

    def __post_init__(self):
        if self.padding is None:
            self.padding = self.stride
