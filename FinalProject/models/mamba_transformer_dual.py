"""
创新2: Mamba-Transformer 双专家路由
频域分析路由器动态分配 Mamba(长程) 和 Transformer(短程) 专家权重
"""
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.MambaBlock import MambaLayer
from layers.frequency_decomp import FreqRouter


class TransformerExpert(nn.Module):
    """Transformer 专家 — 局部窗口注意力"""

    def __init__(self, configs, window_size=32):
        super().__init__()
        self.window_size = window_size

        attn_layers = [
            EncoderLayer(
                attention=AttentionLayer(
                    attention=FullAttention(
                        mask_flag=False,
                        output_attention=False,
                        attention_dropout=configs.dropout,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(configs.d_model))

    def forward(self, x):
        # x: [B, L, d_model]
        out, _ = self.encoder(x)
        return out


class MambaExpert(nn.Module):
    """Mamba 专家 — 线性复杂度的全局建模"""

    def __init__(self, configs):
        super().__init__()
        self.mamba = MambaLayer(
            d_model=configs.d_model,
            n_layers=configs.e_layers,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=configs.dropout,
        )

    def forward(self, x):
        # x: [B, L, d_model]
        return self.mamba(x)


class Model(nn.Module):
    """
    Mamba-Transformer 双专家路由模型
    核心创新:
    1. 频域路由器 — 根据输入频谱特征自适应分配专家权重
    2. Mamba 专家 — 线性复杂度 O(L) 处理长程趋势
    3. Transformer 专家 — 精确建模局部高频波动
    4. 动态加权融合 — 每个样本的权重不同
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # 输入嵌入
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=self.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )

        # 频域路由器
        self.router = FreqRouter(configs.seq_len, self.d_model)

        # 两个专家
        self.mamba_expert = MambaExpert(configs)
        self.transformer_expert = TransformerExpert(configs)

        # 输出投影
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, configs.pred_len * configs.c_out),
        )
        self.c_out = configs.c_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, H, C]

        # Non-stationary normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means) / stdev

        B, H, C = x_enc.shape

        # 嵌入: [B, H, C] -> [B, H, d_model]
        x_embed = self.enc_embedding(x_enc, x_mark_enc)

        # 路由权重: [B, 2]
        routing_weights = self.router(x_enc)  # [B, 2]

        # 专家处理
        mamba_out = self.mamba_expert(x_embed)       # [B, H, d_model]
        transformer_out = self.transformer_expert(x_embed)  # [B, H, d_model]

        # 动态加权融合
        w_mamba = routing_weights[:, 0].unsqueeze(-1).unsqueeze(-1)    # [B, 1, 1]
        w_transformer = routing_weights[:, 1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        fused = (w_mamba * mamba_out + w_transformer * transformer_out)  # [B, H, d_model]
        fused_last = fused[:, -1, :]  # [B, d_model]

        # 投影输出
        output = self.projection(fused_last)  # [B, F*C]
        output = output.reshape(B, self.pred_len, self.c_out)

        # 反归一化
        output = output * stdev + means
        return output
