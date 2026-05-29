"""
创新1: KAN-iTransformer — KAN增强的倒置Transformer
将 iTransformer 的 MLP 前馈层替换为 KAN 层
加入自适应频域分解: 趋势 + 季节 + 残差 三分支
"""
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.kan_layers import KANLayer
from layers.frequency_decomp import AdaptiveFreqDecomp


class KANEncoderLayer(nn.Module):
    """使用 KAN 替代 FFN 的编码器层"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, kan_grid_size=5):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # KAN 层替代传统 MLP
        self.kan_ffn = KANLayer(d_model, d_ff, d_model, grid_size=kan_grid_size)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 自注意力
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # KAN 前馈
        x = x + self.kan_ffn(self.norm2(x))

        return x, attn


class Model(nn.Module):
    """
    KAN-iTransformer: 核心创新模型
    改进点:
    1. 频域分解 — 将输入分解为趋势/季节/残差三分支
    2. KAN前馈 — 用 B-spline 可学习函数替代固定线性变换
    3. 倒置架构 — 注意力在变量间计算，高效处理多变量
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        # 频域分解
        self.freq_decomp = AdaptiveFreqDecomp(top_k=configs.top_k)

        # 三个分支各有一个嵌入层和编码器
        self.branches = nn.ModuleList()
        for _ in range(3):  # 趋势 / 季节 / 残差
            branch = nn.ModuleDict({
                'embedding': DataEmbedding_inverted(
                    c_in=configs.seq_len,
                    d_model=self.d_model,
                    dropout=configs.dropout,
                ),
                'encoder': self._build_kan_encoder(configs),
            })
            self.branches.append(branch)

        # 分支融合权重
        self.branch_gate = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 3),
            nn.Softmax(dim=-1),
        )

        # 输出投影
        self.projector = nn.Linear(self.d_model, configs.pred_len, bias=True)

    def _build_kan_encoder(self, configs):
        """构建使用 KAN 的编码器"""
        layers = []
        for _ in range(configs.e_layers):
            attn = AttentionLayer(
                attention=FullAttention(
                    mask_flag=False,
                    output_attention=configs.output_attention,
                    attention_dropout=configs.dropout,
                ),
                d_model=self.d_model,
                n_heads=self.n_heads,
            )
            layer = KANEncoderLayer(
                attention=attn,
                d_model=self.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                kan_grid_size=5,
            )
            layers.append(layer)

        return nn.ModuleList(layers)

    def _encode_branch(self, x, branch):
        """单个分支的编码过程"""
        # x: [B, H, C] — DataEmbedding_inverted 处理 permute
        enc_out = branch['embedding'](x, None)  # [B, C, d_model]

        for layer in branch['encoder']:
            enc_out, _ = layer(enc_out)

        return enc_out  # [B, C, d_model]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, H, C]

        # Non-stationary normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means) / stdev

        B, H, C = x_enc.shape

        # 频域分解: x -> x_trend + x_seasonal + x_residual
        x_trend, x_seasonal, x_residual = self.freq_decomp(x_enc)

        # 直接使用 [B, H, C] — DataEmbedding_inverted 内部会做 permute
        components = [x_trend, x_seasonal, x_residual]

        # 三个分支分别编码
        branch_outputs = []
        for comp, branch in zip(components, self.branches):
            out = self._encode_branch(comp, branch)  # [B, C, d_model]
            branch_outputs.append(out)

        # 自适应融合
        concat = torch.cat(branch_outputs, dim=-1)  # [B, C, d_model*3]
        gate_weights = self.branch_gate(concat)  # [B, C, 3]

        fused = sum(
            gate_weights[:, :, i:i+1] * branch_outputs[i]
            for i in range(3)
        )  # [B, C, d_model]

        # 投影: [B, C, d_model] -> [B, C, pred_len]
        output = self.projector(fused)

        # 转回: [B, C, F] -> [B, F, C]
        output = output.permute(0, 2, 1)

        # 反归一化
        output = output * stdev + means
        return output
