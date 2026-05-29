"""
TimeKAN — KAN 网络用于时序预测
将 KAN 层替代传统 MLP，增强非线性建模能力
"""
import torch
import torch.nn as nn
from layers.kan_layers import KANLayer
from layers.Embed import PositionalEmbedding


class TimeKANBlock(nn.Module):
    """单个 TimeKAN 块: LayerNorm → KAN → 残差"""

    def __init__(self, d_model, d_ff, grid_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kan = KANLayer(d_model, d_ff, d_model, grid_size=grid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D] 或 [B, C, D]
        residual = x
        x = self.norm(x)
        x = self.kan(x)
        x = self.dropout(x)
        return residual + x


class Model(nn.Module):
    """
    TimeKAN: 使用 KAN (Kolmogorov-Arnold Networks) 进行时序预测
    核心思想: 用可学习的 B-spline 函数替代固定线性变换
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in

        # 输入嵌入: 将每个变量的时间序列映射到 d_model
        self.enc_embedding = nn.Linear(self.seq_len, self.d_model)

        # 位置编码
        self.pos_embedding = PositionalEmbedding(self.d_model)

        # KAN 编码层
        n_layers = configs.e_layers
        d_ff = configs.d_ff
        self.kan_blocks = nn.ModuleList([
            TimeKANBlock(self.d_model, d_ff, grid_size=5, dropout=configs.dropout)
            for _ in range(n_layers)
        ])

        # 输出投影: d_model -> pred_len
        self.projection = nn.Linear(self.d_model, self.pred_len)

        # 归一化
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, H, C]

        # Non-stationary normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means) / stdev

        B, H, C = x_enc.shape

        # 倒置视角: [B, C, H]
        x = x_enc.permute(0, 2, 1)

        # 嵌入: [B, C, H] -> [B, C, d_model]
        x = self.enc_embedding(x)
        x = x + self.pos_embedding(x)

        # KAN 编码
        for block in self.kan_blocks:
            x = block(x)

        x = self.norm(x)

        # 投影: [B, C, d_model] -> [B, C, pred_len]
        output = self.projection(x)

        # 转回: [B, C, F] -> [B, F, C]
        output = output.permute(0, 2, 1)

        # 反归一化
        output = output * stdev + means
        return output
