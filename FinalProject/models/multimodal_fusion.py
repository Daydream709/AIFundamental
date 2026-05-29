"""
创新3: 跨模态对比对齐融合模型
时序数值 + 文本嵌入 + 时序递归图 → 对比对齐 + 门控融合
"""
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.contrastive_loss import InfoNCELoss
from layers.gating_fusion import GatingFusion


class TextEncoder(nn.Module):
    """文本编码器: 将文本嵌入映射到统一维度"""

    def __init__(self, text_dim, d_model, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, text_embed):
        # text_embed: [B, text_dim] 或 None
        if text_embed is None or text_embed.dim() == 1:
            return None
        return self.encoder(text_embed)  # [B, d_model]


class ImageEncoder(nn.Module):
    """图像编码器: 简单CNN处理递归图"""

    def __init__(self, img_size, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((img_size // 2, img_size // 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, img_tensor):
        # img_tensor: [B, 1, img_size, img_size] 或 None
        if img_tensor is None or img_tensor.dim() == 1:
            return None
        x = self.conv(img_tensor)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)  # [B, d_model]


class Model(nn.Module):
    """
    跨模态对比对齐融合预测模型
    核心创新:
    1. 三模态编码 — 时序Transformer + 文本MLP + 图像CNN
    2. InfoNCE对比损失 — 对齐同一时刻不同模态的表示
    3. 自适应门控融合 — 动态学习模态重要性权重
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.use_text = configs.use_text
        self.use_image = configs.use_image
        self.use_contrastive = configs.use_contrastive

        # 时序编码器 (基于 Transformer)
        self.ts_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=self.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )
        attn_layers = [
            EncoderLayer(
                attention=AttentionLayer(
                    attention=FullAttention(
                        mask_flag=False, output_attention=False,
                        attention_dropout=configs.dropout,
                    ),
                    d_model=self.d_model,
                    n_heads=configs.n_heads,
                ),
                d_model=self.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ]
        self.ts_encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(self.d_model))

        # 文本编码器
        self.text_encoder = TextEncoder(configs.text_dim, self.d_model, configs.dropout)

        # 图像编码器
        self.image_encoder = ImageEncoder(configs.img_size, self.d_model, configs.dropout)

        # 门控融合
        self.gating_fusion = GatingFusion(self.d_model, n_modalities=3)

        # 对比损失
        self.contrastive_loss_fn = InfoNCELoss(temperature=0.07) if self.use_contrastive else None

        # 输出投影
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.pred_len * configs.c_out),
        )
        self.c_out = configs.c_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                text_embed=None, img_tensor=None):
        """
        扩展forward: 支持多模态输入
        """
        # x_enc: [B, H, C]

        # Non-stationary normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means) / stdev

        B, H, C = x_enc.shape

        # 1. 时序编码
        ts_embed = self.ts_embedding(x_enc, x_mark_enc)  # [B, H, d_model]
        ts_out, _ = self.ts_encoder(ts_embed)             # [B, H, d_model]
        ts_feat = ts_out[:, -1, :]                         # [B, d_model] 取最后时间步

        # 2. 文本编码
        text_feat = self.text_encoder(text_embed) if self.use_text else None  # [B, d_model] or None

        # 3. 图像编码
        img_feat = self.image_encoder(img_tensor) if self.use_image else None  # [B, d_model] or None

        # 4. 门控融合
        fused_feat, gate_weights = self.gating_fusion(ts_feat, text_feat, img_feat)  # [B, d_model]

        # 5. 对比损失 (训练时计算)
        contrastive_loss = torch.tensor(0.0, device=x_enc.device)
        if self.use_contrastive and self.contrastive_loss_fn is not None:
            if text_feat is not None:
                contrastive_loss = contrastive_loss + self.contrastive_loss_fn(ts_feat, text_feat)
            if img_feat is not None:
                contrastive_loss = contrastive_loss + self.contrastive_loss_fn(ts_feat, img_feat)

        # 6. 投影输出
        output = self.projection(fused_feat)  # [B, F*C]
        output = output.reshape(B, self.pred_len, self.c_out)

        # 反归一化
        output = output * stdev + means

        return output
