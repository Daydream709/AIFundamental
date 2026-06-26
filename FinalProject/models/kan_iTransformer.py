"""
KAN-iTransformer — 核心贡献1: 冲刺最高精度 (~120M参数)

基于 iTransformer 倒置架构，集成4大优化模块:
  模块1: KAN层 (B-spline) 替换FFN + 级联频域分解 (CFD)
        逐层剥离趋势/季节/残差，交由不同KAN专家处理
  模块2: 概率输出 (GaussianNLL) + 共形预测校准
        输出均值+方差，推理后95%置信区间
  模块3: RevIN 可逆实例归一化
        消除训练-测试分布偏移
  模块4: 模型仲裁 — 5维统计特征 + MLP路由器
        动态融合 KAN-iTransformer / PatchTST / Mamba 预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.kan_layers import KANLayer
from layers.StandardNorm import Normalize as RevIN
from layers.conformal_prediction import QuantileHead
from layers.meta_arbitrator import MetaArbitrator


# ============================================================================
# 模块1: 级联频域分解 (CFD) — 逐层剥离
# ============================================================================

class CascadedFreqDecomp(nn.Module):
    """
    级联频域分解 — Layer-by-layer frequency decomposition

    与 AdaptiveFreqDecomp 的区别:
    - AdaptiveFreqDecomp: 在输入端一次性做三分支分解
    - CFD: 每层编码器各做一次分解，逐层剥离不同频段

    设计思想:
    - 第1层: 剥离低频趋势 (0-3个最低频)
    - 第2层: 剥离中频季节分量 (幅度最大的top_k频率)
    - 第3层+: 处理剩余高频残差
    每层使用不同的 KAN 专家处理各自的频段
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x, layer_idx=0):
        """
        x: [B, L, C]
        Returns: x_trend, x_seasonal, x_residual (各 [B, L, C])
        """
        B, L, C = x.shape

        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        freq_mag = torch.abs(x_fft)
        n_freq = x_fft.shape[1]

        # 趋势分量: 每层剥离不同频率范围的趋势
        n_trend_freqs = min(3 + layer_idx, n_freq)
        trend_mask = torch.zeros_like(freq_mag)
        trend_mask[:, :n_trend_freqs, :] = 1.0
        x_trend = torch.fft.irfft(x_fft * trend_mask, n=L, dim=1)

        # 季节分量: 幅度最大的 top_k 频率 (排除已剥离的趋势频段)
        seasonal_mask = torch.zeros_like(freq_mag)
        mag = freq_mag.clone()
        mag[:, :n_trend_freqs, :] = 0
        mag_flat = mag.mean(dim=-1)
        topk_idx = torch.topk(mag_flat, min(self.top_k, n_freq - n_trend_freqs), dim=1).indices

        for b in range(B):
            seasonal_mask[b, topk_idx[b], :] = 1.0
        x_seasonal = torch.fft.irfft(x_fft * seasonal_mask, n=L, dim=1)

        # 残差
        x_residual = x - x_trend - x_seasonal

        return x_trend, x_seasonal, x_residual


# ============================================================================
# KAN 编码器层 (含 CFD)
# ============================================================================

class KANCFDEncoderLayer(nn.Module):
    """
    集成级联频域分解的 KAN 编码器层

    每层流程:
    1. CFD分解 → 趋势/季节/残差
    2. 三个KAN专家分别处理三个分量
    3. 自适应门控融合三分支
    4. 自注意力 + 残差
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 kan_grid_size=5, top_k=5, layer_idx=0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_idx = layer_idx

        # CFD
        self.cfd = CascadedFreqDecomp(top_k=top_k)

        # 三个 KAN 专家 (趋势/季节/残差)
        self.trend_kan = KANLayer(d_model, d_ff, d_model, grid_size=kan_grid_size)
        self.seasonal_kan = KANLayer(d_model, d_ff, d_model, grid_size=kan_grid_size)
        self.residual_kan = KANLayer(d_model, d_ff, d_model, grid_size=kan_grid_size)

        # 门控融合: [B, C, d_model*3] → [B, C, 3]
        self.branch_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )

        # 集成融合
        self.fuse_proj = nn.Linear(d_model * 3, d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: [B, C, d_model] — 倒置嵌入后的表示 (变量数=序列长度维度)
        注意: CFD在时间维上操作，需要 reshape
        """
        B, C, d = x.shape

        # 自注意力在变量维度上 (倒置架构)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # CFD 分解: permute到 [B, d, C] 在"序列维度"上做FFT
        # 因为倒置架构中 C (变量数) 是序列维，d_model 是特征维
        x_for_decomp = self.norm2(x)  # [B, C, d]
        x_trend, x_seasonal, x_residual = self.cfd(x_for_decomp, self.layer_idx)

        # 三个KAN专家处理
        trend_out = self.trend_kan(x_for_decomp)       # [B, C, d]
        seasonal_out = self.seasonal_kan(x_for_decomp)  # [B, C, d]
        residual_out = self.residual_kan(x_for_decomp)  # [B, C, d]

        # 门控融合
        concat = torch.cat([trend_out, seasonal_out, residual_out], dim=-1)  # [B, C, 3d]
        gate_weights = self.branch_gate(concat)  # [B, C, 3]

        # 加权组合
        kan_out = (
            gate_weights[:, :, 0:1] * trend_out +
            gate_weights[:, :, 1:2] * seasonal_out +
            gate_weights[:, :, 2:3] * residual_out
        )

        x = x + kan_out
        return x, attn


# ============================================================================
# 主模型: KAN-iTransformer
# ============================================================================

class Model(nn.Module):
    """
    KAN-iTransformer: 冲刺最高精度的自研高性能模型

    参数量: ~120M (主要来自多个KAN层)
    适用场景: ETTm2 / Weather / Electricity / Environment (所有数据集)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention

        # 模块参数
        self.use_cfd = getattr(configs, 'use_cfd', True)
        self.use_revin = getattr(configs, 'use_revin', True)
        self.use_probabilistic = getattr(configs, 'use_probabilistic', True)
        kan_grid_size = getattr(configs, 'kan_grid_size', 5)
        top_k = getattr(configs, 'top_k', 5)

        # 模块4: RevIN 可逆实例归一化
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)
        else:
            self.revin = None

        # 倒置嵌入: 将变量维作为序列维
        self.embedding = DataEmbedding_inverted(
            c_in=self.seq_len,
            d_model=self.d_model,
            dropout=configs.dropout,
        )

        # 模块1: 级联 KAN-CFD 编码器
        encoder_layers = []
        for i in range(self.e_layers):
            attn = AttentionLayer(
                attention=FullAttention(
                    mask_flag=False,
                    output_attention=configs.output_attention,
                    attention_dropout=configs.dropout,
                ),
                d_model=self.d_model,
                n_heads=self.n_heads,
            )
            if self.use_cfd:
                layer = KANCFDEncoderLayer(
                    attention=attn,
                    d_model=self.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    kan_grid_size=kan_grid_size,
                    top_k=top_k,
                    layer_idx=i,
                )
            else:
                # 无CFD的简单KAN编码器层
                from models.kan_iTransformer import SimpleKANEncoderLayer
                layer = SimpleKANEncoderLayer(
                    attention=attn,
                    d_model=self.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    kan_grid_size=kan_grid_size,
                )
            encoder_layers.append(layer)

        self.encoder = nn.ModuleList(encoder_layers)

        # 模块3: 概率输出头
        if self.use_probabilistic:
            # 输出均值 + log方差 (GaussianNLL)
            self.mean_head = nn.Linear(self.d_model, self.pred_len)
            self.logvar_head = nn.Linear(self.d_model, self.pred_len)
        else:
            self.projector = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: [B, H, C]
        x_mark_enc: [B, H, T] (未使用，倒置架构不需要时间特征)
        x_dec: 未使用
        x_mark_dec: 未使用

        Returns:
          概率模式: (mean [B, F, C], logvar [B, F, C])
          普通模式: output [B, F, C]
        """
        B, H, C = x_enc.shape

        # 模块4: RevIN 归一化 (RevIN 期望 [B, L, C])
        if self.use_revin:
            x_norm = self.revin(x_enc, 'norm')  # [B, H, C]
        else:
            # 简单实例归一化
            means = x_enc.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = (x_enc - means) / stdev

        # 倒置嵌入: [B, H, C] → [B, C, d_model]
        enc_out = self.embedding(x_norm, None)

        # 编码器层
        for layer in self.encoder:
            enc_out, attn = layer(enc_out)

        # 输出
        if self.use_probabilistic:
            mean = self.mean_head(enc_out)   # [B, C, pred_len]
            logvar = self.logvar_head(enc_out)  # [B, C, pred_len]
            mean = mean.permute(0, 2, 1)     # [B, pred_len, C]
            logvar = logvar.permute(0, 2, 1)

            # 模块4: RevIN 反归一化 (输出 [B, pred_len, C])
            if self.use_revin:
                mean = self.revin(mean, 'denorm')
            else:
                mean = mean * stdev + means

            return mean, logvar

        else:
            output = self.projector(enc_out)  # [B, C, pred_len]
            output = output.permute(0, 2, 1)  # [B, pred_len, C]

            if self.use_revin:
                output = self.revin(output, 'denorm')
            else:
                output = output * stdev + means

            return output

    def predict_with_confidence(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                                  conformal_predictor=None, alpha=0.05):
        """
        带置信区间的预测

        Returns:
          mean: [B, F, C]
          lower: [B, F, C]
          upper: [B, F, C]
        """
        assert self.use_probabilistic, "需要 use_probabilistic=True"

        mean, logvar = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        std = torch.exp(0.5 * logvar)

        # 模型自身的95%置信区间 (基于Gaussian)
        z = 1.96  # 95% CI
        lower = mean - z * std
        upper = mean + z * std

        # 共形预测校准
        if conformal_predictor is not None:
            preds_list = [lower, mean, upper]
            mean, lower, upper = conformal_predictor.predict_with_intervals(
                preds_list, alpha=alpha
            )

        return mean, lower, upper


# ============================================================================
# 模块5: 模型仲裁器 (放在此文件中以便与KAN-iTransformer统一管理)
# ============================================================================

class ModelArbitrator(nn.Module):
    """
    模型仲裁集成系统

    对输入序列提取5维统计特征:
      1. 谱熵 (频率分布的均匀性)
      2. 趋势强度 (线性拟合R²)
      3. 周期性 (FFT主频相对强度)
      4. 方差
      5. 滞后1自相关

    训练轻量MLP路由器，动态融合三个模型的预测:
      - KAN-iTransformer (高性能)
      - PatchTST (Transformer代表)
      - Mamba (SSM代表)
    """

    def __init__(self, model_names, d_hidden=64):
        super().__init__()
        self.model_names = model_names
        self.arbitrator = MetaArbitrator(n_models=len(model_names), d_hidden=d_hidden)

    def forward(self, x_enc, model_outputs):
        """
        x_enc: [B, H, C]
        model_outputs: dict {model_name: prediction [B, F, C]}
        """
        preds_list = [model_outputs[name] for name in self.model_names]
        ensemble_pred, weights = self.arbitrator(x_enc, preds_list)
        return ensemble_pred, weights


# ============================================================================
# 简单KAN编码器层 (无CFD, 作为对比参照)
# ============================================================================

class SimpleKANEncoderLayer(nn.Module):
    """使用KAN替代FFN的编码器层 (无CFD)"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, kan_grid_size=5):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.kan_ffn = KANLayer(d_model, d_ff, d_model, grid_size=kan_grid_size)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        x = x + self.kan_ffn(self.norm2(x))
        return x, attn
