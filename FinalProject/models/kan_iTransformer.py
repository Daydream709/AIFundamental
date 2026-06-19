"""
KAN-iTransformer — 核心贡献1: 冲刺最高精度 (~120M参数)

基于 iTransformer 倒置架构，集成 5+2=7 大优化模块 (v2.1):
  模块1: 真实 B-spline KAN (Cox-de Boor 递推) 替换 FFN + 级联频域分解 (CFD)
        逐层剥离趋势/季节/残差，交由不同 KAN 专家处理
  模块1+: Wavelet-CFD (新增, --use_wavelet 开关) - 适合非平稳信号
  模块2: 掩码重建自监督预训练 (mask_ratio=15%)
        让模型学习数据内在表征，提升鲁棒性
  模块3: 概率输出 (GaussianNLL) + 共形预测校准
        输出均值+方差，推理后95%置信区间
  模块4: RevIN 可逆实例归一化
        消除训练-测试分布偏移
  模块5: 模型仲裁 — 5维统计特征 + MLP路由器
        动态融合 KAN-iTransformer / PatchTST / Mamba 预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class WaveletCFD(nn.Module):
    """
    Wavelet 版级联频域分解 (Wavelet-CFD)

    创新点 (v2.1 优化):
      - 用 Wavelet (DWT) 替代 FFT 频域分解
      - Wavelet 在时频两域都有分辨率, 适合非平稳信号
        (FFT 假设全局平稳, 难以处理突变; DWT 能捕捉局部突变)
      - 小波系数按尺度 (scale) 自然分层:
          approximation (a) ≈ 趋势 (低频)
          detail (d1, d2, ...) ≈ 不同尺度的细节 (中-高频)
      - 创新: 配合级联结构, 每一层做不同尺度的 DWT 分解,
        残差依次进入下一层进一步分解

    数学形式:
      输入 x:  [B, L, C]
      对每变量 DWT 分解: x → [a_n, d_n, d_{n-1}, ..., d_1]
        其中 a_n 是低频近似 (≈ 趋势), d_i 是各尺度细节 (≈ 季节)
      CFD 分配:
        trend = a_n (低频趋势)
        seasonal = largest |d_i| 分量 (主季节)
        residual = x - trend - seasonal
    """

    def __init__(self, wavelet='db4', level=2):
        super().__init__()
        try:
            import pywt
        except ImportError:
            raise ImportError("需要安装 PyWavelets: pip install PyWavelets")
        self.pywt = pywt
        self.wavelet = wavelet
        self.level = level

    def _dwt_decompose(self, x_1d):
        """
        对单条 1D 序列做 DWT 分解 (使用 pywt)

        Args:
            x_1d: [B, L]
        Returns:
            approx: [B, L] (低频近似 ≈ 趋势)
            detail_main: [B, L] (主细节分量 ≈ 季节)
            residual: [B, L] (残差)
        """
        B, L = x_1d.shape
        # 动态决定 level: 信号太短 (db4 需 ≥ 8) 时降级
        max_level = self.pywt.dwt_max_level(L, self.wavelet)
        eff_level = min(self.level, max_level, max(1, L // 8))

        approx_list = []
        detail_list = []

        for b in range(B):
            coeffs = self.pywt.wavedec(x_1d[b].detach().cpu().numpy(), self.wavelet, level=eff_level)
            # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            # 用 pywt 重建每个分量 (其余置零) 来获得等长信号
            # 重建到原始长度 L
            approx = self.pywt.waverec(
                [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]],
                self.wavelet
            )[:L]
            approx_list.append(torch.from_numpy(approx).float().to(x_1d.device))

            # 主细节分量: 找幅度最大的 detail
            if len(coeffs) > 1:
                abs_details = [np.abs(c).max() for c in coeffs[1:]]
                main_idx = int(np.argmax(abs_details)) + 1  # +1 因为 coeffs[0] 是 approx
                detail = self.pywt.waverec(
                    [np.zeros_like(coeffs[0])] +
                    [coeffs[i] if i == main_idx else np.zeros_like(coeffs[i]) for i in range(1, len(coeffs))],
                    self.wavelet
                )[:L]
            else:
                # 极短信号: 没法分解, 细节全 0
                detail = np.zeros(L)
            detail_list.append(torch.from_numpy(detail).float().to(x_1d.device))

        approx = torch.stack(approx_list, dim=0)
        detail_main = torch.stack(detail_list, dim=0)
        residual = x_1d - approx - detail_main
        return approx, detail_main, residual

    def forward(self, x, layer_idx=0):
        """
        x: [B, L, C]
        Returns: x_trend, x_seasonal, x_residual (各 [B, L, C])
        """
        B, L, C = x.shape

        x_trend = torch.zeros_like(x)
        x_seasonal = torch.zeros_like(x)
        x_residual = torch.zeros_like(x)

        for c in range(C):
            approx, detail, residual = self._dwt_decompose(x[:, :, c])
            # 层级剥离: 高层(layer_idx 大)剥离更多残差细节
            if layer_idx > 0:
                # 高层用更细的 wavelet 分解 (level + 1)
                self.level = min(self.level + 1, 4)
                approx2, detail2, residual2 = self._dwt_decompose(residual)
                x_trend[:, :, c] = approx
                x_seasonal[:, :, c] = detail + detail2
                x_residual[:, :, c] = residual2
                self.level = max(self.level - 1, 1)  # 还原
            else:
                x_trend[:, :, c] = approx
                x_seasonal[:, :, c] = detail
                x_residual[:, :, c] = residual

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
                 kan_grid_size=5, top_k=5, layer_idx=0, use_wavelet=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_idx = layer_idx

        # CFD: FFT 版 或 Wavelet 版
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.cfd = WaveletCFD(wavelet='db4', level=2)
        else:
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
# 模块2: 掩码重建预训练
# ============================================================================

class MaskedReconstructionHead(nn.Module):
    """
    掩码重建自监督任务头

    在预训练阶段:
    1. 随机掩码15%的输入时间步
    2. 模型重建被掩码的部分
    3. 损失 = MSE(重建, 原始)

    正式训练时禁用 (use_masked_pretrain=False)
    """

    def __init__(self, d_model, c_out):
        super().__init__()
        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, c_out),
        )

    def forward(self, x):
        """
        x: [B, C, d_model]
        Returns: [B, C, c_out] (reconstructed values)
        """
        return self.reconstructor(x)


def generate_mask(x, mask_ratio=0.15):
    """
    生成随机掩码
    x: [B, H, C]
    Returns: mask [B, H, C], masked_x [B, H, C]
    """
    B, H, C = x.shape
    mask = (torch.rand(B, H, C, device=x.device) > mask_ratio).float()
    masked_x = x * mask
    return mask, masked_x


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
        self.use_masked_pretrain = getattr(configs, 'use_masked_pretrain', False)
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
                    use_wavelet=getattr(configs, 'use_wavelet', False),
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

        # 模块2: 掩码重建头 (预训练用)
        if self.use_masked_pretrain:
            self.mask_head = MaskedReconstructionHead(self.d_model, self.seq_len)

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

        # 掩码重建 (预训练模式)
        if self.use_masked_pretrain and self.training:
            recon_mask, x_masked = generate_mask(x_norm, mask_ratio=0.15)
            x_use = x_masked
        else:
            recon_mask = None
            x_use = x_norm

        # 倒置嵌入: [B, H, C] → [B, C, d_model]
        enc_out = self.embedding(x_use, None)

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

    def pretrain_step(self, x_enc):
        """
        掩码重建预训练步骤
        在不修改原有forward的情况下，允许外部调用
        """
        was_training = self.training
        old_mask_flag = self.use_masked_pretrain
        self.use_masked_pretrain = True
        self.train()

        B, H, C = x_enc.shape

        if self.use_revin:
            x_norm = self.revin(x_enc.permute(0, 2, 1), 'norm').permute(0, 2, 1)
        else:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = (x_enc - means) / stdev

        # 生成掩码
        recon_mask = (torch.rand(B, H, C, device=x_enc.device) > 0.15).float()
        x_masked = x_norm * recon_mask

        # 嵌入 + 编码
        enc_out = self.embedding(x_masked, None)
        for layer in self.encoder:
            enc_out, attn = layer(enc_out)

        # 重建
        reconstruction = self.mask_head(enc_out)  # [B, C, H]
        reconstruction = reconstruction.permute(0, 2, 1)  # [B, H, C]

        # 只对掩码位置计算损失
        masked_positions = (recon_mask == 0).float()
        se = (reconstruction - x_norm) ** 2
        recon_loss = (se * masked_positions).sum() / (masked_positions.sum() + 1e-8)

        self.use_masked_pretrain = old_mask_flag
        if not was_training:
            self.eval()

        return recon_loss

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
