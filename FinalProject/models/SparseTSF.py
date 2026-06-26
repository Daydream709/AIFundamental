"""
SparseTSF — 极轻量级稀疏时间序列预测模型
参考: SparseTSF: Modeling Long-term Time Series Forecasting with 1K Parameters

核心思想:
- 跨周期下采样: 将长序列按周期 p 拆分为 p 个短子序列
- 每个子序列独立建模（极简线性层）
- 参数量 < 0.001M (~1K)

v2.1 多模态扩展 (可选):
- 当 forward 传入 text_embed 时，启用 TextEncoder 分支
- text_embed [B, text_dim] → Linear → 残差注入到时序预测
- 引入可学习 gate 控制文本贡献强度
- 原始时序路径保持不变，无 text 时完全退化为原版
"""
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    文本编码器: 将 text_dim 维的句子嵌入映射为与变量空间对齐的 [B, C] 残差

    参数量: text_dim * hidden + hidden * enc_in + enc_in * 1 (gate)
          ≈ 768*64 + 64*C + C ≈ 49K + 65C
          Environment (C=6): ≈ 49.4K
          对比 SparseTSF 原 1K 参数：约 50×，但仍属极轻量
    """

    def __init__(self, text_dim=768, n_vars=6, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_vars),
        )
        # 数据相关门控: 从 text 派生一个 [0, 1] 标量，控制文本贡献
        self.gate = nn.Linear(n_vars, 1)
        # 初始化 gate bias 为 -2.5 → sigmoid ≈ 0.075, 起步几乎不干扰时序
        nn.init.constant_(self.gate.bias, -2.5)
        nn.init.zeros_(self.gate.weight)

    def forward(self, text_embed):
        """
        text_embed: [B, text_dim]
        Returns:    [B, C] 残差贡献（已乘 0.1 缩放 + gate）
        """
        feat = self.encoder(text_embed)               # [B, C]
        g = torch.sigmoid(self.gate(feat))            # [B, 1]
        return feat * g * 0.1                          # [B, C]


class Model(nn.Module):
    """
    SparseTSF: 稀疏线性预测模型
    参数量级: < 0.001M (~1K)

    v2.1: 可选地接受 text_embed 实现多模态融合
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, 'period_len', 24)

        # 确保周期不超过序列长度
        self.period_len = min(self.period_len, self.seq_len)

        # 下采样后的序列长度
        self.down_len = self.seq_len // self.period_len
        if self.down_len < 1:
            self.down_len = 1

        # 每个变量独立建模: 每个变量一个线性层
        # Linear(seg_len, pred_len), 每个变量独立 (individual=True 风格)
        self.linear = nn.ModuleList([
            nn.Linear(self.down_len, self.pred_len)
            for _ in range(self.enc_in)
        ])

        # 极小参数量初始化
        for linear in self.linear:
            nn.init.ones_(linear.weight)
            nn.init.zeros_(linear.bias)

        # v2.1: 多模态文本编码器（始终实例化，forward 时按需启用）
        # 文本输入可能为 None，因此保留模块但延迟激活
        # 重要: text_dim 实际可能是 128 (sentence-transformers) 或 768 (全连接),
        # 用 getattr 读取, 默认 128 兼容已有 cache
        text_dim = getattr(configs, 'text_dim', 128)
        self.text_encoder = TextEncoder(
            text_dim=text_dim, n_vars=self.enc_in, hidden_dim=64
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                text_embed=None):
        """
        x_enc:      [B, L, C] — 输入序列
        text_embed: [B, text_dim] 或 None — 可选文本嵌入
        Returns:    [B, F, C] — 预测
        """
        B, L, C = x_enc.shape
        self.period_len = min(self.period_len, L)
        seg_len = L // self.period_len

        # 跨周期下采样: 将 [B, L, C] 分解为 period 个长度为 seg_len 的子序列
        # 对每个周期位置采样
        downsampled = []
        for p in range(self.period_len):
            # 从位置 p 开始，每隔 period_len 采样一个点
            indices = torch.arange(p, L, self.period_len, device=x_enc.device)
            # 取最后 seg_len 个点（保证长度一致）
            indices = indices[-seg_len:]
            segment = x_enc[:, indices, :]  # [B, seg_len, C]
            downsampled.append(segment)

        # 对每个变量在每个子序列上做线性预测
        outputs = []
        for c in range(C):
            var_outputs = []
            for p in range(self.period_len):
                seg = downsampled[p][:, :, c]  # [B, seg_len]
                # 确保输入维度匹配
                linear = self.linear[c]
                # 如果 seg_len 与 linear.in_features 不匹配，做自适应
                actual_seg_len = seg.shape[1]
                if actual_seg_len != linear.in_features:
                    # 动态调整: 对输入做简单的线性插值
                    seg_adapted = torch.nn.functional.interpolate(
                        seg.unsqueeze(1),
                        size=linear.in_features,
                        mode='linear',
                        align_corners=False,
                    ).squeeze(1)
                    out = linear(seg_adapted)  # [B, pred_len]
                else:
                    out = linear(seg)  # [B, pred_len]
                var_outputs.append(out)

            # 取所有子序列预测的中位数
            stacked = torch.stack(var_outputs, dim=0)  # [period, B, pred_len]
            var_pred = stacked.median(dim=0)[0]  # [B, pred_len]
            outputs.append(var_pred)

        # 组合: [B, C, pred_len] -> [B, pred_len, C]
        output = torch.stack(outputs, dim=-1)  # [B, pred_len, C]

        # v2.1: 文本融合（仅当 text_embed 有效时启用）
        # 检测占位符: dataset_base 在无 text 时返回 torch.zeros(1), shape=(1,)
        if text_embed is not None and text_embed.shape[-1] > 1:
            text_residual = self.text_encoder(text_embed)  # [B, C]
            # 广播到 pred_len 维度作为加性残差
            output = output + text_residual.unsqueeze(1)   # [B, pred_len, C]

        return output
