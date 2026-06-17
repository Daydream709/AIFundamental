"""
SparseTSF — 极轻量级稀疏时间序列预测模型
参考: SparseTSF: Modeling Long-term Time Series Forecasting with 1K Parameters

核心思想:
- 跨周期下采样: 将长序列按周期 p 拆分为 p 个短子序列
- 每个子序列独立建模（极简线性层）
- 参数量 < 0.001M (~1K)

在 v2.0 计划中作为"外部轻量化天花板"对照基准
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SparseTSF: 稀疏线性预测模型
    参数量级: < 0.001M (~1K)
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: [B, L, C] — 输入序列
        Returns: [B, F, C] — 预测
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
        return output
