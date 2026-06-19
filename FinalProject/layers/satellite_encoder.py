"""
卫星图像编码器 — v2.1 多模态
===========================

目的: 把 32×32 grayscale 卫星图编码成与文本同维度的 embedding
  - 空间信息 → 64 维向量 (与 text_dim 匹配)
  - 轻量化: < 50K 参数

设计:
  - 4 层 Conv2d (32 → 16 → 8 → 4 → 2 空间分辨率)
  - 3 个 Conv block (Conv + BN + ReLU + MaxPool)
  - 1 个 Flatten + Linear 投影到 64 维

参数计算:
  - Conv 1: 1→16,  3×3,  9×32×16×16 = 73,728  ← 但有 BN 32 + bias 16
  - Conv 2: 16→32, 3×3,  9×16×32×4  = 18,432
  - Conv 3: 32→64, 3×3,  9×32×64×1  = 18,432  (池化后 1×1)
  - 实际输入 32×32, 池化后 [32→16→8→4→2], 最终 flatten 2×2×64=256
  - Linear: 256 → 64 = 16,448

  参数量 ~ 50K (含 BN)
"""
import torch
import torch.nn as nn


class SatelliteImageEncoder(nn.Module):
    """
    卫星图像编码器: 32×32 grayscale → 64 维 embedding

    输入: [B, 1, 32, 32]  (1 通道灰度)
    输出: [B, embed_dim]   (默认 64 维)
    """

    def __init__(self, in_channels=1, embed_dim=64, img_size=32):
        super().__init__()
        self.embed_dim = embed_dim

        # 3 层 Conv: 1 → 8 → 16 → 32 (大幅缩减 channel)
        self.features = nn.Sequential(
            # [B, 1, 32, 32] → [B, 8, 16, 16]
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [B, 8, 16, 16] → [B, 16, 8, 8]
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # [B, 16, 8, 8] → [B, 32, 4, 4]
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 算 flatten 后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            feat = self.features(dummy)
            self.flatten_dim = feat.view(1, -1).shape[1]

        # 轻量投影: flatten → embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] 灰度图
        Returns:
            [B, embed_dim] 向量
        """
        # 安全检查: 如果输入是 [B, H, W] (没有 channel 维), 加一维
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.proj(feat)


# 单元测试
if __name__ == '__main__':
    enc = SatelliteImageEncoder()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f'SatelliteImageEncoder 参数量: {n_params:,} ({n_params/1e3:.1f}K)')

    # 测试前向
    x = torch.randn(2, 1, 32, 32)
    y = enc(x)
    print(f'  Input: {x.shape}  →  Output: {y.shape}')

    # 测试 backward
    loss = y.sum()
    loss.backward()
    print(f'  Backward OK')

    # 测试无 channel 维输入
    x2 = torch.randn(2, 32, 32)
    y2 = enc(x2)
    print(f'  No-channel input: {x2.shape}  →  {y2.shape}')
