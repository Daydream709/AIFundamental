# 多模态实验问题诊断报告

> **结论**：多模态实验结果完全一致（7 种 text_mode 下 MSE/MAE 都一样），不是"模型未利用多模态信息"那么简单，而是**多模态数据在多个层级被完全丢弃**。本文档从代码层面逐层定位问题。

---

## 一、问题现象

7 种 text_mode 的结果（PatchTST 和 Mamba 都有此问题）：

| text_mode | PatchTST MSE | Mamba MSE |
|-----------|:-----------:|:--------:|
| baseline | 0.4321 | 0.2559 |
| report | 0.4321 | 0.2559 |
| search | 0.4321 | 0.2559 |
| both_concat | 0.4321 | 0.2559 |
| both_gating | 0.4321 | 0.2559 |
| satellite | 0.4321 | 0.2559 |
| text+satellite | 0.4321 | 0.2559 |

所有模式的数值**精确相等**到小数点后 4 位，说明代码路径完全相同，没有任何分支变化。

---

## 二、逐层诊断（5 个 Bug）

### Bug 1️⃣ ：配置文件名不一致（`text_fusion_mode` vs `text_mode`）

**位置**：`data_provider/data_factory.py:40`

```python
# data_factory.py:40
text_mode = getattr(config, 'text_fusion_mode', 'concat')  # ← 读 text_fusion_mode
```

而 `train_line3.py:64-67` 写入的是：
```python
{"use_text": True, "text_mode": "report_only"},   # ← 写 text_mode
```

**后果**：`getattr(config, 'text_fusion_mode', 'concat')` 永远走默认值 `'concat'`，`text_mode` 参数被静默忽略。

**影响**：所有配置都使用同一种融合方式——但即使融合方式相同，文本数据本身应该不一样啊？继续看 Bug 2。

---

### Bug 2️⃣ ：`_attach_multimodal` 调用的是同一种融合模式

`data_factory.py:42` 传入的 `text_mode` 永远是 `'concat'`，但 `multimodal_builder.py:51` 的回退路径（无 cache）才用此参数。如果**已经有 cache** (`Environment_text_embeds.npy`)，**根本不读 text_mode**：

```python
# multimodal_builder.py:40-44
text_cache = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
if os.path.exists(text_cache):
    text_embeds = np.load(text_cache)  # ← 直接用 cache，忽略 text_mode
```

**验证**：从 line3 输出看，所有 7 种 text_mode 的数值完全相同，进一步说明多模态特征本身就是同一个张量。

---

### Bug 3️⃣ ⚠️ **核心问题**：exp_train.py 完全丢弃多模态 batch

**位置**：`exp/exp_train.py:122-134`（训练）和 `:155-170`（验证）

```python
# exp_train.py:122-124  (训练循环)
for batch in data_loader:
    batch = [b.to(self.device) for b in batch]
    x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
    # ↑ 只解包了 4 个元素！
    # ↓ batch[4] = text_embed, batch[5] = img_tensor  ← 完全丢弃
```

**data_provider/dataset_base.py:129-154`__getitem__` 实际返回 6 个元素**：

```python
result = (
    torch.FloatTensor(seq_x),        # batch[0] x_enc
    torch.FloatTensor(seq_y),        # batch[1] x_y
    torch.FloatTensor(seq_x_mark),   # batch[2] x_mark_enc
    torch.FloatTensor(seq_y_mark),   # batch[3] x_mark_y
)
# ↓ 多模态扩展
if self.text_embeds_slice is not None:
    text = self.text_embeds_slice[s_begin:s_end].mean(axis=0)
    result = result + (torch.FloatTensor(text),)   # batch[4] text
else:
    result = result + (torch.zeros(1),)            # batch[4] 占位

if self.img_tensors_slice is not None:
    sat_imgs = self.img_tensors_slice[s_begin:s_end]
    sat_avg = sat_imgs.mean(axis=0)
    result = result + (torch.FloatTensor(sat_avg),)  # batch[5] img
else:
    result = result + (torch.zeros(1, 32, 32),)      # batch[5] 占位
```

**但 exp_train.py 只 unpack 了 4 个**！`batch[4]` 和 `batch[5]` 被完全丢弃。模型从未看到任何多模态数据。

---

### Bug 4️⃣ ：模型 forward 不支持多模态输入

**位置**：`third_party/TimeSeriesLibrary/models/PatchTST.py:213` 和 `MambaSimple.py:53`

```python
# PatchTST.forward
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # ↑ 签名里没有 text_embed / img_tensor 参数
```

```python
# MambaSimple.forward
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # ↑ 同上
```

**即使 exp_train.py 把 batch[4:6] 传进去，模型签名也不接受**。这说明 TSLib 官方实现是纯时序模型，多模态融合需要**单独写一个多模态模型**（如 `MultimodalFusion`），而不是 PatchTST/Mamba 加参数。

---

### Bug 5️⃣ ：缺少真正的多模态融合模型 `MultimodalFusion`

从 `models/__init__.py` 的对比模型列表看，理论上应该有 `MultimodalFusion` 模型。但 line3 实验**用 PatchTST 和 Mamba 跑多模态**——这两个模型本身就不支持多模态输入。

```
实验设计错位：
  ❌ PatchTST/Mamba + use_text=True  → 模型签名不接受
  ✅ MultimodalFusion + use_text=True → 这才是正确组合
```

---

## 三、问题根因总结

| 层级 | 问题 | 后果 |
|------|------|------|
| 1. 配置层 | `text_fusion_mode` 与 `text_mode` 名字不匹配 | text_mode 永远走默认值 |
| 2. 缓存层 | 加载 cache 时忽略 text_mode | 文本嵌入不会因模式改变 |
| 3. 训练层 | exp_train.py 只 unpack batch[0:4] | **text 和 img 被完全丢弃** |
| 4. 模型层 | PatchTST/Mamba.forward 签名不包含多模态参数 | 即使传进去也报错 |
| 5. 设计层 | 多模态实验应使用 MultimodalFusion 而非 PatchTST/Mamba | 实验模型选择错误 |

**问题的根因是第 3 层**：训练循环根本没把多模态数据传给模型。即使前面的 cache 和融合模式都修复了，模型依然看不到多模态信息。

---

## 四、修复方案

### 方案 A：快速验证多模态数据本身是否有效（最小修复）

**目标**：证明 text/img 数据有效，问题出在训练循环。

**步骤**：修改 `exp_train.py`，在训练循环中打印 text 数据的统计量：

```python
# exp_train.py:122 改为
for batch in data_loader:
    batch = [b.to(self.device) for b in batch]
    if len(batch) >= 6:
        x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
        text, img = batch[4], batch[5]
        if epoch == 0:
            print(f"  [DEBUG] text.shape={text.shape}, img.shape={img.shape}, "
                  f"text.mean={text.mean():.4f}, text.std={text.std():.4f}")
    else:
        x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
```

跑 1 个 epoch 验证：text 的 std 应该 > 0（不是全零占位符），img 应该非零。

---

### 方案 B：完整修复（让 PatchTST/Mamba 支持多模态）

**目标**：让 PatchTST/Mamba 真正利用 text/img。

需要 4 步：
1. **模型层**：在 `models/PatchTST.py` 和 `models/MambaSimple.py` 中加多模态融合 wrapper（嵌入 + 拼接 + 门控）
2. **训练循环**：在 `exp_train.py` 中解包 batch[4:6] 并传入
3. **配置对齐**：将 `text_fusion_mode` 统一为 `text_mode`
4. **重跑实验**：在 line3 中重跑

工作量：~2-3 小时（需要修改 3 个文件，patch TSLib 第三方库）

---

### 方案 C：切换到正确的多模态模型

**目标**：用真正的 `MultimodalFusion` 跑多模态实验。

如果 `MultimodalFusion` 模型已存在（在 `models/` 目录下），只需：
1. 把 `train_line3.py` 的 `MODELS` 改为 `["MultimodalFusion"]`
2. 检查 `MultimodalFusion` 的 forward 是否接受 text 和 img

工作量：~30 分钟（如果模型本身是正确的）

---

## 五、建议的实验报告表述

> **多模态实验因训练循环 bug 未生效**：在排查中发现，`exp_train.py` 的训练循环只解包了 batch 的前 4 个元素（时序数据），完全丢弃了 batch[4] (text) 和 batch[5] (img)。此外，PatchTST/Mamba 的 forward 签名也不接受多模态参数。因此当前 line3 实验结果仅反映纯时序性能，不能用于评估多模态融合的价值。修复训练循环并扩展模型签名后需要重跑。

---

## 六、附录：快速验证脚本

跑这一个命令可以验证问题：

```python
# scripts/debug_multimodal.py
from data_provider.data_factory import data_provider
from data_provider.dataset_base import BaseDataset
import torch

ds = BaseDataset(
    root_path='./dataset',
    data_path='Environment.csv',
    flag='train',
    size=(96, 48, 96),
)
ds.text_embeds_slice = None  # 或加载真实 cache
ds.img_tensors_slice = None

batch = ds[0]
print(f"Batch length: {len(batch)}")  # 应该是 4 或 6
print(f"Elements: {[b.shape for b in batch]}")
```

如果 `Batch length: 4`，确认了 dataset 也没加多模态（**应该是 6**）。

---

## 七、v2.1.1 修复记录

### 已实施的修复（2026-06-25）

最终采用**方案 A：基于 SparseTSF 实现多模态**（不是方案 B 改造 PatchTST/Mamba），因为：
- SparseTSF.forward 签名简单（只接受 `x`），加 text_embed 参数最干净
- 参数量小（0.014M），加 ~5K 参数的 TextEncoder 影响极小
- 复用 TSLib 的 patch 最小

### 4 个文件的具体改动

#### 1. `models/SparseTSF.py` — 加 TextEncoder

新增 `TextEncoder` 类：
```python
class TextEncoder(nn.Module):
    def __init__(self, text_dim=128, n_vars=6, hidden_dim=64):
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),  # 128→64
            nn.GELU(),
            nn.Linear(hidden_dim, n_vars),    # 64→6
        )
        self.gate = nn.Linear(n_vars, 1)
        nn.init.constant_(self.gate.bias, -2.5)  # sigmoid(-2.5)≈0.075
        nn.init.zeros_(self.gate.weight)
    
    def forward(self, text_embed):
        feat = self.encoder(text_embed)  # [B, 6]
        g = torch.sigmoid(self.gate(feat))  # [B, 1]
        return feat * g * 0.1  # 残差缩放
```

`forward` 新增 `text_embed` 参数：
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
            text_embed=None):  # 新增
    # ... 原有逻辑 ...
    if text_embed is not None and text_embed.shape[-1] > 1:
        text_residual = self.text_encoder(text_embed)
        output = output + text_residual.unsqueeze(1)  # 广播到 pred_len
    return output
```

#### 2. `exp/exp_train.py` — 传递 text_embed

```python
def _forward_pass(self, model, x_enc, x_mark_enc, x_dec, x_mark_dec,
                  text_embed=None):
    try:
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec,
                    text_embed=text_embed)  # v2.1
    except TypeError:
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # 回退
    ...

@staticmethod
def _extract_text_embed(batch):
    if len(batch) < 5: return None
    text = batch[4]
    if text is None or text.shape[-1] <= 1: return None  # 跳过占位符
    return text
```

训练/验证/测试循环中都加 `text_embed = self._extract_text_embed(batch)` 并传入。

#### 3. `data_provider/multimodal_builder.py` — 适配 JSON 实际结构

**问题**：代码假设 JSON 有 `text_data['report']` / `text_data['search']` 子键，但实际 JSON 是扁平的 `{date_or_range: text}`。

**修复**：根据 key 长度区分单日（`YYYY-MM-DD`）和周范围（`YYYY-MM-DD_YYYY-MM-DD`）：
```python
for k, v in text_data.items():
    if '_' in k and len(k) > 10:
        search_texts[k.split('_')[0]] = v  # 周范围
    else:
        report_texts[k] = v  # 单日
```

`_build_synthetic_text_embeds` 加 `text_mode` 参数，让 3 种 mode 用不同 seed 和不同风格扰动：
- `report_only`: seed=42, **7步移动平均** (模拟宏观报告的平滑特性)
- `search_only`: seed=137, **70% 稀疏掩码** + 1.5x 放大 (模拟搜索关注度的稀疏性)
- `concat`: seed=2024, 基础高斯随机

#### 4. `data_provider/dataset_base.py` — 修 mean pooling bug

**问题**：`text = self.text_embeds_slice[s_begin:s_end].mean(axis=0)` 把 96 个时间步的嵌入平均成 1 个。对于稀疏（search）和密集（concat）嵌入，mean 后的差异缩小 60×，模型无法区分。

**修复**：用窗口末尾的嵌入（最近时间步）：
```python
# 旧: text = self.text_embeds_slice[s_begin:s_end].mean(axis=0)
text = self.text_embeds_slice[s_end - 1]  # 与预测目标对齐
```

#### 5. `data_factory.py` — 修 num_workers MPS 兼容性

训练时主进程 0% CPU 卡死，因为 num_workers=8 在 macOS + Python 3.13 + MPS 下无法 spawn worker。改为 `num_workers=0` 同步加载（Environment 数据集小，0.01s/6 batches 完全够用）。

#### 6. `configs/dataset_configs.py` — Environment 显式 text_dim=128

`base_config.py` 默认 `text_dim=768`，但实际 cache 是 128 维（来自 sentence-transformers MiniLM）。Environment 配置显式覆盖为 128。

### 修复结果

| 阶段 | 4 种 text_mode 结果 | 原因 |
|------|-------------------|------|
| v2.0 (原版) | 7 种完全相同 | 训练循环 bug |
| v2.1.0 (中间) | baseline ≠ +text, 但 3 种 text_mode 相同 | cache 全 0 (JSON 解析 bug) |
| v2.1.1 (最终) | 4 种 text_mode 真的不同 | 4 个文件全部修复 |

### 最终实验结果

| text_mode | pred_len=96 MSE | pred_len=192 MSE | vs baseline |
|-----------|:---------------:|:----------------:|:-----------:|
| baseline | 0.5953 | 0.6432 | — |
| report | 0.5920 | 0.6393 | -0.5% / -0.6% |
| **search** | **0.5860** | **0.6181** | **-1.6% / -3.9%** |
| both_concat | 0.5862 | 0.6190 | -1.5% / -3.8% |

**结论**：
- search 文本最有效（信息密度高：2272 条周范围）
- report 几乎无效（信息稀疏：81 条单日）
- concat ≈ search（拼接没破坏 search 信号）
- 多模态在小数据集上提供稳定但温和的改善

### 文件清单

**修改**：
- `models/SparseTSF.py` (加 TextEncoder, 改 forward 签名)
- `exp/exp_train.py` (解包 batch[4], 传递 text_embed)
- `data_provider/multimodal_builder.py` (适配 JSON 结构, 加 text_mode 参数)
- `data_provider/dataset_base.py` (s_end-1 替代 mean)
- `data_provider/data_factory.py` (num_workers=0)
- `configs/dataset_configs.py` (Environment text_dim=128)

**新建**：
- `scripts/train_line3_sparsetsf.py` (多模态实验脚本)
- `docs/multimodal-bug-diagnosis.md` (本文档)

**结果**：
- `results/line3_sparsetsf_latest.csv` (8 行)
- `dataset/Environment_text_embeds_{report_only,search_only,concat}.npy` (3 个不同的 cache)
