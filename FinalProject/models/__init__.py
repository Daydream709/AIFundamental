"""
模型注册表 — 按照 FinalProject v2.0 计划
7个模型，覆盖5大架构: MLP / Transformer / CNN / SSM / 轻量级

- DLinear (MLP)         — thuml 官方实现
- PatchTST (Transformer) — thuml 官方实现
- TimesNet (CNN)         — thuml 官方实现
- Mamba (SSM)            — thuml 官方实现
- SparseTSF (轻量级)     — 自研实现（thuml无此模型）
- KANiTransformer (自研) — KAN + 倒置Transformer
- LiteSparseNet (自研)   — 稀疏采样 + 分组轻量MLP
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THUML_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary')

# 确保路径: 项目根 > thuml库。TimeSeriesLibrary 也有 layers/ 包，
# 若它排在项目根之前，会遮蔽本项目的 layers.kan_layers 等自研模块。
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
if _THUML_ROOT not in sys.path:
    sys.path.append(_THUML_ROOT)

# thuml 官方实现 — 4个基线模型
from third_party.TimeSeriesLibrary.models.DLinear import Model as DLinear
from third_party.TimeSeriesLibrary.models.PatchTST import Model as PatchTST
from third_party.TimeSeriesLibrary.models.TimesNet import Model as TimesNet
# 使用 MambaSimple (纯PyTorch实现, 无需mamba_ssm包, 接口与Mamba一致)
from third_party.TimeSeriesLibrary.models.MambaSimple import Model as Mamba

# 自研实现 — 3个创新模型
from models.SparseTSF import Model as SparseTSF
from models.kan_iTransformer import Model as KANiTransformer
from models.LiteSparseNet import Model as LiteSparseNet

MODEL_REGISTRY = {
    # 基线 (4个, thuml)
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'TimesNet': TimesNet,
    'Mamba': Mamba,
    # 自研 (3个)
    'SparseTSF': SparseTSF,
    'KANiTransformer': KANiTransformer,
    'LiteSparseNet': LiteSparseNet,
}
