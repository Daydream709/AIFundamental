"""
模型注册表
- thuml 官方模型通过完整路径导入
- 自定义创新模型从项目 models/ 导入
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THUML_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary')

# 确保路径: 项目根 > thuml库 (thuml的utils/masking等需要被找到)
for p in [_PROJECT_ROOT, _THUML_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# thuml 官方实现
from third_party.TimeSeriesLibrary.models.DLinear import Model as DLinear
from third_party.TimeSeriesLibrary.models.PatchTST import Model as PatchTST
from third_party.TimeSeriesLibrary.models.iTransformer import Model as iTransformer
from third_party.TimeSeriesLibrary.models.TimeMixer import Model as TimeMixer
try:
    from third_party.TimeSeriesLibrary.models.Chronos2 import Model as Chronos2
except ImportError:
    Chronos2 = None

# 自定义创新模型
from models.TimeKAN import Model as TimeKAN
from models.kan_iTransformer import Model as KANiTransformer
from models.mamba_transformer_dual import Model as MambaTransformerDual
from models.multimodal_fusion import Model as MultimodalFusion

MODEL_REGISTRY = {
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'TimeMixer': TimeMixer,
    'Chronos2': Chronos2,
    'TimeKAN': TimeKAN,
    'KANiTransformer': KANiTransformer,
    'MambaTransformerDual': MambaTransformerDual,
    'MultimodalFusion': MultimodalFusion,
}
