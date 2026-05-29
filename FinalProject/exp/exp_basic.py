"""
实验基类 — 模型构建、优化器、损失函数
"""
import os
import sys
import importlib
import torch
import torch.nn as nn
from utils.tools import fix_seed

# 确保项目根目录优先于 thuml 库
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THUML_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'TimeSeriesLibrary')


class ExpBasic:
    """所有实验的基类"""

    def __init__(self, config):
        self.config = config
        fix_seed(config.seed)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.gpu}')
        return torch.device('cpu')

    def _build_model(self):
        """根据 config.model 构建模型"""
        model_name = getattr(self.config, 'model', 'DLinear')
        model_cls = self._get_model_class(model_name)
        model = model_cls(self.config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model: {model_name}, Params: {n_params:,}')
        return model

    def _get_model_class(self, model_name):
        """动态导入模型类 — 先确保路径正确"""
        # 确保路径顺序: 项目根 > thuml库
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        if _THUML_ROOT not in sys.path:
            sys.path.insert(1, _THUML_ROOT)

        # thuml 官方模型 — 从 thuml 库导入
        thuml_models = {
            'DLinear': 'third_party.TimeSeriesLibrary.models.DLinear',
            'PatchTST': 'third_party.TimeSeriesLibrary.models.PatchTST',
            'iTransformer': 'third_party.TimeSeriesLibrary.models.iTransformer',
            'TimeMixer': 'third_party.TimeSeriesLibrary.models.TimeMixer',
            'Chronos2': 'third_party.TimeSeriesLibrary.models.Chronos2',
        }
        # 自定义创新模型 — 从项目 models/ 导入
        custom_models = {
            'TimeKAN': 'models.TimeKAN',
            'KANiTransformer': 'models.kan_iTransformer',
            'MambaTransformerDual': 'models.mamba_transformer_dual',
            'MultimodalFusion': 'models.multimodal_fusion',
        }

        if model_name in thuml_models:
            module = importlib.import_module(thuml_models[model_name])
        elif model_name in custom_models:
            module = importlib.import_module(custom_models[model_name])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return module.Model

    def _get_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _get_criterion(self):
        if self.config.loss == 'MSE':
            return nn.MSELoss()
        elif self.config.loss == 'MAE':
            return nn.L1Loss()
        elif self.config.loss == 'GaussianNLL':
            return nn.GaussianNLLLoss()
        else:
            return nn.MSELoss()
