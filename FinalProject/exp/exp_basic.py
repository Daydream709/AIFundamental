"""
实验基类 — 模型构建、优化器、损失函数
FinalProject v2.0: 支持预设配置自动应用
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
    """所有实验的基类 — v2.0"""

    def __init__(self, config):
        self.config = config
        self._apply_model_preset()      # ★ 自动应用预设配置
        fix_seed(config.seed)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _apply_model_preset(self):
        """
        从 model_configs.py 中加载模型在当前数据集上的预设配置。

        优先级: CLI显式参数 > 模型预设 > 数据集默认 > BaseConfig默认

        实现: 只覆盖仍等于 BaseConfig 默认值的参数。
        这意味着:
          - 用户在 CLI 中指定的参数 (--lr 5e-4) 不会被覆盖 ✓
          - 数据集配置中的硬约束 (如 Electricity batch_size=16) 会被保留 ✓
          - 架构参数 (d_model, d_ff, e_layers 等) 会被模型预设覆盖 ✓
        """
        model_name = getattr(self.config, 'model', 'DLinear')
        dataset_name = getattr(self.config, 'data', 'ETTm2')

        try:
            from configs.model_configs import get_model_config
            preset = get_model_config(model_name, dataset_name)

            from configs.base_config import BaseConfig
            default_config = BaseConfig()

            applied = []
            skipped = []
            for key, value in preset.items():
                current = getattr(self.config, key, None)
                default = getattr(default_config, key, None)

                # 只覆盖仍为默认值的参数
                if current == default or current is None:
                    setattr(self.config, key, value)
                    applied.append(key)
                else:
                    skipped.append(f'{key}={current}(kept)')

            if applied:
                print(f'  [Preset] {model_name} on {dataset_name}:')
                for key in applied:
                    val = getattr(self.config, key)
                    if not isinstance(val, (list, dict, bool)):
                        print(f'    {key}={val}')

        except (ImportError, ValueError) as e:
            print(f'  [Preset] Skipped: {e}')

    def _acquire_device(self):
        # Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
        if self.config.use_gpu:
            if torch.cuda.is_available():
                return torch.device(f'cuda:{self.config.gpu}')
            if torch.backends.mps.is_available():
                print("  [Device] MPS (Apple Silicon GPU)")
                return torch.device('mps')
        print("  [Device] CPU")
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
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        if _THUML_ROOT not in sys.path:
            sys.path.insert(1, _THUML_ROOT)

        # thuml 官方基线模型 (v2.0: 4个)
        thuml_models = {
            'DLinear': 'third_party.TimeSeriesLibrary.models.DLinear',
            'PatchTST': 'third_party.TimeSeriesLibrary.models.PatchTST',
            'TimesNet': 'third_party.TimeSeriesLibrary.models.TimesNet',
            'Mamba': 'third_party.TimeSeriesLibrary.models.MambaSimple',
        }
        # 自研创新模型 (v2.0: 3个)
        custom_models = {
            'SparseTSF': 'models.SparseTSF',
            'KANiTransformer': 'models.kan_iTransformer',
            'LiteSparseNet': 'models.LiteSparseNet',
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
