"""
主入口 — FinalProject v2.0 统一CLI接口

7个模型 × 4个数据集
三条实验主线:
  主线一: 全架构对比 (DLinear/PatchTST/TimesNet/Mamba × ETTm2/Weather/Electricity)
  主线二: 自研模型深度评测 (KAN-iTransformer/Lite-SparseNet/SparseTSF/DLinear × 全部4数据集)
  主线三: 多模态有效性消融 (Environment × PatchTST/KAN-iTransformer/Lite-SparseNet × 5组文本设置)

参数优先级: CLI显式指定 > 模型预设(model_configs.py) > 数据集默认 > BaseConfig默认

用法:
    # 使用预设配置
    python run.py --model TimesNet --data ETTm2 --pred_len 96

    # 覆盖架构参数
    python run.py --model PatchTST --data Weather --d_model 256 --e_layers 2

    # 覆盖训练参数
    python run.py --model DLinear --data Electricity --lr 5e-4 --batch_size 32

    # 多模态
    python run.py --model KANiTransformer --data Environment --use_text --text_mode gating
"""
import argparse
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'third_party', 'TimeSeriesLibrary'))

from configs.dataset_configs import get_dataset_config
from exp.exp_train import ExpTrain
from utils.tools import fix_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='KAN-iTransformer v2.0 — Time Series Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --model DLinear --data ETTm2
  python run.py --model KANiTransformer --data Electricity --d_model 512 --e_layers 3
  python run.py --model LiteSparseNet --data Environment --use_text --text_mode gating
  python run.py --model TimesNet --data Weather --d_model 64 --d_ff 64
        """,
    )

    # ==================== 模型和数据 ====================
    parser.add_argument('--model', type=str, default='DLinear',
                       choices=['DLinear', 'PatchTST', 'TimesNet', 'Mamba',
                                'SparseTSF', 'KANiTransformer', 'LiteSparseNet'],
                       help='Model to train (default: DLinear)')
    parser.add_argument('--data', type=str, default='ETTm2',
                       choices=['ETTm2', 'Weather', 'Electricity', 'Environment'],
                       help='Dataset (default: ETTm2)')

    # ==================== 预测参数 ====================
    parser.add_argument('--seq_len', type=int, default=None,
                       help='Input sequence length H (default: 96)')
    parser.add_argument('--pred_len', type=int, default=None,
                       help='Prediction length F (default: 96)')

    # ==================== 通用架构参数 ====================
    arch_group = parser.add_argument_group('Architecture (override model presets)')
    arch_group.add_argument('--d_model', type=int, default=None,
                           help='Model dimension')
    arch_group.add_argument('--n_heads', type=int, default=None,
                           help='Number of attention heads')
    arch_group.add_argument('--e_layers', type=int, default=None,
                           help='Number of encoder layers')
    arch_group.add_argument('--d_ff', type=int, default=None,
                           help='FFN hidden dimension')
    arch_group.add_argument('--dropout', type=float, default=None,
                           help='Dropout rate')

    # ==================== 模型专用参数 ====================
    model_group = parser.add_argument_group('Model-specific parameters')

    # PatchTST
    model_group.add_argument('--patch_len', type=int, default=None,
                            help='PatchTST: patch length (default: 16)')
    model_group.add_argument('--stride', type=int, default=None,
                            help='PatchTST: patch stride (default: 8)')

    # TimesNet
    model_group.add_argument('--top_k', type=int, default=None,
                            help='TimesNet/KAN: top-k frequencies (default: 5)')
    model_group.add_argument('--num_kernels', type=int, default=None,
                            help='TimesNet: inception kernels (default: 6)')

    # Mamba
    model_group.add_argument('--expand', type=int, default=None,
                            help='Mamba: expansion factor (default: 2)')
    model_group.add_argument('--d_conv', type=int, default=None,
                            help='Mamba: conv kernel size (default: 4)')
    model_group.add_argument('--d_state', type=int, default=None,
                            help='Mamba: state dimension (default: 16)')

    # SparseTSF
    model_group.add_argument('--period_len', type=int, default=None,
                            help='SparseTSF: downsampling period (default: 24)')

    # KAN-iTransformer
    model_group.add_argument('--kan_grid_size', type=int, default=None,
                            help='KANiTransformer: B-spline grid size (default: 5)')

    # Lite-SparseNet
    model_group.add_argument('--sparse_ratio', type=int, default=None,
                            help='LiteSparseNet: downsampling factor p (default: 4)')
    model_group.add_argument('--group_size', type=int, default=None,
                            help='LiteSparseNet: group MLP size (default: 16)')
    model_group.add_argument('--fft_residual_k', type=int, default=None,
                            help='LiteSparseNet: FFT residual top-k (default: 2)')

    # ==================== 训练参数 ====================
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=None,
                            help='Training epochs (default: from preset)')
    train_group.add_argument('--batch_size', type=int, default=None,
                            help='Batch size (default: from preset)')
    train_group.add_argument('--lr', type=float, default=None,
                            help='Learning rate (default: from preset)')
    train_group.add_argument('--weight_decay', type=float, default=None,
                            help='Weight decay (default: 1e-5)')
    train_group.add_argument('--patience', type=int, default=None,
                            help='Early stopping patience (default: from preset)')
    train_group.add_argument('--loss', type=str, default=None,
                            choices=['MSE', 'MAE', 'GaussianNLL'],
                            help='Loss function (default: from preset)')
    train_group.add_argument('--seed', type=int, default=None,
                            help='Random seed (default: 42)')

    # ==================== 功能开关 ====================
    feature_group = parser.add_argument_group('Feature toggles')
    feature_group.add_argument('--use_probabilistic', action='store_true', default=None,
                              help='KANiTransformer: enable GaussianNLL probabilistic output')
    feature_group.add_argument('--no_probabilistic', action='store_false', dest='use_probabilistic',
                              help='Disable probabilistic output')
    feature_group.add_argument('--use_cfd', action='store_true', default=None,
                              help='KANiTransformer: enable cascaded freq decomp')
    feature_group.add_argument('--no_cfd', action='store_false', dest='use_cfd',
                              help='Disable CFD')
    feature_group.add_argument('--use_revin', action='store_true', default=None,
                              help='KANiTransformer: enable RevIN normalization')
    feature_group.add_argument('--no_revin', action='store_false', dest='use_revin',
                              help='Disable RevIN')

    # ==================== 多模态 ====================
    mm_group = parser.add_argument_group('Multimodal (Environment dataset)')
    mm_group.add_argument('--use_text', action='store_true', default=None,
                         help='Enable text modality')
    mm_group.add_argument('--no_text', action='store_false', dest='use_text',
                         help='Disable text modality')
    mm_group.add_argument('--text_mode', type=str, default=None,
                         choices=['concat', 'gating', 'report_only', 'search_only'],
                         help='Text fusion mode (default: concat)')

    # ==================== 设备 ====================
    dev_group = parser.add_argument_group('Device')
    dev_group.add_argument('--gpu', type=int, default=0,
                          help='GPU device ID (default: 0)')
    dev_group.add_argument('--use_amp', action='store_true', default=True,
                          help='Use AMP mixed precision (default)')
    dev_group.add_argument('--no_amp', action='store_false', dest='use_amp',
                          help='Disable AMP')

    return parser.parse_args()


def _apply_cli_overrides(config, args):
    """
    将 CLI 指定的参数覆盖到 config 上。

    只覆盖用户显式指定了的值（args 中不为 None 的参数）。
    这样保留模型预设对未指定参数的控制。
    """
    # 映射: CLI参数名 → config属性名 (通常相同)
    cli_params = [
        # 预测
        'seq_len', 'pred_len',
        # 架构
        'd_model', 'n_heads', 'e_layers', 'd_ff', 'dropout',
        # 模型专用
        'patch_len', 'stride', 'top_k', 'num_kernels',
        'expand', 'd_conv', 'd_state',
        'period_len', 'kan_grid_size',
        'sparse_ratio', 'group_size', 'fft_residual_k',
        # 训练
        'train_epochs', 'batch_size', 'learning_rate', 'weight_decay',
        'patience', 'loss', 'seed',
    ]

    overridden = []
    for param in cli_params:
        value = getattr(args, param, None)
        if value is not None:
            # CLI参数名到config属性名的映射
            config_attr = param
            # 特殊映射
            if param == 'train_epochs':
                config_attr = 'train_epochs'
            elif param == 'learning_rate':
                config_attr = 'learning_rate'

            setattr(config, config_attr, value)
            overridden.append(f'{param}={value}')

    # 布尔开关 (只有非 None 时才覆盖)
    bool_params = [
        'use_probabilistic', 'use_cfd', 'use_revin',
        'use_text',
    ]
    for param in bool_params:
        value = getattr(args, param, None)
        if value is not None:
            setattr(config, param, value)
            overridden.append(f'{param}={value}')

    # 字符串参数
    if args.text_mode is not None:
        config.text_fusion_mode = args.text_mode
        overridden.append(f'text_mode={args.text_mode}')

    if overridden:
        print(f'  [CLI] Overrides: {", ".join(overridden)}')


def main():
    args = parse_args()

    # 默认 seq_len/pred_len
    seq_len = args.seq_len if args.seq_len is not None else 96
    pred_len = args.pred_len if args.pred_len is not None else 96

    print("=" * 60)
    print(f"  KAN-iTransformer v2.0 | {args.model} | {args.data}")
    print(f"  H={seq_len} F={pred_len}")
    print("=" * 60)

    # 1. 获取数据集基础配置
    config = get_dataset_config(args.data, seq_len=seq_len, pred_len=pred_len)
    config.model = args.model
    config.gpu = args.gpu
    config.use_amp = args.use_amp
    config.checkpoints = './checkpoints/'

    # 2. 应用 CLI 覆盖（在模型预设之前，因为预设会在 ExpBasic.__init__ 中执行）
    #    这里先设置好 CLI 值，ExpBasic._apply_model_preset 检测到非默认值就不会覆盖
    _apply_cli_overrides(config, args)

    # 3. 多模态默认行为: Environment 数据集自动启用文本
    if args.use_text is None and args.data == 'Environment':
        config.use_text = True
        if args.text_mode is None:
            config.text_fusion_mode = 'concat'

    fix_seed(config.seed if args.seed is not None else 42)
    os.makedirs(config.checkpoints, exist_ok=True)

    # 4. 训练（ExpBasic.__init__ 中会自动应用模型预设，不会覆盖 CLI 指定的值）
    exp = ExpTrain(config)
    results = exp.train()
    exp.save_results()

    if isinstance(results.get('mse'), float):
        print(f"\n  Results: MSE={results['mse']:.6f}, MAE={results['mae']:.6f}")
    else:
        print(f"\n  Results: MSE={results.get('mse', 'N/A')}")
    print("Done!")

    return results


if __name__ == '__main__':
    main()
