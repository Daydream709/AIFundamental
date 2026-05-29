"""
主入口 — 统一CLI接口
用法:
    python run.py --model DLinear --data ETTm2 --seq_len 96 --pred_len 96
    python run.py --model iTransformer --data Weather --pred_len 192
    python run.py --model MultimodalFusion --data Energy --seq_len 24 --pred_len 24
"""
import argparse
import os
import sys

# 确保项目根目录在 path 中（必须优先于 thuml 库）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# thuml库依赖路径（放在第二位，避免覆盖项目的 exp/, layers/, models/, utils/）
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'TimeSeriesLibrary'))

from configs.dataset_configs import get_dataset_config
from exp.exp_train import ExpTrain
from utils.tools import fix_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # 模型和数据
    parser.add_argument('--model', type=str, default='DLinear',
                       choices=['DLinear', 'PatchTST', 'iTransformer', 'TimeMixer',
                                'TimeKAN', 'Chronos2', 'KANiTransformer',
                                'MambaTransformerDual', 'MultimodalFusion'])
    parser.add_argument('--data', type=str, default='ETTm2',
                       choices=['ETTm2', 'Weather', 'Electricity',
                                'Energy', 'Environment', 'Health'])

    # 预测参数
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'MAE', 'GaussianNLL'])
    parser.add_argument('--seed', type=int, default=42)

    # 设备
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Model: {args.model} | Dataset: {args.data} | H={args.seq_len} F={args.pred_len}")
    print("=" * 60)

    # 获取配置
    config = get_dataset_config(args.data, seq_len=args.seq_len, pred_len=args.pred_len)
    config.model = args.model
    config.train_epochs = args.epochs
    config.learning_rate = args.lr
    config.loss = args.loss
    config.seed = args.seed
    config.gpu = args.gpu
    config.use_amp = args.use_amp
    config.checkpoints = './checkpoints/'

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # 多模态配置
    if args.model == 'MultimodalFusion':
        config.use_text = True
        config.use_image = True
        config.use_contrastive = True

    fix_seed(config.seed)
    os.makedirs(config.checkpoints, exist_ok=True)

    # 训练和测试
    if args.model == 'Chronos2':
        from exp.exp_zero_shot import ExpZeroShot
        exp = ExpZeroShot(config)
        results = exp.test()
    else:
        exp = ExpTrain(config)
        results = exp.train()
        exp.save_results()

    print("\nDone!")
    return results


if __name__ == '__main__':
    main()
