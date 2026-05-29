"""
工具函数 — 随机种子、早停、学习率调度
"""
import numpy as np
import torch
import random
import os


def fix_seed(seed: int = 42):
    """固定所有随机种子，确保完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """早停机制 — 验证损失连续 patience 轮不下降则停止"""

    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0.0,
                 save_path: str = 'checkpoints/'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def __call__(self, val_loss, model, model_name: str = 'model'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name: str):
        """验证损失下降时保存模型"""
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model...')
        path = os.path.join(self.save_path, f'{model_name}_checkpoint.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
