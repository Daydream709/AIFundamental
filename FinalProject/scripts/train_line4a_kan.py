"""
Train Line 4a: KAN-iTransformer Ablation (High-Performance)
============================================================

Disables each of the 5 KAN-iTransformer modules in turn to measure
its individual contribution. The baseline (A0) is the full model.

  Model:    KANiTransformer (v2.0: B-spline KAN + 5 modules)
  Datasets: ETTm2, Electricity, Environment (low/mid/high dim)
  Settings (5): A0 baseline + A1-A4 = remove one module each
    A0 - 完整 (full: KAN + CFD + 预训练 + 概率输出 + RevIN + 模型仲裁)
    A1 - w/o CFD (cascade frequency decomp)        --use_cfd=False
    A2 - w/o 预训练 (masked reconstruction)         --use_masked_pretrain=False
    A3 - w/o 概率输出 (use MSE instead of GaussianNLL) --use_probabilistic=False
    A4 - w/o RevIN (use simple instance norm)       --use_revin=False

  注: v2.0 KAN 模型本身不带"关掉 KAN 层"或"关掉模型仲裁"的开关 (那是 v2.1 加的),
      所以 A5(关掉 KAN) 和 A6(关掉仲裁) 在 v2.0 框架下需修改模型代码, 此处省略。
  Total:     5 × 3 × 1 = 15 runs (pred_len=96, by default)

Output files:
  - results/ablation_kan_{ts}.csv  (per-run, with ablation+setting columns)
  - results/ablation_kan.csv       (APPENDED, canonical for viz)

Usage:
  python scripts/train_line4a_kan.py
  python scripts/train_line4a_kan.py --pred_len 192
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse

from _common import (
    setup_path,
    run_experiment,
    save_ablation_results,
    print_summary,
)

setup_path()


# Configuration (matches viz-frontend/src/data/lines.ts LINES[4])
MODEL = "KANiTransformer"
DATASETS = ["ETTm2", "Electricity", "Environment"]
SEQ_LEN = 96
DEFAULT_PRED_LEN = 96  # ablation 默认只跑一个 pred_len (耗时控制)

# 5 个消融设置 — v2.0 KAN-iTransformer 的 5 大模块
# 每个对应关闭一个核心模块的 config flag
KAN_ABLATION_SETTINGS = [
    # (setting_label, extra_config_to_disable_module)
    ("A0 - 完整", {}),  # baseline
    ("A1 - w/o CFD", {"use_cfd": False}),
    ("A2 - w/o 预训练", {"use_masked_pretrain": False}),
    ("A3 - w/o 概率输出", {"use_probabilistic": False}),
    ("A4 - w/o RevIN", {"use_revin": False}),
]
KAN_GROUP_NAME = "KAN 5 Modules"  # 同步到 viz-frontend/src/data/lines.ts KAN_ABLATION_GROUPS


def main():
    parser = argparse.ArgumentParser(description="Train Line 4a: KAN Ablation (v2.0)")
    parser.add_argument("--epochs", type=int, default=30, help="消融训练轮数 (默认 30)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--pred_len",
        type=int,
        default=DEFAULT_PRED_LEN,
        help="预测长度 (默认 96)。要跑全部 4 个长度请改 192/336/720 多次跑",
    )
    args = parser.parse_args()

    total = len(KAN_ABLATION_SETTINGS) * len(DATASETS)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 4a: KAN 消融（高性能，v2.0）" + " " * 30 + "║")
    print(f"║  {len(KAN_ABLATION_SETTINGS)} settings × {len(DATASETS)} datasets × pred_len={args.pred_len} = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for setting_label, setting_config in KAN_ABLATION_SETTINGS:
        for dataset in DATASETS:
            current += 1
            print(f"\n[{current}/{total}]")
            extra = {"label": setting_label, **setting_config}
            r = run_experiment(MODEL, dataset, SEQ_LEN, args.pred_len,
                               extra_config=extra, epochs=args.epochs, gpu=args.gpu)
            r["ablation"] = KAN_GROUP_NAME
            r["setting"] = setting_label
            results.append(r)

    df = pd.DataFrame(results)
    save_ablation_results(df, prefix="kan")
    print_summary(results, "Line 4a (KAN Ablation v2.0)")


if __name__ == "__main__":
    main()
