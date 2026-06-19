"""
Train Line 2: Self-Developed Model Deep Evaluation
==================================================

Compares 3 self-developed models + 4 reference baselines on 3
pure time-series datasets at 4 prediction lengths.

  Self-dev:  KANiTransformer (high-perf), LiteSparseNet (lightweight),
             SparseTSF (external lightweight benchmark)
  Baseline:  DLinear, PatchTST, TimesNet, Mamba (4 thuml architectures)
  Datasets:  ETTm2, Weather, Electricity
  Pred lens: 96, 192, 336, 720
  Total:     7 × 3 × 4 = 84 runs

Output files:
  - results/line2_{ts}.csv   (per-run)
  - results/main_results.csv (APPENDED)

Usage:
  python scripts/train_line2.py
  python scripts/train_line2.py --epochs 50
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse

from _common import (
    setup_path,
    run_experiment,
    save_line_results,
    save_efficiency,
    efficiency_rows_from,
    append_to_partial,
    print_summary,
)

setup_path()


# Configuration (matches viz-frontend/src/data/lines.ts LINES[2])
MODELS = [
    # 自研
    "KANiTransformer",
    "LiteSparseNet",
    "SparseTSF",
    # 参考 (thuml 官方基线)
    "DLinear",
    "PatchTST",
    "TimesNet",
    "Mamba",
]
DATASETS = ["ETTm2", "Weather", "Electricity"]
PRED_LENS = [96, 192, 336, 720]
SEQ_LEN = 96


def main():
    parser = argparse.ArgumentParser(description="Train Line 2: Self-Dev Model Evaluation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    total = len(MODELS) * len(DATASETS) * len(PRED_LENS)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 2: 自研模型深度评测 (Self-Dev Model Evaluation)" + " " * 14 + "║")
    print(f"║  {len(MODELS)} models × {len(DATASETS)} datasets × {len(PRED_LENS)} pred_lens = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for model in MODELS:
        for dataset in DATASETS:
            for pred_len in PRED_LENS:
                current += 1
                print(f"\n[{current}/{total}]")
                # Environment 上 KANiTransformer 开启概率输出
                # (here we don't include Environment so no special case)
                r = run_experiment(model, dataset, SEQ_LEN, pred_len,
                                   epochs=args.epochs, gpu=args.gpu,
                                   on_complete=lambda res: append_to_partial(
                                       {**res, "line": 2}, line=2
                                   ))
                r["line"] = 2
                results.append(r)

    df = pd.DataFrame(results)
    save_line_results(df, line_number=2)
    save_efficiency(efficiency_rows_from(df), line=2)
    print_summary(results, "Line 2")


if __name__ == "__main__":
    main()
