"""
Train Line 1: Cross-Architecture Comparison
=========================================

Compares 4 base architectures (MLP / Transformer / CNN / SSM) on
3 pure time-series datasets at 4 prediction lengths.

  Models:    DLinear (MLP), PatchTST (Transformer), TimesNet (CNN), Mamba (SSM)
  Datasets:  ETTm2, Weather, Electricity
  Pred lens: 96, 192, 336, 720
  Total:     4 × 3 × 4 = 48 runs

Output files:
  - results/line1_{ts}.csv                       (per-run, lowercase columns)
  - results/main_results.csv                     (APPENDED, canonical columns)
  - results/efficiency/flops_params_summary.csv (UPDATED, dedup)

Usage:
  python scripts/train_line1.py
  python scripts/train_line1.py --epochs 50    # override epochs
"""
import os
import sys

# Ensure project root on path before importing _common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from _common import (
    setup_path,
    run_experiment,
    save_line_results,
    save_efficiency,
    efficiency_rows_from,
    append_to_partial,
    print_summary,
    detect_compute,
    print_compute_banner,
)

setup_path()

import argparse

# ---------------------------------------------------------------------------
# Configuration (matches viz-frontend/src/data/lines.ts LINES[1])
# ---------------------------------------------------------------------------
MODELS = ["DLinear", "PatchTST", "TimesNet", "Mamba"]
DATASETS = ["ETTm2", "Weather", "Electricity"]
PRED_LENS = [96, 192, 336, 720]
SEQ_LEN = 96


def main():
    compute = detect_compute()
    print_compute_banner(compute)

    parser = argparse.ArgumentParser(description="Train Line 1: Cross-Architecture Comparison")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数 (默认 100; 在所有设备上保持一致, 仅 AMP 开关不同)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (默认 0)")
    args = parser.parse_args()

    total = len(MODELS) * len(DATASETS) * len(PRED_LENS)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 1: 跨架构对比 (Cross-Architecture Comparison)" + " " * 18 + "║")
    print(f"║  {len(MODELS)} models × {len(DATASETS)} datasets × {len(PRED_LENS)} pred_lens = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for model in MODELS:
        for dataset in DATASETS:
            for pred_len in PRED_LENS:
                current += 1
                print(f"\n[{current}/{total}]")
                r = run_experiment(model, dataset, SEQ_LEN, pred_len,
                                   epochs=args.epochs, gpu=args.gpu,
                                   on_complete=lambda res: append_to_partial(
                                       {**res, "line": 1}, line=1
                                   ),
                                   compute=compute)
                r["line"] = 1
                results.append(r)

    df = pd.DataFrame(results)
    save_line_results(df, line_number=1)
    save_efficiency(efficiency_rows_from(df), line=1)
    print_summary(results, "Line 1")


if __name__ == "__main__":
    main()
