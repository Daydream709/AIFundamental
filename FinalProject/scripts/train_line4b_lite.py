"""
Train Line 4b: Lite-SparseNet Ablation (Lightweight)
====================================================

Disables each of the 3 Lite-SparseNet stages to measure its
contribution. The baseline (B0) is the full model.

  Model:    LiteSparseNet (v2.0: 3-stage design, per-variable weights)
  Datasets: ETTm2, Electricity, Environment (low/mid/high dim)
  Settings (3): B0 baseline + B1, B2 = vary one stage's parameter
    B0 - 完整 (baseline: sparse_ratio=4, group_size=4, fft_residual_k=2)
    B1 - 减弱阶段二 (group_size 减小 → 变量间交互弱化)
    B2 - 减弱阶段三 (fft_residual_k 减小 → FFT 细节修正弱化)

  注: v2.0 Lite 没有"完全关闭某个阶段"的开关 (那是 v2.1 Lite-RevIN/共享权重的变体),
      所以消融采用"减弱该阶段参数"而不是"完全关闭" — 这能反映各阶段的相对贡献。
  Total:     3 × 3 × 1 = 9 runs (pred_len=96, by default)

Output files:
  - results/ablation_lite_{ts}.csv  (per-run)
  - results/ablation_lite.csv       (APPENDED)

Usage:
  python scripts/train_line4b_lite.py
  python scripts/train_line4b_lite.py --pred_len 192
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
    append_to_partial,
    print_summary,
    detect_compute,
    print_compute_banner,
)

setup_path()


# Configuration (matches viz-frontend/src/data/lines.ts LINES[5])
MODEL = "LiteSparseNet"
DATASETS = ["ETTm2", "Electricity", "Environment"]
SEQ_LEN = 96
DEFAULT_PRED_LEN = 96

# 3 个消融设置 — v2.0 Lite-SparseNet 的 3 阶段设计
# v2.0 通过减小/增大关键参数来"减弱"某阶段，而非完全关闭
LITE_ABLATION_SETTINGS = [
    # (setting_label, extra_config_to_weaken_stage)
    ("B0 - 完整", {}),  # baseline
    ("B1 - 减弱阶段二 (group_size 16 → 变量交互弱化)", {"group_size": 16}),
    ("B2 - 减弱阶段三 (fft_residual_k 0 → 无 FFT 修正)", {"fft_residual_k": 0}),
]
LITE_GROUP_NAME = "Lite 3 Stages"  # 同步到 viz-frontend/src/data/lines.ts LITE_ABLATION_GROUPS


def main():
    compute = detect_compute()
    print_compute_banner(compute)

    parser = argparse.ArgumentParser(description="Train Line 4b: Lite Ablation (v2.0)")
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="训练轮数 (默认 30; 在所有设备上保持一致, 仅 AMP 开关不同)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--pred_len",
        type=int,
        default=DEFAULT_PRED_LEN,
        help="预测长度 (默认 96)",
    )
    args = parser.parse_args()

    total = len(LITE_ABLATION_SETTINGS) * len(DATASETS)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 4b: Lite 消融（轻量化，v2.0）" + " " * 29 + "║")
    print(f"║  {len(LITE_ABLATION_SETTINGS)} settings × {len(DATASETS)} datasets × pred_len={args.pred_len} = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for setting_label, setting_config in LITE_ABLATION_SETTINGS:
        for dataset in DATASETS:
            current += 1
            print(f"\n[{current}/{total}]")
            extra = {"label": setting_label, **setting_config}
            r = run_experiment(MODEL, dataset, SEQ_LEN, args.pred_len,
                               extra_config=extra, epochs=args.epochs, gpu=args.gpu,
                               on_complete=lambda res: append_to_partial(
                                   {**res, "ablation": LITE_GROUP_NAME, "setting": setting_label},
                                   ablation_prefix="lite",
                               ),
                               compute=compute)
            r["ablation"] = LITE_GROUP_NAME
            r["setting"] = setting_label
            results.append(r)

    df = pd.DataFrame(results)
    save_ablation_results(df, prefix="lite")
    print_summary(results, "Line 4b (Lite Ablation v2.0)")


if __name__ == "__main__":
    main()
