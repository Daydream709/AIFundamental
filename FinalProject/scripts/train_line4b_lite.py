"""
Train Line 4b: Lite-SparseNet Ablation (Lightweight, v2.1)
==========================================================

Ablation study on the new LinearResidual module (v2.1), which replaces
the v2.0 FFT-based frequency correction. v2.0 results showed that
removing FFT entirely (B2 in the old ablation) improved MSE by 50-67%,
indicating FFT was a net negative. v2.1 introduces a learnable residual
with a per-channel gate; this ablation measures its contribution.

  Model:    LiteSparseNet (v2.1: 3-stage + LinearResidual, per-variable weights)
  Datasets: ETTm2, Electricity, Environment (low/mid/high dim)
  Settings (3): B0 baseline + B1 narrow + B2 off
    B0 - 完整 (baseline: residual_latent_dim=4, 共享下投影 + 通道独享上投影 + 通道独享 gate)
    B1 - 窄瓶颈 (residual_latent_dim=1 → 残差模块表达力降低)
    B2 - 关闭残差 (residual_latent_dim=0 → 0 参数, 完全退化成纯 trend 预测)

  Total:     3 × 3 × 1 = 9 runs (pred_len=96, by default)

Output files:
  - results/ablation_lite_{ts}.csv  (per-run)
  - results/ablation_lite_latest.csv (always-current, for viz)

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

# 3 个消融设置 — v2.1 Lite-SparseNet 的可学习残差模块
# B0 默认 latent_dim=4; B1 缩到 1 (窄瓶颈, 表达力受限);
# B2 设 0 → LinearResidual.__init__ 直接走 enabled=False 分支, 0 参数, 0 计算
LITE_ABLATION_SETTINGS = [
    # (setting_label, extra_config_to_tweak_residual)
    ("B0 - 完整 (residual_latent_dim=4)", {}),  # baseline, default
    ("B1 - 窄瓶颈 (residual_latent_dim=1)", {"residual_latent_dim": 1}),
    ("B2 - 关闭残差 (residual_latent_dim=0)", {"residual_latent_dim": 0}),
]
LITE_GROUP_NAME = "Lite Residual"  # 同步到 viz-frontend/src/data/lines.ts LITE_ABLATION_GROUPS


def main():
    compute = detect_compute()
    print_compute_banner(compute)

    parser = argparse.ArgumentParser(description="Train Line 4b: Lite Residual Ablation (v2.1)")
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
    print("║  Line 4b: Lite 残差消融（轻量化，v2.1）" + " " * 22 + "║")
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
    print_summary(results, "Line 4b (Lite Residual Ablation v2.1)")


if __name__ == "__main__":
    main()
