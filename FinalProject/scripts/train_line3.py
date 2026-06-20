"""
Train Line 3: Multimodal Ablation
=================================

Tests whether text modality helps air-quality prediction on Environment
dataset. Compares 2 architectures (Attention vs SSM) under 5 text
fusion strategies at 2 prediction lengths.

  Models:    PatchTST (Transformer), Mamba (SSM)
  Dataset:   Environment (with text — environment reports + search summaries)
  Pred lens: 96, 192
  Text modes:
    baseline     — pure numerical (no text)
    report       — + environment report (macro context)
    search       — + search summary (real-time public attention)
    both_concat  — both, simple concatenation
    both_gating  — both, gated fusion (adaptive)
  Total:       2 × 1 × 2 × 5 = 20 runs

Output files:
  - results/line3_{ts}.csv       (per-run, with text_mode column)
  - results/main_results.csv     (APPENDED, with text_mode column)

Usage:
  python scripts/train_line3.py
  python scripts/train_line3.py --epochs 30
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
    detect_compute,
    print_compute_banner,
)

setup_path()


# Configuration (matches viz-frontend/src/data/lines.ts LINES[3])
# Models + image-flag mapping (v2.0 data dir has satellite_imgs)
MODELS = ["PatchTST", "Mamba"]
DATASETS = ["Environment"]
PRED_LENS = [96, 192]
SEQ_LEN = 96

# 7 multimodal fusion strategies (5 text + 2 image combinations)
# Modality flags map to multimodal_builder's use_text / use_satellite.
# `use_satellite` reads dataset/satellite_imgs/{date}.png
TEXT_MODES = [
    # (text_mode_label, extra_config_for_dataset)
    ("baseline",          {"use_text": False, "use_satellite": False, "text_mode": "baseline"}),
    ("report",            {"use_text": True,  "use_satellite": False, "text_mode": "report_only"}),
    ("search",            {"use_text": True,  "use_satellite": False, "text_mode": "search_only"}),
    ("both_concat",       {"use_text": True,  "use_satellite": False, "text_mode": "concat"}),
    ("both_gating",       {"use_text": True,  "use_satellite": False, "text_mode": "gating"}),
    # Image modality (卫星图, dataset/satellite_imgs/*.png)
    ("satellite",         {"use_text": False, "use_satellite": True,  "text_mode": "baseline"}),
    ("text+satellite",    {"use_text": True,  "use_satellite": True,  "text_mode": "gating"}),
]


def main():
    compute = detect_compute()
    print_compute_banner(compute)

    parser = argparse.ArgumentParser(description="Train Line 3: Multimodal Ablation")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数 (默认 100; 在所有设备上保持一致, 仅 AMP 开关不同)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    total = len(MODELS) * len(DATASETS) * len(PRED_LENS) * len(TEXT_MODES)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 3: 多模态消融 (Multimodal Ablation on Environment)" + " " * 10 + "║")
    print(f"║  {len(MODELS)} models × {len(TEXT_MODES)} modalities × {len(PRED_LENS)} pred_lens = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for model in MODELS:
        for text_label, text_config in TEXT_MODES:
            for pred_len in PRED_LENS:
                current += 1
                print(f"\n[{current}/{total}]")
                extra = {"label": text_label, **text_config}
                r = run_experiment(model, "Environment", SEQ_LEN, pred_len,
                                   extra_config=extra, epochs=args.epochs, gpu=args.gpu,
                                   on_complete=lambda res: append_to_partial(
                                       {**res, "line": 3, "text_mode": text_label},
                                       line=3,
                                   ),
                                   compute=compute)
                r["line"] = 3
                r["text_mode"] = text_label
                results.append(r)

    df = pd.DataFrame(results)
    save_line_results(df, line_number=3)
    save_efficiency(efficiency_rows_from(df), line=3)
    print_summary(results, "Line 3")


if __name__ == "__main__":
    main()
