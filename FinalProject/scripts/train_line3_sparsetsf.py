"""
Train Line 3 (SparseTSF 多模态版)
==================================

v2.1: 在原 line3 基础上，把 PatchTST/Mamba 换成已支持多模态输入的 SparseTSF。

  Model:    SparseTSF (with TextEncoder branch)
  Dataset:  Environment (with text — environment reports + search summaries)
  Pred len: 96, 192
  Text modes (4 种, 无 satellite):
    baseline     — 纯时序 (use_text=False)
    report       — + 报告文本 (text_mode=report_only)
    search       — + 搜索文本 (text_mode=search_only)
    both_concat  — 两种文本拼接

Total: 1 model × 1 dataset × 2 pred_lens × 4 text_modes = 8 runs

Output:
  - results/line3_sparsetsf_{ts}.csv     (时间戳历史)
  - results/line3_sparsetsf_latest.csv   (最新)
  - results/efficiency/line3_sparsetsf_latest.csv

Usage:
  python scripts/train_line3_sparsetsf.py
  python scripts/train_line3_sparsetsf.py --epochs 1     # smoke test
"""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse

from _common import (
    setup_path,
    run_experiment,
    detect_compute,
    print_compute_banner,
    normalize_columns,
)

setup_path()


# 与原 train_line3 一致, 但去掉了 satellite/text+satellite (不实现图像)
MODELS = ["SparseTSF"]
DATASETS = ["Environment"]
PRED_LENS = [96, 192]
SEQ_LEN = 96

TEXT_MODES = [
    ("baseline",     {"use_text": False, "use_satellite": False, "text_mode": "baseline"}),
    ("report",       {"use_text": True,  "use_satellite": False, "text_mode": "report_only"}),
    ("search",       {"use_text": True,  "use_satellite": False, "text_mode": "search_only"}),
    ("both_concat",  {"use_text": True,  "use_satellite": False, "text_mode": "concat"}),
]

LINE_TAG = "line3_sparsetsf"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"


def save_results(df: pd.DataFrame) -> None:
    """直接写两份文件, 不复用 _common.py 内部 (避免和原 line3 冲突)。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(df)
    if "line" not in df.columns:
        df["line"] = 3

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = RESULTS_DIR / f"{LINE_TAG}_{ts}.csv"
    latest_path = RESULTS_DIR / f"{LINE_TAG}_latest.csv"
    df.to_csv(ts_path, index=False)
    df.to_csv(latest_path, index=False)
    print(f"  💾 Saved: {ts_path} ({df.shape[0]} rows)")
    print(f"  📌 Latest: {latest_path} ({df.shape[0]} rows)")

    # efficiency
    if "Params(M)" in df.columns and "FLOPs(G)" in df.columns:
        eff = df[["model", "dataset", "Params(M)", "FLOPs(G)"]].drop_duplicates(
            subset=["model", "dataset"], keep="last"
        )
        eff_path = EFFICIENCY_DIR / f"{LINE_TAG}_latest.csv"
        eff.to_csv(eff_path, index=False)
        print(f"  ⚡ Efficiency: {eff_path}")


def main():
    compute = detect_compute()
    print_compute_banner(compute)

    parser = argparse.ArgumentParser(
        description="Train Line 3 (SparseTSF 多模态)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    total = len(MODELS) * len(DATASETS) * len(PRED_LENS) * len(TEXT_MODES)
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Line 3 (SparseTSF): 多模态修复版 v2.1" + " " * 25 + "║")
    print(f"║  {len(MODELS)} model × {len(TEXT_MODES)} modes × {len(PRED_LENS)} pred_lens = {total} runs".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")

    results = []
    current = 0
    for model in MODELS:
        for text_label, text_config in TEXT_MODES:
            for pred_len in PRED_LENS:
                current += 1
                print(f"\n[{current}/{total}] {model} | {text_label} | F={pred_len}")
                extra = {"label": text_label, **text_config}
                r = run_experiment(
                    model, "Environment", SEQ_LEN, pred_len,
                    extra_config=extra, epochs=args.epochs, gpu=args.gpu,
                    compute=compute,
                )
                r["line"] = 3
                r["text_mode"] = text_label
                results.append(r)

    df = pd.DataFrame(results)
    save_results(df)

    # 打印对比表
    print("\n" + "=" * 70)
    print("Line 3 (SparseTSF) 4 种 text_mode 对比:")
    print("=" * 70)
    if "mse" in df.columns:
        for pl in PRED_LENS:
            print(f"\n--- pred_len = {pl} ---")
            sub = df[df["pred_len"] == pl][["text_mode", "mse", "mae"]]
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
