"""
Shared helpers for train_line*.py scripts.

Output strategy (same across all 4 line scripts):
  - results/line{N}_{ts}.csv           — per-run CSV (lowercase columns from exp.train)
  - results/main_results.csv           — APPENDED, uppercase canonical columns
  - results/ablation_{prefix}.csv      — APPENDED (for line 4 only)
  - results/efficiency/flops_params_summary.csv — UPDATED (dedup by model+dataset)

The visualization platform (viz-frontend/) reads from these files via
scripts/sync_results.py → viz-frontend/public/data/.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Make project root importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"

# ---------------------------------------------------------------------------
# Column name normalization (canonicalize for visualization)
# ---------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Params(M)": ["Params(M)", "params_M", "params_m", "params(M)"],
    "FLOPs(G)": ["FLOPs(G)", "flops_G", "flops_g", "FLOPs(g)"],
    "InferTime(ms)": ["InferTime(ms)", "infer_time_ms"],
    "GPUMem(MB)": ["GPUMem(MB)", "gpu_mem_mb"],
    "MSE": ["MSE", "mse"],
    "MAE": ["MAE", "mae"],
    "RMSE": ["RMSE", "rmse"],
    "MAPE": ["MAPE", "mape"],
    "SMAPE": ["SMAPE", "smape"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to canonical form. Idempotent."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical and alias not in rename_map.values():
                rename_map[alias] = canonical
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)


def _append_csv(path: Path, df_new: pd.DataFrame) -> int:
    """Append df_new to path (creating if needed). Returns total rows after append."""
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new.copy()
    combined.to_csv(path, index=False)
    return len(combined)


def save_line_results(df: pd.DataFrame, line_number: int) -> Path:
    """Save line results.

    - Saves timestamped copy: results/line{N}_{ts}.csv
    - Appends to results/main_results.csv (canonical uppercase columns)
    Returns the timestamped file path.
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    if "line" not in df.columns:
        df["line"] = line_number

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = RESULTS_DIR / f"line{line_number}_{ts}.csv"
    df.to_csv(ts_path, index=False)
    print(f"  💾 Saved: {ts_path} ({len(df)} rows)")

    # Append to main_results.csv (add timestamp column for traceability)
    df_for_main = df.copy()
    if "timestamp" not in df_for_main.columns:
        df_for_main["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_path = RESULTS_DIR / "main_results.csv"
    total = _append_csv(main_path, df_for_main)
    print(f"  📊 Appended to: {main_path} (now {total} rows total)")
    return ts_path


def save_ablation_results(df: pd.DataFrame, prefix: str) -> Path:
    """Save ablation results.

    - Saves timestamped copy: results/ablation_{prefix}_{ts}.csv
    - Appends to results/ablation_{prefix}.csv (canonical file)
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = RESULTS_DIR / f"ablation_{prefix}_{ts}.csv"
    df.to_csv(ts_path, index=False)
    print(f"  💾 Saved: {ts_path} ({len(df)} rows)")

    canonical_path = RESULTS_DIR / f"ablation_{prefix}.csv"
    total = _append_csv(canonical_path, df)
    print(f"  🔬 Appended to: {canonical_path} (now {total} rows total)")
    return ts_path


def save_efficiency(df: pd.DataFrame) -> None:
    """Save efficiency data (model, dataset, Params(M), FLOPs(G)) to
    results/efficiency/flops_params_summary.csv. Deduplicates by (model, dataset).
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    needed = ["model", "dataset", "Params(M)", "FLOPs(G)"]
    df = df[[c for c in needed if c in df.columns]].drop_duplicates(
        subset=["model", "dataset"], keep="last"
    )
    path = EFFICIENCY_DIR / "flops_params_summary.csv"
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model", "dataset"], keep="last")
    else:
        combined = df
    combined.to_csv(path, index=False)
    print(f"  ⚙️  Saved efficiency: {path} ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    model: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    extra_config: Optional[Dict[str, Any]] = None,
    epochs: int = 100,
    gpu: int = 0,
) -> Dict[str, Any]:
    """Run one experiment, return result dict (lowercase metric keys from exp.train).

    On error, returns dict with status='error' and error message.
    """
    from configs.dataset_configs import get_dataset_config  # noqa: E402
    from exp.exp_train import ExpTrain  # noqa: E402
    from utils.tools import fix_seed  # noqa: E402

    label = f"{model} | {dataset} | F={pred_len}"
    if extra_config and "label" in extra_config:
        label += f" | {extra_config['label']}"
    print(f"  ▶ {label}", flush=True)

    config = get_dataset_config(dataset, seq_len=seq_len, pred_len=pred_len)
    config.model = model
    config.train_epochs = epochs
    config.gpu = gpu
    config.checkpoints = "./checkpoints/"
    os.makedirs(config.checkpoints, exist_ok=True)

    # Apply extra config flags (use_text, text_mode, use_cfd, etc.)
    if extra_config:
        for key, value in extra_config.items():
            if key == "label":
                continue
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Silently attach to config as attribute for downstream use
                try:
                    setattr(config, key, value)
                except Exception:
                    pass

    fix_seed(config.seed)
    try:
        exp = ExpTrain(config)
        result = exp.train()
        result["status"] = "success"
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        traceback.print_exc()
        result = {"status": "error", "error": str(e)}

    result.update(
        {
            "model": model,
            "dataset": dataset,
            "seq_len": seq_len,
            "pred_len": pred_len,
        }
    )
    if extra_config and "label" in extra_config:
        result["label"] = extra_config["label"]
    return result


def efficiency_rows_from(df: pd.DataFrame) -> pd.DataFrame:
    """Extract efficiency columns (model, dataset, Params(M), FLOPs(G)) for save_efficiency."""
    df = normalize_columns(df)
    needed = ["model", "dataset", "Params(M)", "FLOPs(G)"]
    return df[[c for c in needed if c in df.columns]].drop_duplicates(
        subset=["model", "dataset"]
    )


def print_summary(results: List[Dict[str, Any]], line_label: str) -> None:
    """Print a summary table of results."""
    n = len(results)
    n_ok = sum(1 for r in results if r.get("status") == "success")
    print()
    print("=" * 70)
    print(f"  {line_label}: {n_ok}/{n} succeeded")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point helper
# ---------------------------------------------------------------------------
def setup_path() -> None:
    """Add project root + thuml library to sys.path."""
    for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "third_party" / "TimeSeriesLibrary")):
        if p not in sys.path:
            sys.path.insert(0, p)
