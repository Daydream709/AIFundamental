"""
Shared helpers for train_line*.py scripts.

Data flow (per-experiment real-time save + final per-line aggregation):
  1. Single run_experiment() finishes → append 1 row to line{N}_partial.csv
     (so a mid-run crash only loses the currently-running experiment)
  2. All experiments in the script finish → save_line_results() renames
     the partial file to line{N}_{ts}.csv (permanent history) AND writes
     line{N}_latest.csv (always-current for the viz to read).
  3. Efficiency data is written to efficiency/line{N}_{ts}.csv + line{N}_latest.csv
     (per-line, NOT accumulated into a shared file).
  4. exp_train.py's internal ResultLogger.log() is left alone (we just
     never call exp.save_results() / ResultLogger.save() — its records
     are in-memory only and don't leak to disk).

Output files in results/ (per line run, fully self-contained):
  - line{N}_{ts}.csv                (timestamped snapshot, history)
  - line{N}_latest.csv              (always current; viz reads this)
  - line{N}_partial.csv             (live-progress; removed at end)
  - efficiency/line{N}_{ts}.csv     (efficiency snapshot, history)
  - efficiency/line{N}_latest.csv   (efficiency, always current)

  - ablation_{prefix}_{ts}.csv      (timestamped snapshot, history)
  - ablation_{prefix}_latest.csv    (always current; viz reads this)
  - ablation_{prefix}_partial.csv   (live-progress; removed at end)

NOTE: There is NO cross-line aggregation. If a viz page needs data
from another line, it fetches that line's _latest.csv directly.
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
# Output writers (per-line, NO cross-line accumulation)
# ---------------------------------------------------------------------------
def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)


def _partial_path(line: Optional[int] = None, ablation_prefix: Optional[str] = None) -> Path:
    """Resolve the per-script-run live-progress CSV path."""
    if ablation_prefix:
        return RESULTS_DIR / f"ablation_{ablation_prefix}_partial.csv"
    if line is not None:
        return RESULTS_DIR / f"line{line}_partial.csv"
    raise ValueError("Must provide either line or ablation_prefix")


def append_to_partial(result: Dict[str, Any], line: Optional[int] = None,
                      ablation_prefix: Optional[str] = None) -> None:
    """Append a single result row to the live-progress CSV.

    Called immediately after each run_experiment() so a mid-run crash
    only loses the currently-running experiment, not all completed ones.
    """
    _ensure_results_dir()
    df = pd.DataFrame([result])
    df = normalize_columns(df)
    path = _partial_path(line=line, ablation_prefix=ablation_prefix)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(path, index=False)


def _rename_partial_to_timestamped(
    line: Optional[int] = None, ablation_prefix: Optional[str] = None,
) -> Optional[Path]:
    """Move the partial CSV to a timestamped filename.

    Returns the new timestamped path, or None if the partial didn't exist.
    """
    partial = _partial_path(line=line, ablation_prefix=ablation_prefix)
    if not partial.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ablation_prefix:
        new_path = RESULTS_DIR / f"ablation_{ablation_prefix}_{ts}.csv"
    else:
        new_path = RESULTS_DIR / f"line{line}_{ts}.csv"
    os.replace(partial, new_path)
    return new_path


def _write_latest(df: pd.DataFrame, line: Optional[int] = None,
                  ablation_prefix: Optional[str] = None) -> Path:
    """Write to the *_latest.csv file (overwrites any prior latest).

    This is what the visualization platform reads.
    """
    if ablation_prefix:
        path = RESULTS_DIR / f"ablation_{ablation_prefix}_latest.csv"
    else:
        path = RESULTS_DIR / f"line{line}_latest.csv"
    df.to_csv(path, index=False)
    return path


def save_line_results(df: pd.DataFrame, line_number: int) -> Path:
    """Finalize line results.

    - Renames the live-progress file (line{N}_partial.csv) to a timestamped
      snapshot (line{N}_{ts}.csv) when present.
    - Writes line{N}_latest.csv (always current, for the viz to read).
    NOTE: Does NOT accumulate into a shared main_results.csv — each line
    is fully self-contained in its own file.
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    if "line" not in df.columns:
        df["line"] = line_number

    # Prefer the live-progress partial (authoritative); fall back to a
    # synthetic timestamped write if no partial exists (e.g. all errored)
    ts_path = _rename_partial_to_timestamped(line=line_number)
    if ts_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_path = RESULTS_DIR / f"line{line_number}_{ts}.csv"
        df.to_csv(ts_path, index=False)
    print(f"  💾 Saved: {ts_path} ({df.shape[0]} rows)")

    # Always-current file (overwrites previous latest) — the viz reads this
    latest_path = _write_latest(df, line=line_number)
    print(f"  📌 Latest: {latest_path} ({df.shape[0]} rows)")
    return ts_path


def save_ablation_results(df: pd.DataFrame, prefix: str) -> Path:
    """Finalize ablation results.

    - Renames the live-progress file (ablation_{prefix}_partial.csv) to a
      timestamped snapshot (ablation_{prefix}_{ts}.csv).
    - Writes ablation_{prefix}_latest.csv (always current).
    NOTE: Does NOT accumulate into a shared ablation_{prefix}.csv.
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    ts_path = _rename_partial_to_timestamped(ablation_prefix=prefix)
    if ts_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_path = RESULTS_DIR / f"ablation_{prefix}_{ts}.csv"
        df.to_csv(ts_path, index=False)
    print(f"  💾 Saved: {ts_path} ({df.shape[0]} rows)")

    latest_path = _write_latest(df, ablation_prefix=prefix)
    print(f"  📌 Latest: {latest_path} ({df.shape[0]} rows)")
    return ts_path


def save_efficiency(df: pd.DataFrame, line: Optional[int] = None,
                    ablation_prefix: Optional[str] = None) -> None:
    """Save efficiency data per-line to results/efficiency/line{N}_*.csv
    (or ablation efficiency).

    Does NOT write to a shared `flops_params_summary.csv` — efficiency
    is per-line. To see all models' efficiency, look at the relevant
    line's efficiency file.
    """
    _ensure_results_dir()
    df = normalize_columns(df)
    needed = ["model", "dataset", "Params(M)", "FLOPs(G)"]
    df_eff = df[[c for c in needed if c in df.columns]].drop_duplicates(
        subset=["model", "dataset"], keep="last"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ablation_prefix:
        # Ablation runs don't currently emit efficiency — would need
        # model-level measurement. Skip silently.
        return
    if line is None:
        return

    # Timestamped history
    hist_path = EFFICIENCY_DIR / f"line{line}_{ts}.csv"
    df_eff.to_csv(hist_path, index=False)
    # Always-current file (overwrite)
    latest_path = EFFICIENCY_DIR / f"line{line}_latest.csv"
    df_eff.to_csv(latest_path, index=False)
    print(f"  ⚙️  Efficiency: {latest_path} ({len(df_eff)} rows)")


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
    on_complete: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one experiment, return result dict (lowercase metric keys from exp.train).

    On error, returns dict with status='error' and error message.

    If `on_complete(result)` is provided, it's called after each experiment
    (success or failure) BEFORE the function returns — used for real-time
    partial-CSV saving. The callback should not raise.
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
        mse = result.get("mse")
        mae = result.get("mae")
        if isinstance(mse, (int, float)) and isinstance(mae, (int, float)):
            print(f"    ✓ MSE={mse:.6f}, MAE={mae:.6f}")
        else:
            print(f"    ✓ success")
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

    # Real-time per-experiment save: append to partial CSV immediately
    if on_complete is not None:
        try:
            on_complete(result)
        except Exception as cb_err:
            # Callback failures must NOT break the experiment loop
            print(f"    ⚠ on_complete callback failed: {cb_err}")

    return result


def efficiency_rows_from(df: pd.DataFrame) -> pd.DataFrame:
    """Extract efficiency columns (model, dataset, Params(M), FLOPs(G))."""
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
