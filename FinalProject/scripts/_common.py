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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

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
# Compute resource detection (CUDA vs MPS) — AMP only
# ---------------------------------------------------------------------------
# 设计原则: 不同设备上, 训练参数 (batch_size / epochs / lr 等) 必须一致,
# 唯一应该随设备变化的就是 AMP 是否开启 (MPS / CPU 没有 autocast, CUDA 用 BF16).
# 这个模块只负责检测设备和决定 AMP 开关, 不会动 batch_size 或 epochs.
def _classify_gpu_name(gpu_name: str) -> Dict[str, Any]:
    """Map a raw CUDA device name to a short stable device_id + display name."""
    name_lower = gpu_name.lower()
    if "4090" in name_lower:
        return {"device_id": "4090", "display_name": "RTX 4090"}
    if "4080" in name_lower:
        return {"device_id": "4080", "display_name": "RTX 4080"}
    if "3090" in name_lower:
        return {"device_id": "3090", "display_name": "RTX 3090"}
    if "a100" in name_lower:
        return {"device_id": "a100", "display_name": "A100"}
    if "h100" in name_lower:
        return {"device_id": "h100", "display_name": "H100"}
    return {"device_id": "cuda-unknown", "display_name": gpu_name}


def _classify_mps_name() -> Dict[str, Any]:
    """Apple Silicon classifier. Reads chip name from sysctl."""
    try:
        import subprocess
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip() or "Apple Silicon"
    except Exception:
        chip = "Apple Silicon"

    chip_lower = chip.lower()
    if "m5" in chip_lower:
        return {"device_id": "m5", "display_name": "Apple M5"}
    if "m4" in chip_lower:
        return {"device_id": "m4", "display_name": "Apple M4"}
    if "m3" in chip_lower:
        return {"device_id": "m3", "display_name": "Apple M3"}
    if "m2" in chip_lower:
        return {"device_id": "m2", "display_name": "Apple M2"}
    if "m1" in chip_lower:
        return {"device_id": "m1", "display_name": "Apple M1"}
    return {"device_id": "mps-unknown", "display_name": chip}


def detect_compute() -> dict:
    """
    Detect the available compute resource and return its AMP setting.

    The dict only exposes fields that are actually device-dependent:

      - 'backend':    'cuda' | 'mps' | 'cpu'
      - 'device_str': human-friendly description (e.g. "CUDA (RTX 4090)")
      - 'device_id':  short stable id ('4090' | 'm5' | 'cpu' | ...)
      - 'use_amp':    bool — ONLY this (and amp_dtype) is device-dependent
      - 'amp_dtype':  'bfloat16' | 'float16' | None (None = no AMP)

    Training parameters (batch_size, epochs, lr, ...) are intentionally NOT
    part of this dict — they must stay the same on every device.

    Detection priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "NVIDIA GPU"
        profile = _classify_gpu_name(gpu_name)
        bf16_supported = torch.cuda.is_bf16_supported()
        return {
            "backend": "cuda",
            "device_str": f"CUDA ({profile['display_name']})",
            "device_id": profile["device_id"],
            "use_amp": True,
            "amp_dtype": "bfloat16" if bf16_supported else "float16",
        }
    if torch.backends.mps.is_available():
        profile = _classify_mps_name()
        return {
            "backend": "mps",
            "device_str": f"MPS ({profile['display_name']})",
            "device_id": profile["device_id"],
            "use_amp": False,
            "amp_dtype": None,  # MPS has no autocast
        }
    return {
        "backend": "cpu",
        "device_str": "CPU (no GPU detected)",
        "device_id": "cpu",
        "use_amp": False,
        "amp_dtype": None,
    }


def apply_compute_to_config(config: Any, compute: dict) -> None:
    """
    Set device-specific AMP flags on the exp config. Does NOT touch
    batch_size / epochs / lr — those must stay identical on every device.
    """
    config.use_amp = bool(compute["use_amp"])
    config.amp_dtype = compute["amp_dtype"]
    config.device_id = compute["device_id"]


def apply_compute_defaults(args, compute: dict):
    """
    Best-effort safety net for argparse args (dict or Namespace). The
    exp_train path is the real consumer of compute['use_amp']; this helper
    exists so any CLI-level code that wants to read use_amp from argparse
    without going through the exp config still gets the device-correct value.
    """
    def _get(key, default=None):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    def _set(key, value):
        if isinstance(args, dict):
            args[key] = value
        else:
            setattr(args, key, value)

    if _get("use_amp") is None:
        _set("use_amp", compute["use_amp"])
    return args


def print_compute_banner(compute: dict) -> None:
    """Print a one-line banner showing the detected compute resource + AMP setting."""
    print()
    print("─" * 64)
    print(f"  💻 Compute: {compute['device_str']}  [device_id={compute['device_id']}]")
    print(f"  ⚙️  AMP: {'ON (' + compute['amp_dtype'] + ')' if compute['use_amp'] else 'OFF (FP32 only)'}")
    print("─" * 64)
    print()


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
    compute: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one experiment, return result dict (lowercase metric keys from exp.train).

    On error, returns dict with status='error' and error message.

    If `on_complete(result)` is provided, it's called after each experiment
    (success or failure) BEFORE the function returns — used for real-time
    partial-CSV saving. The callback should not raise.

    If `compute` is provided (the dict returned by detect_compute()), its
    settings (use_amp / amp_dtype / batch_size multiplier / device_id) are
    applied to the per-experiment config BEFORE the model is built. If
    compute is None, the config keeps whatever the dataset+model presets set.
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

    # Apply per-compute overrides (use_amp / amp_dtype / batch_size / device_id)
    if compute is not None:
        apply_compute_to_config(config, compute)

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
    """Add project root + thuml library to sys.path.

    ORDER MATTERS: project root MUST come before third_party/TimeSeriesLibrary.
    TSL ships its own `exp/` package (exp_basic.py, exp_long_term_forecasting.py,
    ...), and if TSL is in front of our project root, Python will pick TSL's
    `exp/` and our `from exp.exp_train import ExpTrain` will fail with
    "No module named 'exp.exp_train'".
    """
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    tsl_root = str(PROJECT_ROOT / "third_party" / "TimeSeriesLibrary")
    if tsl_root not in sys.path:
        # Append (not insert) so it goes AFTER the project root, letting our
        # `exp/` win name resolution. TSL's models are still importable when
        # explicitly requested.
        sys.path.append(tsl_root)
