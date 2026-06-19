"""Sync CSV results into the Vue frontend's public/data directory.

Run after any `run_experiments.py` or `run_ablation.py` to refresh the
visualization data. Handles the three column-name variants produced by
the Python pipeline so the frontend always sees a canonical schema.
"""
import shutil
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FRONTEND_DATA = PROJECT_ROOT / "viz-frontend" / "public" / "data"

# Column name normalizations: alias -> canonical
COLUMN_ALIASES = {
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


def normalize_csv(path: Path, dest: Path) -> int:
    """Read CSV, rename alias columns to canonical, write to dest.

    Returns the number of rows normalized.
    """
    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return 0

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print(f"  SKIP (empty): {path}")
        return 0

    header = lines[0].strip().split(",")
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in header and alias != canonical:
                rename_map[alias] = canonical
                break

    new_header = [rename_map.get(h, h) for h in header]
    row_count = len(lines) - 1

    with dest.open("w", encoding="utf-8") as f:
        f.write(",".join(new_header) + "\n")
        f.writelines(lines[1:])

    print(f"  COPIED ({row_count} rows): {path.name} -> {dest.name}")
    if rename_map:
        print(f"    Renamed: {rename_map}")
    return row_count


def main() -> None:
    print("=" * 60)
    print("Sync Results -> viz-frontend/public/data/")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: results/ directory not found: {RESULTS_DIR}")
        return

    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)

    # Map: source path -> destination filename
    # Canonical files (always synced)
    sources = [
        (RESULTS_DIR / "main_results.csv", FRONTEND_DATA / "main_results.csv"),
        (RESULTS_DIR / "efficiency" / "flops_params_summary.csv", FRONTEND_DATA / "efficiency_summary.csv"),
        (RESULTS_DIR / "ablation_kan.csv", FRONTEND_DATA / "ablation_kan.csv"),
        (RESULTS_DIR / "ablation_lite.csv", FRONTEND_DATA / "ablation_lite.csv"),
    ]

    total_rows = 0
    for src, dest in sources:
        total_rows += normalize_csv(src, dest)

    # Also keep timestamped copies for history
    print()
    print("Timestamped line files (line1_*, line2_*, line3_*):")
    for src in sorted(RESULTS_DIR.glob("line*.csv")):
        dest = FRONTEND_DATA / src.name
        total_rows += normalize_csv(src, dest)

    print()
    print("Timestamped ablation files (ablation_kan_*, ablation_lite_*):")
    for src in sorted(RESULTS_DIR.glob("ablation_kan_*.csv")):
        if src.name == "ablation_kan.csv":
            continue
        dest = FRONTEND_DATA / src.name
        total_rows += normalize_csv(src, dest)
    for src in sorted(RESULTS_DIR.glob("ablation_lite_*.csv")):
        if src.name == "ablation_lite.csv":
            continue
        dest = FRONTEND_DATA / src.name
        total_rows += normalize_csv(src, dest)

    # Legacy ablation_synthetic.csv (kept for backward compat with old viz)
    synth = RESULTS_DIR / "ablation_synthetic.csv"
    if synth.exists():
        total_rows += normalize_csv(synth, FRONTEND_DATA / "ablation_results.csv")

    print()
    print("=" * 60)
    print(f"Done. {total_rows} total rows synced to {FRONTEND_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
