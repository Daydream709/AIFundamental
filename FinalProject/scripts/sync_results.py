"""Sync CSV results into the Vue frontend's public/data directory.

Each line has its own per-line files (line{N}_latest.csv, line{N}_{ts}.csv,
ablation_{prefix}_latest.csv, efficiency/line{N}_latest.csv). This script
copies them all to viz-frontend/public/data/ so the visualization
platform can read them.

Per-line separation: NO cross-line aggregation. If a viz page needs
data from another line, it fetches that line's _latest.csv directly.
"""
import shutil
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FRONTEND_DATA = PROJECT_ROOT / "viz-frontend" / "public" / "data"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"

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
    print("Sync Results (per-line) -> viz-frontend/public/data/")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: results/ directory not found: {RESULTS_DIR}")
        return

    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DATA / "efficiency").mkdir(parents=True, exist_ok=True)

    total_rows = 0

    # === Per-line "latest" files (always current, viz reads these) ===
    print()
    print("Per-line latest files (viz entry points):")
    for n in (1, 2, 3):
        latest = RESULTS_DIR / f"line{n}_latest.csv"
        dest = FRONTEND_DATA / f"line{n}_latest.csv"
        total_rows += normalize_csv(latest, dest)
    for prefix in ("kan", "lite"):
        latest = RESULTS_DIR / f"ablation_{prefix}_latest.csv"
        dest = FRONTEND_DATA / f"ablation_{prefix}_latest.csv"
        total_rows += normalize_csv(latest, dest)
    for n in (1, 2, 3):
        eff = EFFICIENCY_DIR / f"line{n}_latest.csv"
        dest = FRONTEND_DATA / "efficiency" / f"line{n}_latest.csv"
        total_rows += normalize_csv(eff, dest)

    # === Line 3 SparseTSF multimodal (v2.1.1: real multimodal results) ===
    # This file replaces the old PatchTST/Mamba line3 which had a training
    # loop bug (see docs/multimodal-bug-diagnosis.md). The SparseTSF
    # 4-mode run is the canonical multimodal result.
    print()
    print("Line 3 SparseTSF multimodal (canonical multimodal data):")
    sparse_latest = RESULTS_DIR / "line3_sparsetsf_latest.csv"
    if sparse_latest.exists():
        total_rows += normalize_csv(sparse_latest, FRONTEND_DATA / "line3_sparsetsf_latest.csv")
    sparse_eff = EFFICIENCY_DIR / "line3_sparsetsf_latest.csv"
    if sparse_eff.exists():
        total_rows += normalize_csv(sparse_eff, FRONTEND_DATA / "efficiency" / "line3_sparsetsf_latest.csv")

    # === v3 efficiency summary (7 models × 4 datasets, the canonical
    #     Params/FLOPs table for the project) ===
    print()
    print("v3 efficiency summary (7 models × 4 datasets):")
    v3 = EFFICIENCY_DIR / "flops_params_summary_v3.csv"
    if v3.exists():
        total_rows += normalize_csv(v3, FRONTEND_DATA / "efficiency" / "flops_params_v3.csv")

    # === Per-line timestamped files (history) ===
    print()
    print("Per-line timestamped files (history):")
    for src in sorted(RESULTS_DIR.glob("line*_*.csv")):
        if src.name.endswith("_latest.csv") or src.name.endswith("_partial.csv"):
            continue
        dest = FRONTEND_DATA / src.name
        total_rows += normalize_csv(src, dest)
    for src in sorted(RESULTS_DIR.glob("ablation_*_*.csv")):
        if src.name.endswith("_latest.csv") or src.name.endswith("_partial.csv"):
            continue
        dest = FRONTEND_DATA / src.name
        total_rows += normalize_csv(src, dest)
    if EFFICIENCY_DIR.exists():
        for src in sorted(EFFICIENCY_DIR.glob("line*_*.csv")):
            if src.name.endswith("_latest.csv"):
                continue
            dest = FRONTEND_DATA / "efficiency" / src.name
            total_rows += normalize_csv(src, dest)

    # === Cleanup: remove legacy files no longer in sync with current viz ===
    # Old main_results.csv / efficiency_summary.csv / ablation_results.csv
    # were the v1 single-file aggregation. The current viz uses per-line
    # _latest.csv only. If a legacy file was copied in the past, remove it.
    print()
    print("Cleanup: removing legacy aggregation files (no longer used):")
    for legacy_name in ("main_results.csv", "efficiency_summary.csv", "ablation_results.csv"):
        legacy_path = FRONTEND_DATA / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"  REMOVED: {legacy_name}")

    print()
    print("=" * 60)
    print(f"Done. {total_rows} total rows synced to {FRONTEND_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
