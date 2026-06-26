/**
 * CSV loader: fetch + PapaParse with column normalization.
 *
 * Data flow is strictly per-line. Each line run produces:
 *   /data/line{N}_latest.csv        (always current; viz reads this)
 *   /data/line{N}_{ts}.csv          (timestamped history)
 *   /data/efficiency/line{N}_latest.csv
 *
 * There is NO shared main_results.csv. If a viz page needs data from
 * another line, it loads that line's _latest.csv directly.
 */
import Papa from "papaparse";
import type { ResultRow, EfficiencyRow, AblationRow } from "@/types/results";

/** Legacy alias columns that might still appear in old CSV files */
const COLUMN_ALIASES: Record<string, string[]> = {
  "Params(M)": ["Params(M)", "params_M", "params_m", "params(M)"],
  "FLOPs(G)": ["FLOPs(G)", "flops_G", "flops_g", "FLOPs(g)"],
  "InferTime(ms)": ["InferTime(ms)", "infer_time_ms"],
  "GPUMem(MB)": ["GPUMem(MB)", "gpu_mem_mb"],
  MSE: ["MSE", "mse"],
  MAE: ["MAE", "mae"],
  RMSE: ["RMSE", "rmse"],
  MAPE: ["MAPE", "mape"],
  SMAPE: ["SMAPE", "smape"],
};

function normalizeRow<T extends Record<string, any>>(row: T): T {
  const renamed: Record<string, any> = { ...row };
  for (const [canonical, aliases] of Object.entries(COLUMN_ALIASES)) {
    for (const alias of aliases) {
      if (alias in renamed && alias !== canonical) {
        renamed[canonical] = renamed[alias];
        delete renamed[alias];
      }
    }
  }
  return renamed as T;
}

async function fetchText(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.text();
}

function parseCSV<T extends Record<string, any>>(text: string): T[] {
  const result = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: (h) => h.trim(),
  });
  if (result.errors.length > 0) {
    console.warn("CSV parse errors:", result.errors.slice(0, 3));
  }
  return result.data.filter((row) => row && Object.keys(row).length > 0);
}

/** Load and parse a CSV file with column normalization. */
export async function loadCSV<T extends Record<string, any>>(
  path: string
): Promise<T[]> {
  try {
    const text = await fetchText(path);
    const rows = parseCSV<T>(text);
    return rows.map(normalizeRow);
  } catch (e) {
    console.warn(`[loadCSV] ${path}:`, e);
    return [];
  }
}

// ============================================================================
// Per-line loaders (Line 1, 2, 3 use these; Line 4 uses ablation loaders)
// ============================================================================

/** Load Line N's latest main results (N ∈ {1, 2, 3}) */
export async function loadLineData(line: number): Promise<ResultRow[]> {
  return loadCSV<ResultRow>(`/data/line${line}_latest.csv`);
}

/** Load Line N's latest efficiency data (N ∈ {1, 2, 3}) */
export async function loadLineEfficiency(
  line: number
): Promise<EfficiencyRow[]> {
  return loadCSV<EfficiencyRow>(`/data/efficiency/line${line}_latest.csv`);
}

/** Load the v3 cross-architecture Params/FLOPs summary (7 models × 4 datasets).
 *  Produced by scripts/measure_v3*.py; supersedes the old efficiency_summary.csv. */
export async function loadFlopsParamsV3(): Promise<EfficiencyRow[]> {
  return loadCSV<EfficiencyRow>(`/data/efficiency/flops_params_v3.csv`);
}

/** Load the canonical Line 3 multimodal results (SparseTSF + 4 text modes).
 *  This is the v2.1.1 "fixed" multimodal data; the older line3_latest.csv
 *  contains PatchTST/Mamba runs where the training loop discarded the
 *  text batch (see docs/multimodal-bug-diagnosis.md). */
export async function loadLine3SparseTSF(): Promise<ResultRow[]> {
  return loadCSV<ResultRow>(`/data/line3_sparsetsf_latest.csv`);
}

/** Load KAN ablation latest data */
export async function loadAblationKan(): Promise<AblationRow[]> {
  return loadCSV<AblationRow>("/data/ablation_kan_latest.csv");
}

/** Load Lite ablation latest data */
export async function loadAblationLite(): Promise<AblationRow[]> {
  return loadCSV<AblationRow>("/data/ablation_lite_latest.csv");
}

// ============================================================================
// Per-line merge: combine main results with efficiency on (model, dataset)
// ============================================================================
export function mergeEfficiency(
  main: ResultRow[],
  eff: EfficiencyRow[]
): ResultRow[] {
  if (eff.length === 0) return main;
  const lookup = new Map<string, EfficiencyRow>();
  for (const row of eff) {
    lookup.set(`${row.model}|${row.dataset}`, row);
  }
  return main.map((row) => {
    const e = lookup.get(`${row.model}|${row.dataset}`);
    if (!e) return row;
    return {
      ...row,
      "Params(M)":
        row["Params(M)"] && row["Params(M)"]! > 0
          ? row["Params(M)"]
          : e["Params(M)"],
      "FLOPs(G)":
        row["FLOPs(G)"] && row["FLOPs(G)"]! > 0
          ? row["FLOPs(G)"]
          : e["FLOPs(G)"],
    };
  });
}

// ============================================================================
// File status (for Overview page)
// ============================================================================
export type FileStatus = Record<string, "ok" | "missing" | "empty">;

const TRACKED_FILES: Array<[string, string]> = [
  ["line1", "/data/line1_latest.csv"],
  ["line2", "/data/line2_latest.csv"],
  ["line3_sparsetsf", "/data/line3_sparsetsf_latest.csv"],
  ["efficiency/line1", "/data/efficiency/line1_latest.csv"],
  ["efficiency/line2", "/data/efficiency/line2_latest.csv"],
  ["efficiency/line3", "/data/efficiency/line3_latest.csv"],
  ["efficiency/line3_sparsetsf", "/data/efficiency/line3_sparsetsf_latest.csv"],
  ["efficiency/flops_v3", "/data/efficiency/flops_params_v3.csv"],
  ["ablation_kan", "/data/ablation_kan_latest.csv"],
  ["ablation_lite", "/data/ablation_lite_latest.csv"],
];

export async function loadFileStatus(): Promise<FileStatus> {
  const out: FileStatus = {};
  for (const [label, path] of TRACKED_FILES) {
    try {
      const res = await fetch(path, { method: "HEAD" });
      if (!res.ok) {
        out[label] = "missing";
        continue;
      }
      const text = await fetchText(path);
      const lines = text.split("\n").filter((l) => l.trim().length > 0);
      out[label] = lines.length > 1 ? "ok" : "empty";
    } catch {
      out[label] = "missing";
    }
  }
  return out;
}
