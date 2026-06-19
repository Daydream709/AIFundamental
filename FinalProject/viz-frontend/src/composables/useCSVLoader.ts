/**
 * CSV loader: fetch + PapaParse with column normalization.
 *
 * Data flow:
 *   public/data/*.csv (synced by scripts/sync_results.py)
 *     -> fetch
 *     -> PapaParse with header:true, dynamicTyping:true
 *     -> normalizeColumns() (handles legacy lowercase aliases)
 *     -> mergeEfficiency() for main results
 */
import Papa from "papaparse";
import type { ResultRow, EfficiencyRow, AblationRow } from "@/types/results";

/** Legacy alias columns that might still appear if sync_results.py was bypassed */
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

/** Load and parse a single CSV file with column normalization */
export async function loadCSV<T extends Record<string, any>>(path: string): Promise<T[]> {
  try {
    const text = await fetchText(path);
    const rows = parseCSV<T>(text);
    return rows.map(normalizeRow);
  } catch (e) {
    console.warn(`[loadCSV] ${path}:`, e);
    return [];
  }
}

/** Load main_results.csv */
export async function loadMainResults(): Promise<ResultRow[]> {
  return loadCSV<ResultRow>("/data/main_results.csv");
}

/** Load efficiency_summary.csv */
export async function loadEfficiency(): Promise<EfficiencyRow[]> {
  return loadCSV<EfficiencyRow>("/data/efficiency_summary.csv");
}

/** Load KAN ablation (results/ablation_kan.csv) */
export async function loadAblationKan(): Promise<AblationRow[]> {
  return loadCSV<AblationRow>("/data/ablation_kan.csv");
}

/** Load Lite ablation (results/ablation_lite.csv) */
export async function loadAblationLite(): Promise<AblationRow[]> {
  return loadCSV<AblationRow>("/data/ablation_lite.csv");
}

/** Load combined ablation data (KAN + Lite + legacy synthetic). */
export async function loadAblation(): Promise<AblationRow[]> {
  const [kan, lite, legacy] = await Promise.all([
    loadAblationKan(),
    loadAblationLite(),
    loadCSV<AblationRow>("/data/ablation_results.csv").catch(() => []),
  ]);
  return [...kan, ...lite, ...legacy];
}

/** Merge main results with efficiency data on (model, dataset) */
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
      "Params(M)": row["Params(M)"] && row["Params(M)"]! > 0 ? row["Params(M)"] : e["Params(M)"],
      "FLOPs(G)": row["FLOPs(G)"] && row["FLOPs(G)"]! > 0 ? row["FLOPs(G)"] : e["FLOPs(G)"],
    };
  });
}

/** Get file availability status for the overview page */
export async function loadFileStatus(): Promise<Record<string, "ok" | "missing" | "empty">> {
  const out: Record<string, "ok" | "missing" | "empty"> = {};
  const files = [
    ["main", "/data/main_results.csv"],
    ["efficiency", "/data/efficiency_summary.csv"],
    ["ablation_kan", "/data/ablation_kan.csv"],
    ["ablation_lite", "/data/ablation_lite.csv"],
  ] as const;

  for (const [label, path] of files) {
    try {
      const res = await fetch(path, { method: "HEAD" });
      if (!res.ok) {
        out[label] = "missing";
        continue;
      }
      // Fetch the body to check for actual data
      const text = await fetchText(path);
      const lines = text.split("\n").filter((l) => l.trim().length > 0);
      out[label] = lines.length > 1 ? "ok" : "empty";
    } catch {
      out[label] = "missing";
    }
  }
  return out;
}
