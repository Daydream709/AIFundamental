/**
 * Type definitions for the visualization platform.
 *
 * Mirrors the data shape after column normalization in
 * scripts/sync_results.py.
 */

/** Single row in main_results.csv (also merged with efficiency data) */
export interface ResultRow {
  model: string;
  dataset: string;
  seq_len: number;
  pred_len: number;
  MSE?: number;
  MAE?: number;
  RMSE?: number;
  MAPE?: number;
  SMAPE?: number;
  "Params(M)"?: number;
  "FLOPs(G)"?: number;
  "InferTime(ms)"?: number;
  "GPUMem(MB)"?: number;
  loss_type?: string;
  text_mode?: string;
  timestamp?: string;
  [key: string]: string | number | undefined;
}

/** Row in efficiency_summary.csv (filled in via merge) */
export interface EfficiencyRow {
  model: string;
  dataset: string;
  "Params(M)"?: number;
  "FLOPs(G)"?: number;
}

/** Row in ablation_results.csv */
export interface AblationRow {
  ablation: string;
  setting: string;
  MSE?: number;
  MAE?: number;
  [key: string]: string | number | undefined;
}

/** All available metric columns */
export type MetricName =
  | "MSE"
  | "MAE"
  | "RMSE"
  | "MAPE"
  | "SMAPE"
  | "Params(M)"
  | "FLOPs(G)"
  | "InferTime(ms)"
  | "GPUMem(MB)";

export const ACCURACY_METRICS: MetricName[] = [
  "MSE",
  "MAE",
  "RMSE",
  "MAPE",
  "SMAPE",
];

export const EFFICIENCY_METRICS: MetricName[] = [
  "Params(M)",
  "FLOPs(G)",
  "InferTime(ms)",
  "GPUMem(MB)",
];

export const ALL_METRICS: MetricName[] = [
  ...ACCURACY_METRICS,
  ...EFFICIENCY_METRICS,
];

/** Chart type keys */
export type ChartKind =
  | "bar"
  | "line"
  | "radar"
  | "heatmap"
  | "pareto"
  | "box"
  | "waterfall"
  | "parallel";

export const CHART_KINDS: ChartKind[] = [
  "bar",
  "line",
  "radar",
  "heatmap",
  "pareto",
  "box",
  "waterfall",
  "parallel",
];

/** Internal model name -> display name */
export const MODEL_DISPLAY: Record<string, string> = {
  DLinear: "DLinear",
  PatchTST: "PatchTST",
  TimesNet: "TimesNet",
  Mamba: "Mamba",
  SparseTSF: "SparseTSF",
  KANiTransformer: "KAN-iTransformer",
  LiteSparseNet: "Lite-SparseNet",
  iTransformer: "iTransformer",
  TimeMixer: "TimeMixer",
  TimeKAN: "TimeKAN",
  MambaTransformerDual: "Mamba-Transformer",
  MultimodalFusion: "Multimodal",
};

export function displayModel(name: string): string {
  return MODEL_DISPLAY[name] ?? name;
}

/** Union of all row types the chart dispatcher accepts */
export type AnyResultRow = ResultRow | AblationRow;
