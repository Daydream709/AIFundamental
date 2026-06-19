/**
 * Metric value formatting with sensible defaults per metric.
 */
import type { MetricName } from "@/types/results";
import { displayModel as _displayModel } from "@/types/results";

export { displayModel } from "@/types/results";

const FORMATTERS: Partial<Record<MetricName, (v: number) => string>> = {
  MSE: (v) => v.toFixed(6),
  MAE: (v) => v.toFixed(6),
  RMSE: (v) => v.toFixed(6),
  MAPE: (v) => `${v.toFixed(4)}%`,
  SMAPE: (v) => `${v.toFixed(4)}%`,
  "Params(M)": (v) => `${v.toFixed(4)} M`,
  "FLOPs(G)": (v) => `${v.toFixed(4)} G`,
  "InferTime(ms)": (v) => `${v.toFixed(3)} ms`,
  "GPUMem(MB)": (v) => `${v.toFixed(2)} MB`,
};

export function formatMetric(metric: MetricName | string, value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const fn = FORMATTERS[metric as MetricName];
  return fn ? fn(value) : value.toFixed(4);
}

/** Short label for filter selectbox */
export function metricLabel(metric: MetricName): string {
  switch (metric) {
    case "Params(M)":
      return "Params (M)";
    case "FLOPs(G)":
      return "FLOPs (G)";
    case "InferTime(ms)":
      return "Infer Time (ms)";
    case "GPUMem(MB)":
      return "GPU Mem (MB)";
    case "MAPE":
    case "SMAPE":
      return `${metric} (%)`;
    default:
      return metric;
  }
}
