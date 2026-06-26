/**
 * Type definitions for filter state and experiment line metadata.
 */
import type { ChartKind, MetricName } from "./results";

/** An experiment line defines the scope of one comparison axis */
export interface ExperimentLine {
  number: number;
  title: string; // Chinese
  subtitle: string; // English
  icons: string[]; // Emoji(s) — single line uses [icon], Line 4a uses [🧠], Line 4b uses [🪶]
  route: string; // Vue router path
  models: string[];
  datasets: string[];
  predLens: number[];
  description: string;
  dataSource: "main" | "ablation";
  /**
   * How to GROUP data for chart x-axis / series. Without this, charts
   * fall back to "model+dataset" which gives a single bar/line for
   * single-model datasets (e.g. Line 3's text_mode ablation, where
   * there is only 1 model × 1 dataset, so we MUST group by text_mode
   * for any meaningful comparison).
   *   - "model+dataset" (default): x = models, series = datasets
   *   - "text_mode+pred_len":     x = text_mode, series = pred_len
   *   - "ablation+setting":       x = setting, series = ablation group
   */
  chartGroupBy?: "model+dataset" | "text_mode+pred_len" | "ablation+setting";
  /** For ablation lines: which ablation groups belong to this line's research topic */
  ablationGroups?: string[];
  extraFilters?: {
    textMode?: string[];
  };
}

/** Mutable filter state stored in Pinia + localStorage */
export interface FilterState {
  selectedModels: string[];
  selectedDatasets: string[];
  selectedPredLen: number | null; // single-select; null = show all horizons
  metric: MetricName;
  chartType: ChartKind;
  textModes: string[];
  selectedAblationGroup: string | null; // single-select; null = all groups
}
