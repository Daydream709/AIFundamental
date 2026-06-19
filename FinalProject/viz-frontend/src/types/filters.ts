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
