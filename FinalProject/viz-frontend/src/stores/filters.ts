/**
 * Shared filter state, persisted to localStorage.
 * Each filter dimension is independent; pages can read/write the subset
 * they care about.
 */
import { defineStore } from "pinia";
import { useStorage } from "@vueuse/core";
import type { ChartKind, MetricName } from "@/types/results";

export const useFiltersStore = defineStore("filters", () => {
  // Persisted slices (useStorage handles SSR safety + serialization)
  const selectedModels = useStorage<string[]>("f:models", []);
  const selectedDatasets = useStorage<string[]>("f:datasets", []);
  // pred_len is single-select: null means "no filter" (show all horizons).
  // Storage key bumped from f:predLens to f:predLen so users with the old
  // multi-select array get a clean start.
  const selectedPredLen = useStorage<number | null>("f:predLen", null);
  const metric = useStorage<MetricName>("f:metric", "MSE");
  const chartType = useStorage<ChartKind>("f:chart", "bar");
  const textModes = useStorage<string[]>("f:textModes", [
    "baseline",
    "report",
    "search",
    "both_concat",
    "both_gating",
  ]);
  // Ablation group is single-select: null means "all groups (compare them all)".
  // Storage key bumped from f:ablationGroups to f:ablationGroup.
  const selectedAblationGroup = useStorage<string | null>("f:ablationGroup", null);

  function resetForLine(line: number) {
    // Always clear pred_len on line change — different lines have different
    // valid horizons (e.g. Line 3 has only [96, 192]; Line 1/2/4 have
    // [96, 192, 336, 720]). A stale 720 from Line 1 would silently filter
    // out all rows on Line 3.
    selectedPredLen.value = null;
    if (line === 3) {
      // Multimodal: keep text mode filter visible
      if (textModes.value.length === 0) {
        textModes.value = ["baseline", "report", "search", "both_concat", "both_gating"];
      }
    }
    if (line === 4 || line === 5) {
      // Ablation pages (Line 4a KAN, Line 4b Lite): default to waterfall
      // chart (only chart that works with ablation rows; other charts need
      // model/dataset columns)
      chartType.value = "waterfall";
      selectedAblationGroup.value = null;
    }
  }

  function resetAll() {
    selectedModels.value = [];
    selectedDatasets.value = [];
    selectedPredLen.value = null;
    metric.value = "MSE";
    chartType.value = "bar";
    textModes.value = ["baseline", "report", "search", "both_concat", "both_gating"];
    selectedAblationGroup.value = null;
  }

  return {
    selectedModels,
    selectedDatasets,
    selectedPredLen,
    metric,
    chartType,
    textModes,
    selectedAblationGroup,
    resetForLine,
    resetAll,
  };
});
