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
  ]);
  // Ablation group is single-select: null means "all groups (compare them all)".
  // Storage key bumped from f:ablationGroups to f:ablationGroup.
  const selectedAblationGroup = useStorage<string | null>("f:ablationGroup", null);

  // One-time migration: drop stale `both_gating` mode from persisted
  // textModes (v2.1.1 no longer uses it; was a placeholder that produced
  // identical results in the old PatchTST/Mamba multimodal bug).
  if (textModes.value.includes("both_gating")) {
    textModes.value = textModes.value.filter((m) => m !== "both_gating");
  }

  function resetForLine(line: number) {
    // BUGFIX: previously only pred_len was reset on line change, which let
    // stale filter state (model/dataset/textModes/ablationGroup) from a
    // previous line silently filter the new line to 0 rows → blank page.
    // Now we reset every cross-page slice on line change. (chartType and
    // metric are user preferences and are preserved.)
    selectedModels.value = [];
    selectedDatasets.value = [];
    selectedPredLen.value = null;
    textModes.value = [];
    selectedAblationGroup.value = null;

    if (line === 3) {
      // Line 3 is the text-mode ablation (SparseTSF × 4 modes). Pre-select
      // all text modes so the default view shows the full comparison.
      textModes.value = ["baseline", "report", "search", "both_concat"];
    }
    if (line === 4 || line === 5) {
      // Ablation pages (Line 4a KAN, Line 4b Lite): default to waterfall
      // chart (only chart that works with ablation rows; other charts need
      // model/dataset columns)
      chartType.value = "waterfall";
    }
  }

  function resetAll() {
    selectedModels.value = [];
    selectedDatasets.value = [];
    selectedPredLen.value = null;
    metric.value = "MSE";
    chartType.value = "bar";
    textModes.value = ["baseline", "report", "search", "both_concat"];
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
