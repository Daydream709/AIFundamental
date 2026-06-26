/**
 * Pinia store for per-line data cache.
 *
 * Strictly per-line: each line has its own slice. Cross-line data
 * references should load the relevant line's _latest.csv directly
 * (see useCSVLoader.ts).
 */
import { defineStore } from "pinia";
import {
  loadAblationKan,
  loadAblationLite,
  loadFileStatus,
  loadFlopsParamsV3,
  loadLine3SparseTSF,
  loadLineData,
  loadLineEfficiency,
  mergeEfficiency,
  type FileStatus,
} from "@/composables/useCSVLoader";
import type { ResultRow, AblationRow } from "@/types/results";

export const useDataStore = defineStore("data", {
  state: () => ({
    loading: false,
    loaded: false,
    lastLoadedAt: 0,

    // Per-line main results (Line 1, 2, 3)
    line1Data: [] as ResultRow[],
    line2Data: [] as ResultRow[],
    line3Data: [] as ResultRow[],

    // Line 3 SparseTSF multimodal (the canonical multimodal results;
    // supersedes line3Data for the Line 3 page).
    line3SparseTSFData: [] as ResultRow[],

    // Per-line efficiency (Line 1, 2, 3)
    line1Efficiency: [] as { model: string; dataset: string; "Params(M)"?: number; "FLOPs(G)"?: number }[],
    line2Efficiency: [] as { model: string; dataset: string; "Params(M)"?: number; "FLOPs(G)"?: number }[],
    line3Efficiency: [] as { model: string; dataset: string; "Params(M)"?: number; "FLOPs(G)"?: number }[],

    // v3 cross-architecture efficiency (7 models × 4 datasets)
    flopsParamsV3: [] as { model: string; dataset: string; "Params(M)"?: number; "FLOPs(G)"?: number }[],

    // Per-prefix ablation (Line 4a KAN, Line 4b Lite)
    kanAblation: [] as AblationRow[],
    liteAblation: [] as AblationRow[],

    // Per-line file status (for Overview page)
    fileStatus: {} as FileStatus,
  }),

  getters: {
    /**
     * Per-line merged data (main + efficiency on model × dataset).
     * Used by the LinePageLayout for Line 1/2/3.
     */
    line1Merged: (state) => mergeEfficiency(state.line1Data, state.line1Efficiency),
    line2Merged: (state) => mergeEfficiency(state.line2Data, state.line2Efficiency),
    line3Merged: (state) => mergeEfficiency(state.line3SparseTSFData, state.line3Efficiency),

    isReady(state) {
      return state.loaded && !state.loading;
    },
  },

  actions: {
    async loadAll() {
      if (this.loaded && Date.now() - this.lastLoadedAt < 60_000) return;
      this.loading = true;
      try {
        const [
          l1, l2, l3,
          l3sparse,
          e1, e2, e3,
          v3,
          kan, lite,
          status,
        ] = await Promise.all([
          loadLineData(1),
          loadLineData(2),
          loadLineData(3),
          loadLine3SparseTSF(),
          loadLineEfficiency(1),
          loadLineEfficiency(2),
          loadLineEfficiency(3),
          loadFlopsParamsV3(),
          loadAblationKan(),
          loadAblationLite(),
          loadFileStatus(),
        ]);
        this.line1Data = l1;
        this.line2Data = l2;
        this.line3Data = l3;
        this.line3SparseTSFData = l3sparse;
        this.line1Efficiency = e1;
        this.line2Efficiency = e2;
        this.line3Efficiency = e3;
        this.flopsParamsV3 = v3;
        this.kanAblation = kan;
        this.liteAblation = lite;
        this.fileStatus = status;
        this.loaded = true;
        this.lastLoadedAt = Date.now();
      } finally {
        this.loading = false;
      }
    },

    async refresh() {
      this.loaded = false;
      await this.loadAll();
    },
  },
});
