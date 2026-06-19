/**
 * Pinia store for CSV data cache.
 * Loads once on first access, shared across all pages.
 */
import { defineStore } from "pinia";
import {
  loadAblation,
  loadEfficiency,
  loadFileStatus,
  loadMainResults,
  mergeEfficiency,
} from "@/composables/useCSVLoader";
import type { ResultRow, AblationRow } from "@/types/results";

export const useDataStore = defineStore("data", {
  state: () => ({
    loading: false,
    loaded: false,
    mainResults: [] as ResultRow[],
    efficiency: [] as { model: string; dataset: string; "Params(M)"?: number; "FLOPs(G)"?: number }[],
    ablation: [] as AblationRow[],
    fileStatus: {} as Record<string, string>,
    lastLoadedAt: 0,
  }),

  getters: {
    mergedMain(state): ResultRow[] {
      return mergeEfficiency(state.mainResults, state.efficiency);
    },
    isReady(state): boolean {
      return state.loaded && !state.loading;
    },
  },

  actions: {
    async loadAll() {
      if (this.loaded && Date.now() - this.lastLoadedAt < 60_000) return;
      this.loading = true;
      try {
        const [main, eff, abl, status] = await Promise.all([
          loadMainResults(),
          loadEfficiency(),
          loadAblation(),
          loadFileStatus(),
        ]);
        this.mainResults = main;
        this.efficiency = eff;
        this.ablation = abl;
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
