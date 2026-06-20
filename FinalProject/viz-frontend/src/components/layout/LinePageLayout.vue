<template>
  <div class="line-page">
    <p class="page-description">{{ line.description }}</p>

    <FilterBar @reset="filters.resetAll()">
      <FilterSelect
        label="模型 (Models)"
        icon="🤖"
        multiple
        :options="modelOptions"
        v-model="filters.selectedModels"
        placeholder="选择模型..."
      />
      <FilterSelect
        label="数据集 (Datasets)"
        icon="📚"
        multiple
        :options="datasetOptions"
        v-model="filters.selectedDatasets"
        placeholder="选择数据集..."
      />
      <FilterSelect
        label="预测长度 (单选)"
        icon="📏"
        :options="predLenOptions"
        v-model="filters.selectedPredLen"
        placeholder="全部 horizon"
      />
      <FilterSelect
        label="指标 (Metric)"
        icon="📐"
        :options="metricOptions"
        v-model="filters.metric"
        placeholder="选择指标"
      />
      <FilterSelect
        label="图表类型"
        icon="📊"
        :options="chartOptions"
        v-model="filters.chartType"
        placeholder="选择图表"
      />
      <FilterSelect
        v-if="line.extraFilters?.textMode"
        label="文本模态 (Line 3)"
        icon="🎭"
        multiple
        :options="textModeOptions"
        v-model="filters.textModes"
        placeholder="选择文本模态..."
      />
      <FilterSelect
        v-if="line.dataSource === 'ablation'"
        label="消融组 (单选)"
        icon="🔬"
        :options="ablationGroupOptions"
        v-model="filters.selectedAblationGroup"
        placeholder="所有消融组"
      />
    </FilterBar>

    <!-- KPI cards -->
    <div class="kpi-row">
      <KPICard
        :title="`Best ${filters.metric}`"
        :value="kpiBestValue"
        :sub="kpiBestModel"
        icon="🏆"
      />
      <KPICard
        :title="`Avg ${filters.metric}`"
        :value="kpiAvgValue"
        icon="📐"
      />
      <KPICard title="Runs" :value="String(filteredData.length)" :sub="`${nModels} models`" icon="🧪" />
      <KPICard title="Datasets" :value="String(nDatasets)" icon="📚" />
    </div>

    <!-- Chart -->
    <EChartWrapper
      v-if="hasData"
      :option="chartOption"
      :min-height="460"
      :empty-message="'当前筛选无数据 — 调整侧边栏的过滤器'"
    />

    <EmptyState
      v-else
      title="数据尚未生成"
      :message="emptyMessage"
      icon="📋"
    />

    <!-- Raw data table -->
    <ResultTable v-if="hasData" :rows="filteredData" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, watch } from "vue";
import { useDataStore } from "@/stores/data";
import { useFiltersStore } from "@/stores/filters";
import { LINES } from "@/data/lines";
import { KPICard } from "@/components/cards";
import FilterBar from "@/components/filters/FilterBar.vue";
import FilterSelect from "@/components/filters/FilterSelect.vue";
import EChartWrapper from "@/components/charts/EChartWrapper.vue";
import ResultTable from "@/components/data/ResultTable.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import { buildChartOption } from "@/composables/useChartOptions";
import { displayModel, formatMetric } from "@/composables/useMetricFormat";
import { ALL_METRICS } from "@/types/results";
import type { ExperimentLine } from "@/types/filters";
import type { AnyResultRow } from "@/types/results";

const props = defineProps<{ line: ExperimentLine }>();
const dataStore = useDataStore();
const filters = useFiltersStore();

onMounted(async () => {
  filters.resetForLine(props.line.number);
  await dataStore.loadAll();
});

// React to line changes (e.g. user navigates between pages).
// IMPORTANT: set chart type BEFORE loadAll() so the first chart render
// uses the right type (line 4 needs waterfall; other lines default to bar).
watch(
  () => props.line.number,
  (n) => {
    filters.resetForLine(n);
  },
  { immediate: true }
);

// ---- Option lists for select boxes ----
const modelOptions = computed(() => props.line.models.map((m) => ({ label: displayModel(m), value: m })));
const datasetOptions = computed(() => props.line.datasets.map((d) => ({ label: d, value: d })));
const predLenOptions = computed(() =>
  props.line.predLens.map((p) => ({ label: String(p), value: p }))
);
const metricOptions = computed(() => ALL_METRICS.map((m) => ({ label: m, value: m })));
// Chart types that work with ablation data (ablation+setting columns only)
const ABLATION_COMPATIBLE_CHARTS = ["waterfall", "bar", "line", "box", "heatmap"];
const chartOptions = computed(() => {
  const all = [
    { label: "📊 柱状图", value: "bar" },
    { label: "📈 折线图", value: "line" },
    { label: "🎯 雷达图", value: "radar" },
    { label: "🌡️ 热力图", value: "heatmap" },
    { label: "⚖️ 帕累托", value: "pareto" },
    { label: "📦 箱线图", value: "box" },
    { label: "💧 瀑布图", value: "waterfall" },
    { label: "☁️ 平行坐标", value: "parallel" },
  ];
  // On Line 4 (ablation), keep all options but mark incompatible ones
  if (props.line.dataSource === "ablation") {
    return all.map((opt) => ({
      ...opt,
      label: ABLATION_COMPATIBLE_CHARTS.includes(opt.value as string)
        ? opt.label
        : `${opt.label} ⚠️`,
      disabled: !ABLATION_COMPATIBLE_CHARTS.includes(opt.value as string),
    }));
  }
  return all;
});
const textModeOptions = computed(() =>
  (props.line.extraFilters?.textMode ?? []).map((m) => ({ label: m, value: m }))
);
const ablationGroupOptions = computed(() => {
  if (props.line.dataSource !== "ablation") return [];
  // Only show groups that belong to this line's research topic.
  // Data source depends on which line: KAN (4) or Lite (5).
  const allowed = props.line.ablationGroups;
  const src: any[] =
    props.line.number === 4 ? dataStore.kanAblation :
    props.line.number === 5 ? dataStore.liteAblation : [];
  const groups = Array.from(
    new Set(
      src
        .filter((r: any) => !allowed || allowed.includes(r.ablation))
        .map((r: any) => r.ablation)
    )
  );
  return groups.map((g: string) => ({ label: g, value: g }));
});

// ---- Data filtering ----
const rawData = computed<AnyResultRow[]>(() => {
  // Strictly per-line: each LineN page reads from its own data slice.
  // No cross-line aggregation here — if a viz needs another line's
  // data, the page loads it explicitly elsewhere.
  if (props.line.number === 4) {
    // KAN ablation
    const allowed = props.line.ablationGroups;
    if (allowed && allowed.length > 0) {
      return dataStore.kanAblation.filter((r) => allowed.includes(r.ablation)) as AnyResultRow[];
    }
    return dataStore.kanAblation as AnyResultRow[];
  }
  if (props.line.number === 5) {
    // Lite ablation
    const allowed = props.line.ablationGroups;
    if (allowed && allowed.length > 0) {
      return dataStore.liteAblation.filter((r) => allowed.includes(r.ablation)) as AnyResultRow[];
    }
    return dataStore.liteAblation as AnyResultRow[];
  }
  if (props.line.number === 1) return dataStore.line1Merged as AnyResultRow[];
  if (props.line.number === 2) {
    // Line 2 only trains the 3 self-dev models. The 4 thuml baselines
    // (DLinear/PatchTST/TimesNet/Mamba) come from Line 1's data — we
    // re-use them here so viz shows a complete 7-model comparison.
    const line1 = dataStore.line1Merged as AnyResultRow[];
    return [...dataStore.line2Merged, ...line1] as AnyResultRow[];
  }
  if (props.line.number === 3) return dataStore.line3Merged as AnyResultRow[];
  return [];
});

const filteredData = computed<AnyResultRow[]>(() => {
  let df: AnyResultRow[] = rawData.value;
  if (df.length === 0) return df;

  // Model filter (only for non-ablation data)
  if (filters.selectedModels.length > 0 && "model" in df[0]) {
    df = df.filter((r) => filters.selectedModels.includes((r as any).model));
  }
  // Dataset filter
  if (filters.selectedDatasets.length > 0 && "dataset" in df[0]) {
    df = df.filter((r) => filters.selectedDatasets.includes((r as any).dataset));
  }
  // Pred len filter (single-select; null = all horizons)
  if (filters.selectedPredLen !== null && "pred_len" in df[0]) {
    df = df.filter((r) => (r as any).pred_len === filters.selectedPredLen);
  }
  // Text mode filter (line 3)
  if (filters.textModes.length > 0 && "text_mode" in df[0]) {
    df = df.filter((r) => filters.textModes.includes((r as any).text_mode));
  }
  // Ablation group filter (line 4, single-select; null = all groups)
  if (filters.selectedAblationGroup !== null && "ablation" in df[0]) {
    df = df.filter((r) => (r as any).ablation === filters.selectedAblationGroup);
  }
  return df;
});

const hasData = computed(() => filteredData.value.length > 0);
const nModels = computed(() => {
  if (filteredData.value.length === 0) return 0;
  if ("model" in filteredData.value[0]) {
    return new Set(filteredData.value.map((r: any) => r.model)).size;
  }
  return 0;
});
const nDatasets = computed(() => {
  if (filteredData.value.length === 0) return 0;
  if ("dataset" in filteredData.value[0]) {
    return new Set(filteredData.value.map((r: any) => r.dataset)).size;
  }
  return 0;
});

// KPIs
const kpiBestValue = computed(() => {
  if (!hasData.value) return "—";
  const m = filters.metric;
  let best: number | null = null;
  for (const r of filteredData.value as any[]) {
    const v = r[m];
    if (typeof v === "number" && !Number.isNaN(v)) {
      if (best === null || v < best) best = v;
    }
  }
  return best === null ? "—" : formatMetric(m, best);
});
const kpiBestModel = computed(() => {
  if (!hasData.value) return "—";
  const m = filters.metric;
  let bestRow: any = null;
  for (const r of filteredData.value as any[]) {
    const v = r[m];
    if (typeof v === "number" && !Number.isNaN(v)) {
      if (!bestRow || v < bestRow[m]) bestRow = r;
    }
  }
  return bestRow?.model ? displayModel(bestRow.model) : "—";
});
const kpiAvgValue = computed(() => {
  if (!hasData.value) return "—";
  const m = filters.metric;
  const vals = (filteredData.value as any[])
    .map((r) => r[m])
    .filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (vals.length === 0) return "—";
  return formatMetric(m, vals.reduce((s, v) => s + v, 0) / vals.length);
});

// Chart option
const chartOption = computed(() => buildChartOption(filters.chartType, filteredData.value, filters.metric));

// Empty state message
const emptyMessage = computed(() => {
  if (dataStore.isReady) {
    if (filteredData.value.length === 0) {
      return `当前筛选条件未命中任何数据。请调整左侧的模型 / 数据集 / 预测长度。`;
    }
    return "请先生成实验数据";
  }
  if (props.line.dataSource === "ablation") {
    return `请先运行 <code>python run_ablation.py</code> 生成消融结果`;
  }
  return `请先运行 <code>python run_experiments.py --line ${props.line.number}</code> 生成实验结果`;
});
</script>

<style scoped>
.line-page {
  max-width: 1500px;
  margin: 0 auto;
}
.page-description {
  font-size: 0.9rem;
  color: var(--color-neu-text-muted);
  line-height: 1.6;
  margin: 0 0 1rem;
  padding: 0.75rem 1rem;
  background: var(--color-neu-bg);
  border-radius: 12px;
  box-shadow:
    inset 3px 3px 6px var(--color-neu-shadow-dark),
    inset -3px -3px 6px var(--color-neu-shadow-light);
}

/* Icon legend section removed — each line now has a single focused icon */

.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}
</style>
