<template>
  <div class="result-table-wrap">
    <div class="table-header">
      <span class="table-title">📋 原始数据</span>
      <span class="table-count">{{ rows.length }} 行</span>
    </div>
    <n-data-table
      :columns="columns"
      :data="rows"
      :pagination="pagination"
      :bordered="false"
      :single-line="false"
      size="small"
      striped
    />
  </div>
</template>

<script setup lang="ts">
import { computed, h } from "vue";
import { NDataTable, NTag } from "naive-ui";
import type { DataTableColumns } from "naive-ui";
import { formatMetric, displayModel } from "@/composables/useMetricFormat";
import type { MetricName } from "@/types/results";

const props = withDefaults(
  defineProps<{
    rows: Record<string, any>[];
    metrics?: MetricName[];
    maxRows?: number;
  }>(),
  { maxRows: 200 }
);

const ALL_METRIC_KEYS: MetricName[] = [
  "MSE",
  "MAE",
  "RMSE",
  "MAPE",
  "SMAPE",
  "Params(M)",
  "FLOPs(G)",
  "InferTime(ms)",
  "GPUMem(MB)",
];

const pagination = computed(() => ({ pageSize: 12, showSizePicker: false }));

const columns = computed<DataTableColumns>(() => {
  const cols: DataTableColumns = [];

  if (props.rows.length > 0 && "model" in props.rows[0]) {
    cols.push({
      title: "Model",
      key: "model",
      width: 140,
      render: (row: any) => h("span", { class: "model-cell" }, displayModel(row.model)),
    });
  }
  if (props.rows.length > 0 && "ablation" in props.rows[0]) {
    cols.push({ title: "Ablation", key: "ablation", width: 130 });
    cols.push({ title: "Setting", key: "setting", width: 110 });
  }
  if (props.rows.length > 0 && "dataset" in props.rows[0]) {
    cols.push({
      title: "Dataset",
      key: "dataset",
      width: 110,
      render: (row: any) => h(NTag, { size: "small", round: true, type: "info" }, { default: () => row.dataset }),
    });
  }
  if (props.rows.length > 0 && "pred_len" in props.rows[0]) {
    cols.push({ title: "pred_len", key: "pred_len", width: 90 });
  }
  if (props.rows.length > 0 && "text_mode" in props.rows[0]) {
    cols.push({ title: "text_mode", key: "text_mode", width: 130 });
  }

  const metrics = props.metrics ?? ALL_METRIC_KEYS;
  for (const m of metrics) {
    if (props.rows.length > 0 && m in props.rows[0]) {
      cols.push({
        title: m,
        key: m,
        width: 110,
        render: (row: any) => formatMetric(m, row[m]),
      });
    }
  }
  return cols;
});
</script>

<style scoped>
.result-table-wrap {
  background: #ffffff;
  border-radius: 16px;
  box-shadow:
    6px 6px 12px var(--color-neu-shadow-dark),
    -6px -6px 12px var(--color-neu-shadow-light);
  padding: 1.25rem 1.5rem;
  margin-top: 1.5rem;
}
.table-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}
.table-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--color-neu-text);
}
.table-count {
  font-size: 0.75rem;
  color: var(--color-neu-text-muted);
  padding: 2px 8px;
  border-radius: 8px;
  background: var(--color-neu-bg);
  box-shadow:
    inset 2px 2px 4px var(--color-neu-shadow-dark),
    inset -2px -2px 4px var(--color-neu-shadow-light);
}
:deep(.model-cell) {
  font-weight: 600;
  color: var(--color-neu-accent);
}
</style>
