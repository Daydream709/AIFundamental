<template>
  <div class="chart-container">
    <div v-if="loading" class="chart-loading">
      <n-spin size="medium" />
      <span>加载中...</span>
    </div>
    <v-chart
      v-else-if="hasData"
      :option="option"
      :autoresize="autoresize"
      :update-options="{ notMerge: true }"
      :style="{ height: heightCss, width: '100%' }"
    />
    <div
      v-else
      class="chart-empty"
      :style="{ minHeight: heightCss }"
    >
      <span class="chart-empty-icon">📊</span>
      <span class="chart-empty-text">{{ emptyMessage }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { NSpin } from "naive-ui";
import VChart from "vue-echarts";
import type { EChartsOption } from "echarts";

const props = withDefaults(
  defineProps<{
    option: EChartsOption;
    loading?: boolean;
    autoresize?: boolean;
    minHeight?: number;
    emptyMessage?: string;
  }>(),
  {
    loading: false,
    autoresize: true,
    minHeight: 400,
    emptyMessage: "暂无数据 — 请调整筛选条件或先生成实验数据",
  }
);

const heightCss = computed(() => `${props.minHeight}px`);

const hasData = computed(() => {
  const series = (props.option as any).series;
  if (!series) return false;
  if (Array.isArray(series)) {
    return series.length > 0;
  }
  return true;
});
</script>

<style scoped>
.chart-container {
  background: #ffffff;
  border-radius: 16px;
  box-shadow:
    6px 6px 12px var(--color-neu-shadow-dark),
    -6px -6px 12px var(--color-neu-shadow-light);
  padding: 1rem 1.25rem 1.25rem;
  min-height: 400px;
  position: relative;
}
.chart-loading,
.chart-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  min-height: 400px;
  color: var(--color-neu-text-muted);
  font-size: 0.9rem;
}
.chart-empty-icon {
  font-size: 2.5rem;
  opacity: 0.4;
}
</style>
