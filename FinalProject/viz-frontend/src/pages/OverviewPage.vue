<template>
  <div class="overview-page">
    <!-- Top KPI row -->
    <div class="kpi-row">
      <KPICard
        title="总实验数"
        :value="String(totalRuns)"
        icon="🧪"
        :sub="`${dataStore.fileStatus.main === 'ok' ? '数据已加载' : '数据缺失'}`"
      />
      <KPICard
        title="模型数"
        :value="String(nModels)"
        icon="🤖"
        :sub="`${nDatasets} 个数据集`"
      />
      <KPICard
        title="最优 MSE"
        :value="bestMSE"
        icon="🏆"
        :sub="bestModel"
      />
      <KPICard
        title="缓存状态"
        :value="cacheStatus"
        icon="🟢"
        :sub="lastLoadedText"
      />
    </div>

    <!-- File status -->
    <div class="section">
      <h2 class="section-title">📁 数据文件状态</h2>
      <div class="status-row">
        <div
          v-for="(status, name) in dataStore.fileStatus"
          :key="name"
          class="status-card"
        >
          <span class="status-emoji">{{ statusEmoji[status] }}</span>
          <span class="status-name">{{ name }}</span>
          <span class="status-label">{{ statusLabel[status] }}</span>
        </div>
      </div>
    </div>

    <!-- 4 line entries -->
    <div class="section">
      <h2 class="section-title">🧭 实验主线</h2>
      <p class="section-desc">从左侧菜单或下方卡片进入各主线详情页</p>
      <div class="line-grid">
        <router-link
          v-for="line in lineList"
          :key="line.number"
          :to="line.route"
          class="line-card press-neu"
        >
          <div class="line-card-icon">{{ line.icons.join(" ") }}</div>
          <div class="line-card-body">
            <div class="line-card-num">Line {{ line.number }}</div>
            <div class="line-card-title">{{ line.title }}</div>
            <div class="line-card-desc">{{ line.description }}</div>
            <div class="line-card-meta">
              <span class="meta-chip">{{ line.models.length }} 模型</span>
              <span class="meta-chip">{{ line.datasets.length }} 数据集</span>
              <span class="meta-chip">{{ line.predLens.length }} 预测长度</span>
            </div>
          </div>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { KPICard } from "@/components/cards";
import { useDataStore } from "@/stores/data";
import { LINES } from "@/data/lines";
import { formatMetric } from "@/composables/useMetricFormat";
import { displayModel } from "@/types/results";

const dataStore = useDataStore();
const lineList = Object.values(LINES);
const tick = ref(0);

onMounted(async () => {
  await dataStore.loadAll();
});

const totalRuns = computed(() => dataStore.mergedMain.length);
const nModels = computed(() => new Set(dataStore.mergedMain.map((r) => r.model)).size);
const nDatasets = computed(() => new Set(dataStore.mergedMain.map((r) => r.dataset)).size);

const bestMSE = computed(() => {
  tick.value;
  if (dataStore.mergedMain.length === 0) return "—";
  const v = dataStore.mergedMain.reduce((min, r) => (r.MSE !== undefined && r.MSE < min ? r.MSE : min), Infinity);
  return v === Infinity ? "—" : formatMetric("MSE", v);
});
const bestModel = computed(() => {
  if (dataStore.mergedMain.length === 0) return "—";
  const best = dataStore.mergedMain.reduce((b, r) => (r.MSE !== undefined && (b.MSE === undefined || r.MSE < b.MSE) ? r : b), {} as any);
  return best.model ? displayModel(best.model) : "—";
});

const cacheStatus = computed(() => (dataStore.loading ? "加载中" : "已就绪"));
const lastLoadedText = computed(() => {
  if (!dataStore.lastLoadedAt) return "尚未加载";
  const sec = Math.floor((Date.now() - dataStore.lastLoadedAt) / 1000);
  return sec < 60 ? `${sec}s 前` : `${Math.floor(sec / 60)}m 前`;
});

const statusEmoji: Record<string, string> = { ok: "✅", missing: "❌", empty: "⚠️" };
const statusLabel: Record<string, string> = { ok: "OK", missing: "缺失", empty: "空" };
</script>

<style scoped>
.overview-page {
  max-width: 1400px;
  margin: 0 auto;
}
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.section {
  margin-bottom: 2.5rem;
}
.section-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--color-neu-text);
  margin: 0 0 0.5rem;
}
.section-desc {
  font-size: 0.85rem;
  color: var(--color-neu-text-muted);
  margin: 0 0 1rem;
}
.status-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
}
.status-card {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.85rem 1rem;
  background: var(--color-neu-bg);
  border-radius: 14px;
  box-shadow:
    inset 3px 3px 6px var(--color-neu-shadow-dark),
    inset -3px -3px 6px var(--color-neu-shadow-light);
  font-size: 0.85rem;
}
.status-emoji {
  font-size: 1.1rem;
}
.status-name {
  font-weight: 600;
  color: var(--color-neu-text);
}
.status-label {
  margin-left: auto;
  color: var(--color-neu-text-muted);
  font-size: 0.78rem;
}
.line-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}
.line-card {
  display: flex;
  gap: 1rem;
  padding: 1.25rem;
  background: var(--color-neu-bg);
  border-radius: 16px;
  text-decoration: none;
  color: var(--color-neu-text);
  box-shadow:
    8px 8px 16px var(--color-neu-shadow-dark),
    -8px -8px 16px var(--color-neu-shadow-light);
}
.line-card:active {
  box-shadow:
    inset 4px 4px 8px var(--color-neu-shadow-dark),
    inset -4px -4px 8px var(--color-neu-shadow-light);
  transform: scale(0.98);
}
.line-card-icon {
  font-size: 2.5rem;
  line-height: 1;
}
.line-card-body {
  flex: 1;
  min-width: 0;
}
.line-card-num {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-neu-text-muted);
  font-weight: 600;
}
.line-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 4px 0 6px;
}
.line-card-desc {
  font-size: 0.78rem;
  color: var(--color-neu-text-muted);
  line-height: 1.5;
  margin-bottom: 0.5rem;
}
.line-card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.meta-chip {
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 8px;
  background: var(--color-neu-bg);
  color: var(--color-neu-text);
  box-shadow:
    2px 2px 4px var(--color-neu-shadow-dark),
    -2px -2px 4px var(--color-neu-shadow-light);
}
</style>
