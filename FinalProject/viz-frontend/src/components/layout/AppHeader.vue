<template>
  <header class="app-header">
    <div class="header-left">
      <h1 class="header-title">{{ pageTitle }}</h1>
      <p v-if="pageSubtitle" class="header-subtitle">{{ pageSubtitle }}</p>
    </div>
    <div class="header-right">
      <button
        class="header-btn press-neu"
        title="重新加载数据"
        @click="dataStore.refresh()"
      >
        <span class="header-btn-icon">🔄</span>
        <span class="header-btn-text">同步数据</span>
      </button>
      <div class="data-badge" :class="`badge-${dataStore.fileStatus.main}`">
        <span class="badge-dot" />
        <span class="badge-text">main: {{ dataStore.fileStatus.main }}</span>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import { useDataStore } from "@/stores/data";

const route = useRoute();
const dataStore = useDataStore();

const pageTitle = computed(() => (route.meta.title as string) ?? "Dashboard");
const pageSubtitle = computed(() => {
  const line = route.meta.line as { subtitle?: string } | undefined;
  return line?.subtitle ?? "";
});
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem 2rem 1rem;
  gap: 1rem;
}
.header-left {
  flex: 1;
  min-width: 0;
}
.header-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-neu-text);
  margin: 0;
  line-height: 1.2;
}
.header-subtitle {
  font-size: 0.85rem;
  color: var(--color-neu-text-muted);
  margin: 4px 0 0;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.header-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 12px;
  background: var(--color-neu-bg);
  color: var(--color-neu-text);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow:
    4px 4px 8px var(--color-neu-shadow-dark),
    -4px -4px 8px var(--color-neu-shadow-light);
  font-family: var(--font-sans);
}
.header-btn:active {
  box-shadow:
    inset 3px 3px 6px var(--color-neu-shadow-dark),
    inset -3px -3px 6px var(--color-neu-shadow-light);
}
.data-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.75rem;
  border-radius: 10px;
  background: var(--color-neu-bg);
  box-shadow:
    inset 2px 2px 4px var(--color-neu-shadow-dark),
    inset -2px -2px 4px var(--color-neu-shadow-light);
  font-size: 0.75rem;
  color: var(--color-neu-text-muted);
  font-weight: 500;
}
.badge-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #94a3b8;
}
.badge-ok .badge-dot {
  background: #10b981;
}
.badge-missing .badge-dot {
  background: #ef4444;
}
.badge-empty .badge-dot {
  background: #f59e0b;
}
</style>
