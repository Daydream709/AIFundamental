<template>
  <div class="kpi-card hover-lift">
    <div class="kpi-title">
      <span v-if="icon" class="kpi-icon">{{ icon }}</span>
      <span>{{ title }}</span>
    </div>
    <div class="kpi-value">{{ value }}</div>
    <div v-if="sub" class="kpi-sub">{{ sub }}</div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  value: string;
  sub?: string;
  icon?: string;
}>();
</script>

<style scoped>
.kpi-card {
  background: var(--color-neu-bg);
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  box-shadow:
    8px 8px 16px var(--color-neu-shadow-dark),
    -8px -8px 16px var(--color-neu-shadow-light);
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 110px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 30%;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.5) 0%, transparent 60%);
  pointer-events: none;
  border-radius: 16px 16px 0 0;
}
.kpi-title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-neu-text-muted);
  font-weight: 600;
  position: relative;
}
.kpi-icon {
  font-size: 0.95rem;
}
.kpi-value {
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--color-neu-accent);
  line-height: 1.1;
  position: relative;
  font-variant-numeric: tabular-nums;
}
.kpi-sub {
  font-size: 0.78rem;
  color: var(--color-neu-text-muted);
  position: relative;
}
</style>
