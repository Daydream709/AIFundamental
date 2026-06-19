<template>
  <div class="filter-bar">
    <div class="filter-grid">
      <slot />
    </div>
    <div v-if="showReset" class="filter-actions">
      <n-button size="small" tertiary @click="$emit('reset')">
        🔄 重置
      </n-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { NButton } from "naive-ui";

withDefaults(
  defineProps<{
    showReset?: boolean;
  }>(),
  { showReset: true }
);
defineEmits<{ reset: [] }>();
</script>

<style scoped>
.filter-bar {
  background: var(--color-neu-bg);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow:
    8px 8px 16px var(--color-neu-shadow-dark),
    -8px -8px 16px var(--color-neu-shadow-light);
  margin-bottom: 1.5rem;
}
.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem 1rem;
}
.filter-actions {
  margin-top: 0.75rem;
  display: flex;
  justify-content: flex-end;
}
</style>
