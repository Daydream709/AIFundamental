<template>
  <div class="empty-state">
    <div class="empty-icon">{{ icon }}</div>
    <h3 class="empty-title">{{ title }}</h3>
    <p class="empty-msg" v-html="message" />
    <slot />
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  message: string;
  icon?: string;
}>();
</script>

<style scoped>
.empty-state {
  text-align: center;
  padding: 3rem 2rem;
  background: var(--color-neu-bg);
  border-radius: 18px;
  border: 2px dashed var(--color-neu-shadow-dark);
  box-shadow:
    inset 6px 6px 12px var(--color-neu-shadow-dark),
    inset -6px -6px 12px var(--color-neu-shadow-light);
  color: var(--color-neu-text);
  max-width: 640px;
  margin: 2rem auto;
}
.empty-icon {
  font-size: 3rem;
  margin-bottom: 0.75rem;
  opacity: 0.85;
}
.empty-title {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--color-neu-text);
  margin: 0 0 0.5rem;
}
.empty-msg {
  font-size: 0.95rem;
  color: var(--color-neu-text-muted);
  line-height: 1.6;
  margin: 0;
}
.empty-msg :deep(code) {
  background: rgba(184, 188, 200, 0.3);
  color: var(--color-neu-text);
  padding: 2px 8px;
  border-radius: 6px;
  font-family: "Monaco", "Menlo", monospace;
  font-size: 0.85em;
}
</style>
