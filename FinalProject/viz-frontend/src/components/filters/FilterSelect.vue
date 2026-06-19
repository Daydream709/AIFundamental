<template>
  <div class="filter-cell">
    <label class="filter-label">
      <span v-if="icon" class="filter-icon">{{ icon }}</span>
      <span>{{ label }}</span>
    </label>
    <n-select
      v-if="multiple"
      multiple
      :value="modelValue"
      :options="options"
      :placeholder="placeholder"
      :max-tag-count="3"
      size="small"
      @update:value="$emit('update:modelValue', $event)"
    />
    <n-select
      v-else
      :value="modelValue"
      :options="options"
      :placeholder="placeholder"
      size="small"
      @update:value="$emit('update:modelValue', $event)"
    />
  </div>
</template>

<script setup lang="ts">
import { NSelect } from "naive-ui";

defineProps<{
  label: string;
  icon?: string;
  options: { label: string; value: string | number }[];
  modelValue: string | number | (string | number)[] | null;
  multiple?: boolean;
  placeholder?: string;
}>();

defineEmits<{ "update:modelValue": [v: any] }>();
</script>

<style scoped>
.filter-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.filter-label {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-neu-text-muted);
  font-weight: 600;
}
.filter-icon {
  font-size: 0.85rem;
}
</style>
