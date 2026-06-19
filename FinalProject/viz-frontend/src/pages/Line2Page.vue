<template>
  <LinePageLayout :line="LINES[2]" />
</template>

<script setup lang="ts">
import LinePageLayout from "@/components/layout/LinePageLayout.vue";
import { LINES } from "@/data/lines";
</script>
