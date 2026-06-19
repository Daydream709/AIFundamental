<template>
  <n-config-provider :theme-overrides="neumorphismOverrides">
    <n-message-provider>
      <n-dialog-provider>
        <div class="app-shell">
          <AppSidebar />
          <main class="app-main">
            <AppHeader />
            <div class="app-content">
              <router-view v-slot="{ Component, route }">
                <transition name="page" mode="out-in">
                  <component :is="Component" :key="route.path" />
                </transition>
              </router-view>
            </div>
          </main>
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { NConfigProvider, NMessageProvider, NDialogProvider } from "naive-ui";
import AppSidebar from "@/components/layout/AppSidebar.vue";
import AppHeader from "@/components/layout/AppHeader.vue";
import { neumorphismOverrides } from "@/styles/naive-overrides";
import { useDataStore } from "@/stores/data";

const dataStore = useDataStore();

onMounted(async () => {
  await dataStore.loadAll();
});
</script>

<style>
.app-shell {
  display: flex;
  min-height: 100vh;
  background: var(--color-neu-bg);
}
.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0; /* prevent flex overflow */
}
.app-content {
  flex: 1;
  padding: 1.5rem 2rem 2.5rem;
  overflow-x: hidden;
}

/* Page transition */
.page-enter-active,
.page-leave-active {
  transition: opacity 250ms ease-out, transform 250ms ease-out;
}
.page-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
.page-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
@media (prefers-reduced-motion: reduce) {
  .page-enter-active,
  .page-leave-active {
    transition: none;
  }
  .page-enter-from,
  .page-leave-to {
    transform: none;
  }
}
</style>
