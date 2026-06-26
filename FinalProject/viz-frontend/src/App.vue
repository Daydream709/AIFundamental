<template>
  <n-config-provider :theme-overrides="neumorphismOverrides">
    <n-message-provider>
      <n-dialog-provider>
        <div v-if="fatalError" class="fatal-error-banner" role="alert">
          <span>⚠️ 界面出现错误:</span>
          <span class="err-msg" :title="fatalError">{{ fatalError }}</span>
          <button type="button" @click="reset">清除缓存并重载</button>
        </div>
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
import { onErrorCaptured, onMounted, ref } from "vue";
import { NConfigProvider, NMessageProvider, NDialogProvider } from "naive-ui";
import AppSidebar from "@/components/layout/AppSidebar.vue";
import AppHeader from "@/components/layout/AppHeader.vue";
import { neumorphismOverrides } from "@/styles/naive-overrides";
import { useDataStore } from "@/stores/data";

const dataStore = useDataStore();

// Global error capture. BUGFIX: previously an uncaught error inside
// a child component (e.g. echarts option builder, CSV parser) would
// tear down the whole <App> tree and the user saw a permanently blank
// page. Now we surface the error in a banner and let the user reload
// or clear localStorage to recover.
const fatalError = ref<string | null>(null);
onErrorCaptured((err) => {
  console.error("[App] captured error:", err);
  fatalError.value = err instanceof Error ? err.message : String(err);
  return false; // don't propagate further
});

onMounted(async () => {
  try {
    await dataStore.loadAll();
  } catch (e) {
    console.error("[App] loadAll failed:", e);
    fatalError.value = e instanceof Error ? e.message : String(e);
  }
});

function reset() {
  // Best-effort recovery: clear filters and localStorage, then reload.
  try {
    localStorage.clear();
  } catch {}
  location.reload();
}
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

/* Fatal-error banner shown if any descendant errored. Without this the
   user just sees a blank page and has no idea what to do. */
.fatal-error-banner {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 9999;
  background: #fee2e2;
  border-bottom: 2px solid #dc2626;
  padding: 1rem 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  color: #991b1b;
  font-size: 0.9rem;
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.2);
}
.fatal-error-banner .err-msg {
  flex: 1;
  font-family: monospace;
  font-size: 0.8rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fatal-error-banner button {
  padding: 0.4rem 0.8rem;
  border: 1px solid #dc2626;
  background: white;
  color: #dc2626;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
}
.fatal-error-banner button:hover {
  background: #fef2f2;
}
</style>
