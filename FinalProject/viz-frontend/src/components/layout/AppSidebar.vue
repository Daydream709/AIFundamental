<template>
  <aside class="sidebar" :class="{ collapsed: uiStore.sidebarCollapsed }">
    <div class="brand">
      <div class="brand-icon">📊</div>
      <transition name="fade">
        <div v-if="!uiStore.sidebarCollapsed" class="brand-text">
          <div class="brand-title">TS Forecaster</div>
          <div class="brand-sub">Neumorphism v1.0</div>
        </div>
      </transition>
    </div>

    <nav class="nav">
      <router-link
        to="/"
        class="nav-item press-neu"
        :class="{ active: route.path === '/' }"
      >
        <span class="nav-icon">🏠</span>
        <transition name="fade">
          <span v-if="!uiStore.sidebarCollapsed" class="nav-label">Overview</span>
        </transition>
      </router-link>

      <div v-if="!uiStore.sidebarCollapsed" class="nav-section">实验主线</div>
      <router-link
        v-for="line in lineList"
        :key="line.number"
        :to="line.route"
        class="nav-item press-neu"
        :class="{ active: route.path === line.route }"
      >
        <span class="nav-icon">{{ line.icons.join(" ") }}</span>
        <transition name="fade">
          <span v-if="!uiStore.sidebarCollapsed" class="nav-label">
            <span class="nav-label-num">Line {{ line.number }}</span>
            <span class="nav-label-text">{{ line.title }}</span>
          </span>
        </transition>
      </router-link>
    </nav>

    <button class="collapse-btn press-neu" @click="uiStore.toggleSidebar()">
      <span>{{ uiStore.sidebarCollapsed ? "›" : "‹" }}</span>
    </button>
  </aside>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import { LINES } from "@/data/lines";
import { useUIStore } from "@/stores/ui";

const route = useRoute();
const uiStore = useUIStore();
const lineList = computed(() => Object.values(LINES));
</script>

<style scoped>
.sidebar {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  padding: 1.5rem 1rem;
  background: var(--color-neu-bg);
  border-right: 1px solid transparent;
  box-shadow: 6px 0 16px var(--color-neu-shadow-dark);
  transition: width 250ms ease-out;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}
.sidebar.collapsed {
  width: 80px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem 1.5rem;
  border-bottom: 1px solid rgba(184, 188, 200, 0.2);
  margin-bottom: 1rem;
}
.brand-icon {
  font-size: 2rem;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  background: var(--color-neu-bg);
  box-shadow:
    4px 4px 8px var(--color-neu-shadow-dark),
    -4px -4px 8px var(--color-neu-shadow-light);
  flex-shrink: 0;
}
.brand-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--color-neu-text);
  line-height: 1.2;
}
.brand-sub {
  font-size: 0.7rem;
  color: var(--color-neu-text-muted);
  margin-top: 2px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  flex: 1;
}
.nav-section {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-neu-text-muted);
  padding: 1rem 0.75rem 0.4rem;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.7rem 0.85rem;
  border-radius: 12px;
  text-decoration: none;
  color: var(--color-neu-text);
  font-size: 0.9rem;
  font-weight: 500;
  background: var(--color-neu-bg);
  box-shadow:
    3px 3px 6px var(--color-neu-shadow-dark),
    -3px -3px 6px var(--color-neu-shadow-light);
  transition: all 200ms ease-out;
}
.nav-item:hover {
  transform: translateY(-1px);
}
.nav-item.active {
  box-shadow:
    inset 3px 3px 6px var(--color-neu-shadow-dark),
    inset -3px -3px 6px var(--color-neu-shadow-light);
  color: var(--color-neu-accent);
  font-weight: 600;
}
.nav-icon {
  font-size: 1.2rem;
  width: 28px;
  text-align: center;
  flex-shrink: 0;
}
.nav-label {
  display: flex;
  flex-direction: column;
  gap: 1px;
  white-space: nowrap;
  overflow: hidden;
}
.nav-label-num {
  font-size: 0.7rem;
  color: var(--color-neu-text-muted);
  font-weight: 500;
}
.nav-label-text {
  font-size: 0.9rem;
}

.collapse-btn {
  align-self: flex-end;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: var(--color-neu-bg);
  color: var(--color-neu-text-muted);
  font-size: 1.2rem;
  cursor: pointer;
  box-shadow:
    3px 3px 6px var(--color-neu-shadow-dark),
    -3px -3px 6px var(--color-neu-shadow-light);
  margin-top: 1rem;
}
.collapse-btn:active {
  box-shadow:
    inset 2px 2px 4px var(--color-neu-shadow-dark),
    inset -2px -2px 4px var(--color-neu-shadow-light);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 150ms ease-out;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
