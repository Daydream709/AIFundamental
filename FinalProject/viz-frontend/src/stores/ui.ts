/**
 * UI state store: sidebar collapse, current theme, etc.
 */
import { defineStore } from "pinia";
import { useStorage } from "@vueuse/core";

export const useUIStore = defineStore("ui", () => {
  const sidebarCollapsed = useStorage<boolean>("ui:sidebar", false);
  const chartAccent = useStorage<string>("ui:accent", "blue"); // blue | warm | mixed

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value;
  }

  return {
    sidebarCollapsed,
    chartAccent,
    toggleSidebar,
  };
});
