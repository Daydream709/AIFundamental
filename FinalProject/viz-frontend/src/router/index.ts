import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";
import OverviewPage from "@/pages/OverviewPage.vue";
import Line1Page from "@/pages/Line1Page.vue";
import Line2Page from "@/pages/Line2Page.vue";
import Line3Page from "@/pages/Line3Page.vue";
import Line4aPage from "@/pages/Line4aPage.vue";
import Line4bPage from "@/pages/Line4bPage.vue";
import { LINES } from "@/data/lines";

/**
 * Static page-component map. Vite's dynamic-import-vars plugin requires
 * the file extension (.vue) to be in the static part of the import path,
 * so we can't use `${line.pageFile}` directly. Static imports + this map
 * keep the build warning-free.
 */
const PAGE_COMPONENTS: Record<number, () => Promise<unknown>> = {
  1: () => import("@/pages/Line1Page.vue"),
  2: () => import("@/pages/Line2Page.vue"),
  3: () => import("@/pages/Line3Page.vue"),
  4: () => import("@/pages/Line4aPage.vue"),
  5: () => import("@/pages/Line4bPage.vue"),
};

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "overview",
    component: OverviewPage,
    meta: { title: "项目总览" },
  },
  ...Object.values(LINES).map((line) => ({
    path: line.route,
    name: `line-${line.number}`,
    component: PAGE_COMPONENTS[line.number]!,
    meta: { title: line.title, line },
  })),
  { path: "/:pathMatch(.*)*", redirect: "/" },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
