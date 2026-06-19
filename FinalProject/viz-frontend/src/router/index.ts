import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";
import { LINES } from "@/data/lines";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "overview",
    component: () => import("@/pages/OverviewPage.vue"),
    meta: { title: "项目总览" },
  },
  ...Object.values(LINES).map((line) => ({
    path: line.route,
    name: `line-${line.number}`,
    component: () => import(`@/pages/Line${line.number}Page.vue`),
    meta: { title: line.title, line },
  })),
  { path: "/:pathMatch(.*)*", redirect: "/" },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
