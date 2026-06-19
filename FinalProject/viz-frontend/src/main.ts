import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { router } from "./router";
import "./styles/main.css";

// Register ECharts components globally
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import {
  BarChart,
  LineChart,
  RadarChart,
  HeatmapChart,
  ScatterChart,
  BoxplotChart,
  ParallelChart,
} from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  PolarComponent,
  VisualMapComponent,
  ParallelComponent,
  MarkLineComponent,
  DatasetComponent,
} from "echarts/components";

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  RadarChart,
  HeatmapChart,
  ScatterChart,
  BoxplotChart,
  ParallelChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  PolarComponent,
  VisualMapComponent,
  ParallelComponent,
  MarkLineComponent,
  DatasetComponent,
]);

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
