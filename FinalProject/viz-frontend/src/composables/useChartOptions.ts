/**
 * Build ECharts option objects from raw data + filter state.
 *
 * Each chart kind has its own builder. All builders return a base
 * option (transparent background, flat color palette) that the
 * EChartWrapper renders inside a flat white card with neumorphism
 * rounded corners.
 */
import type { EChartsOption } from "echarts";
import type { AnyResultRow, MetricName } from "@/types/results";

// Flat color palette (high contrast for data readability)
const PALETTE = [
  "#0080FF",
  "#FF8500",
  "#3D4852",
  "#6C63FF",
  "#10B981",
  "#F43F5E",
  "#0EA5E9",
  "#A855F7",
];

const BASE_FONT = '"Plus Jakarta Sans", system-ui, sans-serif';
const TEXT_COLOR = "#3d4852";
const GRID_COLOR = "rgba(61, 72, 82, 0.1)";
const AXIS_COLOR = "#94a3b8";

function baseAxis() {
  return {
    axisLine: { lineStyle: { color: AXIS_COLOR } },
    axisLabel: { color: TEXT_COLOR, fontFamily: BASE_FONT },
    splitLine: { lineStyle: { color: GRID_COLOR } },
  };
}

function baseOption(): EChartsOption {
  return {
    backgroundColor: "transparent",
    color: PALETTE,
    textStyle: { fontFamily: BASE_FONT, color: TEXT_COLOR },
    grid: { left: 60, right: 30, top: 60, bottom: 50, containLabel: true },
    legend: { textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT }, top: 10 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255, 255, 255, 0.98)",
      borderColor: "#b8bcc8",
      borderWidth: 1,
      textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 13 },
      axisPointer: { lineStyle: { color: "#6c63ff" } },
    },
  };
}

function emptyOption(title: string, message: string): EChartsOption {
  return {
    ...baseOption(),
    title: {
      text: title,
      left: "center",
      top: "middle",
      textStyle: { color: "#94a3b8", fontSize: 14, fontWeight: "normal" },
    },
    graphic: {
      type: "text",
      left: "center",
      bottom: "40%",
      style: { text: message, fill: "#94a3b8", fontSize: 12, fontFamily: BASE_FONT },
    },
  };
}

// -----------------------------------------------------------------
// Per-chart builders
// -----------------------------------------------------------------

function isAblationData(data: AnyResultRow[]): boolean {
  return data.length > 0 && "ablation" in data[0] && !("model" in data[0]);
}

function buildBar(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const ablation = isAblationData(data);
  if (ablation) {
    // Ablation: x = setting, group = ablation group
    const groups = Array.from(new Set(typed.map((r) => r.ablation)));
    const settings = Array.from(new Set(typed.map((r) => r.setting)));
    const series = groups.map((g) => ({
      name: g,
      type: "bar" as const,
      data: settings.map((s) => {
        const row = typed.find((r) => r.ablation === g && r.setting === s);
        return row?.[metric] ?? null;
      }),
    }));
    return {
      ...baseOption(),
      title: { text: `${metric} · Bar (by Setting)`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
      legend: { ...baseOption().legend, data: groups, top: 30 },
      xAxis: { ...baseAxis(), type: "category", data: settings, name: "Setting" },
      yAxis: { ...baseAxis(), type: "value", name: metric },
      series,
    };
  }
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const datasets = Array.from(new Set(typed.map((r) => r.dataset)));
  const series = datasets.map((ds) => ({
    name: ds,
    type: "bar" as const,
    data: models.map((m) => {
      const row = typed.find((r) => r.model === m && r.dataset === ds);
      return row?.[metric] ?? null;
    }),
  }));
  return {
    ...baseOption(),
    title: { text: `${metric} · Bar`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    legend: { ...baseOption().legend, data: datasets, top: 30 },
    xAxis: { ...baseAxis(), type: "category", data: models },
    yAxis: { ...baseAxis(), type: "value", name: metric },
    series,
  };
}

function buildLine(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const ablation = isAblationData(data);
  if (ablation) {
    // Ablation line: one line per ablation group, x = setting order
    const groups = Array.from(new Set(typed.map((r) => r.ablation)));
    const settings = Array.from(new Set(typed.map((r) => r.setting)));
    const series = groups.map((g) => ({
      name: g,
      type: "line" as const,
      smooth: true,
      symbolSize: 8,
      lineStyle: { width: 2.5 },
      data: settings.map((s) => {
        const row = typed.find((r) => r.ablation === g && r.setting === s);
        return row?.[metric] ?? null;
      }),
    }));
    return {
      ...baseOption(),
      title: { text: `${metric} vs Setting`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
      legend: { ...baseOption().legend, data: groups, top: 30 },
      xAxis: { ...baseAxis(), type: "category", data: settings, name: "Setting" },
      yAxis: { ...baseAxis(), type: "value", name: metric },
      series,
    };
  }
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const predLens = Array.from(new Set(typed.map((r) => r.pred_len))).sort((a, b) => a - b);
  const series = models.map((m) => ({
    name: m,
    type: "line" as const,
    smooth: true,
    symbolSize: 8,
    lineStyle: { width: 2.5 },
    data: predLens.map((p) => {
      const rows = typed.filter((r) => r.model === m && r.pred_len === p);
      if (rows.length === 0) return null;
      const avg = rows.reduce((s, r) => s + (r[metric] ?? 0), 0) / rows.length;
      return Number(avg.toFixed(6));
    }),
  }));
  return {
    ...baseOption(),
    title: { text: `${metric} vs Pred Length`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    legend: { ...baseOption().legend, data: models, top: 30 },
    xAxis: { ...baseAxis(), type: "category", data: predLens.map(String), name: "pred_len" },
    yAxis: { ...baseAxis(), type: "value", name: metric },
    series,
  };
}

function buildRadar(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const dims = [
    { name: "Accuracy", max: 100 },
    { name: "Speed", max: 100 },
    { name: "Compactness", max: 100 },
    { name: "Robustness", max: 100 },
    { name: "Coverage", max: 100 },
  ];

  const perModel = models.map((m) => {
    const rows = typed.filter((r) => r.model === m);
    const mse = rows.reduce((s, r) => s + (r.MSE ?? 1), 0) / rows.length || 1;
    const infer = rows.reduce((s, r) => s + (r["InferTime(ms)"] ?? 100), 0) / rows.length || 100;
    const params = rows.reduce((s, r) => s + (r["Params(M)"] ?? 10), 0) / rows.length || 10;
    const flops = rows.reduce((s, r) => s + (r["FLOPs(G)"] ?? 10), 0) / rows.length || 10;
    const mseStd = rows.length > 1 ? rows.reduce((s, r) => s + Math.pow((r.MSE ?? 0) - mse, 2), 0) / rows.length : 0;
    const robustness = Math.max(0, 100 - Math.sqrt(mseStd) * 50);
    return { model: m, mse, infer, params, flops, robustness };
  });

  const mins = {
    mse: Math.min(...perModel.map((p) => p.mse)),
    infer: Math.min(...perModel.map((p) => p.infer)),
    params: Math.min(...perModel.map((p) => p.params)),
    flops: Math.min(...perModel.map((p) => p.flops)),
  };
  const maxs = {
    mse: Math.max(...perModel.map((p) => p.mse)),
    infer: Math.max(...perModel.map((p) => p.infer)),
    params: Math.max(...perModel.map((p) => p.params)),
    flops: Math.max(...perModel.map((p) => p.flops)),
  };
  const norm = (v: number, lo: number, hi: number) => (hi > lo ? (100 * (hi - v)) / (hi - lo) : 50);

  return {
    ...baseOption(),
    title: { text: "Model Fingerprint (5-Dim Radar)", left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: { trigger: "item", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
    legend: { ...baseOption().legend, data: models, top: 30 },
    radar: {
      indicator: dims,
      center: ["50%", "58%"],
      radius: 110,
      axisName: { color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 12 },
      splitLine: { lineStyle: { color: GRID_COLOR } },
      splitArea: { areaStyle: { color: ["rgba(224, 229, 236, 0.3)", "rgba(224, 229, 236, 0.1)"] } },
    },
    series: [
      {
        type: "radar",
        data: perModel.map((p) => ({
          name: p.model,
          value: [
            norm(p.mse, mins.mse, maxs.mse),
            norm(p.infer, mins.infer, maxs.infer),
            norm(p.params, mins.params, maxs.params),
            p.robustness,
            norm(p.flops, mins.flops, maxs.flops),
          ],
        })),
        areaStyle: { opacity: 0.2 },
        lineStyle: { width: 2 },
      },
    ],
  };
}

function buildHeatmap(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const ablation = isAblationData(data);
  if (ablation) {
    const groups = Array.from(new Set(typed.map((r) => r.ablation)));
    const settings = Array.from(new Set(typed.map((r) => r.setting)));
    const matrix: [number, number, number][] = [];
    for (let i = 0; i < groups.length; i++) {
      for (let j = 0; j < settings.length; j++) {
        const row = typed.find((r) => r.ablation === groups[i] && r.setting === settings[j]);
        const v = row?.[metric];
        matrix.push([j, i, v !== undefined && v !== null ? Number(v) : NaN]);
      }
    }
    const values = matrix.map((m) => m[2]).filter((v) => !Number.isNaN(v));
    const minV = values.length ? Math.min(...values) : 0;
    const maxV = values.length ? Math.max(...values) : 1;
    return {
      ...baseOption(),
      title: { text: `${metric} Heatmap (Ablation × Setting)`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
      tooltip: { position: "top", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
      grid: { left: 100, right: 30, top: 60, bottom: 60 },
      xAxis: { ...baseAxis(), type: "category", data: settings, name: "Setting" },
      yAxis: { ...baseAxis(), type: "category", data: groups, name: "Ablation" },
      visualMap: {
        min: minV,
        max: maxV,
        calculable: true,
        orient: "vertical",
        right: 0,
        top: "center",
        inRange: { color: ["#10B981", "#FFD700", "#F43F5E"] },
        textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT },
      },
      series: [
        {
          type: "heatmap",
          data: matrix,
          label: { show: true, color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 11, formatter: (p: any) => Number(p.value[2]).toFixed(3) },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.2)" } },
        },
      ],
    };
  }
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const datasets = Array.from(new Set(typed.map((r) => r.dataset)));
  const matrix: [number, number, number][] = [];
  for (let i = 0; i < models.length; i++) {
    for (let j = 0; j < datasets.length; j++) {
      const row = typed.find((r) => r.model === models[i] && r.dataset === datasets[j]);
      const v = row?.[metric];
      matrix.push([j, i, v !== undefined && v !== null ? Number(v) : NaN]);
    }
  }
  const values = matrix.map((m) => m[2]).filter((v) => !Number.isNaN(v));
  const minV = values.length ? Math.min(...values) : 0;
  const maxV = values.length ? Math.max(...values) : 1;
  return {
    ...baseOption(),
    title: { text: `${metric} Heatmap (Model × Dataset)`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: { position: "top", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
    grid: { left: 80, right: 30, top: 60, bottom: 60 },
    xAxis: { ...baseAxis(), type: "category", data: datasets, name: "Dataset" },
    yAxis: { ...baseAxis(), type: "category", data: models, name: "Model" },
    visualMap: {
      min: minV,
      max: maxV,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "center",
      inRange: { color: ["#10B981", "#FFD700", "#F43F5E"] },
      textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT },
    },
    series: [
      {
        type: "heatmap",
        data: matrix,
        label: { show: true, color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 11, formatter: (p: any) => Number(p.value[2]).toFixed(3) },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.2)" } },
      },
    ],
  };
}

function buildPareto(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const points = models.map((m, i) => {
    const rows = typed.filter((r) => r.model === m);
    const x = rows.reduce((s, r) => s + (r["InferTime(ms)"] ?? 0), 0) / rows.length;
    const y = rows.reduce((s, r) => s + (r[metric] ?? 0), 0) / rows.length;
    return {
      name: m,
      value: [Number(x.toFixed(3)), Number(y.toFixed(6))],
      itemStyle: { color: PALETTE[i % PALETTE.length] },
    };
  });

  const sorted = [...points].sort((a, b) => a.value[0] - b.value[0]);
  const front: typeof sorted = [];
  let bestY = Infinity;
  for (const p of sorted) {
    if (p.value[1] <= bestY) {
      front.push(p);
      bestY = p.value[1];
    }
  }

  return {
    ...baseOption(),
    title: { text: `Pareto: ${metric} vs Inference Time`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(255, 255, 255, 0.98)",
      borderColor: "#b8bcc8",
      textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT },
      formatter: (p: any) => `<b>${p.name}</b><br>${metric}: ${p.value[1]}<br>InferTime: ${p.value[0]}ms`,
    },
    grid: { left: 70, right: 30, top: 60, bottom: 50 },
    xAxis: { ...baseAxis(), type: "value", name: "Inference Time (ms)" },
    yAxis: { ...baseAxis(), type: "value", name: metric },
    series: [
      {
        type: "scatter",
        symbolSize: 22,
        data: points,
        label: { show: true, position: "top", color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 11, formatter: (p: any) => p.name },
      },
      {
        type: "line",
        smooth: false,
        symbol: "none",
        data: front.map((p) => p.value),
        lineStyle: { color: "#6c63ff", width: 2, type: "dashed" },
        name: "Pareto Front",
      },
    ],
  };
}

function buildBox(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const ablation = isAblationData(data);
  if (ablation) {
    // Ablation box: one box per ablation group
    const groups = Array.from(new Set(typed.map((r) => r.ablation)));
    const boxData = groups.map((g) => {
      const values = typed.filter((r) => r.ablation === g).map((r) => r[metric]).filter((v) => v != null) as number[];
      if (values.length === 0) return [0, 0, 0, 0, 0];
      const sorted = [...values].sort((a, b) => a - b);
      const q = (p: number) => sorted[Math.floor((sorted.length - 1) * p)] ?? 0;
      return [sorted[0], q(0.25), q(0.5), q(0.75), sorted[sorted.length - 1]];
    });
    return {
      ...baseOption(),
      title: { text: `${metric} Distribution by Ablation Group`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
      tooltip: { trigger: "item", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
      grid: { left: 100, right: 30, top: 50, bottom: 50 },
      xAxis: { ...baseAxis(), type: "value", name: metric },
      yAxis: { ...baseAxis(), type: "category", data: groups, name: "Ablation" },
      series: [
        {
          type: "boxplot",
          data: boxData,
          itemStyle: { color: "rgba(108, 99, 255, 0.5)", borderColor: "#6c63ff" },
        },
      ],
    };
  }
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const boxData = models.map((m) => {
    const values = typed.filter((r) => r.model === m).map((r) => r[metric]).filter((v) => v != null) as number[];
    if (values.length === 0) return [0, 0, 0, 0, 0];
    const sorted = [...values].sort((a, b) => a - b);
    const q = (p: number) => sorted[Math.floor((sorted.length - 1) * p)] ?? 0;
    return [sorted[0], q(0.25), q(0.5), q(0.75), sorted[sorted.length - 1]];
  });
  return {
    ...baseOption(),
    title: { text: `${metric} Distribution by Model`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: { trigger: "item", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
    grid: { left: 70, right: 30, top: 50, bottom: 50 },
    xAxis: { ...baseAxis(), type: "category", data: models },
    yAxis: { ...baseAxis(), type: "value", name: metric },
    series: [
      {
        type: "boxplot",
        data: boxData,
        itemStyle: { color: "rgba(108, 99, 255, 0.5)", borderColor: "#6c63ff" },
      },
    ],
  };
}

function buildWaterfall(data: AnyResultRow[]): EChartsOption {
  if (data.length === 0) return emptyOption("无消融数据", "请先运行 python run_ablation.py");
  const firstGroup = (data[0] as any).ablation ?? "Ablation";
  const sub = data.filter((r: any) => r.ablation === firstGroup);
  if (sub.length === 0) return emptyOption("无消融数据", "请先运行 python run_ablation.py");
  const baseline = (sub[0] as any).MSE ?? 0;
  const categories = sub.map((r: any) => r.setting);
  const deltas = sub.map((r: any) => ((r as any).MSE ?? 0) - baseline);
  const helper = [baseline, ...deltas.slice(1)];
  return {
    ...baseOption(),
    title: { text: `Ablation Waterfall: ${firstGroup}`, left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: { trigger: "axis", backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
    grid: { left: 60, right: 30, top: 60, bottom: 80 },
    xAxis: { ...baseAxis(), type: "category", data: categories, axisLabel: { ...baseAxis().axisLabel, rotate: 30 } },
    yAxis: { ...baseAxis(), type: "value", name: "MSE" },
    series: [
      { type: "bar", stack: "total", data: helper, itemStyle: { color: "transparent" }, silent: true },
      {
        type: "bar",
        stack: "total",
        data: deltas,
        label: { show: true, position: "top", color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 11, formatter: (p: any) => Number(p.value).toFixed(4) },
        itemStyle: { color: (p: any) => (p.value >= 0 ? "#F43F5E" : "#10B981") },
      },
    ],
  };
}

function buildParallel(data: AnyResultRow[], metric: MetricName): EChartsOption {
  const typed = data as any[];
  if (typed.length === 0) return emptyOption("无数据", "当前筛选条件无匹配数据");
  const models = Array.from(new Set(typed.map((r) => r.model)));
  const dimNames: MetricName[] = ["MSE", "MAE", "Params(M)", "FLOPs(G)", "InferTime(ms)"];
  const perModel = models.map((m) => {
    const rows = typed.filter((r) => r.model === m);
    return dimNames.map((dim) => {
      const vals = rows.map((r) => r[dim] ?? 0).filter((v) => v !== 0);
      if (vals.length === 0) return 0;
      return vals.reduce((s, v) => s + v, 0) / vals.length;
    });
  });
  return {
    ...baseOption(),
    title: { text: "Model Fingerprint (Parallel Coordinates)", left: "center", top: 0, textStyle: { color: TEXT_COLOR, fontSize: 16, fontWeight: 600 } },
    tooltip: { backgroundColor: "rgba(255, 255, 255, 0.98)", borderColor: "#b8bcc8", textStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT } },
    grid: { left: 60, right: 30, top: 60, bottom: 50 },
    parallelAxis: dimNames.map((dim, i) => ({
      dim: i,
      name: dim,
      nameTextStyle: { color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 12 },
      axisLine: { lineStyle: { color: AXIS_COLOR } },
      axisLabel: { color: TEXT_COLOR, fontFamily: BASE_FONT, fontSize: 10 },
    })),
    series: [
      {
        type: "parallel",
        lineStyle: { width: 2, opacity: 0.7 },
        data: perModel.map((values, i) => ({
          name: models[i],
          value: values,
          lineStyle: { color: PALETTE[i % PALETTE.length] },
        })),
      },
    ],
  };
}

// -----------------------------------------------------------------
// Public dispatcher
// -----------------------------------------------------------------

export function buildChartOption(
  kind: string,
  data: AnyResultRow[],
  metric: MetricName
): EChartsOption {
  const isAblation = isAblationData(data);

  switch (kind) {
    case "bar":
      return buildBar(data, metric);
    case "line":
      return buildLine(data, metric);
    case "radar":
      // Radar requires model aggregation — not meaningful for ablation
      if (isAblation) return emptyOption("雷达图不适用于消融", "请选择其他图表（瀑布/柱/线/热力/箱）");
      return buildRadar(data, metric);
    case "heatmap":
      return buildHeatmap(data, metric);
    case "pareto":
      // Pareto requires infer time — not in ablation data
      if (isAblation) return emptyOption("帕累托图不适用于消融", "请选择其他图表");
      return buildPareto(data, metric);
    case "box":
      return buildBox(data, metric);
    case "waterfall":
      if (isAblation) return buildWaterfall(data);
      return emptyOption("需要消融数据", "请选择 Line 4 消融实验");
    case "parallel":
      // Parallel requires efficiency metrics
      if (isAblation) return emptyOption("平行坐标不适用于消融", "请选择其他图表");
      return buildParallel(data, metric);
    default:
      return baseOption();
  }
}
