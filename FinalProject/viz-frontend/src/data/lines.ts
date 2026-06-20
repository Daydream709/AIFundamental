/**
 * 5 experiment lines matching project-plan-v2.0.md section 5.
 *
 * Line 4 was originally a single page but the two research topics
 * (KAN-iTransformer high-performance vs Lite-SparseNet lightweight)
 * have DIFFERENT ablation groups, so the UI is split into:
 *   - Line 4a (KAN, 🧠)   — KAN-iTransformer 5 模块消融
 *   - Line 4b (Lite, 🪶)  — Lite-SparseNet 3 阶段消融
 * Each line's `ablationGroups` constrains which rows from the
 * shared ablation_results.csv are shown on that page.
 */
import type { ExperimentLine } from "@/types/filters";

/** Ablation groups relevant to the KAN-iTransformer (high-performance) study.
 *  Must match the `ablation` column values produced by scripts/train_line4a_kan.py. */
export const KAN_ABLATION_GROUPS = ["KAN 5 Modules"];

/** Ablation groups relevant to the Lite-SparseNet (lightweight) study.
 *  Must match the `ablation` column values produced by scripts/train_line4b_lite.py.
 *  v2.1: changed from "Lite 3 Stages" (FFT-based) to "Lite Residual" (LinearResidual). */
export const LITE_ABLATION_GROUPS = ["Lite Residual"];

export const LINES: Record<number, ExperimentLine> = {
  1: {
    number: 1,
    title: "全架构对比",
    subtitle: "Cross-Architecture Comparison",
    icons: ["🏛️"],
    route: "/line1",
    models: ["DLinear", "PatchTST", "TimesNet", "Mamba"],
    datasets: ["ETTm2", "Weather", "Electricity"],
    predLens: [96, 192, 336, 720],
    description:
      "MLP / Transformer / CNN / SSM 四大架构在 3 个基准数据集上的横向对比，回答「哪种架构最适合时序预测」的核心问题。",
    dataSource: "main",
  },
  2: {
    number: 2,
    title: "自研模型深度评测",
    subtitle: "Self-Developed Model Deep Evaluation",
    icons: ["🚀"],
    route: "/line2",
    models: [
      // 自研
      "KANiTransformer",
      "LiteSparseNet",
      "SparseTSF",
      // 参考 (thuml 官方基线)
      "DLinear",
      "PatchTST",
      "TimesNet",
      "Mamba",
    ],
    datasets: ["ETTm2", "Weather", "Electricity"],
    predLens: [96, 192, 336, 720],
    description:
      "自研 KAN-iTransformer (高性能) + Lite-SparseNet (轻量化) vs SparseTSF (外部标杆) + DLinear / PatchTST / TimesNet / Mamba (4 大架构基线)。覆盖 7 个模型的全对比。",
    dataSource: "main",
  },
  3: {
    number: 3,
    title: "多模态消融",
    subtitle: "Multimodal Ablation (Environment)",
    icons: ["🎭"],
    route: "/line3",
    models: ["PatchTST", "Mamba"],
    datasets: ["Environment"],
    predLens: [96, 192],
    description:
      "Environment 数据集上 5 种文本模态 (baseline / report / search / both_concat / both_gating) 的消融对比。PatchTST (Attention) vs Mamba (SSM) — 两种归纳偏置对文本的响应差异。",
    dataSource: "main",
    extraFilters: {
      textMode: [
        "baseline",
        "report",
        "search",
        "both_concat",
        "both_gating",
      ],
    },
  },
  4: {
    number: 4,
    title: "KAN 消融（高性能）",
    subtitle: "KAN-iTransformer Ablation (High-Performance)",
    icons: ["🧠"],
    route: "/line4-kan",
    models: ["KANiTransformer"],
    datasets: ["ETTm2", "Electricity", "Environment"],
    predLens: [96, 192, 336, 720],
    description:
      "KAN-iTransformer (v2.0) 5 大模块的消融研究 (A0-A4)：KAN 层 / CFD / 预训练 / 概率输出 / RevIN。回答「KAN 每个设计选择是否都不可替代」。",
    dataSource: "ablation",
    ablationGroups: KAN_ABLATION_GROUPS,
  },
  5: {
    number: 5,
    title: "Lite 消融（轻量化）",
    subtitle: "Lite-SparseNet Ablation (Lightweight)",
    icons: ["🪶"],
    route: "/line4-lite",
    models: ["LiteSparseNet"],
    datasets: ["ETTm2", "Electricity", "Environment"],
    predLens: [96, 192, 336, 720],
    description:
      "Lite-SparseNet (v2.0) 3 阶段设计的消融研究 (B0-B2)：稀疏趋势 / 分组 MLP / FFT 残差。回答「Lite 极简架构是否真的够用」。",
    dataSource: "ablation",
    ablationGroups: LITE_ABLATION_GROUPS,
  },
};

export const TEXT_MODE_LABELS: Record<string, string> = {
  baseline: "📊 基线 (无文本)",
  report: "📄 + Report (宏观)",
  search: "🔍 + Search (实时)",
  both_concat: "➕ + 两者拼接",
  both_gating: "🔀 + 门控融合",
};
