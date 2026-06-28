# Experiment Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. Data Sources (4 datasets)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    │
│  │  ETTm2   │    │ Weather  │    │ Electricity │    │   Environment    │    │
│  │ 7 vars   │    │ 21 vars  │    │  321 vars   │    │  6 vars + text   │    │
│  │  15min   │    │  10min   │    │     1h      │    │ daily / 156 rpts │    │
│  │  69,680  │    │  52,696  │    │   26,304    │    │  15,979 + 2272   │    │
│  └──────────┘    └──────────┘    └─────────────┘    └──────────────────┘    │
│       ↓                ↓               ↓                  ↓                │
└───────┼────────────────┼───────────────┼──────────────────┼────────────────┘
        └────────────────┴───────────────┴──────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                2. Unified Preprocessing (data_provider/dataset_base)         │
│   • Z-score normalization using train-set [0%, 70%] mean / std               │
│       `data = (data - mean) / std`  (std=0 → 1.0 fallback)                  │
│   • Time-ordered split: train 70% / val 15% / test 15%  (7 : 1.5 : 1.5)    │
│       border_ratios = [0.0, 0.7, 0.85, 1.0]                                 │
│   • seq_len = 96, pred_len ∈ {96, 192, 336, 720}                            │
│   • Time features: month / day / weekday / hour → normalized to [0, 1]      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  Line 1 (48 runs)│  Line 2 (32 runs)│  Line 3 (8 runs) │ Line 4 (21 runs) │
│  Architecture    │  Self-developed  │  Multimodal      │  Ablation        │
│  benchmark       │  model eval      │  ablation        │  study           │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│  4 models ×      │  2 self-dev ×    │  1 model ×       │ (KAN 4 + Lite 3) │
│  3 datasets ×    │  4 datasets ×    │  1 dataset ×     │  configs ×       │
│  4 pred_len      │  4 pred_len      │  4 text_mode     │  3 datasets      │
│                  │                  │  × 2 pred_len    │                  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│  DLinear         │  KAN-iTransformer│  SparseTSF       │ KAN: A0/A1/A2/A3 │
│  PatchTST        │  LiteSparseNet   │                  │ Lite: B0/B1/B2   │
│  TimesNet        │                  │  Environment     │                  │
│  Mamba           │  ETTm2           │                  │ ETTm2            │
│                  │  Weather         │  baseline        │ Weather          │
│  ETTm2           │  Electricity     │  report          │ Electricity      │
│  Weather         │  Environment     │  search          │                  │
│  Electricity     │                  │  both_concat     │                  │
│                  │  96/192/336/720  │                  │                  │
│  96/192/336/720  │                  │  pred_len 96,192 │  pred_len 96     │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│  Goal: evaluate  │  Goal: verify    │  Goal: quantify  │ Goal: quantify   │
│  the fit of 4    │  self-developed  │  the real gain   │ each innovative  │
│  architectures   │  models across   │  brought by text │ module's real    │
│  to data         │  4 datasets      │  (reports/search)│ contribution     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. Evaluation Metrics (§4.2)                        │
│   MSE (primary) · MAE · RMSE · MAPE · SMAPE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          4. Analysis & Visualization                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tables (§5.1 ~ 5.5)                │  Figures (figures/)                    │
│  • 5.1 Line 1 MSE/MAE, 4 models     │  • fig1   Baseline heatmap             │
│  • 5.2 Line 2 self-dev vs baselines │  • fig2   F=96→720 degradation         │
│  • 5.3 Line 3 multimodal text_mode  │  • fig3   Cross-dataset CV             │
│  • 5.4 Line 4 ablation A0~A3,B0~B2  │  • fig4   Arch-dataset fit             │
│                                     │  • fig5   Self-dev vs baselines        │
│  Efficiency (7 models, 3-4 sets)    │  • fig6   Params comparison            │
│  • Params (M)                       │  • fig7   Pareto frontier              │
│  • FLOPs (G)                        │  • fig8   Multimodal experiment        │
│  • InferTime (ms)                   │  • fig9   Ablation bar chart           │
│  • GPUMem (MB)                      │  • fig10  FLOPs (7 models × 4 sets)    │
│                                     │  • fig11  Infer time + GPU mem         │
│                                     │           (7 models × 3 sets)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                              【 Report / Slides 】
```

## Line Summary

| Line | # Runs | Key variables | Output |
|------|--------|---------------|--------|
| **Line 1** Architecture benchmark | 4×3×4 = **48 runs** | Models (DLinear / PatchTST / TimesNet / Mamba) × Datasets (ETTm2 / Weather / Electricity) × pred_len (96 / 192 / 336 / 720) | Tables 5-1 ~ 5-4 + fig1~fig4 |
| **Line 2** Self-developed eval | 2×4×4 = **32 runs** | Self-dev (KAN-iTransformer / LiteSparseNet) × Datasets (4) × pred_len (4) | Table 5-5 + fig5~fig7 |
| **Line 3** Multimodal ablation | 1×1×4×2 = **8 runs** | SparseTSF × Environment × text_mode (baseline / report / search / both_concat) × pred_len (96 / 192) | Tables 5-6, 5-9, 5-10 + fig8 |
| **Line 4** Ablation study | (4+3)×3 = **21 runs** | KAN 4 configs + Lite 3 configs × 3 datasets | Table 5-7 + fig9 |
| **Efficiency** | 7 models × 3-4 datasets | fvcore + line1/line2 CSV | Table 5-8 + fig10/fig11 |
| **Total** | **109 training runs** | — | — |
