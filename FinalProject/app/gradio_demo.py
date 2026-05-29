"""
Gradio 交互式 Demo — 时序预测可视化展示

启动方式:
    python app/gradio_demo.py
"""
import os
import sys
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from configs.dataset_configs import get_dataset_config
from data_provider.data_factory import data_provider
from utils.tools import fix_seed
from utils.metrics import metric

import torch


def load_model(model_name, dataset, seq_len, pred_len):
    """加载指定模型"""
    config = get_dataset_config(dataset, seq_len=seq_len, pred_len=pred_len)
    config.model = model_name
    config.checkpoints = './checkpoints/'

    if model_name == 'MultimodalFusion':
        config.use_text = True
        config.use_image = True

    fix_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from exp.exp_basic import ExpBasic
        exp = ExpBasic(config)
        return exp.model, config, device
    except Exception as e:
        return None, config, device


def predict(model_name, dataset, seq_len, pred_len, show_confidence):
    """执行预测并返回图表"""
    model, config, device = load_model(model_name, dataset, seq_len, pred_len)

    if model is None:
        return None, "Model loading failed. Please check if the model is trained."

    model.eval()
    model.to(device)

    # 加载测试数据
    test_loader = data_provider(config, 'test')

    # 获取一个batch
    for batch in test_loader:
        break

    batch = [b.to(device) for b in batch]
    x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
    x_dec = torch.zeros_like(x_y[:, -pred_len:, :])
    x_mark_dec = x_mark_y[:, -pred_len:, :]
    true = x_y[:, -pred_len:, :]

    with torch.no_grad():
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # 转为 numpy
    pred_np = pred[0, :, 0].cpu().numpy()  # 第一个样本，第一个变量
    true_np = true[0, :, 0].cpu().numpy()

    # 计算指标
    mse, mae, rmse, mape, smape = metric(pred[0:1].cpu().numpy(), true[0:1].cpu().numpy())

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true_np, 'k-', linewidth=2, label='Ground Truth')
    ax.plot(pred_np, 'r-', linewidth=1.5, label=f'{model_name} Prediction')

    if show_confidence and pred_len > 0:
        std = np.std(pred_np - true_np)
        ax.fill_between(range(pred_len),
                        pred_np - 1.96 * std,
                        pred_np + 1.96 * std,
                        alpha=0.2, color='red', label='95% CI (approx.)')

    ax.set_title(f'{model_name} on {dataset} (H={seq_len}, F={pred_len})')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 指标文本
    metrics_text = (
        f"Model: {model_name}\n"
        f"Dataset: {dataset}\n"
        f"H={seq_len}, F={pred_len}\n"
        f"---\n"
        f"MSE:  {mse:.6f}\n"
        f"MAE:  {mae:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAPE: {mape:.4f}%\n"
        f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )

    return fig, metrics_text


def arbitrate(dataset, seq_len, pred_len):
    """模型仲裁模式 — 对比所有模型并推荐最优"""
    models = ['DLinear', 'PatchTST', 'iTransformer', 'TimeMixer',
              'TimeKAN', 'KANiTransformer', 'MambaTransformerDual']

    results = {}
    for model_name in models:
        try:
            model, config, device = load_model(model_name, dataset, seq_len, pred_len)
            if model is None:
                continue
            model.eval()
            model.to(device)

            test_loader = data_provider(config, 'test')
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = [b.to(device) for b in batch]
                    x_enc, x_y, x_mark_enc, x_mark_y = batch[0], batch[1], batch[2], batch[3]
                    x_dec = torch.zeros_like(x_y[:, -pred_len:, :])
                    x_mark_dec = x_mark_y[:, -pred_len:, :]
                    pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    preds.append(pred.cpu().numpy())
                    trues.append(x_y[:, -pred_len:, :].cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            mse, mae, _, _, _ = metric(preds, trues)
            results[model_name] = {'MSE': mse, 'MAE': mae}

        except Exception as e:
            print(f"  {model_name}: Error - {e}")
            continue

    if not results:
        return None, "No models available. Please train models first."

    # 找最优模型
    best_model = min(results, key=lambda k: results[k]['MSE'])

    # 绘制对比柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    mses = [results[n]['MSE'] for n in names]
    colors = ['gold' if n == best_model else 'steelblue' for n in names]

    ax.barh(names, mses, color=colors)
    ax.set_xlabel('MSE')
    ax.set_title(f'Model Arbitration — {dataset} (F={pred_len})')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 标注最优
    best_idx = names.index(best_model)
    ax.text(mses[best_idx], best_idx, f'  BEST ({mses[best_idx]:.4f})',
            va='center', fontweight='bold', color='darkgoldenrod')

    recommendation = (
        f"Recommended: {best_model}\n"
        f"MSE: {results[best_model]['MSE']:.6f}\n"
        f"MAE: {results[best_model]['MAE']:.6f}\n"
        f"\nAll {len(results)} models evaluated."
    )

    return fig, recommendation


# ==================== Gradio UI ====================

def create_demo():
    with gr.Blocks(title='Time Series Forecasting Demo') as demo:
        gr.Markdown('# Time Series Forecasting Demo')
        gr.Markdown('Multimodal Long-term Time Series Prediction System')

        with gr.Tab('Single Model Prediction'):
            with gr.Row():
                with gr.Column(scale=1):
                    dataset_dd = gr.Dropdown(
                        choices=['ETTm2', 'Weather', 'Electricity', 'Energy', 'Environment', 'Health'],
                        value='ETTm2', label='Dataset'
                    )
                    model_dd = gr.Dropdown(
                        choices=['DLinear', 'PatchTST', 'iTransformer', 'TimeMixer',
                                'TimeKAN', 'KANiTransformer', 'MambaTransformerDual',
                                'MultimodalFusion', 'Chronos2'],
                        value='DLinear', label='Model'
                    )
                    seq_len_sl = gr.Slider(96, 336, value=96, step=96, label='History Length (H)')
                    pred_len_sl = gr.Slider(96, 720, value=96, step=96, label='Prediction Length (F)')
                    confidence_cb = gr.Checkbox(label='Show 95% Confidence Interval')
                    predict_btn = gr.Button('Predict', variant='primary')

                with gr.Column(scale=2):
                    plot_output = gr.Plot(label='Prediction Plot')
                    metrics_output = gr.Textbox(label='Metrics', lines=10)

            predict_btn.click(
                fn=predict,
                inputs=[model_dd, dataset_dd, seq_len_sl, pred_len_sl, confidence_cb],
                outputs=[plot_output, metrics_output],
            )

        with gr.Tab('Model Arbitration'):
            with gr.Row():
                with gr.Column(scale=1):
                    arb_dataset = gr.Dropdown(
                        choices=['ETTm2', 'Weather', 'Electricity', 'Energy', 'Environment', 'Health'],
                        value='ETTm2', label='Dataset'
                    )
                    arb_seq = gr.Slider(96, 336, value=96, step=96, label='History Length')
                    arb_pred = gr.Slider(96, 720, value=96, step=96, label='Prediction Length')
                    arb_btn = gr.Button('Run Arbitration', variant='primary')

                with gr.Column(scale=2):
                    arb_plot = gr.Plot(label='Model Comparison')
                    arb_rec = gr.Textbox(label='Recommendation', lines=6)

            arb_btn.click(
                fn=arbitrate,
                inputs=[arb_dataset, arb_seq, arb_pred],
                outputs=[arb_plot, arb_rec],
            )

    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, theme=gr.themes.Soft())
