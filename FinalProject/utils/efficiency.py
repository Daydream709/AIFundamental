"""
效率统计工具 — 参数量 / FLOPs / 推理时间 / GPU显存
"""
import torch
import time
import numpy as np


def count_parameters(model):
    """统计模型参数量 (单位: M)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6  # total_M, trainable_M


def measure_inference_time(model, input_shape, device='cuda', n_runs=100, use_amp=True):
    """
    测量推理时间 (单位: ms)

    Args:
        model: PyTorch 模型
        input_shape: (B, H, C)
        device: 'cuda' 或 'cpu'
        n_runs: 运行次数 (取平均)
        use_amp: 是否使用混合精度
    """
    model.eval()
    B, H, C = input_shape
    device_obj = torch.device(device)

    # 构造输入
    x_enc = torch.randn(B, H, C, device=device_obj)
    x_mark = torch.randn(B, H, 4, device=device_obj)
    x_dec = torch.zeros(B, 96, C, device=device_obj)  # pred_len=96 default
    x_mark_dec = torch.randn(B, 96, 4, device=device_obj)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            with torch.cuda.amp.autocast(enabled=use_amp):
                _ = model(x_enc, x_mark, x_dec, x_mark_dec)

    # 测量
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(n_runs):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _ = model(x_enc, x_mark, x_dec, x_mark_dec)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / n_runs  # ms
    else:
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x_enc, x_mark, x_dec, x_mark_dec)
        elapsed = (time.time() - t0) * 1000 / n_runs  # ms

    return elapsed


def measure_gpu_memory(model, input_shape, device='cuda'):
    """测量 GPU 显存占用 (单位: MB)"""
    if device != 'cuda' or not torch.cuda.is_available():
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.train()
    B, H, C = input_shape
    device_obj = torch.device(device)

    x_enc = torch.randn(B, H, C, device=device_obj)
    x_mark = torch.randn(B, H, 4, device=device_obj)
    x_dec = torch.zeros(B, 96, C, device=device_obj)
    x_mark_dec = torch.randn(B, 96, 4, device=device_obj)

    output = model(x_enc, x_mark, x_dec, x_mark_dec)
    loss = output.sum()
    loss.backward()

    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.empty_cache()
    return mem_mb


def get_model_efficiency(model, input_shape, device='cuda'):
    """获取完整的效率指标"""
    total_m, trainable_m = count_parameters(model)
    infer_time = measure_inference_time(model, input_shape, device)
    gpu_mem = measure_gpu_memory(model, input_shape, device)

    return {
        'total_params_M': round(total_m, 3),
        'trainable_params_M': round(trainable_m, 3),
        'inference_time_ms': round(infer_time, 2),
        'gpu_memory_MB': round(gpu_mem, 1),
    }
