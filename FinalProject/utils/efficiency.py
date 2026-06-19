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


def measure_flops(model, input_shape, device='cuda', freq='h', use_amp=False):
    """
    计算模型 FLOPs (单位: GFLOPs, 即 10^9 次浮点运算)

    使用 fvcore.FlopCountAnalysis, 符合主流论文惯例.

    Args:
        model: PyTorch 模型
        input_shape: (B, H, C)
        device: 'cuda' 或 'cpu'
        freq: 时间频率 — 'h'/'t'/'d'/'w' 等, 决定时间特征维度
        use_amp: 是否使用混合精度 (默认 False, FLOPs 统计不依赖实际精度)

    Returns:
        float: FLOPs in GFLOPs, 失败返回 0.0
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print('  Warning: fvcore not installed, FLOPs will be 0')
        return 0.0

    # 不同 freq 对应不同数量时间特征: h→4, d→3, t→5, s→6, w→2, m→1
    freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
    n_time_features = freq_map.get(freq, 4)

    B, H, C = input_shape
    device_obj = torch.device(device)
    model.eval()

    x_enc = torch.randn(B, H, C, device=device_obj)
    x_mark = torch.randn(B, H, n_time_features, device=device_obj)
    pred_len = getattr(model, 'pred_len', 96) if hasattr(model, 'pred_len') else 96
    x_dec = torch.zeros(B, pred_len, C, device=device_obj)
    x_mark_dec = torch.randn(B, pred_len, n_time_features, device=device_obj)

    try:
        flops_analyzer = FlopCountAnalysis(model, (x_enc, x_mark, x_dec, x_mark_dec))
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        total_flops = flops_analyzer.total()
        return total_flops / 1e9  # → GFLOPs
    except Exception as e:
        print(f'  Warning: FLOPs analysis failed: {e}')
        return 0.0


def measure_inference_time(model, input_shape, device='cuda', n_runs=100, use_amp=True, freq='h'):
    """
    测量推理时间 (单位: ms)

    Args:
        model: PyTorch 模型
        input_shape: (B, H, C)
        device: 'cuda' 或 'cpu'
        n_runs: 运行次数 (取平均)
        use_amp: 是否使用混合精度
        freq: 时间频率
    """
    model.eval()
    freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
    n_tf = freq_map.get(freq, 4)
    B, H, C = input_shape
    device_obj = torch.device(device)

    x_enc = torch.randn(B, H, C, device=device_obj)
    x_mark = torch.randn(B, H, n_tf, device=device_obj)
    x_dec = torch.zeros(B, 96, C, device=device_obj)
    x_mark_dec = torch.randn(B, 96, n_tf, device=device_obj)

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


def measure_gpu_memory(model, input_shape, device='cuda', freq='h'):
    """测量 GPU 显存占用 (单位: MB)"""
    if device != 'cuda' or not torch.cuda.is_available():
        return 0.0

    freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
    n_tf = freq_map.get(freq, 4)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.train()
    B, H, C = input_shape
    device_obj = torch.device(device)

    x_enc = torch.randn(B, H, C, device=device_obj)
    x_mark = torch.randn(B, H, n_tf, device=device_obj)
    x_dec = torch.zeros(B, 96, C, device=device_obj)
    x_mark_dec = torch.randn(B, 96, n_tf, device=device_obj)

    output = model(x_enc, x_mark, x_dec, x_mark_dec)
    loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()
    loss.backward()

    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.empty_cache()
    return mem_mb


def get_model_efficiency(model, input_shape, device='cuda'):
    """获取完整的效率指标"""
    total_m, trainable_m = count_parameters(model)
    flops_g = measure_flops(model, input_shape, device)
    infer_time = measure_inference_time(model, input_shape, device)
    gpu_mem = measure_gpu_memory(model, input_shape, device)

    return {
        'total_params_M': round(total_m, 3),
        'trainable_params_M': round(trainable_m, 3),
        'flops_G': round(flops_g, 3),
        'inference_time_ms': round(infer_time, 2),
        'gpu_memory_MB': round(gpu_mem, 1),
    }
