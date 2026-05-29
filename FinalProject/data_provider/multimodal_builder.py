"""
多模态数据构建 — 文本嵌入 + 递归图
"""
import os
import numpy as np


def load_or_build_multimodal(dataset_name, data_dir, text_dim=768, img_size=32):
    """
    加载或构建多模态数据

    Returns:
        text_embeds: [N, text_dim] numpy array 或 None
        img_tensors: [N, 1, img_size, img_size] numpy array 或 None
    """
    text_embeds = None
    img_tensors = None

    # 加载预计算的文本嵌入
    text_cache = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
    if os.path.exists(text_cache):
        text_embeds = np.load(text_cache)
        print(f'  Loaded text embeddings: {text_embeds.shape}')
    else:
        # 生成合成文本嵌入
        text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim)

    # 加载预计算的递归图
    img_cache = os.path.join(data_dir, f'{dataset_name}_recurrence.npy')
    if os.path.exists(img_cache):
        img_tensors = np.load(img_cache)
        print(f'  Loaded recurrence plots: {img_tensors.shape}')
    else:
        # 从时序数据生成递归图
        img_tensors = _build_recurrence_plots(dataset_name, data_dir, img_size)

    return text_embeds, img_tensors


def _build_synthetic_text_embeds(dataset_name, data_dir, text_dim):
    """生成合成的文本嵌入（用于测试和非EPA数据集）"""
    import pandas as pd
    data_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    n = len(df)

    # 用随机向量模拟文本嵌入
    np.random.seed(42)
    # 降低维度用于实际训练
    actual_dim = min(text_dim, 128)
    text_embeds = np.random.randn(n, actual_dim).astype(np.float32) * 0.1

    cache_path = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
    np.save(cache_path, text_embeds)
    print(f'  Generated synthetic text embeddings: {text_embeds.shape} -> {cache_path}')
    return text_embeds


def _build_recurrence_plots(dataset_name, data_dir, img_size=32):
    """从时序数据生成递归图 (Recurrence Plot)"""
    import pandas as pd
    data_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    cols = [c for c in df.columns if c != 'date']
    data = df[cols].values.astype(np.float32)

    # 对第一个变量生成递归图
    series = data[:, 0]
    n = len(series)

    # 滑窗生成递归图
    rp_list = []
    window = img_size * 2  # 每个窗口的大小
    stride = 1
    for i in range(0, n - window + 1, stride):
        seg = series[i:i + window]
        seg_norm = (seg - seg.mean()) / (seg.std() + 1e-8)
        rp = _compute_recurrence_matrix(seg_norm, img_size)
        rp_list.append(rp)

    result = np.stack(rp_list, axis=0).astype(np.float32)  # [N', 1, img_size, img_size]
    result = result[:n]  # 对齐长度

    # 如果不够长，用最后一个填充
    if len(result) < n:
        pad = np.tile(result[-1:], (n - len(result), 1, 1, 1))
        result = np.concatenate([result, pad], axis=0)

    cache_path = os.path.join(data_dir, f'{dataset_name}_recurrence.npy')
    np.save(cache_path, result)
    print(f'  Generated recurrence plots: {result.shape} -> {cache_path}')
    return result


def _compute_recurrence_matrix(series, img_size):
    """计算单个递归图矩阵"""
    n = len(series)
    # 计算距离矩阵
    dist = np.abs(series[:, None] - series[None, :])
    # 用百分位阈值二值化
    threshold = np.percentile(dist, 10)
    rp = (dist < threshold).astype(np.float32)
    # resize 到目标尺寸
    if rp.shape[0] != img_size:
        from PIL import Image
        img = Image.fromarray((rp * 255).astype(np.uint8))
        img = img.resize((img_size, img_size), Image.BILINEAR)
        rp = np.array(img, dtype=np.float32) / 255.0
    return rp[np.newaxis, :, :]  # [1, img_size, img_size]
