"""
多模态数据构建 — v2.0 支持文本模态 (report + search)

Environment 数据集的文本来源:
  - report: 环境报告 (宏观政策/年度总结, ~156条)
  - search: 相关搜索摘要 (公众关注度, ~2,272条)

文本融合模式:
  - concat: 简单拼接 report 和 search 嵌入
  - gating: 自适应门控融合
  - report_only: 仅使用 report 文本
  - search_only: 仅使用 search 文本
"""
import os
import json
import numpy as np


def load_or_build_multimodal(dataset_name, data_dir, text_dim=768, img_size=32,
                             text_mode='concat'):
    """
    加载或构建多模态数据

    Args:
        dataset_name: 数据集名称 (Environment / Energy / Health)
        data_dir: 数据目录 (./dataset/)
        text_dim: 文本嵌入维度
        img_size: 递归图尺寸 (未在v2.0中使用)
        text_mode: 文本融合模式

    Returns:
        text_embeds: [N, text_dim] numpy array 或 None
        img_tensors: [N, 1, img_size, img_size] 或 None
    """
    text_embeds = None
    img_tensors = None

    # 1. 尝试加载预计算的文本嵌入缓存
    text_cache = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
    if os.path.exists(text_cache):
        text_embeds = np.load(text_cache)
        print(f'  Loaded text embeddings: {text_embeds.shape}')
    else:
        # 2. 尝试从 Time-MMD JSON 文件构建真实文本嵌入
        text_embeds = _build_from_timemmd_json(dataset_name, data_dir, text_dim)

        if text_embeds is None:
            # 3. 回退: 生成合成文本嵌入 (仅用于测试)
            text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim)

    # 递归图 (v2.0 中未使用，保留兼容性)
    img_cache = os.path.join(data_dir, f'{dataset_name}_recurrence.npy')
    if os.path.exists(img_cache):
        img_tensors = np.load(img_cache)
    else:
        img_tensors = _build_recurrence_plots(dataset_name, data_dir, img_size)

    return text_embeds, img_tensors


def _build_from_timemmd_json(dataset_name, data_dir, text_dim):
    """
    从 Time-MMD 的 text JSON 构建真实文本嵌入

    Time-MMD 的 Environment 数据包含两类文本:
      - report: 环境报告 (宏观政策/年度总结)
      - search: 相关搜索摘要 (公众关注度)

    使用 sentence-transformers 进行文本编码
    """
    text_json_path = os.path.join(data_dir, f'{dataset_name}_text.json')
    if not os.path.exists(text_json_path):
        print(f'  No text JSON found at {text_json_path}')
        return None

    csv_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(csv_path):
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)
    n_samples = len(df)
    date_col = df['date'] if 'date' in df.columns else df.iloc[:, 0]

    # 加载 JSON 文本数据
    with open(text_json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    # 提取 report 和 search 文本
    report_texts = text_data.get('report', {}) if isinstance(text_data, dict) else {}
    search_texts = text_data.get('search', {}) if isinstance(text_data, dict) else {}

    # 为每个样本分配文本嵌入
    text_embeds = np.zeros((n_samples, text_dim), dtype=np.float32)

    texts_available = bool(report_texts) or bool(search_texts)
    if texts_available:
        # 使用 sentence-transformers 编码文本 (如果可用)
        try:
            text_embeds = _encode_texts_with_model(
                date_col, report_texts, search_texts, n_samples, text_dim
            )
            print(f'  Encoded text with sentence-transformers: {text_embeds.shape}')
        except ImportError:
            print('  sentence-transformers not available, using synthetic text embeddings')
            text_embeds = None
        except Exception as e:
            print(f'  Text encoding failed: {e}')
            text_embeds = None

    # 缓存
    if text_embeds is not None:
        cache_path = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
        np.save(cache_path, text_embeds)

    return text_embeds


def _encode_texts_with_model(date_col, report_texts, search_texts, n_samples, text_dim):
    """
    使用 sentence-transformers 模型编码文本

    Environment 数据集:
      - report: 按日期键的环境报告文本 (~156条)
      - search: 按日期键的搜索摘要文本 (~2,272条)

    编码策略:
      1. 对每条 report 文本编码 → 768维
      2. 对每条 search 文本编码 → 768维
      3. 按日期对齐到时序样本
      4. 拼接: [report_embed | search_embed] → 1536维
      5. 降维到 text_dim
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384维, 轻量
    embed_dim = encoder.get_sentence_embedding_dimension()

    # 汇总所有文本并编码
    all_keys = set()
    for date_val in date_col:
        date_key = str(date_val)
        all_keys.add(date_key)

    report_embeds = {}
    search_embeds = {}

    # 编码 report 文本
    if report_texts:
        report_items = list(report_texts.items())
        report_values = [v if isinstance(v, str) else str(v) for _, v in report_items]
        report_encoded = encoder.encode(report_values, show_progress_bar=False,
                                        convert_to_numpy=True)
        for (key, _), emb in zip(report_items, report_encoded):
            report_embeds[str(key)] = emb

    # 编码 search 文本
    if search_texts:
        search_items = list(search_texts.items())
        search_values = [v if isinstance(v, str) else str(v) for _, v in search_items]
        search_encoded = encoder.encode(search_values, show_progress_bar=False,
                                        convert_to_numpy=True)
        for (key, _), emb in zip(search_items, search_encoded):
            search_embeds[str(key)] = emb

    # 对齐到时序样本
    text_embeds = np.zeros((n_samples, text_dim), dtype=np.float32)
    for i, date_val in enumerate(date_col):
        date_key = str(date_val)
        rep_emb = report_embeds.get(date_key, np.zeros(embed_dim, dtype=np.float32))
        sea_emb = search_embeds.get(date_key, np.zeros(embed_dim, dtype=np.float32))

        # 拼接: [report | search] → 降维到 text_dim
        combined = np.concatenate([rep_emb, sea_emb])  # 768维 (2*384)
        if combined.shape[0] > text_dim:
            # 简单截断到 text_dim
            text_embeds[i] = combined[:text_dim]
        elif combined.shape[0] < text_dim:
            text_embeds[i, :combined.shape[0]] = combined
        else:
            text_embeds[i] = combined

    return text_embeds


def _build_synthetic_text_embeds(dataset_name, data_dir, text_dim):
    """生成合成的文本嵌入（回退方案，仅用于测试）"""
    import pandas as pd
    data_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    n = len(df)

    # 用随机向量模拟文本嵌入
    np.random.seed(42)
    actual_dim = min(text_dim, 128)
    text_embeds = np.random.randn(n, actual_dim).astype(np.float32) * 0.1

    cache_path = os.path.join(data_dir, f'{dataset_name}_text_embeds.npy')
    np.save(cache_path, text_embeds)
    print(f'  Generated synthetic text embeddings: {text_embeds.shape}')
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

    series = data[:, 0]
    n = len(series)

    rp_list = []
    window = img_size * 2
    stride = 1
    for i in range(0, n - window + 1, stride):
        seg = series[i:i + window]
        seg_norm = (seg - seg.mean()) / (seg.std() + 1e-8)
        rp = _compute_recurrence_matrix(seg_norm, img_size)
        rp_list.append(rp)

    result = np.stack(rp_list, axis=0).astype(np.float32)
    result = result[:n]

    if len(result) < n:
        pad = np.tile(result[-1:], (n - len(result), 1, 1, 1))
        result = np.concatenate([result, pad], axis=0)

    cache_path = os.path.join(data_dir, f'{dataset_name}_recurrence.npy')
    np.save(cache_path, result)
    print(f'  Generated recurrence plots: {result.shape}')
    return result


def _compute_recurrence_matrix(series, img_size):
    """计算单个递归图矩阵"""
    n = len(series)
    dist = np.abs(series[:, None] - series[None, :])
    threshold = np.percentile(dist, 10)
    rp = (dist < threshold).astype(np.float32)
    if rp.shape[0] != img_size:
        from PIL import Image
        img = Image.fromarray((rp * 255).astype(np.uint8))
        img = img.resize((img_size, img_size), Image.BILINEAR)
        rp = np.array(img, dtype=np.float32) / 255.0
    return rp[np.newaxis, :, :]
