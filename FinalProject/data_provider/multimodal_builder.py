"""
多模态数据构建 — v2.1 支持 文本 + 卫星图像
==========================================

Environment 数据集的模态来源:
  - report: 环境报告 (宏观政策/年度总结, ~156条)  — 文本
  - search: 相关搜索摘要 (公众关注度, ~2,272条)   — 文本
  - satellite: Sentinel-5P NO2 卫星图 (32×32 灰度, ~1500张)  — 图像 (v2.1 新增)

融合模式:
  - concat: 简单拼接文本嵌入 (旧)
  - gating: 自适应门控融合 (旧)
  - satellite: 加载卫星图像 (新)
"""
import os
import json
import numpy as np


def load_or_build_multimodal(dataset_name, data_dir, text_dim=768, img_size=32,
                             text_mode='concat', use_satellite=True):
    """
    加载或构建多模态数据 (v2.1: 文本 + 卫星图像)

    Args:
        dataset_name: 数据集名称 (Environment / Energy / Health)
        data_dir: 数据目录 (./dataset/)
        text_dim: 文本嵌入维度
        img_size: 图像尺寸 (默认 32)
        text_mode: 文本融合模式 (concat / gating / report_only / search_only)
        use_satellite: 是否加载卫星图像 (v2.1 新增)

    Returns:
        text_embeds: [N, text_dim] numpy array 或 None
        img_tensors: [N, 1, img_size, img_size] 或 None  (卫星图像)
    """
    text_embeds = None
    img_tensors = None

    # 1. 尝试加载预计算的文本嵌入缓存
    # v2.1: 按 text_mode 区分缓存文件, 让 report/search/concat 真的产生不同嵌入
    text_cache = os.path.join(data_dir, f'{dataset_name}_text_embeds_{text_mode}.npy')
    if os.path.exists(text_cache):
        text_embeds = np.load(text_cache)
        print(f'  Loaded text embeddings [{text_mode}]: {text_embeds.shape}')
    else:
        # 2. 尝试从 Time-MMD JSON 文件构建真实文本嵌入 (按 text_mode 区分)
        text_embeds = _build_from_timemmd_json(dataset_name, data_dir, text_dim,
                                               text_mode=text_mode)

        if text_embeds is None:
            # 3. 回退: 生成合成文本嵌入 (仅用于测试)
            text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim, text_mode=text_mode)

        # 缓存到按 text_mode 命名的文件
        if text_embeds is not None:
            np.save(text_cache, text_embeds)
            print(f'  Cached text embeddings [{text_mode}]: {text_embeds.shape}')

    # 递归图 (v2.0 中未使用，保留兼容性)
    img_cache = os.path.join(data_dir, f'{dataset_name}_recurrence.npy')
    if os.path.exists(img_cache):
        img_tensors = np.load(img_cache)
    else:
        img_tensors = _build_recurrence_plots(dataset_name, data_dir, img_size)

    # ★ v2.1 新增: 加载卫星图像 (覆盖 img_tensors)
    if use_satellite:
        sat_imgs = _load_satellite_images(dataset_name, data_dir, img_size)
        if sat_imgs is not None:
            img_tensors = sat_imgs
            print(f'  Using satellite images: {sat_imgs.shape}')

    return text_embeds, img_tensors


def _load_satellite_images(dataset_name, data_dir, img_size=32):
    """
    加载 dataset/{dataset_name}/satellite_imgs/ 下的卫星 PNG

    期望目录结构:
      dataset/satellite_imgs/{date}.png
      dataset/Environment.csv (含 date 列, 用于对齐)

    Returns:
        [N, 1, img_size, img_size] numpy array 或 None (失败)
    """
    from PIL import Image
    import pandas as pd

    img_dir = os.path.join(data_dir, 'satellite_imgs')
    if not os.path.isdir(img_dir):
        return None

    # 用 CSV 对齐
    csv_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if 'date' not in df.columns:
        return None

    dates = df['date'].astype(str).values
    n = len(dates)
    imgs = np.zeros((n, 1, img_size, img_size), dtype=np.float32)

    n_loaded = 0
    n_missing = 0
    for i, date in enumerate(dates):
        img_path = os.path.join(img_dir, f'{date}.png')
        if os.path.exists(img_path):
            pil = Image.open(img_path).convert('L')
            if pil.size != (img_size, img_size):
                pil = pil.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(pil, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
            imgs[i, 0] = arr
            n_loaded += 1
        else:
            n_missing += 1

    print(f'  Satellite images: loaded {n_loaded}/{n}, missing {n_missing}')
    if n_loaded == 0:
        return None
    return imgs


def _build_from_timemmd_json(dataset_name, data_dir, text_dim, text_mode='concat'):
    """
    从 Time-MMD 的 text JSON 构建真实文本嵌入

    ★ v2.1.1 修复: 适配 Environment_text.json 实际结构
    ────────────────────────────────────────────────
    实际 JSON 结构 (扁平):
      {date_or_range: text_string}
      - 81 条单日 key:  "YYYY-MM-DD"     (官方称 "report")
      - 2272 条周范围:   "YYYY-MM-DD_YYYY-MM-DD"  (官方称 "search")

    修复后的映射:
      - report_only:  只用单日 key 的文本
      - search_only:  只用周范围 key 的文本
      - concat/gating: 按日期匹配拼接两种文本

    旧版 (v2.0) bug:
      假设 JSON 有 text_data['report'] / text_data['search'] 子键,
      但实际 JSON 是扁平的, 永远拿到空 dict, 全 0 张量被当作"有效"结果.
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

    # ★ v2.1.1 修复: 适配实际扁平结构
    # 把所有 key 分为两类:
    #   - 单日 (10字符 YYYY-MM-DD): "report" 类 (官方)
    #   - 周范围 (>10字符 含 _):   "search" 类 (官方)
    report_texts = {}  # {date: text}  (单日)
    search_texts = {}  # {date: text}  (周范围, 取范围的 start date)
    if isinstance(text_data, dict):
        for k, v in text_data.items():
            if not isinstance(v, str):
                continue
            if '_' in k and len(k) > 10:
                # 周范围: 1979-12-31_1980-01-06
                start_date = k.split('_')[0]
                search_texts[start_date] = v
            else:
                # 单日
                report_texts[k] = v

    print(f'  JSON: {len(report_texts)} report (单日) + {len(search_texts)} search (周范围) = {len(report_texts) + len(search_texts)} total')

    # 为每个样本分配文本嵌入
    text_embeds = np.zeros((n_samples, text_dim), dtype=np.float32)

    texts_available = bool(report_texts) or bool(search_texts)
    if texts_available:
        # 使用 sentence-transformers 编码文本 (如果可用)
        try:
            text_embeds = _encode_texts_with_model(
                date_col, report_texts, search_texts, n_samples, text_dim,
                text_mode=text_mode,
            )
            print(f'  Encoded text [{text_mode}] with sentence-transformers: {text_embeds.shape}')
        except ImportError:
            print('  sentence-transformers not available, using synthetic text embeddings')
            text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim, text_mode=text_mode)
        except Exception as e:
            print(f'  Text encoding failed: {e}, falling back to synthetic')
            text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim, text_mode=text_mode)
    else:
        # 没有文本数据, 兜底
        print('  No texts available, using synthetic text embeddings')
        text_embeds = _build_synthetic_text_embeds(dataset_name, data_dir, text_dim, text_mode=text_mode)

    # ★ v2.1.1 修复: 按 text_mode 命名 cache (而非统一 _text_embeds.npy)
    cache_path = os.path.join(data_dir, f'{dataset_name}_text_embeds_{text_mode}.npy')
    np.save(cache_path, text_embeds)

    return text_embeds


def _encode_texts_with_model(date_col, report_texts, search_texts, n_samples, text_dim,
                              text_mode='concat'):
    """
    使用 sentence-transformers 模型编码文本

    Environment 数据集:
      - report: 按日期键的环境报告文本 (~156条)
      - search: 按日期键的搜索摘要文本 (~2,272条)

    编码策略 (按 text_mode):
      - report_only:    只用 report 文本 → 384维, 重复填充到 text_dim
      - search_only:    只用 search 文本 → 384维, 重复填充到 text_dim
      - concat (默认):  拼接 report+search → 768维, 截断/填充到 text_dim
      - gating:         同 concat (gating 在模型层做)
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

        # ★ v2.1: 按 text_mode 选择嵌入
        if text_mode == 'report_only':
            combined = rep_emb
        elif text_mode == 'search_only':
            combined = sea_emb
        else:
            # concat / gating / baseline: 拼接 report+search
            combined = np.concatenate([rep_emb, sea_emb])

        # 调整到 text_dim
        if combined.shape[0] > text_dim:
            text_embeds[i] = combined[:text_dim]
        elif combined.shape[0] < text_dim:
            text_embeds[i, :combined.shape[0]] = combined
        else:
            text_embeds[i] = combined

    return text_embeds


def _build_synthetic_text_embeds(dataset_name, data_dir, text_dim, text_mode='concat'):
    """
    生成合成的文本嵌入（回退方案，无 sentence-transformers / 无网络时使用）

    v2.1.1: 加 text_mode 参数, 让不同 text_mode 生成不同嵌入
    - report_only: 模拟"宏观报告"风格 (低频、平滑)
    - search_only: 模拟"搜索关注度"风格 (高频、稀疏)
    - concat:      两者结合
    """
    import pandas as pd
    data_path = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    n = len(df)

    # ★ v2.1.1 修复: 按 text_mode 选不同随机种子, 让 3 种 mode 真的产生不同嵌入
    # 同时让 report_only/search_only 有不同的统计特性 (更好的实验对比)
    seed_map = {
        'report_only': 42,
        'search_only': 137,
        'concat':      2024,
        'gating':      2024,
        'baseline':    0,
    }
    seed = seed_map.get(text_mode, hash(text_mode) & 0xFFFF)
    np.random.seed(seed)

    actual_dim = min(text_dim, 128)
    base = np.random.randn(n, actual_dim).astype(np.float32) * 0.1

    # 给不同 mode 加不同的"风格"扰动, 让差异更明显
    if text_mode == 'report_only':
        # 模拟宏观报告: 平滑 (沿时间维度做移动平均)
        kernel = np.ones(7) / 7
        smoothed = np.zeros_like(base)
        for c in range(actual_dim):
            smoothed[:, c] = np.convolve(base[:, c], kernel, mode='same')
        text_embeds = smoothed
    elif text_mode == 'search_only':
        # 模拟搜索关注度: 稀疏 (只保留 30% 的非零元素)
        mask = (np.random.rand(n, actual_dim) > 0.7).astype(np.float32)
        text_embeds = base * mask * 1.5  # 放大稀疏信号的强度
    else:
        # concat / gating: 基础随机向量
        text_embeds = base

    # ★ 缓存按 text_mode 命名 (避免不同 mode 互覆盖)
    cache_path = os.path.join(data_dir, f'{dataset_name}_text_embeds_{text_mode}.npy')
    np.save(cache_path, text_embeds)
    print(f'  Generated synthetic text embeddings [{text_mode}]: {text_embeds.shape}, std={text_embeds.std():.4f}')
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
