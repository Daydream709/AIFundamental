"""
统计检验 — Wilcoxon 符号秩检验
"""
import numpy as np
from scipy import stats


def wilcoxon_test(scores_a, scores_b, alpha=0.05):
    """
    Wilcoxon 符号秩检验

    Args:
        scores_a: 模型A在各设置下的指标 (list/array)
        scores_b: 模型B在各设置下的指标 (list/array)
        alpha: 显著性水平

    Returns:
        statistic: 检验统计量
        p_value: p值
        significant: 是否显著 (p < alpha)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    diff = scores_a - scores_b

    # 去除差值为0的样本
    diff = diff[diff != 0]

    if len(diff) < 2:
        return 0.0, 1.0, False

    try:
        statistic, p_value = stats.wilcoxon(diff)
    except ValueError:
        return 0.0, 1.0, False

    return statistic, p_value, p_value < alpha


def pairwise_wilcoxon(results_dict, metric='MSE'):
    """
    对所有模型对进行 Wilcoxon 检验

    Args:
        results_dict: {model_name: [metric_values]} 各模型在不同设置下的指标
        metric: 指标名称

    Returns:
        DataFrame 格式的 p-value 矩阵
    """
    import pandas as pd
    models = list(results_dict.keys())
    n = len(models)

    p_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, p_val, _ = wilcoxon_test(
                results_dict[models[i]],
                results_dict[models[j]]
            )
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    return pd.DataFrame(p_matrix, index=models, columns=models)
