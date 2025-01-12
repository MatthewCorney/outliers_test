import numpy as np
from scipy.stats import t
import math
from typing import Union


def grubbs_test(data: Union[np.ndarray, list], alpha: float = 0.05):
    """
    Perform Grubbs' test for outlier detection.

    This function implements the two-sided version of Grubbs' test to detect
    a single outlier in a dataset. The test statistic is compared against a
    critical value derived from Student's t-distribution to determine if an
    extreme data point is an outlier.

    :param data: array-like data
    :param alpha: Significance level (default 0.05).
    :returns: A dictionary with the following keys:
        * ``'zscore'`` (float):
          The maximum absolute z-score in the data.
        * ``'g_crit'`` (float):
          The critical value for the Grubbs' test.
        * ``'p_value'`` (float):
          The approximate two-sided p-value.
        * ``'is_outlier'`` (bool):
          True if the detected point is an outlier, False otherwise.
        * ``'outlier_index'`` (int):
          The index of the detected outlier in the data array.
        * ``'outlier_value'`` (float):
          The value of the detected outlier.
    """
    x = np.array(data, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Grubbs' test requires at least 3 data points.")

    mean_x = np.mean(data)
    sd_x = np.std(data, ddof=1)

    z_scores = (data - mean_x) / sd_x

    idx_outlier = int(np.argmax(np.abs(z_scores)))
    g = abs(z_scores[idx_outlier])

    df_t = n - 2
    t_crit = t.ppf(1 - alpha / (2 * n), df_t)
    g_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t_crit ** 2 / (df_t + t_crit ** 2))

    is_outlier = g > g_crit

    t_stat = g * math.sqrt((n - 1) / n)
    p_value = 2.0 * (1.0 - t.cdf(t_stat, df_t))

    return {
        "z_score": float(g),
        "g_crit": float(g_crit),
        "p_value": float(p_value),
        "is_outlier": bool(is_outlier),
        "outlier_index": idx_outlier,
        "outlier_value": float(data[idx_outlier]),
    }
