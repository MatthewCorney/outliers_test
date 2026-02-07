"""Grubbs' test for detecting a single outlier in a univariate dataset."""

import math
from typing import Dict, Union

import numpy as np
from scipy.stats import t


def grubbs_test(data: Union[np.ndarray, list], alpha: float = 0.05) -> Dict[str, Union[float, bool, int]]:
    """
    Perform Grubbs' test for outlier detection.

    This function implements the two-sided version of Grubbs' test to detect
    a single outlier in a dataset. The test statistic is compared against a
    critical value derived from Student's t-distribution.

    References:
        Grubbs, F.E. (1950). "Sample criteria for testing outlying observations."
        *Annals of Mathematical Statistics*, 21(1), 27-58.

    :param data: array-like data (must have at least 3 elements).
    :param alpha: Significance level (default 0.05).
    :returns: A dictionary with the following keys:
        * ``'zscore'`` (float):
          The maximum absolute z-score in the data (Grubbs' G statistic).
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

    mean_x = np.mean(x)
    sd_x = np.std(x, ddof=1)

    if sd_x == 0:
        raise ValueError("All data points are identical; Grubbs' test is not applicable.")

    z_scores = (x - mean_x) / sd_x

    idx_outlier = int(np.argmax(np.abs(z_scores)))
    g = abs(z_scores[idx_outlier])

    df_t = n - 2
    t_crit = t.ppf(1 - alpha / (2 * n), df_t)
    g_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t_crit ** 2 / (df_t + t_crit ** 2))

    is_outlier = g > g_crit

    t_stat = g * math.sqrt((n - 1) / n)
    p_value = 2.0 * (1.0 - t.cdf(t_stat, df_t))

    return {
        "zscore": float(g),
        "g_crit": float(g_crit),
        "p_value": float(p_value),
        "is_outlier": bool(is_outlier),
        "outlier_index": idx_outlier,
        "outlier_value": float(x[idx_outlier]),
    }
