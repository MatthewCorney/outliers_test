"""Grubbs' test for detecting a single outlier in a univariate dataset."""

import math
from typing import Dict, List, Union

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
          Approximate two-sided p-value (Bonferroni upper bound; conservative).
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

    # Recover the underlying t-statistic from G using the exact relationship:
    #   G = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
    # Solving for t^2 gives:
    s2 = n * g ** 2 / (n - 1) ** 2
    if s2 >= 1.0:
        p_value = 0.0
    else:
        t2 = (n - 2) * s2 / (1.0 - s2)
        t_stat = math.sqrt(t2)
        # Bonferroni correction: G is the max of n studentized deviations
        p_value = min(1.0, 2.0 * n * (1.0 - t.cdf(t_stat, df_t)))

    return {
        "zscore": float(g),
        "g_crit": float(g_crit),
        "p_value": float(p_value),
        "is_outlier": bool(is_outlier),
        "outlier_index": idx_outlier,
        "outlier_value": float(x[idx_outlier]),
    }


def grubbs_test_iterative(
    data: Union[np.ndarray, list],
    alpha: float = 0.05,
    max_outliers: int = 0,
) -> Dict[str, Union[List[Dict], List[int], np.ndarray, int]]:
    """
    Iterative Grubbs' test: repeatedly apply Grubbs' test, removing one
    outlier at a time, until no more are detected or *max_outliers* is reached.

    Each iteration tests H0: "no outliers remain in the (reduced) dataset"
    at significance level *alpha*. The test stops when either Grubbs' test
    fails to reject or the dataset is reduced to fewer than 3 observations.

    References:
        Grubbs, F.E. (1969). "Procedures for Detecting Outlying Observations
        in Samples." *Technometrics*, 11(1), 1-21.

    :param data: array-like data (must have at least 3 elements).
    :param alpha: Significance level for each individual Grubbs' test
        (default 0.05).
    :param max_outliers: Maximum number of outliers to remove. 0 (default)
        means no limit — iterate until the test stops rejecting.
    :returns: A dictionary with the following keys:
        * ``'iterations'`` (list[dict]): One dict per iteration with the
          single-Grubbs result plus the ``'original_index'`` of the removed
          point.
        * ``'outlier_indices'`` (list[int]): Indices (in the original data)
          of all detected outliers, in order of removal.
        * ``'outlier_values'`` (numpy.ndarray): Corresponding values.
        * ``'n_outliers'`` (int): Number of outliers detected.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < 3:
        raise ValueError("Grubbs' test requires at least 3 data points.")
    if max_outliers < 0:
        raise ValueError(f"max_outliers must be >= 0, got {max_outliers}.")

    # Track mapping from working array index → original index.
    original_indices = np.arange(n)
    working = x.copy()

    iterations: List[Dict] = []
    outlier_indices: List[int] = []
    outlier_values: List[float] = []

    while len(working) >= 3:
        if max_outliers > 0 and len(outlier_indices) >= max_outliers:
            break

        try:
            result = grubbs_test(working, alpha=alpha)
        except ValueError:
            # Remaining data is identical or otherwise untestable — stop.
            break

        if not result["is_outlier"]:
            break

        local_idx = result["outlier_index"]
        orig_idx = int(original_indices[local_idx])

        iteration_record = {
            "original_index": orig_idx,
            "outlier_value": result["outlier_value"],
            "zscore": result["zscore"],
            "g_crit": result["g_crit"],
            "p_value": result["p_value"],
        }
        iterations.append(iteration_record)
        outlier_indices.append(orig_idx)
        outlier_values.append(result["outlier_value"])

        # Remove the outlier from working arrays.
        working = np.delete(working, local_idx)
        original_indices = np.delete(original_indices, local_idx)

    return {
        "iterations": iterations,
        "outlier_indices": outlier_indices,
        "outlier_values": np.array(outlier_values),
        "n_outliers": len(outlier_indices),
    }
