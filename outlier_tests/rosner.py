"""Rosner's generalized ESD test for detecting multiple outliers."""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import t

from outlier_tests.basic_logger import logger


def rosner_test(data: Union[np.ndarray, list], k: int = 3, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Rosner's generalized extreme Studentized deviate (ESD) test for outlier detection.

    References:
        Rosner, B. (1983). "Percentage points for a generalized ESD many-outlier
        procedure." *Technometrics*, 25(2), 165-172.

    :param data: array-like data (must not contain NaN/Inf).
    :param k: Maximum number of outliers to test (1 <= k <= n-2).
    :param alpha: Significance level for two-sided test (0 < alpha < 1).
    :return: A dictionary with the following keys:
         * ``'stat'`` (numpy.ndarray): R_i values (test statistics) for each iteration.
         * ``'crit_value'`` (numpy.ndarray): Lambda_i values (critical values).
         * ``'n_outliers'`` (int): Number of outliers found.
         * ``'all_stats'`` (pandas.DataFrame): Iteration-by-iteration details including
           mean, sd, removed value/index, test statistic, critical value, and outlier flag.
         * ``'bad_obs'`` (int): Count of non-finite values detected (always 0 if
           the function returns without error).
    """
    x = np.asarray(data, dtype=float)
    obs_num = np.arange(len(x)) + 1  # 1-based indices to mimic R

    if len(x) < 3:
        raise ValueError("Rosner test requires at least 3 observations.")
    if k < 1 or k > (len(x) - 2):
        raise ValueError(f"k must be between 1 and {len(x) - 2} (n - 2), got {k}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    mask_finite = np.isfinite(x)
    bad_obs = np.sum(~mask_finite)
    if bad_obs > 0:
        raise ValueError("Data must not contain NA/NaN/Inf values.")
    n = len(x)
    # Optional warnings about large k
    if k > 10 or k > (n // 2):
        logger.warning(f"The error may be larger when K is larger than 10 or larger than {len(x) / 2}."
                       )

    r_vals = np.full(k, np.nan)  # R_i (test statistics)
    mean_vals = np.full(k, np.nan)  # mean at each iteration
    sd_vals = np.full(k, np.nan)  # std dev at each iteration
    val_removed = np.full(k, np.nan)  # data value removed
    obs_removed = np.full(k, np.nan)  # index of observation removed

    current_data = x.copy()
    current_obs = obs_num.copy()

    for i in range(k):
        mu = current_data.mean()
        sig = current_data.std(ddof=1)

        mean_vals[i] = mu
        sd_vals[i] = sig

        if sig <= 0:
            r_vals[i:] = np.nan
            break

        z = np.abs(current_data - mu) / sig
        idx_max = np.argmax(z)

        r_vals[i] = z[idx_max]
        val_removed[i] = current_data[idx_max]
        obs_removed[i] = current_obs[idx_max]

        current_data = np.delete(current_data, idx_max)
        current_obs = np.delete(current_obs, idx_max)

        # If we have fewer than 2 points left, we must stop
        if len(current_data) < 2:
            break

    lambdas = np.full(k, np.nan)
    for i in range(k):
        l_val = i  # i in 0..k-1 => R's "k_i" = i+1, so l = i
        degrees_freedom = (n - l_val - 2)
        if degrees_freedom < 1:
            # Cannot compute if df < 1
            lambdas[i] = np.nan
        else:
            p_i = 1.0 - ((alpha / 2.0) / (n - l_val))
            t_crit = t.ppf(p_i, degrees_freedom)
            numerator = t_crit * (n - l_val - 1)
            denominator = np.sqrt((n - l_val - 2 + t_crit ** 2) * (n - l_val))
            lambdas[i] = numerator / denominator

    is_outlier = r_vals > lambdas
    if np.any(is_outlier):
        last_true_idx = np.max(np.where(is_outlier)[0])
        is_outlier[: last_true_idx + 1] = True

    n_outliers = np.sum(is_outlier)

    df_stats = pd.DataFrame({
        "i": np.arange(k),
        "mean": mean_vals,
        "sd": sd_vals,
        "outlier_value": val_removed,
        "outlier_index": obs_removed,
        "r": r_vals,
        "lambda": lambdas,
        "Outlier": is_outlier
    })

    results = {
        "stat": r_vals,
        "crit_value": lambdas,
        "n_outliers": n_outliers,
        "all_stats": df_stats,
        "bad_obs": bad_obs,
    }

    return results
