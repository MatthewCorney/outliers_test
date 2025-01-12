import numpy as np
import pandas as pd
from scipy.stats import t
from src.outlier_tests.basic_logger import logger
from typing import Union


def rosner_test(data: Union[np.ndarray, list], k: int = 3, alpha: float = 0.05):
    """
    Rosner test for outlier detection.

    :param data: array-like data
    :param k: Maximum number of outliers to test
    :param alpha: Significance level for two-sided test
    :return: A dictionary with the following keys:
         * ``'stat'``: array of R_i values (test statistics)
         * ``'crit_value'``: array of lambda_i values (critical values)
         * ``'n_outliers'``: number of outliers found
         * ``'all_stats'``: a pandas DataFrame of iteration-by-iteration details
    """
    x = np.asarray(data, dtype=float)
    obs_num = np.arange(len(x)) + 1  # 1-based indices to mimic R

    if len(x) < 3:
        logger.error('There must be at least 3 observations.)')
        raise ValueError
    if k < 1 or k > (len(x) - 2):
        logger.error('alpha must bein (0,1)')
        raise ValueError
    if not (0 < alpha < 1):
        logger.error('alpha must bein (0,1) ')
        raise ValueError

    mask_finite = np.isfinite(x)
    bad_obs = np.sum(~mask_finite)
    if bad_obs > 0:
        logger.error('NA/NaN/Inf Should not be present in data')
        raise ValueError
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
