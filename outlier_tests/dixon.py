"""Dixon's Q test for detecting a single outlier in small samples (n=3..30)."""

import math
from typing import Dict, Optional, Union

import numpy as np
from scipy.stats import t


def compute_q_critical(
        n: int,
        alpha: float,
        two_sided: bool = True,
        degrees_freedom: Optional[int] = None
) -> float:
    """
    Compute the approximate critical Q-value for Dixon's Q Test.

    Uses the t-distribution approximation described in:
    Dixon, W.J. (1950). "Analysis of extreme values."
    *Annals of Mathematical Statistics*, 21(4), 488-506.

    Degrees of freedom default to ``n - 2`` following Rorabacher (1991).

    :param n: The sample size.
    :param alpha: Significance level (default 0.05).
    :param two_sided: If ``True``, use a two-sided alpha. Otherwise, use one-sided.
    :param degrees_freedom: Degrees of freedom (default ``n - 2``).
    :return: The approximate critical Q value for Dixon's Q Test as a float.
    """
    if degrees_freedom is None:
        degrees_freedom = n - 2

    alpha_adj = alpha / 2 if two_sided else alpha

    t_critical = t.ppf(1 - alpha_adj, degrees_freedom)

    numerator = t_critical * (n - 1)
    denominator = math.sqrt(n) * math.sqrt(degrees_freedom + t_critical ** 2)
    q_critical = numerator / denominator

    return float(q_critical)


def dixon_test(
        data: Union[np.ndarray, list],
        alpha: float = 0.05,
        mode: str = "two_sided"
) -> Dict[str, Union[float, bool, int, None]]:
    r"""
    Perform Dixon's Q test for outlier detection.

    References:
        Dixon, W.J. (1950). *Annals of Mathematical Statistics*, 21(4), 488-506.
        Rorabacher, D.B. (1991). *Analytical Chemistry*, 63(2), 139-146.

    :param data: array-like data
    :param alpha: Significance level (default 0.05).
    :param mode: The mode for the test. One of:
        * ``"two_sided"``: Test for an outlier on either end of the distribution.
        * ``"less"``: Test only for a low-end outlier.
        * ``"greater"``: Test only for a high-end outlier.
    :return:
        A dictionary with the following keys:
        * ``'statistic'`` (float): The computed Dixon Q statistic.
        * ``'critical_value'`` (float): The approximate critical Q value.
        * ``'p_value'`` (float or None): None (exact p-value computation is not
          supported for Dixon's Q; use ``reject`` for decision-making).
        * ``'reject'`` (bool): Whether to reject the null hypothesis (i.e., detect an outlier).
        * ``'outlier_index'`` (int or None): The index of the outlier if found, else None.
        * ``'outlier_value'`` (float or None): The value of the outlier if found, else None.
    """
    # Convert data to a NumPy array
    x = np.array(data, dtype=float)
    n = len(x)

    # Check sample size validity
    if not (3 <= n <= 30):
        raise ValueError("Dixon's Q test is typically valid for 3 <= n <= 30.")

    # Sort data while retaining original indices
    sorted_indices = np.argsort(x)
    sorted_data = x[sorted_indices]

    data_range = sorted_data[-1] - sorted_data[0]
    if data_range == 0:
        return {
            "statistic": 0.0,
            "critical_value": float(compute_q_critical(n, alpha, two_sided=(mode == "two_sided"))),
            "p_value": 1.0,
            "reject": False,
            "outlier_index": None,
            "outlier_value": None,
        }

    # Decide how to compute Q and which side is candidate outlier
    if mode == "less":
        gap = sorted_data[1] - sorted_data[0]
        data_range = sorted_data[-1] - sorted_data[0]
        q_stat = gap / data_range

        outlier_index = int(sorted_indices[0])
        outlier_value = float(sorted_data[0])

        # Compute one-sided critical value
        q_crit = compute_q_critical(n, alpha, two_sided=False)

    elif mode == "greater":
        gap = sorted_data[-1] - sorted_data[-2]
        data_range = sorted_data[-1] - sorted_data[0]
        q_stat = gap / data_range

        outlier_index = int(sorted_indices[-1])
        outlier_value = float(sorted_data[-1])

        # Compute one-sided critical value
        q_crit = compute_q_critical(n, alpha, two_sided=False)

    elif mode == "two_sided":
        # Compute candidate outliers at both ends
        gap_low = sorted_data[1] - sorted_data[0]
        gap_high = sorted_data[-1] - sorted_data[-2]
        data_range = sorted_data[-1] - sorted_data[0]

        q_low = gap_low / data_range
        q_high = gap_high / data_range

        # Determine which side is more likely an outlier
        if q_low > q_high:
            q_stat = q_low
            outlier_index = int(sorted_indices[0])
            outlier_value = float(sorted_data[0])
        else:
            q_stat = q_high
            outlier_index = int(sorted_indices[-1])
            outlier_value = float(sorted_data[-1])

        # Compute two-sided critical value
        q_crit = compute_q_critical(n, alpha, two_sided=True)

    else:
        raise ValueError("mode must be 'less', 'greater', or 'two_sided'.")

    # Compare Q statistic to Q critical
    reject = q_stat >= q_crit

    # Exact p-value computation is not straightforward for Dixon's Q test;
    # return None to avoid misleading consumers.
    p_value = None

    # If no outlier is found, reset to None
    if not reject:
        outlier_index = None
        outlier_value = None

    return {
        "statistic": float(q_stat),
        "critical_value": float(q_crit),
        "p_value": p_value,
        "reject": bool(reject),
        "outlier_index": outlier_index,
        "outlier_value": outlier_value
    }
