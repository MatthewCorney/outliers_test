"""IQR-based outlier detection using Tukey's fences.

A non-parametric method that flags points outside [Q1 - k*IQR, Q3 + k*IQR],
where IQR = Q3 - Q1. The default multiplier k=1.5 defines "outliers" and
k=3.0 defines "far outliers", following Tukey (1977).
"""

from typing import Dict, List, Union

import numpy as np


def iqr_test(
    data: Union[np.ndarray, list],
    k: float = 1.5,
) -> Dict[str, Union[np.ndarray, float, int, List[int]]]:
    """
    Detect outliers using Tukey's fences (IQR method).

    A point x_i is flagged as an outlier if::

        x_i < Q1 - k * IQR   or   x_i > Q3 + k * IQR

    where Q1 and Q3 are the first and third quartiles and IQR = Q3 - Q1.

    References:
        Tukey, J.W. (1977). *Exploratory Data Analysis.* Addison-Wesley.

    :param data: array-like data (must have at least 3 elements).
    :param k: Fence multiplier (default 1.5 for standard outliers;
        use 3.0 for "far outliers").
    :return: A dictionary with the following keys:
        * ``'q1'`` (float): First quartile.
        * ``'q3'`` (float): Third quartile.
        * ``'iqr'`` (float): Interquartile range (Q3 - Q1).
        * ``'lower_fence'`` (float): Q1 - k * IQR.
        * ``'upper_fence'`` (float): Q3 + k * IQR.
        * ``'outlier_indices'`` (list[int]): Indices of detected outliers.
        * ``'outlier_values'`` (numpy.ndarray): Values of detected outliers.
        * ``'n_outliers'`` (int): Number of outliers detected.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < 3:
        raise ValueError("IQR test requires at least 3 data points.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1

    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    outlier_mask = (x < lower_fence) | (x > upper_fence)
    outlier_indices = list(np.where(outlier_mask)[0])

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_indices": outlier_indices,
        "outlier_values": x[outlier_mask],
        "n_outliers": int(np.sum(outlier_mask)),
    }
