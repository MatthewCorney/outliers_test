"""Modified Z-Score outlier detection using the Median Absolute Deviation (MAD).

Robust to the very outliers it is trying to detect, unlike mean/std-based methods.
"""

from typing import Dict, List, Union

import numpy as np


def mad_test(
    data: Union[np.ndarray, list],
    threshold: float = 3.5,
) -> Dict[str, Union[np.ndarray, float, int, List[int]]]:
    """
    Detect outliers using the Modified Z-Score based on the MAD.

    The modified z-score for each observation is:

        M_i = 0.6745 * (x_i - median) / MAD

    where MAD = median(|x_i - median|). The constant 0.6745 is the 0.75th
    quantile of the standard normal distribution, making the MAD a consistent
    estimator of the standard deviation for normal data.

    References:
        Iglewicz, B. and Hoaglin, D.C. (1993). "Volume 16: How to Detect and
        Handle Outliers." *The ASQC Basic References in Quality Control:
        Statistical Techniques.*

    :param data: array-like data (must have at least 3 elements).
    :param threshold: Absolute modified z-score above which a point is
        flagged as an outlier (default 3.5, as recommended by Iglewicz & Hoaglin).
    :return: A dictionary with the following keys:
        * ``'modified_z_scores'`` (numpy.ndarray): Modified z-score for each point.
        * ``'mad'`` (float): The median absolute deviation.
        * ``'median'`` (float): The sample median.
        * ``'outlier_indices'`` (list[int]): Indices of detected outliers.
        * ``'outlier_values'`` (numpy.ndarray): Values of detected outliers.
        * ``'n_outliers'`` (int): Number of outliers detected.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < 3:
        raise ValueError("MAD test requires at least 3 data points.")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    if mad == 0.0:
        # MAD is zero when >50% of values are identical.
        # Fall back: flag nothing (no robust scale estimate possible).
        return {
            "modified_z_scores": np.zeros(n),
            "mad": 0.0,
            "median": med,
            "outlier_indices": [],
            "outlier_values": np.array([]),
            "n_outliers": 0,
        }

    modified_z = 0.6745 * (x - med) / mad

    outlier_mask = np.abs(modified_z) > threshold
    outlier_indices = list(np.where(outlier_mask)[0])

    return {
        "modified_z_scores": modified_z,
        "mad": mad,
        "median": med,
        "outlier_indices": outlier_indices,
        "outlier_values": x[outlier_mask],
        "n_outliers": int(np.sum(outlier_mask)),
    }
