import pytest
import json
import numpy as np
import pandas as pd

from outlier_tests.rosner import rosner_test
from outlier_tests.grubb import grubbs_test
from outlier_tests.dixon import dixon_test

PATH = r"test_data\results.json"


@pytest.fixture(scope="module")
def rosner_data():
    """Fixture for data entries that contain 'rosner' key."""
    with open(PATH, "r") as f:
        results = json.load(f)
    return [(data_name, data_results)
            for data_name, data_results in results.items()
            if 'rosner' in data_results]


@pytest.fixture(scope="module")
def grubbs_data():
    """Fixture for data entries that contain 'grubbs' key (referred as 'grubs' in JSON)."""
    with open(PATH, "r") as f:
        results = json.load(f)
    return [(data_name, data_results)
            for data_name, data_results in results.items()
            if 'grubs' in data_results]


@pytest.fixture(scope="module")
def dixon_data():
    """Fixture for data entries that contain 'dixon' key."""
    with open(PATH, "r") as f:
        results = json.load(f)
    return [(data_name, data_results)
            for data_name, data_results in results.items()
            if 'dixon' in data_results]


def test_rosner(rosner_data):
    """
    Test that the Python rosner_test matches the R-based rosner output
    (rosnor_q, number of outliers, etc.).
    """
    for data_name, data_results in rosner_data:
        data_array = data_results["data"]
        rosner_q = data_results["rosner"]["rosnor_q"]
        all_stats = pd.DataFrame(data_results["rosner"]["all_stats"])

        rosner_result = rosner_test(data=data_array, k=len(rosner_q))
        q_result = [round(x, 4).item() for x in list(rosner_result["stat"])]

        assert np.allclose(rosner_q, q_result, rtol=0, atol=0.01), (
            f"\nData Name: {data_name}"
            f"\nR Q-Values:      {rosner_q}"
            f"\nPython Q-Values: {q_result}"
        )

        expected_outlier_count = sum(all_stats["Outlier"])
        python_outlier_count = rosner_result["n_outliers"]
        assert expected_outlier_count == python_outlier_count, (
            f"\nData Name: {data_name}"
            f"\nR Outlier Count:      {expected_outlier_count}"
            f"\nPython Outlier Count: {python_outlier_count}"
        )


def test_grubbs(grubbs_data):
    """
    Test that the Python grubbs_test matches the R-based grubbs output
    (z-score, p-value < 0.05, etc.).
    """
    for data_name, data_results in grubbs_data:
        data_array = data_results["data"]

        r_statistic = data_results["grubbs"]["statistic"][0]
        r_p_value = data_results["grubbs"]["p_value"][0]

        result = grubbs_test(data_array)
        py_zscore = result["zscore"]
        py_is_outlier = result["is_outlier"]

        assert np.allclose(py_zscore, r_statistic, rtol=0, atol=0.01), (
            f"\nData Name: {data_name}"
            f"\nR z-score:      {round(r_statistic, 4)}"
            f"\nPython z-score: {round(py_zscore, 4)}"
        )
        r_is_outlier = r_p_value < 0.05
        assert py_is_outlier == r_is_outlier, (
            f"\nData Name: {data_name}"
            f"\nR Outlier:      {r_is_outlier}"
            f"\nPython Outlier: {py_is_outlier}"
        )


def test_dixon(dixon_data):
    """
    Test that the Python dixon_test matches the R-based dixon output
    (statistic, p-value, etc.) for each mode.
    """
    for data_name, data_results in dixon_data:
        data_array = data_results["data"]
        for mode_key, mode_value in data_results["dixon"].items():
            python_mode = mode_key.replace('.', '_')
            result = dixon_test(data=data_array, mode=python_mode)
            r_statistic = mode_value["statistic"][0]
            r_p_value = mode_value["p_value"][0]

            assert np.allclose(result["statistic"], r_statistic, rtol=0, atol=0.01), (
                f"\nData Name: {data_name}"
                f"\nMode: {python_mode}"
                f"\nR Statistic:      {r_statistic}"
                f"\nPython Statistic: {result['statistic']}"
            )
