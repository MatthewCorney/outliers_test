"""Tests validating Python outlier tests against R reference implementations."""

import numpy as np
import pytest

from outlier_tests.dixon import dixon_test
from outlier_tests.grubb import grubbs_test
from outlier_tests.rosner import rosner_test


# ---------------------------------------------------------------------------
# Rosner test -- validated against R's EnvStats::rosnerTest
# ---------------------------------------------------------------------------

def test_rosner_q_values(rosner_case):
    data_array = rosner_case["data"]
    rosner_q = rosner_case["rosner"]["rosner_q"]

    result = rosner_test(data=data_array, k=len(rosner_q))
    q_result = [round(x, 4).item() for x in list(result["stat"])]

    assert np.allclose(rosner_q, q_result, rtol=0, atol=0.01)


def test_rosner_outlier_count(rosner_case):
    data_array = rosner_case["data"]
    rosner_ref = rosner_case["rosner"]
    expected_count = sum(row["Outlier"] for row in rosner_ref["all_stats"])

    result = rosner_test(data=data_array, k=len(rosner_ref["rosner_q"]))
    assert result["n_outliers"] == expected_count


# ---------------------------------------------------------------------------
# Grubbs test -- validated against R's outliers::grubbs.test
# ---------------------------------------------------------------------------

def test_grubbs_zscore(grubbs_case):
    data_array = grubbs_case["data"]
    r_statistic = grubbs_case["grubbs"]["statistic"][0]

    result = grubbs_test(data_array)
    assert np.allclose(result["zscore"], r_statistic, rtol=0, atol=0.01)


def test_grubbs_outlier_detection(grubbs_case):
    data_array = grubbs_case["data"]
    r_p_value = grubbs_case["grubbs"]["p_value"][0]

    result = grubbs_test(data_array)
    assert result["is_outlier"] == (r_p_value < 0.05)


# ---------------------------------------------------------------------------
# Dixon test -- validated against R's outliers::dixon.test
# ---------------------------------------------------------------------------

def test_dixon_statistic_and_decision(dixon_case):
    """Validate Q statistic AND reject/no-reject decision against R."""
    data_array = dixon_case["data"]
    for mode_key, mode_value in dixon_case["dixon"].items():
        python_mode = mode_key.replace(".", "_")
        result = dixon_test(data=data_array, mode=python_mode)
        r_statistic = mode_value["statistic"][0]
        r_p_value = mode_value["p_value"][0]

        assert np.allclose(result["statistic"], r_statistic, rtol=0, atol=0.01)

        # Verify the rejection decision matches R
        r_rejects = r_p_value < 0.05
        assert result["reject"] == r_rejects, (
            f"Mode={python_mode}: reject={result['reject']} but R p={r_p_value}"
        )


# ---------------------------------------------------------------------------
# Error path / edge-case tests
# ---------------------------------------------------------------------------

class TestDixonValidation:
    def test_too_few_elements(self):
        with pytest.raises(ValueError, match="3 <= n <= 30"):
            dixon_test([1, 2])

    def test_too_many_elements(self):
        with pytest.raises(ValueError, match="3 <= n <= 30"):
            dixon_test(list(range(31)))

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            dixon_test([1, 2, 3], mode="invalid")

    def test_identical_values(self):
        result = dixon_test([5, 5, 5, 5, 5])
        assert result["reject"] is False
        assert result["statistic"] == 0.0
        assert result["outlier_index"] is None

    def test_list_input(self):
        result = dixon_test([1.0, 2.0, 3.0, 10.0])
        assert isinstance(result["statistic"], float)


class TestGrubbsValidation:
    def test_too_few_elements(self):
        with pytest.raises(ValueError, match="at least 3"):
            grubbs_test([1, 2])

    def test_identical_values(self):
        with pytest.raises(ValueError, match="identical"):
            grubbs_test([5, 5, 5, 5])

    def test_list_input(self):
        result = grubbs_test([1.0, 2.0, 3.0, 10.0])
        assert isinstance(result["zscore"], float)


class TestRosnerValidation:
    def test_too_few_elements(self):
        with pytest.raises(ValueError, match="at least 3"):
            rosner_test([1, 2])

    def test_k_too_large(self):
        with pytest.raises(ValueError, match="k must be between"):
            rosner_test([1, 2, 3, 4, 5], k=4)

    def test_k_zero(self):
        with pytest.raises(ValueError, match="k must be between"):
            rosner_test([1, 2, 3, 4, 5], k=0)

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            rosner_test([1, 2, 3, 4, 5], alpha=1.5)

    def test_nan_in_data(self):
        with pytest.raises(ValueError, match="NaN/Inf"):
            rosner_test([1, 2, float("nan"), 4, 5])

    def test_inf_in_data(self):
        with pytest.raises(ValueError, match="NaN/Inf"):
            rosner_test([1, 2, float("inf"), 4, 5])

    def test_list_input(self):
        result = rosner_test([1, 2, 3, 4, 5, 6, 100], k=1)
        assert isinstance(result["n_outliers"], (int, np.integer))

    def test_all_stats_is_list_of_dicts(self):
        result = rosner_test([1, 2, 3, 4, 5, 6, 100], k=2)
        assert isinstance(result["all_stats"], list)
        assert isinstance(result["all_stats"][0], dict)
        assert "r" in result["all_stats"][0]
        assert "lambda" in result["all_stats"][0]
        assert "Outlier" in result["all_stats"][0]
