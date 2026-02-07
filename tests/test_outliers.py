"""Tests validating Python outlier tests against R reference implementations."""

import numpy as np
import pytest

from outlier_tests.dixon import dixon_test
from outlier_tests.grubb import grubbs_test
from outlier_tests.iqr import iqr_test
from outlier_tests.mad import mad_test
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


class TestMADValidation:
    def test_detects_obvious_outlier(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = mad_test(data)
        assert result["n_outliers"] >= 1
        assert 9 in result["outlier_indices"]  # index of 100

    def test_no_outliers_in_clean_data(self):
        data = [10.0, 10.1, 9.9, 10.2, 9.8, 10.05, 9.95]
        result = mad_test(data)
        assert result["n_outliers"] == 0
        assert result["outlier_indices"] == []

    def test_manual_z_score(self):
        """Verify the formula: M_i = 0.6745 * (x_i - median) / MAD."""
        data = [1, 2, 3, 4, 5]
        result = mad_test(data)
        med = np.median(data)
        mad = np.median(np.abs(np.array(data) - med))
        expected_z = 0.6745 * (np.array(data) - med) / mad
        np.testing.assert_allclose(result["modified_z_scores"], expected_z)
        assert result["median"] == med
        assert result["mad"] == mad

    def test_identical_values(self):
        result = mad_test([5, 5, 5, 5, 5])
        assert result["mad"] == 0.0
        assert result["n_outliers"] == 0
        assert np.all(result["modified_z_scores"] == 0.0)

    def test_too_few_elements(self):
        with pytest.raises(ValueError, match="at least 3"):
            mad_test([1, 2])

    def test_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold must be positive"):
            mad_test([1, 2, 3], threshold=-1)

    def test_zero_threshold(self):
        with pytest.raises(ValueError, match="threshold must be positive"):
            mad_test([1, 2, 3], threshold=0)

    def test_custom_threshold(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
        strict = mad_test(data, threshold=2.0)
        lenient = mad_test(data, threshold=5.0)
        assert strict["n_outliers"] >= lenient["n_outliers"]

    def test_list_input(self):
        result = mad_test([1, 2, 3, 4, 5, 6, 100])
        assert isinstance(result["n_outliers"], int)
        assert isinstance(result["outlier_indices"], list)

    def test_return_keys(self):
        result = mad_test([1, 2, 3, 4, 5])
        expected_keys = {"modified_z_scores", "mad", "median",
                         "outlier_indices", "outlier_values", "n_outliers"}
        assert set(result.keys()) == expected_keys


class TestIQRValidation:
    def test_detects_obvious_outlier(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = iqr_test(data)
        assert result["n_outliers"] >= 1
        assert 9 in result["outlier_indices"]  # index of 100

    def test_no_outliers_in_clean_data(self):
        data = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
        result = iqr_test(data)
        assert result["n_outliers"] == 0

    def test_manual_fences(self):
        """Verify fences: lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = iqr_test(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        assert result["q1"] == q1
        assert result["q3"] == q3
        assert result["iqr"] == iqr
        assert result["lower_fence"] == q1 - 1.5 * iqr
        assert result["upper_fence"] == q3 + 1.5 * iqr

    def test_far_outlier_multiplier(self):
        """k=3.0 should detect fewer outliers than k=1.5."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
        standard = iqr_test(data, k=1.5)
        far = iqr_test(data, k=3.0)
        assert standard["n_outliers"] >= far["n_outliers"]

    def test_identical_values(self):
        result = iqr_test([5, 5, 5, 5, 5])
        assert result["iqr"] == 0.0
        assert result["n_outliers"] == 0

    def test_too_few_elements(self):
        with pytest.raises(ValueError, match="at least 3"):
            iqr_test([1, 2])

    def test_negative_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            iqr_test([1, 2, 3], k=-1)

    def test_zero_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            iqr_test([1, 2, 3], k=0)

    def test_lower_outlier(self):
        """Detect outlier below the lower fence."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, -50]
        result = iqr_test(data)
        assert 9 in result["outlier_indices"]  # index of -50
        assert -50.0 in result["outlier_values"]

    def test_list_input(self):
        result = iqr_test([1, 2, 3, 4, 5, 6, 100])
        assert isinstance(result["n_outliers"], int)
        assert isinstance(result["outlier_indices"], list)

    def test_return_keys(self):
        result = iqr_test([1, 2, 3, 4, 5])
        expected_keys = {"q1", "q3", "iqr", "lower_fence", "upper_fence",
                         "outlier_indices", "outlier_values", "n_outliers"}
        assert set(result.keys()) == expected_keys
