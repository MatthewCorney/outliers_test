"""Shared test fixtures for outlier_tests."""

import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).parent / "test_data" / "results.json"


def _load_results():
    with open(DATA_PATH) as f:
        return json.load(f)


def _collect_cases(key):
    """Collect test cases from results.json that contain the given key."""
    results = _load_results()
    return [
        pytest.param(data_results, id=data_name)
        for data_name, data_results in results.items()
        if key in data_results
    ]


@pytest.fixture(params=_collect_cases("rosner"))
def rosner_case(request):
    return request.param


@pytest.fixture(params=_collect_cases("grubbs"))
def grubbs_case(request):
    return request.param


@pytest.fixture(params=_collect_cases("dixon"))
def dixon_case(request):
    return request.param
