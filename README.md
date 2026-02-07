# outlier-tests

A Python library for statistical outlier detection, providing classical hypothesis-testing methods for identifying anomalous observations in univariate data.

## Installation

**pip**

```bash
pip install outlier-tests
```

**Poetry**

```bash
poetry add outlier-tests
```

## Quick Start

```python
from outlier_tests import dixon_test, grubbs_test, rosner_test
```

### Dixon's Q Test

Best suited for small sample sizes (3--30 observations).

```python
data = [0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.465]

result = dixon_test(data, alpha=0.05, mode="two_sided")
print(result)
```

The `mode` parameter accepts `"two_sided"`, `"left"`, or `"right"` to control which tail is tested.

### Grubbs' Test

Detects a single outlier in a roughly normal dataset of any size.

```python
data = [12.1, 12.3, 12.5, 12.4, 12.2, 12.3, 18.9]

result = grubbs_test(data, alpha=0.05)
print(result)
```

### Rosner Test

Identifies up to *k* outliers simultaneously, avoiding the masking effect that can occur when testing one point at a time. Recommended for larger datasets (n >= 25).

```python
data = [1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.4, 1.3, 1.2, 50.0, -40.0]

result = rosner_test(data, k=3, alpha=0.05)
print(result)
```

## When to Use Each Test

| Test | Sample Size | Outliers Detected | Key Advantage |
|---|---|---|---|
| `dixon_test` | 3--30 | 1 | Designed for very small samples |
| `grubbs_test` | Any | 1 | Simple, widely accepted single-outlier test |
| `rosner_test` | >= 25 recommended | Up to *k* | Handles multiple outliers without masking |

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/):

```bash
pytest
```

To run with verbose output and coverage:

```bash
pytest -v --cov=outlier_tests
```

## License

This project is distributed under the terms of the MIT License. See `LICENSE` for details.
