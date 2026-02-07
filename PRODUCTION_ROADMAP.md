# Production Readiness Audit -- outlier_tests

## Codebase Summary

Python statistical library (~450 lines) implementing three classical outlier detection tests:
**Dixon's Q**, **Grubbs'**, and **Rosner's**. Validates results against R reference implementations.
Uses Poetry for packaging, pytest for testing, and depends on NumPy, SciPy, and pandas.

---

## P0: Critical (Tests broken / blocking issues)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 1 | **Tests fail on Linux/macOS** -- `PATH = r"test_data\results.json"` uses Windows backslash. All 3 tests throw `FileNotFoundError`. | `tests/test_outliers.py:10` | Replace with `pathlib.Path(__file__).parent / "test_data" / "results.json"` |
| 2 | **No `.gitignore`** -- `.pyc`, `__pycache__/`, `.pytest_cache/`, `dist/`, `*.egg-info` will be committed accidentally. | project root | Add a standard Python `.gitignore` |
| 3 | **`pytest` is a runtime dependency** -- listed under `[tool.poetry.dependencies]` instead of dev dependencies. | `pyproject.toml:13` | Move to `[tool.poetry.group.dev.dependencies]` |

## P1: High (Correctness and reliability)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 4 | **Misleading error messages in `rosner_test`** -- `k` range validation says "alpha must be in (0,1)" when it should describe `k`. Also has typo: "bein". | `rosner.py:28-31` | Fix messages to match the actual parameter being validated |
| 5 | **Bare `raise ValueError`** in rosner -- no message string, making debugging difficult. | `rosner.py:26,29,32,38` | Add descriptive messages to all `raise ValueError` calls |
| 6 | **`grubbs_test` computes on raw `data` instead of `x`** -- converts to numpy array `x` but uses original `data` param for mean/std/z_scores. Breaks if `data` is a plain list. | `grubb.py:37-40` | Replace `data` with `x` on lines 37-40 |
| 7 | **Division by zero risk in Dixon** -- if all values are identical, `data_range` = 0 causes `ZeroDivisionError`. | `dixon.py:81,92,106` | Guard: if `data_range == 0`, return early (no outlier possible) |
| 8 | **Grubbs `std(ddof=1)` with zero std dev** -- no guard for `sd_x == 0`. | `grubb.py:38` | Add zero standard deviation check |
| 9 | **Missing `__version__`** in package `__init__.py`. | `outlier_tests/__init__.py` | Add `__version__ = "0.1.0"` |

## P2: Medium (Library best practices and packaging)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 10 | **`logging.basicConfig()` at import time** -- libraries must not configure the root logger. | `basic_logger.py` | Use `logging.getLogger(__name__)` with `NullHandler`. Remove `basicConfig`. |
| 11 | **Empty README** -- no installation, usage examples, or API docs. | `README.md` | Write full README with examples for each test |
| 12 | **Empty package description** in pyproject.toml. | `pyproject.toml:4` | Add description string |
| 13 | **Missing return type annotations** on `grubbs_test` and `rosner_test`. | `grubb.py:7`, `rosner.py:8` | Add `-> dict` or typed `Dict` return annotations |
| 14 | **Inconsistent return key naming** -- Grubbs returns `"z_score"` but tests read `"zscore"`. | `grubb.py:55`, `tests/test_outliers.py:83` | Align the key name |
| 15 | **No `py.typed` marker** -- PEP 561 marker needed for type checkers. | `outlier_tests/` | Add empty `py.typed` file |

## P3: Standard (Testing and CI)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 16 | **No CI/CD pipeline** -- no automated testing on push/PR. | project root | Add GitHub Actions workflow for lint, type-check, test across Python 3.10-3.12 |
| 17 | **No test coverage tracking**. | `pyproject.toml` | Add `pytest-cov`, configure coverage reporting |
| 18 | **No linting/formatting** configured. | `pyproject.toml` | Add `ruff` as dev dependency with config |
| 19 | **No type checking** -- hints exist but `mypy` is not configured. | `pyproject.toml` | Add `mypy` to dev deps, configure `[tool.mypy]` |
| 20 | **Tests don't cover error paths** -- no tests for invalid inputs. | `tests/` | Add parametrized tests for all validation branches |
| 21 | **Tests use manual loops** instead of `pytest.mark.parametrize` -- one failure stops the entire test. | `tests/test_outliers.py` | Refactor to parametrize |
| 22 | **No `conftest.py`** -- shared fixtures belong in conftest. | `tests/` | Move fixtures to `tests/conftest.py` |

## P4: Low (Code quality polish)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 23 | **Typo in JSON key** -- `"grubs"` (one 'b') vs `grubbs_data` in tests. | `tests/test_outliers.py:30`, `tests/test_data/results.json` | Use consistent spelling |
| 24 | **Dixon p-value is a placeholder** -- returns alpha if reject, else 1.0. | `dixon.py:129` | Compute a real p-value or mark field as `None` |
| 25 | **Magic number `n - 2` degrees of freedom** -- no literature reference. | `dixon.py:29`, `grubb.py:45`, `rosner.py:82` | Add citations in docstrings |
| 26 | **Rosner output mixes numpy arrays and pandas DataFrame**. | `rosner.py:111-117` | Document clearly or convert for consistency |
| 27 | **No module-level docstrings** on source files. | all `.py` files | Add brief module docstrings |
| 28 | **Typo `"rosnor_q"`** in test data/code (should be `"rosner_q"`). | `tests/test_data/results.json`, `tests/test_outliers.py:50` | Fix spelling |
| 29 | **Extra closing parenthesis** in error message `'...observations.)'`. | `rosner.py:25` | Remove extra `)` |

---

## Summary

| Priority | Count | Theme |
|----------|-------|-------|
| P0 Critical | 3 | Tests broken, missing gitignore, wrong dep classification |
| P1 High | 6 | Correctness bugs, misleading errors, missing version |
| P2 Medium | 6 | Library anti-patterns, packaging, docs |
| P3 Standard | 7 | CI/CD, coverage, linting, type checking, test structure |
| P4 Low | 7 | Typos, code style, documentation niceties |
| **Total** | **29** | |

**Recommended execution order:** P0 first (gets tests passing), then P1 (correctness), then P2+P3 together (CI and packaging), P4 as polish.
