"""Statistical outlier detection tests (Dixon, Grubbs, Rosner)."""

from .dixon import dixon_test
from .grubb import grubbs_test
from .rosner import rosner_test

__version__ = "0.1.0"
__all__ = ["dixon_test", "grubbs_test", "rosner_test"]