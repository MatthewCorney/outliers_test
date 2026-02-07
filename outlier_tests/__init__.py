"""Statistical outlier detection tests."""

from .dixon import dixon_test
from .grubb import grubbs_test
from .mad import mad_test
from .rosner import rosner_test

__version__ = "0.2.0"
__all__ = ["dixon_test", "grubbs_test", "mad_test", "rosner_test"]