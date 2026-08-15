"""Compatibility facade for published result indexing and comparison output."""

from .publication_services import results as _impl
from .publication_services.results import ResultEntry, index_results, write_comparison

__all__ = ["ResultEntry", "index_results", "write_comparison"]


def __getattr__(name: str):
    return getattr(_impl, name)
