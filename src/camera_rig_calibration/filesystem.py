"""Compatibility facade for cross-platform storage filesystem helpers."""

from .storage_services import filesystem as _impl
from .storage_services.filesystem import promote_directory, rename_with_retry

__all__ = ["promote_directory", "rename_with_retry"]


def __getattr__(name: str):
    return getattr(_impl, name)
