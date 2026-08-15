"""Compatibility facade for storage cleanup planning and execution."""

from .storage_services import cleanup as _impl
from .storage_services.cleanup import (
    CleanupPlan,
    CleanupTarget,
    build_cleanup_plan,
    build_preparation_cache_cleanup_plan,
    build_results_cleanup_plan,
    build_temporary_cleanup_plan,
    combine_cleanup_plans,
    execute_cleanup,
)

__all__ = [
    "CleanupPlan",
    "CleanupTarget",
    "build_cleanup_plan",
    "build_preparation_cache_cleanup_plan",
    "build_results_cleanup_plan",
    "build_temporary_cleanup_plan",
    "combine_cleanup_plans",
    "execute_cleanup",
]


def __getattr__(name: str):
    return getattr(_impl, name)
