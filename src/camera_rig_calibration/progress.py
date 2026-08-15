"""Compatibility facade for runtime progress helpers."""

from .runtime_services import progress as _impl
from .runtime_services.progress import (
    COUNT_PATTERNS,
    STRUCTURED_PROGRESS,
    ProgressClock,
    ProgressEvent,
    progress_text,
    terminal_lines,
)

__all__ = [
    "COUNT_PATTERNS",
    "STRUCTURED_PROGRESS",
    "ProgressClock",
    "ProgressEvent",
    "progress_text",
    "terminal_lines",
]


def __getattr__(name: str):
    return getattr(_impl, name)
