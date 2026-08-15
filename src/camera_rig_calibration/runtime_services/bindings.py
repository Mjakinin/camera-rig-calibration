"""Late-bound hooks for runtime product policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeBindings:
    command_heartbeat_seconds: float
    colmap_artifact_fingerprint: Callable[..., str]
    resolve_selections: Callable[..., Any]
    freeze_selections: Callable[..., Any]


def current_runtime_bindings() -> RuntimeBindings:
    from . import api as runtime
    from ..observation_services import api as observations

    return RuntimeBindings(
        command_heartbeat_seconds=runtime.COMMAND_HEARTBEAT_SECONDS,
        colmap_artifact_fingerprint=runtime.colmap_artifact_fingerprint,
        resolve_selections=observations.resolve_selections,
        freeze_selections=observations.freeze_selections,
    )


__all__ = ["RuntimeBindings", "current_runtime_bindings"]
