"""Late-bound hooks used by queue policies and integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Hook = Callable[..., Any]


@dataclass(frozen=True)
class QueueBindings:
    pipeline_orchestrator: type
    run_queue_preflight: Hook
    publish_preparation_transaction: Hook
    publish_queue_transaction: Hook
    freeze_selections: Hook
    method_preflight_coverage: Hook
    subprocess_run: Hook


def current_queue_bindings() -> QueueBindings:
    from . import api as queueing

    return QueueBindings(
        pipeline_orchestrator=queueing.PipelineOrchestrator,
        run_queue_preflight=queueing.run_queue_preflight,
        publish_preparation_transaction=(
            queueing.publish_preparation_transaction
        ),
        publish_queue_transaction=queueing.publish_queue_transaction,
        freeze_selections=queueing.freeze_selections,
        method_preflight_coverage=queueing._method_preflight_coverage,
        subprocess_run=queueing.subprocess.run,
    )


__all__ = ["QueueBindings", "current_queue_bindings"]
