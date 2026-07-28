"""Register the maintained built-in components exactly once."""

from __future__ import annotations

from ..methods.ap01.pipeline import AP01Method
from ..methods.ap02.pipeline import AP02Method
from ..methods.ap03.pipeline import AP03Method
from ..registry import (
    calibration_methods,
    evaluators,
    experiment_providers,
    input_adapters,
)
from .evaluation import MarkerConsistencyEvaluator
from .experiments import ColmapMatcherExperiments
from .inputs import (
    FilesystemInputAdapter,
    McapInputAdapter,
    PreparedInputAdapter,
    SimulationInputAdapter,
)


def register_builtin_components() -> None:
    """Populate all registries while preserving idempotent startup."""

    components = (
        (input_adapters, PreparedInputAdapter()),
        (input_adapters, SimulationInputAdapter()),
        (input_adapters, FilesystemInputAdapter()),
        (input_adapters, McapInputAdapter()),
        (calibration_methods, AP01Method()),
        (calibration_methods, AP02Method()),
        (calibration_methods, AP03Method()),
        (evaluators, MarkerConsistencyEvaluator()),
        (experiment_providers, ColmapMatcherExperiments()),
    )
    for registry, component in components:
        if component.id not in registry:
            registry.register(component)
