"""Built-in rigcal components and their public registration entry point.

The actual calibration implementations live below :mod:`camera_rig_calibration.methods`.
This package contains only the adapters that connect inputs, methods,
evaluators and experiment variants to the runtime registries.

Public adapters are imported lazily.  In particular, importing
``components.common`` from one calibration method must not initialize the
registration module and recursively import that same partially initialized
method.
"""


def register_builtin_components() -> None:
    """Register all built-in adapters without eager package-level imports."""
    from .registration import register_builtin_components as _register

    _register()


def __getattr__(name: str):
    """Resolve the historical public component exports on first access."""
    if name == "AP01Method":
        from ..methods.ap01.pipeline import AP01Method

        return AP01Method
    if name == "AP02Method":
        from ..methods.ap02.pipeline import AP02Method

        return AP02Method
    if name == "AP03Method":
        from ..methods.ap03.pipeline import AP03Method

        return AP03Method
    if name == "MarkerConsistencyEvaluator":
        from .evaluation import MarkerConsistencyEvaluator

        return MarkerConsistencyEvaluator
    if name == "ColmapMatcherExperiments":
        from .experiments import ColmapMatcherExperiments

        return ColmapMatcherExperiments
    if name in {
        "FilesystemInputAdapter",
        "McapInputAdapter",
        "PreparedInputAdapter",
        "SimulationInputAdapter",
    }:
        from .inputs import (
            FilesystemInputAdapter,
            McapInputAdapter,
            PreparedInputAdapter,
            SimulationInputAdapter,
        )

        return {
            "FilesystemInputAdapter": FilesystemInputAdapter,
            "McapInputAdapter": McapInputAdapter,
            "PreparedInputAdapter": PreparedInputAdapter,
            "SimulationInputAdapter": SimulationInputAdapter,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AP01Method",
    "AP02Method",
    "AP03Method",
    "ColmapMatcherExperiments",
    "FilesystemInputAdapter",
    "MarkerConsistencyEvaluator",
    "McapInputAdapter",
    "PreparedInputAdapter",
    "SimulationInputAdapter",
    "register_builtin_components",
]
