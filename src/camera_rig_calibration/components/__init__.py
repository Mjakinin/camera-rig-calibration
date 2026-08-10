"""Built-in rigcal components and their public registration entry point.

The actual calibration implementations live below :mod:`camera_rig_calibration.methods`.
This package contains only the adapters that connect inputs, methods,
evaluators and experiment variants to the runtime registries.

Method adapters and the registration routine are imported lazily.  This keeps
``components.common`` usable from an individual method module without forcing
``components.registration`` to import that same method again while it is only
partially initialized.
"""

from .evaluation import MarkerConsistencyEvaluator
from .experiments import ColmapMatcherExperiments
from .inputs import (
    FilesystemInputAdapter,
    McapInputAdapter,
    PreparedInputAdapter,
    SimulationInputAdapter,
)


def register_builtin_components() -> None:
    """Register all built-in adapters without eager method-package imports."""
    from .registration import register_builtin_components as _register

    _register()


def __getattr__(name: str):
    """Preserve the historical public method-adapter imports lazily."""
    if name == "AP01Method":
        from ..methods.ap01.pipeline import AP01Method

        return AP01Method
    if name == "AP02Method":
        from ..methods.ap02.pipeline import AP02Method

        return AP02Method
    if name == "AP03Method":
        from ..methods.ap03.pipeline import AP03Method

        return AP03Method
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
