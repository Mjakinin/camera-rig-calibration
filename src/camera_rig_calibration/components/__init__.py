"""Built-in rigcal components and their public registration entry point.

The actual calibration implementations live below :mod:`camera_rig_calibration.methods`.
This package contains only the adapters that connect inputs, methods,
evaluators and experiment variants to the runtime registries.
"""

from .evaluation import MarkerConsistencyEvaluator
from .experiments import ColmapMatcherExperiments
from .inputs import (
    FilesystemInputAdapter,
    McapInputAdapter,
    PreparedInputAdapter,
    SimulationInputAdapter,
)
from .registration import register_builtin_components
from ..methods.ap01.pipeline import AP01Method
from ..methods.ap02.pipeline import AP02Method
from ..methods.ap03.pipeline import AP03Method

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
