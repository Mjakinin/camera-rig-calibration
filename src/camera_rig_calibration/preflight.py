"""Compatibility facade for phased queue preflight."""
from __future__ import annotations

from .methods.ap02.graph_diagnostics import (
    AP02GraphDiagnosis,
    diagnose_ap02_graph,
    graph_components,
)
from .components import register_builtin_components
from .config.models import RigConfig, effective_observation_quality
from .contracts import RunContext
from .methods.ap02.frame_selection import (
    AP02FrameSelectionError,
    select_ap02_frames,
    write_ap02_frame_selection,
)
from .observation_quality import (
    ObservationFilterResult,
    ObservationQualityError,
    filter_observations,
)
from .observations import (
    ResolvedSelections,
    resolve_selections,
    write_selection_candidates_csv,
)
from .preflight_services.coordinator import run_queue_preflight
from .preflight_services.core import (
    CameraObservationCoverage,
    PreflightJob,
    PreflightJobResult,
    QueuePreflightResult,
    _copy_filter_artifacts,
    _observation_camera_id,
    _read_observation_rows,
    _write_ap02_graph_diagnosis,
    _write_json,
    build_queue_camera_coverage,
)
from .registry import calibration_methods
