"""Late-bound policy hooks for queue preflight phases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Hook = Callable[..., Any]


@dataclass(frozen=True)
class PreflightDependencies:
    register_builtin_components: Hook
    observation_camera_id: Hook
    read_observation_rows: Hook
    write_json: Hook
    write_ap02_graph_diagnosis: Hook
    filter_observations: Hook
    resolve_selections: Hook
    graph_components: Hook
    diagnose_ap02_graph: Hook
    select_ap02_frames: Hook
    write_ap02_frame_selection: Hook
    write_selection_candidates_csv: Hook
    effective_observation_quality: Hook

    @classmethod
    def current(cls) -> "PreflightDependencies":
        from . import api as preflight
        from ..observation_services import api as observations

        return cls(
            register_builtin_components=preflight.register_builtin_components,
            observation_camera_id=preflight._observation_camera_id,
            read_observation_rows=preflight._read_observation_rows,
            write_json=preflight._write_json,
            write_ap02_graph_diagnosis=preflight._write_ap02_graph_diagnosis,
            filter_observations=preflight.filter_observations,
            resolve_selections=observations.resolve_selections,
            graph_components=preflight.graph_components,
            diagnose_ap02_graph=preflight.diagnose_ap02_graph,
            select_ap02_frames=preflight.select_ap02_frames,
            write_ap02_frame_selection=preflight.write_ap02_frame_selection,
            write_selection_candidates_csv=(
                preflight.write_selection_candidates_csv
            ),
            effective_observation_quality=(
                preflight.effective_observation_quality
            ),
        )
