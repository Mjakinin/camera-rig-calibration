"""Late-bound dependencies used by the AP01 compatibility facade."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AP01CoreBindings:
    """Resolve facade hooks at execution time so monkey-patches keep working."""

    parse_args: Callable[..., Any]
    load_camera_info: Callable[..., Any]
    run_colmap: Callable[..., Any]
    parse_colmap_poses: Callable[..., Any]
    read_csv: Callable[..., Any]
    prepare_observations: Callable[..., Any]
    robust_scale: Callable[..., Any]
    write_csv: Callable[..., Any]
    best_static_by_camera_marker: Callable[..., Any]
    moving_by_marker: Callable[..., Any]
    direct_candidates: Callable[..., Any]
    relay_candidates: Callable[..., Any]
    aggregate_candidates: Callable[..., Any]
    write_status: Callable[..., Any]

    @classmethod
    def current(cls) -> "AP01CoreBindings":
        from . import core

        return cls(
            parse_args=core.parse_args,
            load_camera_info=core.load_camera_info,
            run_colmap=core.run_colmap,
            parse_colmap_poses=core.parse_colmap_poses,
            read_csv=core.read_csv,
            prepare_observations=core.prepare_observations,
            robust_scale=core.robust_scale,
            write_csv=core.write_csv,
            best_static_by_camera_marker=core.best_static_by_camera_marker,
            moving_by_marker=core.moving_by_marker,
            direct_candidates=core.direct_candidates,
            relay_candidates=core.relay_candidates,
            aggregate_candidates=core.aggregate_candidates,
            write_status=core.write_status,
        )
