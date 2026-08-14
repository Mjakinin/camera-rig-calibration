"""ROS-independent scene generation and optional isolated RViz sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .scene import ensure_visualization_artifacts
from .session import launch_visualization_directory


def ensure_fallback_visualization_artifacts(
    experiment_root: Path,
) -> dict[str, Any]:
    """Build a non-AP03 fallback scene without loading it during package import."""

    from .fallback_scene import (
        ensure_fallback_visualization_artifacts as build_fallback,
    )

    return build_fallback(experiment_root)


def launch_isolated_rviz(
    experiment_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Launch the best already-published scene without rerunning calibration.

    Visualization source priority:
    AP03 Multi COLMAP -> AP01 moving COLMAP -> pose-only common/native 6DOF.
    """

    experiment_root = Path(experiment_root).resolve()
    primary = ensure_visualization_artifacts(experiment_root)
    primary_source = primary.get("point_cloud_source", {})
    if (
        primary.get("available")
        and isinstance(primary_source, dict)
        and primary_source.get("method") == "ap03_multi"
    ):
        return launch_visualization_directory(
            experiment_root / "visualization",
            experiment_root,
            repository_root,
            scene_label="ap03_colmap",
        )

    fallback = ensure_fallback_visualization_artifacts(experiment_root)
    if fallback.get("available"):
        return launch_visualization_directory(
            experiment_root / "visualization",
            experiment_root,
            repository_root,
            scene_label=str(fallback.get("selected_source") or "fallback"),
        )

    if primary.get("available"):
        return launch_visualization_directory(
            experiment_root / "visualization",
            experiment_root,
            repository_root,
            scene_label="canonical_6dof",
        )

    from .ap02_native import discover_ap02_native_scenes, ensure_ap02_native_scene

    native_scenes = discover_ap02_native_scenes(experiment_root)
    if native_scenes:
        native_scenes.sort(
            key=lambda item: (
                item.get("role") != "primary",
                str(item.get("scene_id", "")),
            )
        )
        selected = native_scenes[0]
        visualization_root = ensure_ap02_native_scene(
            experiment_root, selected
        )
        return launch_visualization_directory(
            visualization_root,
            experiment_root,
            repository_root,
            scene_label=str(selected.get("scene_id") or "ap02_native"),
        )

    reasons = [
        str(payload.get("reason") or payload.get("status") or "")
        for payload in (primary, fallback)
        if payload
    ]
    raise RuntimeError(
        "RViz visualization unavailable: "
        + "; ".join(reason for reason in reasons if reason)
    )


__all__ = [
    "ensure_visualization_artifacts",
    "ensure_fallback_visualization_artifacts",
    "launch_isolated_rviz",
]
