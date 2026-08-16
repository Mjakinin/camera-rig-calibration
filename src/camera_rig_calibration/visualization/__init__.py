"""ROS-independent scene generation and optional isolated RViz sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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


def _set_rviz_visible_methods(
    rviz_path: Path,
    visible_methods: Iterable[str],
) -> None:
    """Select camera-pose layers without changing any published result data."""
    selected = {str(method).strip().lower() for method in visible_methods}
    if not selected:
        raise ValueError("At least one RViz method must be selected")
    lines = rviz_path.read_text(encoding="utf-8").splitlines()
    method_display = False
    enabled_value: bool | None = None
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Name: "):
            name = stripped.removeprefix("Name: ").strip()
            # Camera-pose display names are exactly method/label. Auxiliary
            # layers append descriptive suffixes and remain off by default.
            method_display = (
                "/" in name
                and not name.endswith(" anchor edges")
                and not name.endswith(" estimate-to-GT errors")
            )
            enabled_value = None
            if method_display:
                method = name.split("/", 1)[0].strip().lower()
                enabled_value = method in selected
        if method_display and enabled_value is not None and stripped.startswith("Enabled:"):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}Enabled: {'true' if enabled_value else 'false'}"
            method_display = False
            enabled_value = None
        updated.append(line)
    rviz_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def launch_isolated_rviz(
    experiment_root: Path,
    repository_root: Path,
    *,
    visible_methods: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Launch the best already-published scene without rerunning calibration.

    Visualization source priority:
    AP03 Multi COLMAP -> AP01 moving COLMAP -> pose-only common/native 6DOF.
    ``visible_methods`` only controls which already-published camera layers are
    initially enabled in RViz; it never changes calibration or evaluation.
    """

    experiment_root = Path(experiment_root).resolve()
    primary = ensure_visualization_artifacts(experiment_root)
    primary_source = primary.get("point_cloud_source", {})
    if (
        primary.get("available")
        and isinstance(primary_source, dict)
        and primary_source.get("method") == "ap03_multi"
    ):
        if visible_methods is not None:
            _set_rviz_visible_methods(
                experiment_root / "visualization" / "rigcal_result.rviz",
                visible_methods,
            )
        return launch_visualization_directory(
            experiment_root / "visualization",
            experiment_root,
            repository_root,
            scene_label="ap03_colmap",
        )

    fallback = ensure_fallback_visualization_artifacts(experiment_root)
    if fallback.get("available"):
        if visible_methods is not None:
            rviz_path = experiment_root / "visualization" / "rigcal_result.rviz"
            if rviz_path.is_file():
                _set_rviz_visible_methods(rviz_path, visible_methods)
        return launch_visualization_directory(
            experiment_root / "visualization",
            experiment_root,
            repository_root,
            scene_label=str(fallback.get("selected_source") or "fallback"),
        )

    if primary.get("available"):
        if visible_methods is not None:
            _set_rviz_visible_methods(
                experiment_root / "visualization" / "rigcal_result.rviz",
                visible_methods,
            )
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
        selected_scene = native_scenes[0]
        visualization_root = ensure_ap02_native_scene(
            experiment_root, selected_scene
        )
        return launch_visualization_directory(
            visualization_root,
            experiment_root,
            repository_root,
            scene_label=str(selected_scene.get("scene_id") or "ap02_native"),
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
    "_set_rviz_visible_methods",
    "launch_isolated_rviz",
]
