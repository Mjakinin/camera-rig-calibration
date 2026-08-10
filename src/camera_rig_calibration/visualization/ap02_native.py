from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..anchor_export.adapters import load_camera_poses
from ..anchor_export.geometry import pose_payload
from . import scene as common_scene


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scene"


def _primary_reference(result_payload: dict[str, Any]) -> int | None:
    config = result_payload.get("config_summary", {})
    if not isinstance(config, dict):
        return None
    for key in ("resolved_reference_marker_id", "reference_marker_id"):
        try:
            return int(config.get(key))
        except (TypeError, ValueError):
            continue
    return None


def discover_ap02_native_scenes(experiment_root: Path) -> list[dict[str, Any]]:
    """Return independently valid AP02 local-frame scenes.

    Disconnected components are deliberately separate scenes. They are never
    overlaid with one another or with common-anchor methods because the required
    cross-component transform is not observable from AP02 itself.
    """

    root = Path(experiment_root).resolve()
    scenes: list[dict[str, Any]] = []
    for result_path in sorted((root / "methods" / "ap02").glob("*/RESULT.json")):
        result_root = result_path.parent
        result = _read_json(result_path)
        label = str(result.get("label") or result_root.name)
        component_path = (
            result_root
            / "diagnostics"
            / "method"
            / "component_diagnostics"
            / "AP02_COMPONENT_RESULTS.json"
        )
        components = _read_json(component_path)
        rows = [
            item
            for item in components.get("components", [])
            if isinstance(item, dict)
        ]
        primary_id = str(components.get("primary_component_id") or "")
        primary_reference = _primary_reference(result)

        if primary_reference is not None:
            primary = next(
                (
                    item
                    for item in rows
                    if str(item.get("component_id")) == primary_id
                ),
                {},
            )
            cameras = list(primary.get("static_cameras", []))
            if not cameras:
                cameras = sorted(load_camera_poses(result_root))
            if cameras:
                scenes.append(
                    {
                        "scene_id": f"ap02__{label}__{primary_id or 'primary'}",
                        "display_name": (
                            f"AP02 primary {primary_id or 'component'} — "
                            f"marker {primary_reference} native frame, "
                            f"{len(cameras)} cameras"
                        ),
                        "method": "ap02",
                        "label": label,
                        "component_id": primary_id or "primary",
                        "reference_marker_id": primary_reference,
                        "pose_root": result_root,
                        "static_cameras": sorted(map(str, cameras)),
                        "role": "primary",
                    }
                )

        for item in rows:
            component_id = str(item.get("component_id") or "")
            if component_id == primary_id:
                continue
            if str(item.get("execution_status")) != "available":
                continue
            try:
                reference = int(
                    item.get("local_reference_marker_id", item.get("anchor_marker_id"))
                )
            except (TypeError, ValueError):
                continue
            cameras = sorted(map(str, item.get("static_cameras", [])))
            pose_root = (
                result_root
                / "diagnostics"
                / "method"
                / "component_diagnostics"
                / component_id
            )
            if len(cameras) < 2 or not (pose_root / "camera_extrinsics.csv").is_file():
                continue
            scenes.append(
                {
                    "scene_id": f"ap02__{label}__{component_id}",
                    "display_name": (
                        f"AP02 {component_id} — marker {reference} native frame, "
                        f"{len(cameras)} cameras"
                    ),
                    "method": "ap02",
                    "label": label,
                    "component_id": component_id,
                    "reference_marker_id": reference,
                    "pose_root": pose_root,
                    "static_cameras": cameras,
                    "role": "diagnostic_component",
                }
            )
    return scenes


def _native_rviz_config(*, fixed_frame: str, method: str, label: str) -> str:
    camera_topic = common_scene._topic(method, label, "cameras")
    edge_topic = common_scene._topic(method, label, "anchor_edges")
    return (
        "Panels:\n"
        "  - Class: rviz_common/Displays\n"
        "Visualization Manager:\n"
        f"  Global Options:\n    Fixed Frame: {fixed_frame}\n"
        "    Background Color: 35; 35; 35\n"
        "  Displays:\n"
        + common_scene._rviz_display(
            name="Local AP02 reference marker",
            topic="/rigcal/scene/anchor",
            enabled=True,
        )
        + "\n"
        + common_scene._rviz_display(
            name=f"{method}/{label} cameras",
            topic=camera_topic,
            enabled=True,
        )
        + "\n"
        + common_scene._rviz_display(
            name=f"{method}/{label} reference edges",
            topic=edge_topic,
            enabled=True,
        )
        + "\n"
        "  Tools:\n"
        "    - Class: rviz_default_plugins/Interact\n"
        "    - Class: rviz_default_plugins/MoveCamera\n"
        "Window Geometry:\n  Width: 1400\n  Height: 900\n"
    )


def ensure_ap02_native_scene(
    experiment_root: Path,
    scene_descriptor: dict[str, Any],
) -> Path:
    root = Path(experiment_root).resolve()
    pose_root = Path(scene_descriptor["pose_root"]).resolve()
    reference = int(scene_descriptor["reference_marker_id"])
    cameras = set(map(str, scene_descriptor.get("static_cameras", [])))
    poses = {
        camera: transform
        for camera, transform in load_camera_poses(pose_root).items()
        if not cameras or camera in cameras
    }
    if not poses:
        raise RuntimeError(
            f"AP02 native RViz scene has no camera poses: {pose_root}"
        )

    scene_id = _safe(str(scene_descriptor["scene_id"]))
    output = root / "visualization" / "native_ap02" / scene_id
    output.mkdir(parents=True, exist_ok=True)
    label = _safe(
        f"{scene_descriptor.get('label', 'ap02')}__"
        f"{scene_descriptor.get('component_id', 'component')}__native_ref{reference}"
    )
    fixed_frame = f"ap02_marker_{reference}"
    camera_rows = [
        {"camera_id": camera, **pose_payload(transform)}
        for camera, transform in sorted(poses.items())
    ]
    pose_variant = {
        "method": "ap02",
        "label": label,
        "camera_topic": common_scene._topic("ap02", label, "cameras"),
        "anchor_edges_topic": common_scene._topic(
            "ap02", label, "anchor_edges"
        ),
        "error_lines_topic": common_scene._topic("ap02", label, "error_lines"),
        "cameras": camera_rows,
    }
    frustums = []
    for camera, transform in sorted(poses.items()):
        intrinsics = common_scene._intrinsics(common_scene._camera_info(root, camera))
        if intrinsics is None:
            # Visualization-only fallback. It changes only frustum shape, never
            # the published camera pose or any calibration/evaluation quantity.
            intrinsics = (1.0, 1.0, 0.5, 0.5, 1, 1)
        frustums.append(
            {
                "camera_id": camera,
                "points": common_scene._frustum(transform, intrinsics, 0.34),
            }
        )

    common_scene._write_ply(output / "scene_anchor_frame.ply", [])
    common_scene._write_json(
        output / "poses_anchor_frame.json",
        {
            "schema_version": 2,
            "fixed_frame": fixed_frame,
            "variants": [pose_variant],
            "ground_truth": {
                "available": False,
                "cameras": [],
                "reason": "native AP02 local-frame view; no GT alignment is applied",
            },
        },
    )
    common_scene._write_json(
        output / "camera_frustums.json",
        {
            "schema_version": 2,
            "display_depth_m": 0.34,
            "variants": [
                {"method": "ap02", "label": label, "frustums": frustums}
            ],
        },
    )
    common_scene._atomic_text(
        output / "rigcal_result.rviz",
        _native_rviz_config(
            fixed_frame=fixed_frame,
            method="ap02",
            label=label,
        ),
    )
    common_scene._write_json(
        output / "visualization_manifest.json",
        {
            "schema_version": 2,
            "contract": "rigcal_rviz_native_ap02_v1",
            "available": True,
            "status": "OK_NATIVE_PARTIAL",
            "fixed_frame": fixed_frame,
            "point_cloud": "scene_anchor_frame.ply",
            "point_count": 0,
            "poses": "poses_anchor_frame.json",
            "frustums": "camera_frustums.json",
            "rviz_config": "rigcal_result.rviz",
            "variants": [
                {
                    "method": "ap02",
                    "label": label,
                    "component_id": scene_descriptor.get("component_id"),
                    "reference_marker_id": reference,
                    "camera_count": len(camera_rows),
                    "default_visible": True,
                    "anchor_edges_default_visible": True,
                }
            ],
            "scientific_scope": (
                "native AP02 component only; cross-component extrinsics are not "
                "observable and no alignment to AP03/common evaluation is applied"
            ),
            "ground_truth_used": False,
        },
    )
    return output


__all__ = ["discover_ap02_native_scenes", "ensure_ap02_native_scene"]
