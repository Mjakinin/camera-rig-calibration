"""Fallback RViz scenes from already-published calibration artifacts.

Source priority is deliberately visualization-only:
1. AP03 is handled by :mod:`visualization.scene`.
2. If AP03 context is unavailable, reuse AP01 moving-COLMAP + metric scale.
3. If no point cloud is available, render anchor-relative/native 6DOF poses only.

No calibration method is rerun and no scientific result is modified.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..anchor_export.adapters import load_camera_poses
from ..anchor_export.geometry import (
    invert_transform,
    robust_pose_average,
    validate_transform,
)
from ..methods.ap01.core_geometry import T_from_observation, parse_colmap_poses
from . import scene as common_scene


FALLBACK_CONTRACT = "rigcal_rviz_fallback_scene_v1"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int_value(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _score(row: dict[str, str]) -> float:
    try:
        value = float(row.get("selection_score") or 1.0)
    except ValueError:
        value = 1.0
    return max(value, 1e-9)


def _anchor_method_transform(
    variant: dict[str, Any],
    native: dict[str, np.ndarray],
    root_camera: str,
) -> np.ndarray | None:
    payload = variant.get("payload", {})
    alignment = payload.get("anchor_alignment", {})
    matrix = (
        alignment.get("transform_anchor_method", {}).get("matrix")
        if isinstance(alignment, dict)
        else None
    )
    if matrix is not None:
        try:
            return validate_transform(np.asarray(matrix, dtype=np.float64))
        except (TypeError, ValueError):
            pass

    anchored = {
        str(item.get("camera_id")): item
        for item in payload.get("cameras", [])
        if isinstance(item, dict) and item.get("camera_id")
    }
    root_payload = anchored.get(root_camera, {})
    if root_payload.get("matrix") is None or root_camera not in native:
        return None
    try:
        anchor_root = validate_transform(
            np.asarray(root_payload["matrix"], dtype=np.float64)
        )
        return anchor_root @ invert_transform(native[root_camera])
    except (TypeError, ValueError):
        return None


def _ap01_colmap_source(
    experiment_root: Path,
    variants: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    candidates = [item for item in variants if item.get("method") == "ap01"]
    candidates.sort(
        key=lambda item: (
            item.get("label") != "baseline",
            -len(item.get("payload", {}).get("cameras", [])),
            str(item.get("label", "")),
        )
    )
    for variant in candidates:
        result_root = Path(variant["path"]).parent
        colmap_root = (
            result_root
            / "diagnostics"
            / "method"
            / "01_moving_colmap"
            / "sparse_txt_best"
        )
        points_path = colmap_root / "points3D.txt"
        images_path = colmap_root / "images.txt"
        scale_path = (
            result_root
            / "diagnostics"
            / "method"
            / "02_metric_scale"
            / "metric_scale.txt"
        )
        observations_path = (
            result_root
            / "diagnostics"
            / "preflight"
            / "accepted_observations.csv"
        )
        missing = [
            path
            for path in (points_path, images_path, scale_path, observations_path)
            if not path.is_file()
        ]
        if missing:
            warnings.append(
                f"{variant['label']}: AP01 visualization inputs are incomplete: "
                + ", ".join(path.name for path in missing)
            )
            continue
        try:
            scale = float(scale_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            warnings.append(f"{variant['label']}: AP01 metric scale is invalid")
            continue
        if not np.isfinite(scale) or scale <= 0.0:
            warnings.append(f"{variant['label']}: AP01 metric scale is non-positive")
            continue
        try:
            colmap_poses = parse_colmap_poses(images_path)
        except RuntimeError as exc:
            warnings.append(f"{variant['label']}: {exc}")
            continue
        native = load_camera_poses(result_root)
        if not native:
            warnings.append(f"{variant['label']}: AP01 camera poses are unavailable")
            continue
        identity = np.eye(4, dtype=np.float64)
        root_camera = min(
            native,
            key=lambda camera: float(np.linalg.norm(native[camera] - identity)),
        )
        transform_anchor_method = _anchor_method_transform(
            variant, native, root_camera
        )
        if transform_anchor_method is None:
            warnings.append(
                f"{variant['label']}: AP01 common-anchor transform is unavailable"
            )
            continue

        observation_rows = _rows(observations_path)
        static_by_marker: dict[int, dict[str, str]] = {}
        for row in observation_rows:
            if str(row.get("observer_type", "")).lower() != "static":
                continue
            camera = str(row.get("observer_id") or row.get("camera_name") or "")
            if camera != root_camera:
                continue
            marker = _int_value(row.get("marker_id"))
            if marker is None:
                continue
            previous = static_by_marker.get(marker)
            if previous is None or _score(row) > _score(previous):
                static_by_marker[marker] = row

        transforms: list[np.ndarray] = []
        weights: list[float] = []
        supports: list[dict[str, Any]] = []
        for row in observation_rows:
            if str(row.get("observer_type", "")).lower() != "moving":
                continue
            marker = _int_value(row.get("marker_id"))
            frame = _int_value(row.get("frame_id"))
            if marker is None or frame is None:
                continue
            static = static_by_marker.get(marker)
            colmap_pose = colmap_poses.get(frame)
            if static is None or colmap_pose is None:
                continue
            try:
                root_marker = T_from_observation(static)
                moving_marker = T_from_observation(row)
            except RuntimeError:
                continue
            root_moving = root_marker @ invert_transform(moving_marker)
            metric_world_to_moving = np.asarray(
                colmap_pose, dtype=np.float64
            ).copy()
            metric_world_to_moving[:3, 3] *= scale
            root_colmap_world = root_moving @ metric_world_to_moving
            if not np.all(np.isfinite(root_colmap_world)):
                continue
            transforms.append(root_colmap_world)
            weights.append((_score(static) * _score(row)) ** 0.5)
            supports.append(
                {
                    "marker_id": marker,
                    "frame_id": frame,
                    "static_camera": root_camera,
                }
            )
        if not transforms:
            warnings.append(
                f"{variant['label']}: no accepted AP01 marker/COLMAP alignment chain exists"
            )
            continue
        try:
            aggregate = robust_pose_average(transforms, weights)
        except ValueError as exc:
            warnings.append(f"{variant['label']}: AP01 COLMAP alignment failed: {exc}")
            continue
        transform_anchor_colmap = (
            transform_anchor_method @ aggregate.transform
        )
        raw_points = common_scene._read_colmap_points(points_path)
        if not raw_points:
            warnings.append(f"{variant['label']}: AP01 COLMAP point cloud is empty")
            continue
        return (
            {
                **variant,
                "root": result_root,
                "points_path": points_path,
                "images_path": images_path,
                "scale_path": scale_path,
                "observations_path": observations_path,
                "scale": scale,
                "raw_points": raw_points,
                "point_count": len(raw_points),
                "root_camera": root_camera,
                "alignment_support_count": len(transforms),
                "alignment_inlier_count": len(aggregate.inlier_indices),
                "alignment_supports": supports,
                "transform_anchor_colmap": transform_anchor_colmap,
            },
            warnings,
        )
    return None, warnings


def _variant_frame(item: dict[str, Any]) -> str:
    return str(item.get("payload", {}).get("parent_frame") or "")


def _select_pose_variants(
    variants: list[dict[str, Any]],
    preferred_frame: str | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    usable = [
        item
        for item in variants
        if _variant_frame(item)
        and item.get("payload", {}).get("cameras")
    ]
    if preferred_frame:
        selected = [item for item in usable if _variant_frame(item) == preferred_frame]
        if selected:
            return preferred_frame, selected
    frames: dict[str, list[dict[str, Any]]] = {}
    for item in usable:
        frames.setdefault(_variant_frame(item), []).append(item)
    if not frames:
        return None, []
    frame = min(
        frames,
        key=lambda value: (
            -sum(len(item["payload"].get("cameras", [])) for item in frames[value]),
            value,
        ),
    )
    return frame, frames[frame]


def _rviz_config(
    fixed_frame: str,
    variants: list[dict[str, Any]],
    *,
    point_cloud_name: str | None,
    ground_truth_available: bool,
) -> str:
    displays: list[str] = []
    if point_cloud_name:
        displays.append(
            common_scene._rviz_display(
                name=point_cloud_name,
                topic="/rigcal/scene/points",
                enabled=True,
                class_name="rviz_default_plugins/PointCloud2",
            )
        )
    displays.append(
        common_scene._rviz_display(
            name="Anchor / local origin",
            topic="/rigcal/scene/anchor",
            enabled=True,
        )
    )
    if ground_truth_available:
        displays.append(
            common_scene._rviz_display(
                name="Ground truth cameras",
                topic="/rigcal/ground_truth/cameras",
                enabled=True,
            )
        )
    for item in variants:
        name = f"{item['method']}/{item['label']}"
        displays.append(
            common_scene._rviz_display(
                name=name,
                topic=common_scene._topic(item["method"], item["label"], "cameras"),
                enabled=True,
            )
        )
        displays.append(
            common_scene._rviz_display(
                name=f"{name} anchor edges",
                topic=common_scene._topic(
                    item["method"], item["label"], "anchor_edges"
                ),
                enabled=False,
            )
        )
    return (
        "Panels:\n"
        "  - Class: rviz_common/Displays\n"
        "Visualization Manager:\n"
        f"  Global Options:\n    Fixed Frame: {fixed_frame}\n"
        "    Background Color: 35; 35; 35\n"
        "  Displays:\n"
        + "\n".join(displays)
        + "\n"
        "  Tools:\n"
        "    - Class: rviz_default_plugins/Interact\n"
        "    - Class: rviz_default_plugins/MoveCamera\n"
        "Window Geometry:\n  Width: 1400\n  Height: 900\n"
    )


def _launch_script() -> str:
    return (
        "import json\n"
        "from pathlib import Path\n"
        "from camera_rig_calibration.visualization import launch_isolated_rviz\n\n"
        "if __name__ == '__main__':\n"
        "    experiment = Path(__file__).resolve().parent.parent\n"
        "    repository = next((p for p in experiment.parents if (p / 'pyproject.toml').is_file()), Path.cwd())\n"
        "    print(json.dumps(launch_isolated_rviz(experiment, repository), indent=2))\n"
    )


def _write_scene(
    experiment_root: Path,
    output: Path,
    variants: list[dict[str, Any]],
    *,
    fixed_frame: str,
    point_source: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    point_cloud_name: str | None = None
    transformed_points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    source_files = [Path(item["path"]) for item in variants]
    if point_source is not None:
        scale = float(point_source["scale"])
        transform = validate_transform(
            np.asarray(point_source["transform_anchor_colmap"], dtype=np.float64)
        )
        transformed_points = [
            (
                transform[:3, :3] @ (scale * point) + transform[:3, 3],
                color,
            )
            for point, color in point_source["raw_points"]
        ]
        source_files.extend(
            [
                Path(point_source["points_path"]),
                Path(point_source["images_path"]),
                Path(point_source["scale_path"]),
                Path(point_source["observations_path"]),
            ]
        )
        point_cloud_name = "AP01 moving-COLMAP context"
    common_scene._write_ply(output / "scene_anchor_frame.ply", transformed_points)

    anchor_marker_id: int | None = None
    for item in variants:
        anchor_marker_id = _int_value(item.get("payload", {}).get("anchor_marker_id"))
        if anchor_marker_id is not None:
            break
    gt_cameras: list[dict[str, Any]] = []
    gt_source: str | None = None
    if anchor_marker_id is not None and fixed_frame.startswith("evaluation_anchor_marker_"):
        gt_cameras, gt_source = common_scene._ground_truth_cameras(
            experiment_root, anchor_marker_id=anchor_marker_id
        )

    pose_variants: list[dict[str, Any]] = []
    frustum_variants: list[dict[str, Any]] = []
    for item in variants:
        cameras = [
            camera
            for camera in item.get("payload", {}).get("cameras", [])
            if isinstance(camera, dict) and camera.get("matrix") is not None
        ]
        pose_variants.append(
            {
                "method": item["method"],
                "label": item["label"],
                "camera_topic": common_scene._topic(
                    item["method"], item["label"], "cameras"
                ),
                "anchor_edges_topic": common_scene._topic(
                    item["method"], item["label"], "anchor_edges"
                ),
                "error_lines_topic": common_scene._topic(
                    item["method"], item["label"], "error_lines"
                ),
                "cameras": cameras,
            }
        )
        frustums = []
        for camera in cameras:
            camera_id = str(camera.get("camera_id", ""))
            intrinsics = common_scene._intrinsics(
                common_scene._camera_info(experiment_root, camera_id)
            )
            if intrinsics is None:
                intrinsics = (1.0, 1.0, 0.5, 0.5, 1, 1)
            transform = validate_transform(
                np.asarray(camera["matrix"], dtype=np.float64)
            )
            frustums.append(
                {
                    "camera_id": camera_id,
                    "points": common_scene._frustum(transform, intrinsics, 0.34),
                }
            )
        frustum_variants.append(
            {
                "method": item["method"],
                "label": item["label"],
                "frustums": frustums,
            }
        )

    common_scene._write_json(
        output / "poses_anchor_frame.json",
        {
            "schema_version": 2,
            "fixed_frame": fixed_frame,
            "variants": pose_variants,
            "ground_truth": {
                "namespace": "ground_truth",
                "topic": "/rigcal/ground_truth/cameras",
                "source": gt_source,
                "cameras": gt_cameras,
            },
        },
    )
    common_scene._write_json(
        output / "camera_frustums.json",
        {
            "schema_version": 2,
            "display_depth_m": 0.34,
            "variants": frustum_variants,
        },
    )
    common_scene._atomic_text(
        output / "rigcal_result.rviz",
        _rviz_config(
            fixed_frame,
            variants,
            point_cloud_name=point_cloud_name,
            ground_truth_available=bool(gt_cameras),
        ),
    )
    common_scene._atomic_text(output / "launch_rviz.py", _launch_script())

    fingerprint = hashlib.sha256(FALLBACK_CONTRACT.encode("utf-8"))
    for path in sorted(set(source_files)):
        if path.is_file():
            fingerprint.update(path.read_bytes())
    if point_source is not None:
        status = "OK_AP01_COLMAP_FALLBACK"
        selected_source = "ap01_colmap"
        warnings = [
            *warnings,
            "AP03 point-cloud context was unavailable; RViz reuses the already-published AP01 moving-COLMAP reconstruction and metric scale.",
            "This is visualization-only; AP01 calibration outputs are not recomputed or changed.",
        ]
    else:
        status = "OK_POSE_ONLY_6DOF"
        selected_source = "pose_only"
        warnings = [
            *warnings,
            "Pose-only RViz scene: the anchor/reference frame is the origin and camera 6DOF poses are shown without a COLMAP point cloud.",
        ]
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "contract": FALLBACK_CONTRACT,
        "fingerprint": fingerprint.hexdigest(),
        "status": status,
        "available": True,
        "fixed_frame": fixed_frame,
        "point_cloud": "scene_anchor_frame.ply",
        "point_count": len(transformed_points),
        "poses": "poses_anchor_frame.json",
        "frustums": "camera_frustums.json",
        "rviz_config": "rigcal_result.rviz",
        "publisher": "camera_rig_calibration.visualization.ros_scene",
        "source_priority": ["ap03_multi", "ap01_colmap", "pose_only"],
        "selected_source": selected_source,
        "variants": [
            {"method": item["method"], "label": item["label"]}
            for item in variants
        ],
        "ground_truth": {
            "available": bool(gt_cameras),
            "camera_count": len(gt_cameras),
            "source": gt_source,
        },
        "warnings": warnings,
    }
    if point_source is not None:
        manifest["point_cloud_source"] = {
            "method": "ap01",
            "label": point_source["label"],
            "points3D": str(
                Path(point_source["points_path"])
                .relative_to(experiment_root)
                .as_posix()
            ),
            "scale_m_per_colmap_unit": float(point_source["scale"]),
            "alignment_support_count": int(
                point_source["alignment_support_count"]
            ),
            "alignment_inlier_count": int(
                point_source["alignment_inlier_count"]
            ),
            "root_camera": point_source["root_camera"],
        }
    common_scene._write_json(output / "visualization_manifest.json", manifest)
    common_scene._update_result_status(experiment_root, status)
    return manifest


def ensure_fallback_visualization_artifacts(
    experiment_root: Path,
) -> dict[str, Any]:
    """Build the best non-AP03 scene without running any calibration method."""

    experiment_root = Path(experiment_root).resolve()
    output = experiment_root / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    variants = common_scene._method_variants(experiment_root)

    ap01_source, ap01_warnings = _ap01_colmap_source(
        experiment_root, variants
    )
    if ap01_source is not None:
        preferred_frame = _variant_frame(ap01_source)
        fixed_frame, selected = _select_pose_variants(
            variants, preferred_frame=preferred_frame
        )
        if fixed_frame and selected:
            return _write_scene(
                experiment_root,
                output,
                selected,
                fixed_frame=fixed_frame,
                point_source=ap01_source,
                warnings=ap01_warnings,
            )

    fixed_frame, selected = _select_pose_variants(variants)
    if fixed_frame and selected:
        return _write_scene(
            experiment_root,
            output,
            selected,
            fixed_frame=fixed_frame,
            point_source=None,
            warnings=ap01_warnings,
        )

    manifest = {
        "schema_version": 2,
        "contract": FALLBACK_CONTRACT,
        "status": "UNAVAILABLE_NO_VISUALIZABLE_6DOF",
        "available": False,
        "reason": (
            "No AP01 point-cloud alignment and no anchor/native 6DOF pose set "
            "is available for RViz."
        ),
        "warnings": ap01_warnings,
    }
    common_scene._write_json(output / "visualization_manifest.json", manifest)
    return manifest


__all__ = ["ensure_fallback_visualization_artifacts"]
