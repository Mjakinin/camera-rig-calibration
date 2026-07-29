from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.adapters import load_camera_poses
from ..anchor_export.geometry import (
    invert_transform,
    pose_payload,
    validate_transform,
)


SCENE_CONTRACT = "rigcal_rviz_scene_v2"
SOURCE_MISMATCH = "UNAVAILABLE_AP03_SOURCE_MISMATCH"


def _ros_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_text(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _update_result_status(experiment_root: Path, status: str) -> None:
    for result_path in sorted(
        (experiment_root / "methods").glob("*/*/RESULT.json")
    ):
        payload = _read_json(result_path)
        if not payload:
            continue
        payload["visualization_status"] = status
        payload["visualization_manifest"] = (
            "../../../visualization/visualization_manifest.json"
        )
        _write_json(result_path, payload)


def _method_variants(experiment_root: Path) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for anchor_path in sorted(
        (experiment_root / "methods").glob(
            "*/*/camera_extrinsics_anchor.json"
        )
    ):
        payload = _read_json(anchor_path)
        status = payload.get("anchor_export_status", {})
        method = str(
            payload.get("method") or anchor_path.parents[1].name
        )
        if method == "ap03":
            # The shared container is provenance, not a third AP03 estimate.
            continue
        if not isinstance(status, dict) or not status.get("available"):
            continue
        variants.append(
            {
                "method": method,
                "label": str(
                    payload.get("label") or anchor_path.parent.name
                ),
                "path": anchor_path,
                "payload": payload,
            }
        )
    return variants


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _count_colmap_points(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _ap03_source(
    experiment_root: Path,
    variants: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve only the explicitly documented AP03 Multi best model."""
    warnings: list[str] = []
    candidates = [
        item for item in variants if item["method"] == "ap03_multi"
    ]
    candidates.sort(
        key=lambda item: (
            item["label"] != "baseline",
            -len(item["payload"].get("cameras", [])),
            item["label"],
        )
    )
    for variant in candidates:
        derived_root = Path(variant["path"]).parent
        provenance_path = (
            derived_root / "provenance" / "derived_result.json"
        )
        provenance = _read_json(provenance_path)
        container_value = provenance.get("shared_colmap_container")
        metadata_value = provenance.get("scale_metadata")
        camera_value = provenance.get("camera_pose_source")
        best_model_value = provenance.get("shared_colmap_best_model")
        if not all(
            value is not None
            for value in (
                container_value,
                metadata_value,
                camera_value,
                best_model_value,
            )
        ):
            warnings.append(
                f"{variant['label']}: derived AP03 provenance is incomplete"
            )
            continue
        container = experiment_root / str(container_value)
        metadata_path = experiment_root / str(metadata_value)
        camera_path = experiment_root / str(camera_value)
        best_model = str(best_model_value)
        if (
            not _within(container, experiment_root / "methods" / "ap03")
            or not _within(metadata_path, container)
            or not _within(camera_path, container)
        ):
            warnings.append(
                f"{variant['label']}: AP03 sources do not belong to the "
                "declared shared container"
            )
            continue
        metadata = _read_json(metadata_path)
        if str(metadata.get("best_model")) != best_model:
            warnings.append(
                f"{variant['label']}: best-model metadata mismatch"
            )
            continue
        points_path = (
            container
            / "diagnostics"
            / "method"
            / "colmap"
            / "reconstruction"
            / "sparse_txt"
            / best_model
            / "points3D.txt"
        )
        if not points_path.is_file():
            warnings.append(
                f"{variant['label']}: exact sparse_txt/{best_model}/"
                "points3D.txt is missing"
            )
            continue
        try:
            scale = float(provenance["scale_m_per_colmap_unit"])
            metadata_scale = float(metadata["scale_m_per_colmap_unit"])
            expected_points = int(metadata["num_sparse_points3d"])
        except (KeyError, TypeError, ValueError):
            warnings.append(
                f"{variant['label']}: scale or point-count metadata is invalid"
            )
            continue
        actual_points = _count_colmap_points(points_path)
        if (
            not np.isclose(scale, metadata_scale, rtol=0.0, atol=1e-15)
            or actual_points != expected_points
        ):
            warnings.append(
                f"{variant['label']}: scale or point-count consistency failed"
            )
            continue
        alignment = _read_json(
            derived_root / "diagnostics" / "anchor_alignment.json"
        ).get("alignment", {})
        transform_value = alignment.get("transform_anchor_method", {}).get(
            "matrix"
        )
        try:
            transform_anchor_method = validate_transform(
                np.asarray(transform_value, dtype=np.float64)
            )
        except (TypeError, ValueError):
            warnings.append(
                f"{variant['label']}: AP03 Multi anchor transform is invalid"
            )
            continue
        native = load_camera_poses(derived_root)
        anchored = {
            str(item.get("camera_id")): validate_transform(
                np.asarray(item["matrix"], dtype=np.float64)
            )
            for item in variant["payload"].get("cameras", [])
            if isinstance(item, dict)
            and item.get("camera_id")
            and item.get("matrix") is not None
        }
        if set(native) != set(anchored) or any(
            not np.allclose(
                transform_anchor_method @ native[camera],
                anchored[camera],
                rtol=0.0,
                atol=1e-8,
            )
            for camera in native
        ):
            warnings.append(
                f"{variant['label']}: transformed AP03 camera poses do not "
                "match the published Multi anchor export"
            )
            continue
        return (
            {
                **variant,
                "root": derived_root,
                "container": container,
                "provenance_path": provenance_path,
                "metadata": metadata,
                "metadata_path": metadata_path,
                "camera_path": camera_path,
                "points_path": points_path,
                "best_model": best_model,
                "scale": scale,
                "point_count": actual_points,
                "transform_anchor_method": transform_anchor_method,
            },
            warnings,
        )
    return None, warnings


def _read_colmap_points(
    path: Path,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 7:
            continue
        try:
            point = np.asarray(
                [float(fields[1]), float(fields[2]), float(fields[3])],
                dtype=np.float64,
            )
            color = tuple(
                max(0, min(255, int(fields[index])))
                for index in (4, 5, 6)
            )
        except ValueError:
            continue
        if np.all(np.isfinite(point)):
            points.append((point, color))
    return points


def _write_ply(
    path: Path,
    points: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
        f"{color[0]} {color[1]} {color[2]}"
        for point, color in points
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def _camera_info(
    experiment_root: Path, camera_id: str
) -> dict[str, Any]:
    return _read_json(
        experiment_root
        / "raw_images"
        / "camera_info"
        / f"{camera_id}.json"
    )


def _intrinsics(
    info: dict[str, Any],
) -> tuple[float, float, float, float, int, int] | None:
    matrix = info.get("K")
    if not isinstance(matrix, list):
        matrix = (info.get("camera_matrix") or {}).get("data")
    if not isinstance(matrix, list) or len(matrix) < 9:
        return None
    try:
        return (
            float(matrix[0]),
            float(matrix[4]),
            float(matrix[2]),
            float(matrix[5]),
            int(info.get("width") or info.get("image_width")),
            int(info.get("height") or info.get("image_height")),
        )
    except (TypeError, ValueError):
        return None


def _frustum(
    transform: np.ndarray,
    intrinsics: tuple[float, float, float, float, int, int],
    depth: float,
) -> list[list[float]]:
    fx, fy, cx, cy, width, height = intrinsics
    camera_points = [np.zeros(3)]
    for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
        camera_points.append(
            np.asarray(
                [
                    (u - cx) * depth / fx,
                    (v - cy) * depth / fy,
                    depth,
                ]
            )
        )
    return [
        [
            float(value)
            for value in transform[:3, :3] @ point + transform[:3, 3]
        ]
        for point in camera_points
    ]


def _topic(method: str, label: str, layer: str) -> str:
    return (
        f"/rigcal/methods/{_ros_token(method)}/"
        f"{_ros_token(label)}/{layer}"
    )


def _rviz_display(
    *,
    name: str,
    topic: str,
    enabled: bool,
    class_name: str = "rviz_default_plugins/MarkerArray",
) -> str:
    return (
        f"    - Class: {class_name}\n"
        f"      Name: {name}\n"
        f"      Enabled: {'true' if enabled else 'false'}\n"
        "      Topic:\n"
        f"        Value: {topic}"
    )


def _rviz_config(
    fixed_frame: str,
    variants: list[dict[str, Any]],
    *,
    ground_truth_available: bool,
) -> str:
    displays = [
        _rviz_display(
            name="AP03 Multi COLMAP context",
            topic="/rigcal/scene/points",
            enabled=True,
            class_name="rviz_default_plugins/PointCloud2",
        ),
        _rviz_display(
            name="Common anchor",
            topic="/rigcal/scene/anchor",
            enabled=True,
        ),
    ]
    if ground_truth_available:
        displays.append(
            _rviz_display(
                name="Ground truth cameras",
                topic="/rigcal/ground_truth/cameras",
                enabled=True,
            )
        )
    for variant in variants:
        name = f"{variant['method']}/{variant['label']}"
        enabled = variant["method"] == "ap03_multi"
        displays.extend(
            [
                _rviz_display(
                    name=name,
                    topic=_topic(
                        variant["method"], variant["label"], "cameras"
                    ),
                    enabled=enabled,
                ),
                _rviz_display(
                    name=f"{name} anchor edges",
                    topic=_topic(
                        variant["method"], variant["label"], "anchor_edges"
                    ),
                    enabled=False,
                ),
            ]
        )
        if ground_truth_available:
            displays.append(
                _rviz_display(
                    name=f"{name} estimate-to-GT errors",
                    topic=_topic(
                        variant["method"], variant["label"], "error_lines"
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


def _ground_truth_cameras(
    experiment_root: Path,
    *,
    anchor_marker_id: int,
) -> tuple[list[dict[str, Any]], str | None]:
    # Local import avoids evaluation.reporting -> visualization.scene cycles.
    from ..evaluation.simulation_ground_truth import (
        ensure_simulation_ground_truth,
    )

    dataset = _read_json(experiment_root / "dataset.json")
    category = str(
        dataset.get("category")
        or dataset.get("dataset", {}).get("category")
        or ""
    )
    snapshot = (
        experiment_root
        / "metadata"
        / "simulation"
        / "world_snapshot.sdf"
    )
    if category != "simulation" and not snapshot.is_file():
        return [], None
    gt = ensure_simulation_ground_truth(experiment_root, backfilled=True)
    if gt.get("status") != "available":
        return [], None
    marker = gt.get("markers", {}).get(str(anchor_marker_id), {})
    transform = marker.get("T_world_marker_opencv")
    if transform is None:
        return [], None
    anchor_world = invert_transform(
        validate_transform(np.asarray(transform, dtype=np.float64))
    )
    cameras = []
    for camera_id, value in sorted(gt.get("static_cameras", {}).items()):
        camera = anchor_world @ validate_transform(
            np.asarray(value, dtype=np.float64)
        )
        cameras.append(
            {
                "camera_id": str(camera_id),
                "parent_frame": f"evaluation_anchor_marker_{anchor_marker_id}",
                "child_frame": f"ground_truth/{camera_id}_optical_frame",
                **pose_payload(camera),
            }
        )
    gt_path = (
        experiment_root
        / "metadata"
        / "simulation"
        / "ground_truth.json"
    )
    return cameras, gt_path.relative_to(experiment_root).as_posix()


def _unavailable(
    experiment_root: Path,
    output: Path,
    *,
    status: str,
    reason: str,
    warnings: list[str],
) -> dict[str, Any]:
    for name in (
        "scene_anchor_frame.ply",
        "poses_anchor_frame.json",
        "camera_frustums.json",
        "rigcal_result.rviz",
        "launch_rviz.py",
    ):
        (output / name).unlink(missing_ok=True)
    manifest = {
        "schema_version": 2,
        "contract": SCENE_CONTRACT,
        "status": status,
        "available": False,
        "reason": reason,
        "warnings": warnings,
    }
    _write_json(output / "visualization_manifest.json", manifest)
    _update_result_status(experiment_root, status)
    return manifest


def ensure_visualization_artifacts(
    experiment_root: Path,
) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    ensure_experiment_anchor_exports(experiment_root)
    output = experiment_root / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    variants = _method_variants(experiment_root)
    source, source_warnings = _ap03_source(experiment_root, variants)
    if source is None:
        return _unavailable(
            experiment_root,
            output,
            status=(
                SOURCE_MISMATCH
                if any("mismatch" in item or "consistency" in item for item in source_warnings)
                else "UNAVAILABLE_NO_AP03_RECONSTRUCTION"
            ),
            reason=(
                "RViz requires the exact declared AP03 Multi best-model "
                "points3D file, Multi scale, and matching anchor export. "
                "No alternative model and no new COLMAP run are used."
            ),
            warnings=source_warnings,
        )
    source_files = [
        Path(source["points_path"]),
        Path(source["metadata_path"]),
        Path(source["provenance_path"]),
        *[Path(item["path"]) for item in variants],
    ]
    fingerprint_hash = hashlib.sha256(SCENE_CONTRACT.encode("utf-8"))
    for path in sorted(source_files):
        fingerprint_hash.update(path.read_bytes())
    fingerprint = fingerprint_hash.hexdigest()
    transform_anchor_method = source["transform_anchor_method"]
    scale = float(source["scale"])
    raw_points = _read_colmap_points(Path(source["points_path"]))
    if len(raw_points) != int(source["point_count"]):
        return _unavailable(
            experiment_root,
            output,
            status=SOURCE_MISMATCH,
            reason="The parsed point count changed during scene generation.",
            warnings=source_warnings,
        )
    transformed_points = [
        (
            transform_anchor_method[:3, :3] @ (scale * point)
            + transform_anchor_method[:3, 3],
            color,
        )
        for point, color in raw_points
    ]
    _write_ply(output / "scene_anchor_frame.ply", transformed_points)
    marker_length = float(
        source["metadata"].get("marker_length_m") or 0.17
    )
    fixed_frame = str(source["payload"]["parent_frame"])
    try:
        anchor_marker_id = int(source["payload"]["anchor_marker_id"])
    except (KeyError, TypeError, ValueError):
        return _unavailable(
            experiment_root,
            output,
            status=SOURCE_MISMATCH,
            reason="AP03 Multi has no valid frozen anchor ID.",
            warnings=source_warnings,
        )
    gt_cameras, gt_source = _ground_truth_cameras(
        experiment_root, anchor_marker_id=anchor_marker_id
    )
    pose_variants: list[dict[str, Any]] = []
    frustum_variants: list[dict[str, Any]] = []
    for variant in variants:
        cameras = [
            camera
            for camera in variant["payload"].get("cameras", [])
            if isinstance(camera, dict)
        ]
        pose_variants.append(
            {
                "method": variant["method"],
                "label": variant["label"],
                "camera_topic": _topic(
                    variant["method"], variant["label"], "cameras"
                ),
                "anchor_edges_topic": _topic(
                    variant["method"], variant["label"], "anchor_edges"
                ),
                "error_lines_topic": _topic(
                    variant["method"], variant["label"], "error_lines"
                ),
                "cameras": cameras,
            }
        )
        frustums = []
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            info = _intrinsics(_camera_info(experiment_root, camera_id))
            if info is None:
                continue
            transform = validate_transform(
                np.asarray(camera["matrix"], dtype=np.float64)
            )
            frustums.append(
                {
                    "camera_id": camera_id,
                    "points": _frustum(
                        transform, info, 2.0 * marker_length
                    ),
                }
            )
        frustum_variants.append(
            {
                "method": variant["method"],
                "label": variant["label"],
                "frustums": frustums,
            }
        )
    poses_payload = {
        "schema_version": 2,
        "fixed_frame": fixed_frame,
        "variants": pose_variants,
        "ground_truth": {
            "namespace": "ground_truth",
            "topic": "/rigcal/ground_truth/cameras",
            "source": gt_source,
            "cameras": gt_cameras,
        },
    }
    frustum_payload = {
        "schema_version": 2,
        "display_depth_m": 2.0 * marker_length,
        "variants": frustum_variants,
    }
    _write_json(output / "poses_anchor_frame.json", poses_payload)
    _write_json(output / "camera_frustums.json", frustum_payload)
    _atomic_text(
        output / "rigcal_result.rviz",
        _rviz_config(
            fixed_frame,
            variants,
            ground_truth_available=bool(gt_cameras),
        ),
    )
    _atomic_text(
        output / "launch_rviz.py",
        (
            "from camera_rig_calibration.visualization.session import "
            "main_for_generated_scene\n\n"
            "if __name__ == '__main__':\n"
            "    main_for_generated_scene(__file__)\n"
        ),
    )
    manifest = {
        "schema_version": 2,
        "contract": SCENE_CONTRACT,
        "fingerprint": fingerprint,
        "status": "OK",
        "available": True,
        "fixed_frame": fixed_frame,
        "point_cloud": "scene_anchor_frame.ply",
        "point_cloud_publisher_count": 1,
        "point_cloud_display_count": 1,
        "point_count": len(transformed_points),
        "poses": "poses_anchor_frame.json",
        "frustums": "camera_frustums.json",
        "rviz_config": "rigcal_result.rviz",
        "publisher": "camera_rig_calibration.visualization.ros_scene",
        "point_cloud_source": {
            "method": "ap03_multi",
            "container_method": "ap03",
            "container_label": source["label"],
            "model_id": source["best_model"],
            "points3D": str(
                Path(source["points_path"])
                .relative_to(experiment_root)
                .as_posix()
            ),
            "point_count": source["point_count"],
            "scale_m_per_colmap_unit": scale,
            "anchor_source": str(
                Path(source["path"])
                .relative_to(experiment_root)
                .as_posix()
            ),
            "ground_truth_source": gt_source,
        },
        "variants": [
            {
                "method": item["method"],
                "label": item["label"],
                "default_visible": item["method"] == "ap03_multi",
                "anchor_edges_default_visible": False,
                "error_lines_default_visible": False,
            }
            for item in variants
        ],
        "ground_truth": {
            "available": bool(gt_cameras),
            "camera_count": len(gt_cameras),
            "default_visible": bool(gt_cameras),
            "source": gt_source,
        },
        "consistency_checks": {
            "strict_best_model": True,
            "shared_container_membership": True,
            "scale_matches_multi_metadata": True,
            "point_count_matches_metadata": True,
            "transformed_cameras_match_ap03_multi_export": True,
        },
        "warnings": [
            *source_warnings,
            (
                "Visible ghosting, if present, belongs to this one unmodified "
                "best-model reconstruction; Ground Truth is never used to "
                "deform or align the point cloud."
            ),
        ],
        "note": (
            "The single point cloud is AP03 Multi/COLMAP context. Camera "
            "poses retain their own method/variant identity."
        ),
    }
    _write_json(output / "visualization_manifest.json", manifest)
    _update_result_status(experiment_root, "OK")
    return manifest
