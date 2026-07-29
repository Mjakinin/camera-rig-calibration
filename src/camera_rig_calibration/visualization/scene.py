from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..anchor_export import ensure_experiment_anchor_exports
from ..config import load_config
from ..anchor_export.geometry import validate_transform


SCENE_CONTRACT = "rigcal_rviz_scene_v1"


def _ros_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_text(path: Path, text: str) -> None:
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
        (experiment_root / "methods").glob("*/*/camera_extrinsics_anchor.json")
    ):
        payload = _read_json(anchor_path)
        status = payload.get("anchor_export_status", {})
        if not isinstance(status, dict) or not status.get("available"):
            continue
        variants.append(
            {
                "method": str(payload.get("method") or anchor_path.parents[1].name),
                "label": str(payload.get("label") or anchor_path.parent.name),
                "path": anchor_path,
                "payload": payload,
            }
        )
    return variants


def _ap03_source(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for variant in variants:
        if variant["method"] != "ap03":
            continue
        root = Path(variant["path"]).parent
        metadata_path = (
            root
            / "diagnostics"
            / "method"
            / "scale_multi"
            / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
        )
        metadata = _read_json(metadata_path)
        best_model = str(metadata.get("best_model", "0"))
        points = (
            root
            / "diagnostics"
            / "method"
            / "colmap"
            / "reconstruction"
            / "sparse_txt"
            / best_model
            / "points3D.txt"
        )
        if not points.is_file():
            alternatives = sorted(
                (
                    root
                    / "diagnostics"
                    / "method"
                    / "colmap"
                    / "reconstruction"
                    / "sparse_txt"
                ).glob("*/points3D.txt")
            )
            points = alternatives[0] if alternatives else points
        alignment = _read_json(
            root / "diagnostics" / "anchor_alignment.json"
        ).get("alignment", {})
        transform = alignment.get("transform_anchor_method", {}).get("matrix")
        if (
            not points.is_file()
            or metadata.get("scale_m_per_colmap_unit") is None
            or transform is None
        ):
            continue
        candidates.append(
            {
                **variant,
                "root": root,
                "metadata": metadata,
                "metadata_path": metadata_path,
                "points_path": points,
                "transform_anchor_method": transform,
                "camera_count": len(variant["payload"].get("cameras", [])),
                "point_count": int(metadata.get("num_sparse_points3d") or 0),
            }
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["label"] != "baseline",
            -item["camera_count"],
            -item["point_count"],
            item["label"],
        ),
    )


def _read_colmap_points(path: Path) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
                max(0, min(255, int(fields[index]))) for index in (4, 5, 6)
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


def _camera_info(experiment_root: Path, camera_id: str) -> dict[str, Any]:
    return _read_json(
        experiment_root / "raw_images" / "camera_info" / f"{camera_id}.json"
    )


def _intrinsics(info: dict[str, Any]) -> tuple[float, float, float, float, int, int] | None:
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
            for value in (
                transform[:3, :3] @ point + transform[:3, 3]
            )
        ]
        for point in camera_points
    ]


def _rviz_config(
    fixed_frame: str,
    variants: list[dict[str, Any]],
) -> str:
    displays = [
        """    - Class: rviz_default_plugins/TF
      Name: Coordinate frames
      Enabled: true
    - Class: rviz_default_plugins/PointCloud2
      Name: AP03 COLMAP context
      Enabled: true
      Topic:
        Value: /rigcal/scene/points
    - Class: rviz_default_plugins/MarkerArray
      Name: Common anchor
      Enabled: true
      Topic:
        Value: /rigcal/scene/anchor"""
    ]
    for variant in variants:
        topic = (
            f"/rigcal/methods/{_ros_token(variant['method'])}/"
            f"{_ros_token(variant['label'])}/markers"
        )
        displays.append(
            "    - Class: rviz_default_plugins/MarkerArray\n"
            f"      Name: {variant['method']}/{variant['label']}\n"
            "      Enabled: true\n"
            "      Topic:\n"
            f"        Value: {topic}"
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


def ensure_visualization_artifacts(
    experiment_root: Path,
) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    ensure_experiment_anchor_exports(experiment_root)
    output = experiment_root / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    variants = _method_variants(experiment_root)
    source = _ap03_source(variants)
    if source is None:
        for name in (
            "scene_anchor_frame.ply",
            "poses_anchor_frame.json",
            "camera_frustums.json",
            "rigcal_result.rviz",
            "launch_rviz.py",
        ):
            (output / name).unlink(missing_ok=True)
        manifest = {
            "schema_version": 1,
            "contract": SCENE_CONTRACT,
            "status": "UNAVAILABLE_NO_AP03_RECONSTRUCTION",
            "available": False,
            "reason": (
                "A scaled AP03 Multi sparse reconstruction and a valid AP03 "
                "common-anchor export are required. No new COLMAP run is started."
            ),
        }
        _write_json(output / "visualization_manifest.json", manifest)
        _update_result_status(experiment_root, str(manifest["status"]))
        return manifest
    source_files = [
        Path(source["points_path"]),
        Path(source["metadata_path"]),
        *[Path(item["path"]) for item in variants],
    ]
    fingerprint_hash = hashlib.sha256(SCENE_CONTRACT.encode("utf-8"))
    for path in sorted(source_files):
        fingerprint_hash.update(path.read_bytes())
    fingerprint = fingerprint_hash.hexdigest()
    current = _read_json(output / "visualization_manifest.json")
    if (
        current.get("fingerprint") == fingerprint
        and (output / "scene_anchor_frame.ply").is_file()
        and (output / "rigcal_result.rviz").is_file()
    ):
        return current
    transform_anchor_method = validate_transform(
        np.asarray(source["transform_anchor_method"], dtype=np.float64)
    )
    scale = float(source["metadata"]["scale_m_per_colmap_unit"])
    raw_points = _read_colmap_points(Path(source["points_path"]))
    transformed_points = [
        (
            transform_anchor_method[:3, :3] @ (scale * point)
            + transform_anchor_method[:3, 3],
            color,
        )
        for point, color in raw_points
    ]
    _write_ply(output / "scene_anchor_frame.ply", transformed_points)
    marker_length = float(source["metadata"].get("marker_length_m") or 0.17)
    fixed_frame = str(source["payload"]["parent_frame"])
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
                "topic": (
                    f"/rigcal/methods/{_ros_token(variant['method'])}/"
                    f"{_ros_token(variant['label'])}/markers"
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
        "schema_version": 1,
        "fixed_frame": fixed_frame,
        "variants": pose_variants,
    }
    frustum_payload = {
        "schema_version": 1,
        "display_depth_m": 2.0 * marker_length,
        "variants": frustum_variants,
    }
    _write_json(output / "poses_anchor_frame.json", poses_payload)
    _write_json(output / "camera_frustums.json", frustum_payload)
    _atomic_text(
        output / "rigcal_result.rviz",
        _rviz_config(fixed_frame, variants),
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
        "schema_version": 1,
        "contract": SCENE_CONTRACT,
        "fingerprint": fingerprint,
        "status": "OK",
        "available": True,
        "fixed_frame": fixed_frame,
        "point_cloud": "scene_anchor_frame.ply",
        "point_count": len(transformed_points),
        "poses": "poses_anchor_frame.json",
        "frustums": "camera_frustums.json",
        "rviz_config": "rigcal_result.rviz",
        "publisher": "camera_rig_calibration.visualization.ros_scene",
        "source": {
            "method": "ap03",
            "label": source["label"],
            "points3D": str(Path(source["points_path"]).relative_to(experiment_root)),
            "scale_m_per_colmap_unit": scale,
        },
        "variants": [
            {"method": item["method"], "label": item["label"]}
            for item in variants
        ],
        "note": (
            "The point cloud is AP03/COLMAP context. Camera poses retain their "
            "own method/variant identity."
        ),
    }
    _write_json(output / "visualization_manifest.json", manifest)
    _update_result_status(experiment_root, str(manifest["status"]))
    return manifest
