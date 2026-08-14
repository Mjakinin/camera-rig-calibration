from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.visualization.fallback_scene import (
    ensure_fallback_visualization_artifacts,
)


def _matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _anchor_payload(method: str, label: str, camera: str) -> dict:
    return {
        "method": method,
        "label": label,
        "anchor_marker_id": 14,
        "parent_frame": "evaluation_anchor_marker_14",
        "anchor_export_status": {"available": True, "code": "OK"},
        "anchor_alignment": {
            "transform_anchor_method": {"matrix": _matrix()}
        },
        "cameras": [
            {
                "camera_id": camera,
                "matrix": _matrix(),
            }
        ],
    }


def test_pose_only_ap02_uses_anchor_as_origin_without_colmap(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    result = experiment / "methods" / "ap02" / "combined"
    result.mkdir(parents=True)
    (result / "camera_extrinsics_anchor.json").write_text(
        json.dumps(_anchor_payload("ap02", "combined", "cam_a")),
        encoding="utf-8",
    )

    manifest = ensure_fallback_visualization_artifacts(experiment)

    assert manifest["available"] is True
    assert manifest["selected_source"] == "pose_only"
    assert manifest["fixed_frame"] == "evaluation_anchor_marker_14"
    assert manifest["point_count"] == 0
    rviz = (experiment / "visualization" / "rigcal_result.rviz").read_text(
        encoding="utf-8"
    )
    assert "Fixed Frame: evaluation_anchor_marker_14" in rviz
    assert "ap02/combined" in rviz
    assert "PointCloud2" not in rviz


def test_ap01_colmap_fallback_is_scaled_and_aligned_to_anchor(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    result = experiment / "methods" / "ap01" / "baseline"
    colmap = result / "diagnostics" / "method" / "01_moving_colmap" / "sparse_txt_best"
    scale_root = result / "diagnostics" / "method" / "02_metric_scale"
    preflight = result / "diagnostics" / "preflight"
    colmap.mkdir(parents=True)
    scale_root.mkdir(parents=True)
    preflight.mkdir(parents=True)

    (result / "camera_extrinsics_anchor.json").write_text(
        json.dumps(_anchor_payload("ap01", "baseline", "cam_root")),
        encoding="utf-8",
    )
    (result / "camera_extrinsics.csv").write_text(
        "entity_id,x_m,y_m,z_m,rvec_x,rvec_y,rvec_z\n"
        "cam_root,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (colmap / "images.txt").write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "1 1 0 0 0 0 0 0 1 moving_0000.png\n"
        "0 0 -1\n",
        encoding="utf-8",
    )
    (colmap / "points3D.txt").write_text(
        "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        "1 1 2 3 255 0 0 0.1 1 0\n",
        encoding="utf-8",
    )
    (scale_root / "metric_scale.txt").write_text("2.0\n", encoding="utf-8")

    fields = [
        "observer_type",
        "observer_id",
        "camera_name",
        "frame_id",
        "marker_id",
        "selection_score",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x_m",
        "tvec_y_m",
        "tvec_z_m",
    ]
    rows = [
        {
            "observer_type": "static",
            "observer_id": "cam_root",
            "camera_name": "cam_root",
            "frame_id": "",
            "marker_id": "5",
            "selection_score": "10",
            "rvec_x": "0",
            "rvec_y": "0",
            "rvec_z": "0",
            "tvec_x_m": "0",
            "tvec_y_m": "0",
            "tvec_z_m": "2",
        },
        {
            "observer_type": "moving",
            "observer_id": "moving_calib_camera",
            "camera_name": "moving_calib_camera",
            "frame_id": "0",
            "marker_id": "5",
            "selection_score": "9",
            "rvec_x": "0",
            "rvec_y": "0",
            "rvec_z": "0",
            "tvec_x_m": "0",
            "tvec_y_m": "0",
            "tvec_z_m": "2",
        },
    ]
    with (preflight / "accepted_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = ensure_fallback_visualization_artifacts(experiment)

    assert manifest["available"] is True
    assert manifest["selected_source"] == "ap01_colmap"
    assert manifest["point_count"] == 1
    assert manifest["point_cloud_source"]["method"] == "ap01"
    assert manifest["point_cloud_source"]["scale_m_per_colmap_unit"] == 2.0
    ply = (experiment / "visualization" / "scene_anchor_frame.ply").read_text(
        encoding="utf-8"
    )
    assert "2 4 6 255 0 0" in ply
    rviz = (experiment / "visualization" / "rigcal_result.rviz").read_text(
        encoding="utf-8"
    )
    assert "AP01 moving-COLMAP context" in rviz
    assert "PointCloud2" in rviz
