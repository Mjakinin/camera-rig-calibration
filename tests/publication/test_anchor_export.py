from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from camera_rig_calibration.anchor_export.exporter import (
    export_method_anchor_poses,
)
from camera_rig_calibration.anchor_export.geometry import (
    invert_transform,
    make_transform,
    pose_payload,
    rigid_fit,
    rotation_to_quaternion,
)
from camera_rig_calibration.config import load_config, save_config
from camera_rig_calibration.config.models import RigConfig
from camera_rig_calibration.observations import (
    freeze_selections,
    resolve_selections,
)
from camera_rig_calibration.visualization.scene import (
    ensure_visualization_artifacts,
)
from camera_rig_calibration.visualization.session import _reserve_domain
from camera_rig_calibration.evaluation.reporting import (
    _anchor_camera_gt_rows,
)


POSE_FIELDS = [
    "reference_frame",
    "transform_convention",
    "entity_type",
    "entity_id",
    "source",
    "x_m",
    "y_m",
    "z_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "rvec_x",
    "rvec_y",
    "rvec_z",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _method_root(
    tmp_path: Path,
    config: RigConfig,
    method: str,
    label: str = "baseline",
) -> Path:
    root = tmp_path / "methods" / method / label
    (root / "provenance").mkdir(parents=True)
    resolved = config.model_copy(
        update={
            "methods": config.methods.model_copy(
                update={"enabled": [method]}, deep=True
            ),
            "evaluation": config.evaluation.model_copy(
                update={
                    "anchor_marker_id": 7,
                    "anchor_selection_mode": "explicit",
                }
            ),
        },
        deep=True,
    )
    save_config(resolved, root / "provenance" / "resolved_config.yaml")
    (root / "RESULT.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "method": method,
                "label": label,
                "artifact_status": "available",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "camera_extrinsics.csv",
        POSE_FIELDS,
        [
            {
                "reference_frame": "native",
                "transform_convention": "T_reference_camera",
                "entity_type": "static_camera",
                "entity_id": camera,
                "source": "fixture",
                "x_m": index,
                "y_m": 0,
                "z_m": 0,
                "roll_deg": 0,
                "pitch_deg": 0,
                "yaw_deg": 0,
                "rvec_x": 0,
                "rvec_y": 0,
                "rvec_z": 0,
            }
            for index, camera in enumerate(
                [item.id for item in resolved.static_cameras]
            )
        ],
    )
    return root


def test_geometry_contract_and_rigid_fit() -> None:
    angle = math.radians(30)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    expected = make_transform(rotation, np.array([1.0, 2.0, 3.0]))
    source = np.array(
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
        dtype=float,
    )
    target = (rotation @ source.T).T + expected[:3, 3]
    fitted, rmse = rigid_fit(source, target)

    assert np.allclose(fitted, expected)
    assert rmse < 1e-12
    assert np.allclose(invert_transform(expected) @ expected, np.eye(4))
    assert rotation_to_quaternion(rotation)[3] >= 0
    assert pose_payload(expected)["roll_rad"] == 0.0


def test_ap02_identity_anchor_exports_all_cameras(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = _method_root(tmp_path, prepared_config, "ap02")
    config_path = root / "provenance" / "resolved_config.yaml"
    config = load_config(config_path)
    config = config.model_copy(
        update={
            "methods": config.methods.model_copy(
                update={
                    "ap02": config.methods.ap02.model_copy(
                        update={"reference_marker_id": 7}
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )
    save_config(config, config_path)

    status = export_method_anchor_poses(root)
    payload = json.loads(
        (root / "camera_extrinsics_anchor.json").read_text(encoding="utf-8")
    )

    assert status["available"] is True
    assert status["code"] == "OK"
    assert len(payload["cameras"]) == len(prepared_config.static_cameras)
    assert payload["pairwise_invariance"]["passed"] is True
    assert (root / "camera_extrinsics_anchor.yaml").is_file()
    assert (root / "camera_extrinsics_anchor.csv").is_file()
    assert (root / "diagnostics" / "anchor_alignment.json").is_file()


def test_ap01_robust_anchor_aggregates_static_observations(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = _method_root(tmp_path, prepared_config, "ap01")
    accepted = (
        root / "diagnostics" / "preflight" / "accepted_observations.csv"
    )
    _write_csv(
        accepted,
        [
            "observer_type",
            "observer_id",
            "frame_id",
            "image_path",
            "marker_id",
            "rvec_x",
            "rvec_y",
            "rvec_z",
            "tvec_x_m",
            "tvec_y_m",
            "tvec_z_m",
            "selection_score",
        ],
        [
            {
                "observer_type": "static",
                "observer_id": prepared_config.static_cameras[0].id,
                "frame_id": "0",
                "image_path": "first.png",
                "marker_id": 7,
                "rvec_x": 0,
                "rvec_y": 0,
                "rvec_z": 0,
                "tvec_x_m": 2,
                "tvec_y_m": 0,
                "tvec_z_m": 0,
                "selection_score": 0.8,
            },
            {
                "observer_type": "static",
                "observer_id": prepared_config.static_cameras[1].id,
                "frame_id": "0",
                "image_path": "second.png",
                "marker_id": 7,
                "rvec_x": 0,
                "rvec_y": 0,
                "rvec_z": 0,
                "tvec_x_m": 1,
                "tvec_y_m": 0,
                "tvec_z_m": 0,
                "selection_score": 0.7,
            },
        ],
    )

    status = export_method_anchor_poses(root)
    alignment = json.loads(
        (root / "diagnostics" / "anchor_alignment.json").read_text(
            encoding="utf-8"
        )
    )

    assert status["code"] == "OK"
    transform = alignment["alignment"]["transform_method_anchor"]["matrix"]
    assert np.allclose(np.asarray(transform)[:3, 3], [2, 0, 0])
    assert alignment["alignment"]["inlier_count"] == 2


def test_ap03_anchor_uses_existing_scale_without_scale_fit(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = _method_root(tmp_path, prepared_config, "ap03")
    scale_root = root / "diagnostics" / "method" / "scale_multi"
    scale_root.mkdir(parents=True)
    (scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json").write_text(
        json.dumps({"scale_m_per_colmap_unit": 0.5}),
        encoding="utf-8",
    )
    half = prepared_config.markers.length_m / 2.0
    ideal = [
        (-half, half, 0.0),
        (half, half, 0.0),
        (half, -half, 0.0),
        (-half, -half, 0.0),
    ]
    # Raw COLMAP corners are divided by the already determined 0.5 scale.
    _write_csv(
        scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv",
        [
            "marker_id",
            "corner_idx",
            "status",
            "x_colmap",
            "y_colmap",
            "z_colmap",
            "obs_count",
            "inlier_count",
            "median_reproj_px",
        ],
        [
            {
                "marker_id": 7,
                "corner_idx": index,
                "status": "OK",
                "x_colmap": 2 * (point[0] + 1.0),
                "y_colmap": 2 * (point[1] + 2.0),
                "z_colmap": 2 * (point[2] + 3.0),
                "obs_count": 10,
                "inlier_count": 9,
                "median_reproj_px": 0.3,
            }
            for index, point in enumerate(ideal)
        ],
    )

    status = export_method_anchor_poses(root)
    alignment = json.loads(
        (root / "diagnostics" / "anchor_alignment.json").read_text(
            encoding="utf-8"
        )
    )

    assert status["code"] == "OK"
    assert alignment["alignment"]["alignment"] == "rigid_kabsch_no_scale_fit"
    assert alignment["alignment"]["square_fit_rmse_m"] < 1e-12


def test_rviz_scene_uses_existing_ap03_points_only(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    root = _method_root(
        tmp_path, prepared_config, "ap03_multi", "baseline"
    )
    container = tmp_path / "methods" / "ap03" / "baseline"
    scale_root = container / "diagnostics" / "method" / "scale_multi"
    scale_root.mkdir(parents=True)
    (scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json").write_text(
        json.dumps(
            {
                "scale_m_per_colmap_unit": 0.5,
                "marker_length_m": 0.17,
                "best_model": "0",
                "num_sparse_points3d": 1,
            }
        ),
        encoding="utf-8",
    )
    half = prepared_config.markers.length_m / 2.0
    ideal = [
        (-half, half, 0.0),
        (half, half, 0.0),
        (half, -half, 0.0),
        (-half, -half, 0.0),
    ]
    shared_geometry = (
        container
        / "diagnostics"
        / "derived"
        / "shared_anchor_geometry"
        / "marker_7_corners_colmap.json"
    )
    shared_geometry.parent.mkdir(parents=True)
    shared_geometry.write_text(
        json.dumps(
            {
                "anchor_marker_id": 7,
                "corners": [
                    {
                        "corner_idx": index,
                        "x_colmap": 2 * point[0],
                        "y_colmap": 2 * point[1],
                        "z_colmap": 2 * point[2],
                    }
                    for index, point in enumerate(ideal)
                ],
            }
        ),
        encoding="utf-8",
    )
    camera_source = (
        scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
    )
    camera_source.write_bytes((root / "camera_extrinsics.csv").read_bytes())
    (root / "provenance" / "derived_result.json").write_text(
        json.dumps(
            {
                "shared_colmap_container": "methods/ap03/baseline",
                "shared_colmap_best_model": "0",
                "shared_anchor_geometry": (
                    "methods/ap03/baseline/diagnostics/derived/"
                    "shared_anchor_geometry/marker_7_corners_colmap.json"
                ),
                "scale_metadata": (
                    "methods/ap03/baseline/diagnostics/method/scale_multi/"
                    "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
                ),
                "camera_pose_source": (
                    "methods/ap03/baseline/diagnostics/method/scale_multi/"
                    "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
                ),
                "scale_m_per_colmap_unit": 0.5,
            }
        ),
        encoding="utf-8",
    )
    points = (
        container
        / "diagnostics"
        / "method"
        / "colmap"
        / "reconstruction"
        / "sparse_txt"
        / "0"
        / "points3D.txt"
    )
    points.parent.mkdir(parents=True)
    points.write_text(
        "# fixture\n1 2 4 6 10 20 30 0.1\n", encoding="utf-8"
    )
    info_root = tmp_path / "raw_images" / "camera_info"
    info_root.mkdir(parents=True)
    for camera in prepared_config.static_cameras:
        (info_root / f"{camera.id}.json").write_text(
            json.dumps(
                {
                    "width": 640,
                    "height": 480,
                    "K": [500, 0, 320, 0, 500, 240, 0, 0, 1],
                }
            ),
            encoding="utf-8",
        )

    manifest = ensure_visualization_artifacts(tmp_path)

    assert manifest["status"] == "OK"
    assert manifest["point_count"] == 1
    assert manifest["point_cloud_display_count"] == 1
    assert manifest["point_cloud_source"]["model_id"] == "0"
    ply = (tmp_path / "visualization" / "scene_anchor_frame.ply").read_text(
        encoding="utf-8"
    )
    assert "1 2 3 10 20 30" in ply
    assert (tmp_path / "visualization" / "rigcal_result.rviz").is_file()
    assert (
        tmp_path / "visualization" / "poses_anchor_frame.json"
    ).is_file()


def test_rviz_scene_is_explicitly_unavailable_without_ap03(
    tmp_path: Path,
) -> None:
    manifest = ensure_visualization_artifacts(tmp_path)
    assert manifest["available"] is False
    assert manifest["status"] == "UNAVAILABLE_NO_AP03_RECONSTRUCTION"


def test_rviz_sessions_reserve_distinct_ros_domains(tmp_path: Path) -> None:
    first_domain, first_lock = _reserve_domain(tmp_path)
    second_domain, second_lock = _reserve_domain(tmp_path)

    assert first_domain != second_domain
    assert first_lock.is_file()
    assert second_lock.is_file()


def test_manual_common_anchor_can_freeze_warned_raw_detection(
    prepared_config: RigConfig, tmp_path: Path
) -> None:
    observations = tmp_path / "observations"
    rows = [
        {
            "observer_type": observer_type,
            "observer_id": observer_id,
            "frame_id": frame_id,
            "marker_id": 1,
            "area_px2": 100,
            "pnp_reprojection_rmse_px": 0.5,
            "pnp_success": True,
        }
        for observer_type, observer_id, frame_id in (
            ("static", prepared_config.static_cameras[0].id, "static"),
            ("static", prepared_config.static_cameras[1].id, "static"),
            ("moving", "moving_0", "0"),
            ("moving", "moving_1", "1"),
        )
    ]
    _write_csv(
        observations / "shared_all_aruco_observations.csv",
        list(rows[0]),
        rows,
    )
    config = prepared_config.model_copy(
        update={
            "evaluation": prepared_config.evaluation.model_copy(
                update={"anchor_selection_mode": "review_once"}
            )
        },
        deep=True,
    )
    resolved = resolve_selections(config, observations)
    resolved = replace(
        resolved,
        payload={
            **resolved.payload,
            "raw_marker_inventory": [
                {"id": 1, "raw_observations": 4},
                {"id": 99, "raw_observations": 1},
            ],
        },
    )

    frozen = freeze_selections(
        config,
        resolved,
        {"evaluation_anchor_marker_id": 99},
    )

    assert frozen.evaluation.anchor_marker_id == 99
    assert frozen.evaluation.anchor_selection_mode == "explicit"


def test_simulation_anchor_gt_is_direct_posthoc_comparison() -> None:
    world_anchor = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    world_camera = make_transform(np.eye(3), np.array([2.0, 0.0, 0.0]))
    estimated = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))

    rows = _anchor_camera_gt_rows(
        "ap02",
        "baseline",
        {
            "cameras": [
                {
                    "camera_id": "front",
                    "matrix": pose_payload(estimated)["matrix"],
                }
            ]
        },
        anchor_marker_id=7,
        gt_cameras={"front": world_camera},
        gt_markers={7: world_anchor},
    )

    assert rows[0]["translation_error_cm"] == 0.0
    assert rows[0]["rotation_error_deg"] == 0.0
    assert (
        rows[0]["evaluation"]
        == "direct_anchor_relative_posthoc_gt_no_fit_no_scale"
    )
