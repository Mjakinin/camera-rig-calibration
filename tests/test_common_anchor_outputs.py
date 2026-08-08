from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.common_anchor_authority_policy import (
    install_common_anchor_authority_policy,
)
from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    InputSourceKind,
    MethodSettings,
    RigConfig,
    StaticCameraSettings,
)
from camera_rig_calibration.marker_preference_policy import install_marker_preference_policy
from camera_rig_calibration.product_policy import install_product_policy
from camera_rig_calibration.result_output_policy import (
    _sixdof_text,
    install_result_output_policy,
)
from camera_rig_calibration.submission_policy import install_submission_policy


install_product_policy()
install_submission_policy()
install_marker_preference_policy()
install_common_anchor_authority_policy()
install_result_output_policy()

from camera_rig_calibration import observations  # noqa: E402
from camera_rig_calibration.anchor_export import exporter  # noqa: E402
from camera_rig_calibration.visualization import scene  # noqa: E402


FIELDS = [
    "observer_type",
    "observer_id",
    "camera_name",
    "frame_id",
    "marker_id",
    "pnp_success",
    "selection_score",
    "pnp_reprojection_rmse_px",
    "marker_area_ratio",
]


def _write_selection_observations(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        # Preferred marker 0 is deliberately not visible in the AP01 root.
        {"observer_type": "static", "observer_id": "cam_a", "camera_name": "cam_a", "frame_id": "", "marker_id": 0, "pnp_success": "true", "selection_score": 10, "pnp_reprojection_rmse_px": 0.3, "marker_area_ratio": 0.02},
        {"observer_type": "static", "observer_id": "cam_a", "camera_name": "cam_a", "frame_id": "", "marker_id": 5, "pnp_success": "true", "selection_score": 9, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.02},
        {"observer_type": "static", "observer_id": "cam_b", "camera_name": "cam_b", "frame_id": "", "marker_id": 5, "pnp_success": "true", "selection_score": 8, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.02},
        {"observer_type": "moving", "observer_id": "moving_calib_camera", "camera_name": "moving_calib_camera", "frame_id": "0", "marker_id": 0, "pnp_success": "true", "selection_score": 7, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.015},
        {"observer_type": "moving", "observer_id": "moving_calib_camera", "camera_name": "moving_calib_camera", "frame_id": "1", "marker_id": 0, "pnp_success": "true", "selection_score": 7, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.015},
        {"observer_type": "moving", "observer_id": "moving_calib_camera", "camera_name": "moving_calib_camera", "frame_id": "2", "marker_id": 5, "pnp_success": "true", "selection_score": 6, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.015},
        {"observer_type": "moving", "observer_id": "moving_calib_camera", "camera_name": "moving_calib_camera", "frame_id": "3", "marker_id": 5, "pnp_success": "true", "selection_score": 6, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.015},
    ]
    with (root / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path) -> RigConfig:
    methods = MethodSettings(enabled=["ap01"])
    methods = methods.model_copy(
        update={
            "ap01": methods.ap01.model_copy(
                update={"root_camera": "cam_b", "direct_target_camera": "auto"}
            )
        },
        deep=True,
    )
    return RigConfig(
        dataset=DatasetSettings(
            id="common_anchor_root_visibility",
            category=DatasetCategory.REAL_VEHICLE,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=tmp_path,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_a"),
            StaticCameraSettings(id="cam_b"),
        ],
        methods=methods,
        evaluation=EvaluationSettings(
            enabled=True,
            anchor_marker_id=0,
            anchor_selection_mode="auto",
        ),
    )


def test_preferred_anchor_does_not_require_ap01_root_visibility(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    _write_selection_observations(root)
    resolved = observations.resolve_selections(_config(tmp_path), root)
    assert resolved.root_camera == "cam_b"
    assert resolved.evaluation_anchor_marker_id == 0
    anchor = resolved.payload["evaluation_anchor"]
    assert anchor["selected"] == 0
    assert 0 in anchor["automatic_observation_candidates"]
    assert anchor["fallback_used"] is False
    assert "root visibility is not required" in anchor["reason"]


def test_anchor_export_prefers_published_common_anchor(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    method_root = experiment / "methods" / "ap01" / "baseline"
    (method_root / "provenance").mkdir(parents=True)
    config = _config(experiment).model_copy(
        update={
            "evaluation": EvaluationSettings(
                enabled=True,
                anchor_marker_id=2,
                anchor_selection_mode="auto",
            )
        },
        deep=True,
    )
    save_config(config, method_root / "provenance" / "resolved_config.yaml")
    selected = experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps({"anchor_marker_id": 0}), encoding="utf-8")
    effective = exporter._config_for_result(method_root)
    assert effective is not None
    assert effective.evaluation.anchor_marker_id == 0


def test_common_anchor_sixdof_export_is_yaml_like_and_in_radians(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    (experiment / "evaluations").mkdir(parents=True)
    (experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 0}), encoding="utf-8"
    )
    for method, label, x in (
        ("ap01", "baseline", 1.0),
        ("ap02", "baseline", 2.0),
        ("ap03_multi", "baseline", 3.0),
    ):
        root = experiment / "methods" / method / label
        root.mkdir(parents=True)
        (root / "camera_extrinsics_anchor.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "label": label,
                    "anchor_marker_id": 0,
                    "parent_frame": "evaluation_anchor_marker_0",
                    "anchor_export_status": {"available": True},
                    "cameras": [
                        {
                            "camera_id": "cam_edge_0",
                            "x_m": x,
                            "y_m": 0.0,
                            "z_m": 1.5,
                            "roll_rad": 0.01,
                            "pitch_rad": 0.02,
                            "yaw_rad": 0.03,
                            "qx": 0.0,
                            "qy": 0.0,
                            "qz": 0.0,
                            "qw": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    text, payload = _sixdof_text(experiment)
    assert payload["anchor_marker_id"] == 0
    assert "Reference frame: evaluation_anchor_marker_0" in text
    assert "roll: 0.01" in text
    assert "pitch: 0.02" in text
    assert "yaw: 0.03" in text
    assert (experiment / "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml").is_file()
    assert (experiment / "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv").is_file()


def test_rviz_enables_primary_ap01_ap02_ap03_cameras_and_anchor_edges() -> None:
    variants = [
        {"method": "ap01", "label": "baseline"},
        {"method": "ap02", "label": "ref_marker_0__ref_mode_auto"},
        {"method": "ap03_multi", "label": "multi_markers_auto__single_marker_0"},
    ]
    config = scene._rviz_config(
        "evaluation_anchor_marker_0",
        variants,
        ground_truth_available=False,
    )
    for name in (
        "ap01/baseline",
        "ap02/ref_marker_0__ref_mode_auto",
        "ap03_multi/multi_markers_auto__single_marker_0",
    ):
        assert f"Name: {name}\n      Enabled: true" in config
        assert f"Name: {name} anchor edges\n      Enabled: true" in config
