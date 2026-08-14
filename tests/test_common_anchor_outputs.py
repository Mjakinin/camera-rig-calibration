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
    "distance_m",
    "center_u",
    "center_v",
    "area_px2",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "corner0_u",
    "corner0_v",
    "corner1_u",
    "corner1_v",
    "corner2_u",
    "corner2_v",
    "corner3_u",
    "corner3_v",
]


def _observation(**values) -> dict:
    """Return a minimal but scientifically valid AP01 observation fixture."""

    marker = int(values["marker_id"])
    frame_text = str(values.get("frame_id", ""))
    frame = int(frame_text) if frame_text else 0
    # Keep every synthetic pose finite and deterministic.  The test exercises
    # selection/anchor authority, not pose accuracy; nevertheless it deliberately
    # satisfies the same PnP/geometry contract consumed by AP01 production code.
    z = 2.0 + 0.05 * marker + 0.01 * frame
    center_u = 320.0 + 3.0 * marker
    center_v = 240.0 + 2.0 * frame
    half = 40.0
    row = {
        "distance_m": z,
        "center_u": center_u,
        "center_v": center_v,
        "area_px2": (2.0 * half) ** 2,
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": 0.02 * marker,
        "tvec_y_m": 0.01 * frame,
        "tvec_z_m": z,
        "corner0_u": center_u - half,
        "corner0_v": center_v - half,
        "corner1_u": center_u + half,
        "corner1_v": center_v - half,
        "corner2_u": center_u + half,
        "corner2_v": center_v + half,
        "corner3_u": center_u - half,
        "corner3_v": center_v + half,
    }
    row.update(values)
    return row


def _write_selection_observations(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        # Preferred marker 0 is deliberately not visible in the AP01 root.
        _observation(observer_type="static", observer_id="cam_a", camera_name="cam_a", frame_id="", marker_id=0, pnp_success="true", selection_score=10, pnp_reprojection_rmse_px=0.3, marker_area_ratio=0.02),
        _observation(observer_type="static", observer_id="cam_a", camera_name="cam_a", frame_id="", marker_id=5, pnp_success="true", selection_score=9, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.02),
        _observation(observer_type="static", observer_id="cam_b", camera_name="cam_b", frame_id="", marker_id=5, pnp_success="true", selection_score=8, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.02),
        _observation(observer_type="moving", observer_id="moving_calib_camera", camera_name="moving_calib_camera", frame_id="0", marker_id=0, pnp_success="true", selection_score=7, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.015),
        _observation(observer_type="moving", observer_id="moving_calib_camera", camera_name="moving_calib_camera", frame_id="1", marker_id=0, pnp_success="true", selection_score=7, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.015),
        _observation(observer_type="moving", observer_id="moving_calib_camera", camera_name="moving_calib_camera", frame_id="2", marker_id=5, pnp_success="true", selection_score=6, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.015),
        _observation(observer_type="moving", observer_id="moving_calib_camera", camera_name="moving_calib_camera", frame_id="3", marker_id=5, pnp_success="true", selection_score=6, pnp_reprojection_rmse_px=0.4, marker_area_ratio=0.015),
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
            ),
            # This is a real-vehicle fixture. Keep inactive AP02/AP03 on the
            # public real-vehicle marker contract as well. The strict AP02
            # baseline and AP03 marker-14 defaults belong to simulation.
            "ap02": methods.ap02.model_copy(
                update={
                    "reference_marker_selection_mode": "auto",
                    "reference_marker_id": 0,
                }
            ),
            "ap03": methods.ap03.model_copy(
                update={
                    "single": methods.ap03.single.model_copy(
                        update={"scale_marker_id": 0}
                    )
                },
                deep=True,
            ),
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
    # The final product policy may strengthen the wording to the canonical
    # real-vehicle marker-0 rule; the observable contract is the same: marker 0
    # remains selected even though the AP01 root is another camera.
    assert (
        "root visibility is not required" in anchor["reason"]
        or "marker 0" in anchor["reason"]
    )


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
