from __future__ import annotations

import csv
from pathlib import Path

import pytest

from camera_rig_calibration.policies.common_anchor_authority_policy import (
    install_common_anchor_authority_policy,
)
from camera_rig_calibration.config.models import (
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    InputSourceKind,
    MethodSettings,
    RigConfig,
    StaticCameraSettings,
)
from camera_rig_calibration.policies.marker_preference_policy import install_marker_preference_policy
from camera_rig_calibration.policies.product_policy import install_product_policy
from camera_rig_calibration.policies.queue_anchor_preference_policy import (
    install_queue_anchor_preference_policy,
)
from camera_rig_calibration.policies.real_partial_evaluation_policy import (
    _install_nonblocking_preflight,
    install_real_partial_evaluation_policy,
)
from camera_rig_calibration.policies.real_vehicle_marker_zero_policy import (
    install_real_vehicle_marker_zero_policy,
)
from camera_rig_calibration.policies.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.policies.submission_bindings import install_submission_bindings
from camera_rig_calibration.policies.submission_policy import install_submission_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_common_anchor_authority_policy()
install_real_vehicle_marker_zero_policy()
install_queue_anchor_preference_policy()
install_real_partial_evaluation_policy()
install_submission_bindings()

from camera_rig_calibration import preflight  # noqa: E402


@pytest.fixture(autouse=True)
def _reassert_final_real_evaluation_preflight() -> None:
    """Make this policy contract independent of pytest collection order."""
    _install_nonblocking_preflight()


FIELDS = [
    "observer_type",
    "observer_id",
    "camera_name",
    "frame_id",
    "marker_id",
    "detection_success",
    "pnp_success",
    "marker_length_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion_model",
    *(f"d{index}" for index in range(8)),
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "distance_m",
    *(f"corner{index}_{axis}" for index in range(4) for axis in ("u", "v")),
    "area_px2",
    "marker_area_ratio",
    "image_width_px",
    "image_height_px",
    "center_u",
    "center_v",
    "pnp_reprojection_rmse_px",
    "selection_score",
]


def _row(
    observer_type: str,
    observer_id: str,
    marker_id: int,
    *,
    frame_id: str = "",
) -> dict[str, object]:
    width = 640.0
    height = 480.0
    fx = fy = 500.0
    cx = width / 2.0
    cy = height / 2.0
    marker_length_m = 0.17
    half = marker_length_m / 2.0
    z = 1.0 + 0.01 * marker_id
    projected = [
        (cx + fx * (-half) / z, cy + fy * half / z),
        (cx + fx * half / z, cy + fy * half / z),
        (cx + fx * half / z, cy + fy * (-half) / z),
        (cx + fx * (-half) / z, cy + fy * (-half) / z),
    ]
    side_px = fx * marker_length_m / z
    area_px2 = side_px * side_px
    row: dict[str, object] = {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": observer_id,
        "frame_id": frame_id,
        "marker_id": marker_id,
        "detection_success": "true",
        "pnp_success": "true",
        "marker_length_m": marker_length_m,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_model": "plumb_bob",
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": 0.0,
        "tvec_y_m": 0.0,
        "tvec_z_m": z,
        "distance_m": z,
        "area_px2": area_px2,
        "marker_area_ratio": area_px2 / (width * height),
        "image_width_px": int(width),
        "image_height_px": int(height),
        "center_u": cx,
        "center_v": cy,
        "pnp_reprojection_rmse_px": 0.0,
        "selection_score": 100.0 - marker_id,
    }
    for index in range(8):
        row[f"d{index}"] = 0.0
    for index, (u, v) in enumerate(projected):
        row[f"corner{index}_u"] = u
        row[f"corner{index}_v"] = v
    return row


def _write_dataset(root: Path) -> None:
    for path in (
        root / "raw_images" / "static" / "cam_a.png",
        root / "raw_images" / "static" / "cam_b.png",
        root / "raw_images" / "moving" / "frame_000000.png",
        root / "raw_images" / "camera_info" / "cam_a.json",
        root / "raw_images" / "camera_info" / "cam_b.json",
        root / "raw_images" / "camera_info" / "moving_calib_camera.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _write_observations(path: Path, *, sparse_zero: bool = False) -> None:
    rows = [
        # Marker 0 is intentionally not visible in AP01 root cam_b. cam_a sees it,
        # which is enough because AP01 can express a marker through any solved,
        # reachable static camera.
        _row("static", "cam_a", 0),
        _row("static", "cam_a", 5),
        _row("static", "cam_b", 5),
        _row("moving", "moving_calib_camera", 0, frame_id="0"),
        _row("moving", "moving_calib_camera", 5, frame_id="0"),
        _row("moving", "moving_calib_camera", 5, frame_id="1"),
    ]
    if not sparse_zero:
        rows.append(_row("moving", "moving_calib_camera", 0, frame_id="1"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _config(root: Path, method_id: str) -> RigConfig:
    methods = MethodSettings(enabled=[method_id])
    methods = methods.model_copy(
        update={
            "ap01": methods.ap01.model_copy(
                update={"root_camera": "cam_b", "direct_target_camera": "cam_a"}
            ),
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
                    ),
                    "multi": methods.ap03.multi.model_copy(update={"marker_ids": "auto"}),
                },
                deep=True,
            ),
        },
        deep=True,
    )
    return RigConfig(
        dataset=DatasetSettings(
            id=f"queue_anchor_{method_id}",
            category=DatasetCategory.REAL_VEHICLE,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=root,
        ),
        static_cameras=[StaticCameraSettings(id="cam_a"), StaticCameraSettings(id="cam_b")],
        methods=methods,
        evaluation=EvaluationSettings(
            enabled=True,
            anchor_marker_id=0,
            anchor_selection_mode="auto",
        ),
    )


def test_real_vehicle_zero_stays_common_anchor_without_ap01_root_visibility(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    raw = tmp_path / "raw_observations.csv"
    _write_observations(raw)
    jobs = (
        preflight.PreflightJob("ap01", _config(dataset, "ap01")),
        preflight.PreflightJob("ap03", _config(dataset, "ap03")),
    )
    result = preflight.run_queue_preflight(
        jobs,
        raw_observations_csv=raw,
        dataset_root=dataset,
        output_directory=tmp_path / "preflight",
        repository_root=tmp_path,
    )
    assert result.ready is True, [(job.job_id, job.status, job.errors) for job in result.jobs]
    assert result.common_evaluation_anchor_marker_id == 0
    for job in result.jobs:
        assert job.selections is not None
        assert job.selections.evaluation_anchor_marker_id == 0
        policy = job.selections.payload["real_vehicle_marker_zero_policy"]
        assert policy["marker_zero_observed"] is True
        assert policy["calibration_gating"] is False
        assert policy["common_evaluation_requested"] is True
        assert policy["unobservable_method_results_are_reported_not_failed"] is True


def test_real_vehicle_observed_but_insufficient_zero_is_reported_without_blocking_calibration(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    raw = tmp_path / "raw_observations.csv"
    _write_observations(raw, sparse_zero=True)
    jobs = (
        preflight.PreflightJob("ap01", _config(dataset, "ap01")),
        preflight.PreflightJob("ap03", _config(dataset, "ap03")),
    )
    result = preflight.run_queue_preflight(
        jobs,
        raw_observations_csv=raw,
        dataset_root=dataset,
        output_directory=tmp_path / "preflight",
        repository_root=tmp_path,
    )

    assert result.ready is True
    assert result.status == "READY_WITH_WARNINGS"
    assert result.common_evaluation_anchor_marker_id == 0
    assert all(job.status != "FAILED_PREFLIGHT" for job in result.jobs)
    assert all(not job.errors for job in result.jobs)
    assert any(
        "evaluation remains enabled and does not gate calibration" in warning
        for job in result.jobs
        for warning in job.warnings
    )
    assert any(
        "Non-blocking evaluation preflight removed evaluation-only readiness errors"
        in detail
        for job in result.jobs
        for detail in job.details
    )
    for job in result.jobs:
        assert job.selections is not None
        assert job.selections.evaluation_anchor_marker_id == 0
        policy = job.selections.payload["real_vehicle_marker_zero_policy"]
        assert policy["marker_zero_observed"] is True
        assert policy["calibration_gating"] is False
        assert policy["common_evaluation_requested"] is True
        assert policy["unobservable_method_results_are_reported_not_failed"] is True
