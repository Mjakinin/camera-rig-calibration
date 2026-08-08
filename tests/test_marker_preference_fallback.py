from __future__ import annotations

import csv
from pathlib import Path

from camera_rig_calibration.config.models import (
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    InputSourceKind,
    MethodSettings,
    RigConfig,
    SelectionSettings,
    StaticCameraSettings,
)
from camera_rig_calibration.marker_preference_policy import (
    install_marker_preference_policy,
)
from camera_rig_calibration.product_policy import install_product_policy
from camera_rig_calibration.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.submission_bindings import install_submission_bindings
from camera_rig_calibration.submission_policy import install_submission_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_submission_bindings()

from camera_rig_calibration import observations  # noqa: E402


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


def _write_observations(root: Path, marker_ids: tuple[int, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for marker in marker_ids:
        for camera_index, camera in enumerate(("cam_a", "cam_b")):
            rows.append(
                {
                    "observer_type": "static",
                    "observer_id": camera,
                    "camera_name": camera,
                    "frame_id": "",
                    "marker_id": marker,
                    "pnp_success": "true",
                    "selection_score": 100.0 - marker - camera_index,
                    "pnp_reprojection_rmse_px": 0.3 + 0.05 * camera_index,
                    "marker_area_ratio": 0.02,
                }
            )
        for frame in (0, 1, 2):
            rows.append(
                {
                    "observer_type": "moving",
                    "observer_id": "moving_calib_camera",
                    "camera_name": "moving_calib_camera",
                    "frame_id": frame,
                    "marker_id": marker,
                    "pnp_success": "true",
                    "selection_score": 80.0 - marker - frame * 0.01,
                    "pnp_reprojection_rmse_px": 0.4,
                    "marker_area_ratio": 0.015,
                }
            )
    with (root / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _config(
    category: DatasetCategory,
    preferred: int,
    prepared_root: Path,
) -> RigConfig:
    methods = MethodSettings(enabled=["ap02", "ap03"])
    ap02 = methods.ap02.model_copy(
        update={
            "reference_marker_selection_mode": "auto",
            "reference_marker_id": preferred,
        }
    )
    ap03 = methods.ap03.model_copy(
        update={
            "single": methods.ap03.single.model_copy(
                update={"scale_marker_id": preferred}
            ),
            "multi": methods.ap03.multi.model_copy(
                update={"marker_ids": "auto"}
            ),
        },
        deep=True,
    )
    methods = methods.model_copy(
        update={"ap02": ap02, "ap03": ap03}, deep=True
    )
    return RigConfig(
        dataset=DatasetSettings(
            id="marker_preference_test",
            category=category,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=prepared_root,
            input_root=prepared_root,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_a"),
            StaticCameraSettings(id="cam_b"),
        ],
        methods=methods,
        selection=SelectionSettings(mode="auto"),
        evaluation=EvaluationSettings(
            enabled=True,
            anchor_marker_id=preferred,
            anchor_selection_mode="auto",
        ),
    )


def test_real_vehicle_prefers_zero_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    _write_observations(root, (0, 5))
    resolved = observations.resolve_selections(
        _config(DatasetCategory.REAL_VEHICLE, 0, tmp_path), root
    )
    assert resolved.ap02_reference_marker_id == 0
    assert resolved.ap03_single_scale_marker_id == 0
    assert resolved.evaluation_anchor_marker_id == 0
    policy = resolved.payload["category_marker_preference"]
    assert policy["ap02"]["fallback_used"] is False
    assert policy["ap03_single"]["fallback_used"] is False
    assert policy["evaluation_anchor"]["fallback_used"] is False
    assert policy["ground_truth_used"] is False


def test_real_vehicle_zero_missing_falls_back_to_auto(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    _write_observations(root, (5,))
    resolved = observations.resolve_selections(
        _config(DatasetCategory.REAL_VEHICLE, 0, tmp_path), root
    )
    assert resolved.ap02_reference_marker_id == 5
    assert resolved.ap03_single_scale_marker_id == 5
    assert resolved.evaluation_anchor_marker_id == 5
    policy = resolved.payload["category_marker_preference"]
    assert policy["ap02"]["fallback_used"] is True
    assert policy["ap03_single"]["fallback_used"] is True
    assert policy["evaluation_anchor"]["fallback_used"] is True


def test_simulation_fourteen_missing_falls_back_to_auto(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    _write_observations(root, (5,))
    resolved = observations.resolve_selections(
        _config(DatasetCategory.SIMULATION, 14, tmp_path), root
    )
    assert resolved.ap02_reference_marker_id == 5
    assert resolved.ap03_single_scale_marker_id == 5
    assert resolved.evaluation_anchor_marker_id == 5
    policy = resolved.payload["category_marker_preference"]
    assert policy["category_default_marker_id"] == 14
    assert policy["ap02"]["fallback_used"] is True
    assert policy["ap03_single"]["fallback_used"] is True
    assert policy["evaluation_anchor"]["fallback_used"] is True
