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
    StaticCameraSettings,
)
from camera_rig_calibration.marker_preference_policy import (
    install_marker_preference_policy,
)
from camera_rig_calibration.product_policy import install_product_policy
from camera_rig_calibration.queue_anchor_preference_policy import (
    install_queue_anchor_preference_policy,
)
from camera_rig_calibration.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.submission_bindings import install_submission_bindings
from camera_rig_calibration.submission_policy import install_submission_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_queue_anchor_preference_policy()
install_submission_bindings()

from camera_rig_calibration import preflight  # noqa: E402


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
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
]


def _row(
    observer_type: str,
    observer_id: str,
    marker_id: int,
    *,
    frame_id: str = "",
) -> dict[str, object]:
    return {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": observer_id,
        "frame_id": frame_id,
        "marker_id": marker_id,
        "pnp_success": "true",
        "selection_score": 100.0 - marker_id,
        "pnp_reprojection_rmse_px": 0.4,
        "marker_area_ratio": 0.02,
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": 0.0,
        "tvec_y_m": 0.0,
        "tvec_z_m": 1.0 + 0.01 * marker_id,
    }


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


def _write_observations(path: Path) -> None:
    rows = [
        # Marker 0 exists, but the AP01 root cam_b never sees it. It must not be
        # forced as the queue-wide common anchor.
        _row("static", "cam_a", 0),
        _row("static", "cam_a", 5),
        _row("static", "cam_b", 5),
        _row("moving", "moving_calib_camera", 0, frame_id="0"),
        _row("moving", "moving_calib_camera", 0, frame_id="1"),
        _row("moving", "moving_calib_camera", 5, frame_id="0"),
        _row("moving", "moving_calib_camera", 5, frame_id="1"),
    ]
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
                update={
                    "root_camera": "cam_b",
                    "direct_target_camera": "cam_a",
                }
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
                    "multi": methods.ap03.multi.model_copy(
                        update={"marker_ids": "auto"}
                    ),
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


def test_real_vehicle_preferred_zero_falls_back_at_queue_common_anchor(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    observations = tmp_path / "raw_observations.csv"
    _write_observations(observations)

    jobs = (
        preflight.PreflightJob("ap01", _config(dataset, "ap01")),
        preflight.PreflightJob("ap03", _config(dataset, "ap03")),
    )
    result = preflight.run_queue_preflight(
        jobs,
        raw_observations_csv=observations,
        dataset_root=dataset,
        output_directory=tmp_path / "preflight",
        repository_root=tmp_path,
    )

    assert result.ready is True
    assert all(job.runnable for job in result.jobs)
    assert result.common_evaluation_anchor_marker_id == 5
    for job in result.jobs:
        assert job.selections is not None
        assert job.selections.evaluation_anchor_marker_id == 5
        preference = job.selections.payload["category_marker_preference"][
            "evaluation_anchor"
        ]
        assert preference == {
            "preferred": 0,
            "selected": 5,
            "fallback_used": True,
        }
