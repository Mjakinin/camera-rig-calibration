from __future__ import annotations

import csv
from pathlib import Path

from camera_rig_calibration.ap01_auto_direct import automatic_ap01_direct_target
from camera_rig_calibration.config.models import (
    DatasetCategory,
    DatasetSettings,
    InputSourceKind,
    MethodSettings,
    RigConfig,
    StaticCameraSettings,
)


def _observation(
    camera: str,
    marker: int,
    *,
    rmse: float,
    score: float,
    x: float,
) -> dict[str, object]:
    # 100x100 px marker at roughly the image centre and 2 m distance easily
    # clears the baseline AP01 direct area/distance/combined-quality gates.
    return {
        "observer_type": "static",
        "observer_id": camera,
        "camera_name": camera,
        "marker_id": marker,
        "pnp_success": "true",
        "selection_score": score,
        "pnp_reprojection_rmse_px": rmse,
        "marker_area_ratio": 0.01,
        "distance_m": 2.0,
        "center_u": 640.0,
        "center_v": 360.0,
        "corner0_u": 590.0,
        "corner0_v": 310.0,
        "corner1_u": 690.0,
        "corner1_v": 310.0,
        "corner2_u": 690.0,
        "corner2_v": 410.0,
        "corner3_u": 590.0,
        "corner3_v": 410.0,
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": x,
        "tvec_y_m": marker * 0.001,
        "tvec_z_m": 2.0,
    }


def _config(tmp_path: Path, camera_ids: list[str]) -> RigConfig:
    methods = MethodSettings()
    methods = methods.model_copy(
        update={
            "enabled": ["ap01"],
            "ap01": methods.ap01.model_copy(
                update={
                    "direct_target_camera": "auto",
                    "root_camera": camera_ids[0],
                }
            ),
        },
        deep=True,
    )
    return RigConfig(
        dataset=DatasetSettings(
            id="ap01_auto_direct",
            category=DatasetCategory.SIMULATION,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=tmp_path,
            input_root=tmp_path,
        ),
        static_cameras=[StaticCameraSettings(id=value) for value in camera_ids],
        methods=methods,
    )


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (path / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_direct_selector_requires_two_quality_mad_inlier_markers(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations"
    _write(
        observations,
        [
            _observation("cam_root", 7, rmse=0.5, score=18.0, x=0.0),
            _observation("cam_root", 14, rmse=0.4, score=20.0, x=0.0),
            _observation("cam_direct", 7, rmse=0.6, score=17.0, x=0.10),
            _observation("cam_direct", 14, rmse=0.5, score=19.0, x=0.10),
            # Better single-marker RMSE is deliberately insufficient.
            _observation("cam_relay", 14, rmse=0.1, score=30.0, x=0.20),
        ],
    )
    selected, candidates = automatic_ap01_direct_target(
        _config(tmp_path, ["cam_root", "cam_direct", "cam_relay"]),
        observations,
        "cam_root",
    )
    assert selected == "cam_direct"
    direct = next(item for item in candidates if item["id"] == "cam_direct")
    relay = next(item for item in candidates if item["id"] == "cam_relay")
    assert direct["quality_filtered_markers"] == 2
    assert direct["independent_inlier_markers"] == 2
    assert direct["quality_filter_fallback_used"] is False
    assert direct["compatible"] is True
    assert relay["compatible"] is False


def test_exact_direct_selector_returns_none_for_relay_only_case(tmp_path: Path) -> None:
    observations = tmp_path / "observations"
    _write(
        observations,
        [
            _observation("cam_root", 14, rmse=0.4, score=20.0, x=0.0),
            _observation("cam_other", 14, rmse=0.5, score=19.0, x=0.10),
        ],
    )
    selected, candidates = automatic_ap01_direct_target(
        _config(tmp_path, ["cam_root", "cam_other"]),
        observations,
        "cam_root",
    )
    assert selected is None
    assert candidates[0]["compatible"] is False
