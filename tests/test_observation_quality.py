from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_rig_calibration.config.models import (
    MarkerSettings,
    ObservationQualitySettings,
)
from camera_rig_calibration.observation_quality import (
    ObservationQualityError,
    filter_observations,
)


def _row(
    observer_type: str,
    observer_id: str,
    marker_id: int,
    *,
    frame_id: str = "",
    area: float = 400.0,
    z: float = 1.0,
    corner_shift: float = 0.0,
) -> dict[str, object]:
    projected_half_edge = 10.0 / abs(z) if z else 10.0
    corners = [
        (50 - projected_half_edge, 50 + projected_half_edge),
        (50 + projected_half_edge, 50 + projected_half_edge),
        (50 + projected_half_edge, 50 - projected_half_edge),
        (50 - projected_half_edge, 50 - projected_half_edge),
    ]
    row: dict[str, object] = {
        "observer_type": observer_type,
        "observer_id": observer_id,
        "camera_name": observer_id if observer_type == "static" else "",
        "frame_id": frame_id,
        "image_path": f"frame_{frame_id}.png" if frame_id else f"{observer_id}.png",
        "marker_id": marker_id,
        "marker_length_m": 0.2,
        "detection_success": True,
        "pnp_success": True,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
        "distortion_model": "plumb_bob",
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": 0.0,
        "tvec_y_m": 0.0,
        "tvec_z_m": z,
        "distance_m": abs(z),
        "center_u": 50.0,
        "center_v": 50.0,
        "image_width_px": 100,
        "image_height_px": 100,
        "area_px2": area,
        "marker_area_ratio": area / 10_000.0,
    }
    for index, (u, v) in enumerate(corners):
        row[f"corner{index}_u"] = u + corner_shift
        row[f"corner{index}_v"] = v
    for index in range(8):
        row[f"d{index}"] = 0.0
    return row


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_filter_recomputes_rmse_and_writes_complete_audit(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        _row("static", "left", 7),
        _row("moving", "moving_1", 7, frame_id="1", area=25.0),
        _row("moving", "moving_2", 8, frame_id="2", z=3.0),
        _row("moving", "moving_3", 9, frame_id="3", corner_shift=10.0),
    ]
    _write(source, rows)

    result = filter_observations(
        source,
        tmp_path / "filter",
        job_id="ap03_quality",
        marker_settings=MarkerSettings(),
        quality=ObservationQualitySettings(
            maximum_pnp_reprojection_error_px=3.0,
            minimum_marker_area_ratio=0.01,
            maximum_marker_distance_m=2.0,
        ),
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 3
    summary = json.loads(
        (result.output_directory / "observation_filter_summary.json").read_text()
    )
    assert summary["filter"] == "observation_quality_v2"
    assert summary["decision_counts"] == {
        "accepted": 1,
        "marker_area_ratio_below_minimum": 1,
        "marker_distance_above_maximum": 1,
        "pnp_reprojection_error_above_maximum": 1,
    }
    with result.accepted_path.open(newline="", encoding="utf-8") as handle:
        accepted = list(csv.DictReader(handle))
    assert float(accepted[0]["pnp_reprojection_rmse_px"]) == pytest.approx(0.0)
    with (
        result.filtered_observations_root
        / "shared_all_aruco_observations.csv"
    ).open(newline="", encoding="utf-8") as handle:
        filtered = list(csv.DictReader(handle))
    assert [int(row["marker_id"]) for row in filtered] == [7]


def test_positive_depth_check_is_enabled_by_default(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    _write(source, [_row("static", "left", 7, z=-1.0)])

    result = filter_observations(
        source,
        tmp_path / "filter",
        job_id="fixed_checks",
        marker_settings=MarkerSettings(),
        quality=ObservationQualitySettings(),
    )

    assert result.accepted_count == 0
    with result.rejected_path.open(newline="", encoding="utf-8") as handle:
        rejected = list(csv.DictReader(handle))
    assert rejected[0]["reason"] == "marker_depth_not_positive"


def test_positive_depth_check_can_be_explicitly_disabled(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    _write(source, [_row("static", "left", 7, z=-1.0)])

    result = filter_observations(
        source,
        tmp_path / "filter",
        job_id="diagnostic_depth",
        marker_settings=MarkerSettings(),
        quality=ObservationQualitySettings(
            require_positive_depth=False,
            maximum_pnp_reprojection_error_px="disabled",
        ),
    )

    assert result.accepted_count == 1


def test_missing_legacy_reprojection_inputs_fail_preflight(tmp_path: Path) -> None:
    source = tmp_path / "legacy.csv"
    row = _row("static", "left", 7)
    row.pop("corner3_v")
    _write(source, [row])

    with pytest.raises(ObservationQualityError, match="missing columns"):
        filter_observations(
            source,
            tmp_path / "filter",
            job_id="legacy",
            marker_settings=MarkerSettings(),
            quality=ObservationQualitySettings(),
        )


def test_missing_legacy_distortion_data_fails_preflight(tmp_path: Path) -> None:
    source = tmp_path / "legacy.csv"
    row = _row("static", "left", 7)
    row.pop("d0")
    _write(source, [row])

    with pytest.raises(ObservationQualityError, match="missing columns: d0"):
        filter_observations(
            source,
            tmp_path / "filter",
            job_id="legacy",
            marker_settings=MarkerSettings(),
            quality=ObservationQualitySettings(),
        )


def test_dimensions_are_recovered_from_referenced_image(
    tmp_path: Path,
) -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), image)
    row = _row("static", "left", 7, area=240.0)
    row.pop("image_width_px")
    row.pop("image_height_px")
    row.pop("marker_area_ratio")
    row["image_path"] = image_path.name
    source = tmp_path / "raw.csv"
    _write(source, [row])

    result = filter_observations(
        source,
        tmp_path / "filter",
        job_id="dimension_recovery",
        marker_settings=MarkerSettings(),
        quality=ObservationQualitySettings(),
    )

    with result.accepted_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        accepted = next(csv.DictReader(handle))
    assert int(accepted["image_width_px"]) == 200
    assert int(accepted["image_height_px"]) == 120
    assert float(accepted["marker_area_ratio"]) == pytest.approx(0.01)


def test_missing_dimensions_and_unreadable_image_stop_preflight(
    tmp_path: Path,
) -> None:
    row = _row("static", "left", 7)
    row.pop("image_width_px")
    row.pop("image_height_px")
    row.pop("marker_area_ratio")
    row["image_path"] = "missing.png"
    source = tmp_path / "raw.csv"
    _write(source, [row])

    with pytest.raises(
        ObservationQualityError,
        match="exact image dimensions are missing",
    ):
        filter_observations(
            source,
            tmp_path / "filter",
            job_id="missing_dimensions",
            marker_settings=MarkerSettings(),
            quality=ObservationQualitySettings(),
        )
