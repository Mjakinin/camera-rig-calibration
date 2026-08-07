from __future__ import annotations

from pathlib import Path

import pytest

from parity.main_route2_v1.observation_parity import (
    compare_semantic_rows,
    load_observation_csv,
    semantic_row_keys,
)


def _row(
    *,
    camera: str = "moving_calib_camera",
    frame: str = "1",
    marker: str = "14",
) -> dict[str, str]:
    row = {
        "observer_type": "moving",
        "observer_id": "moving_frame_000001",
        "camera_name": camera,
        "frame_id": frame,
        "marker_id": marker,
        "marker_length_m": "0.17",
        "fx": "900",
        "fy": "900",
        "cx": "640",
        "cy": "360",
        "pnp_success": "True",
        "rvec_x": "0.1",
        "rvec_y": "0.2",
        "rvec_z": "0.3",
        "tvec_x_m": "0.01",
        "tvec_y_m": "0.02",
        "tvec_z_m": "1.0",
        "distance_m": "1.0002499687578101",
        "center_u": "640",
        "center_v": "360",
        "area_px2": "10000",
        "distortion_model": "plumb_bob",
        "image_width": "1280",
        "image_height": "720",
        "pnp_reprojection_rmse_px": "0.25",
    }
    corners = ((590, 310), (690, 310), (690, 410), (590, 410))
    for index, (u, v) in enumerate(corners):
        row[f"corner{index}_u"] = str(u)
        row[f"corner{index}_v"] = str(v)
    for index in range(8):
        row[f"d{index}"] = "0"
    return row


def test_semantic_row_keys_normalize_frames_and_preserve_duplicates() -> None:
    keys = semantic_row_keys([_row(), _row()])
    assert keys[0] == {
        "source_kind": "moving",
        "camera_id": "moving_calib_camera",
        "frame_id": "000001",
        "marker_id": 14,
        "occurrence_index": 0,
    }
    assert keys[1]["occurrence_index"] == 1


def test_duplicate_semantic_rows_are_not_collapsed() -> None:
    report, differences = compare_semantic_rows(
        [_row(), _row()], [_row(), _row()]
    )
    assert not differences
    assert report["set_content_parity"] is True
    assert report["main_duplicate_base_key_count"] == 1


def test_corner_order_mismatch_is_classified_explicitly() -> None:
    main = _row()
    wizard = _row()
    for index, source in enumerate((1, 2, 3, 0)):
        wizard[f"corner{index}_u"] = main[f"corner{source}_u"]
        wizard[f"corner{index}_v"] = main[f"corner{source}_v"]

    report, differences = compare_semantic_rows([main], [wizard])
    assert report["set_content_parity"] is False
    assert differences[0]["reason"] == "corner_order_mismatch"


def test_set_content_and_original_order_are_reported_separately() -> None:
    first = _row(frame="1")
    second = _row(frame="2")
    report, differences = compare_semantic_rows(
        [first, second], [second, first]
    )
    assert not differences
    assert report["set_content_parity"] is True
    assert report["original_order_parity"] is False


def test_corner_numeric_tolerance_is_explicit() -> None:
    main = _row()
    within = _row()
    within["corner0_u"] = str(float(within["corner0_u"]) + 5e-10)
    equal, differences = compare_semantic_rows([main], [within])
    assert not differences
    assert equal["set_content_parity"] is True

    outside = _row()
    outside["corner0_u"] = str(float(outside["corner0_u"]) + 2e-9)
    mismatch, differences = compare_semantic_rows([main], [outside])
    assert mismatch["set_content_parity"] is False
    assert differences[0]["field"] == "corner0_u"


def test_observation_parity_rejects_ground_truth_paths_before_read(
    tmp_path: Path,
) -> None:
    forbidden = tmp_path / "ground_truth" / "observations.csv"
    with pytest.raises(PermissionError, match="Ground Truth"):
        load_observation_csv(forbidden)
