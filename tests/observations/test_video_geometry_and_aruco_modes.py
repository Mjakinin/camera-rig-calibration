from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

import camera_rig_calibration.input.video_geometry as video_geometry
from camera_rig_calibration.input.video_geometry import (
    apply_display_rotation,
    probe_video_geometry,
)
from camera_rig_calibration.methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    detect_markers_with_diagnostics,
    effective_detector_config,
)


@pytest.mark.parametrize(
    ("rotation", "expected_rotation", "expected_size"),
    [
        (0, 0, (1920, 1080)),
        (90, 90, (1080, 1920)),
        (180, 180, (1920, 1080)),
        (270, -90, (1080, 1920)),
        (-90, -90, (1080, 1920)),
    ],
)
def test_ffprobe_geometry_normalizes_display_rotation(
    tmp_path: Path,
    monkeypatch,
    rotation: int,
    expected_rotation: int,
    expected_size: tuple[int, int],
) -> None:
    video = tmp_path / "portrait.mov"
    video.write_bytes(b"fixture")
    payload = {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "side_data_list": [{"rotation": rotation}],
            }
        ]
    }
    monkeypatch.setattr(
        video_geometry.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    geometry = probe_video_geometry(video)

    assert geometry.display_rotation_degrees == expected_rotation
    assert (geometry.output_width, geometry.output_height) == expected_size
    assert geometry.orientation_policy == "apply_ffprobe_display_rotation"


def test_negative_ninety_rotation_preserves_pixels_and_becomes_portrait() -> None:
    frame = np.arange(2 * 3, dtype=np.uint8).reshape(2, 3)

    rotated = apply_display_rotation(frame, -90)

    assert rotated.shape == (3, 2)
    assert np.array_equal(rotated, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))


def _small_dark_marker_fixture() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    rng = np.random.default_rng(4)
    image = None
    # The eighth deterministic sample is deliberately below the baseline's
    # useful contrast/size envelope but remains confirmed by both gamma passes.
    for _ in range(8):
        size = int(rng.integers(12, 55))
        low = int(rng.integers(0, 130))
        high = int(rng.integers(max(low + 5, 40), 190))
        blur = int(rng.choice([1, 3, 5, 7]))
        noise = float(rng.choice([0, 2, 4, 8]))
        marker = np.zeros((size, size), dtype=np.uint8)
        if hasattr(cv2.aruco, "generateImageMarker"):
            cv2.aruco.generateImageMarker(
                dictionary, 17, size, marker, 1
            )
        else:
            marker = cv2.aruco.drawMarker(dictionary, 17, size)
        marker = (
            low + marker.astype(np.float64) / 255.0 * (high - low)
        ).astype(np.uint8)
        image = np.full((800, 800), high, dtype=np.uint8)
        offset = 400 - size // 2
        image[offset:offset + size, offset:offset + size] = marker
        if blur > 1:
            image = cv2.GaussianBlur(image, (blur, blur), 0)
        if noise:
            image = np.clip(
                image + rng.normal(0, noise, image.shape),
                0,
                255,
            ).astype(np.uint8)
    assert image is not None
    return image


def test_high_sensitivity_recovers_confirmed_small_dark_marker() -> None:
    image = _small_dark_marker_fixture()

    baseline, _ = detect_markers_with_diagnostics(
        image, detection_mode="baseline"
    )
    sensitive, evidence = detect_markers_with_diagnostics(
        image, detection_mode="high_sensitivity"
    )

    assert baseline == []
    assert [
        (
            item["marker_id"],
            item["detection_source"],
            item["detection_support"],
        )
        for item in sensitive
    ] == [(17, "gamma_consensus", 2)]
    assert {
        item["pass"]
        for item in evidence
        if item["accepted"] and item["marker_id"] == 17
    } == {"gamma_0.60", "gamma_0.65"}


def test_detector_contract_records_reproducible_parameters() -> None:
    config = effective_detector_config(
        "high_sensitivity", "DICT_4X4_50"
    )

    assert config["contract"] == DETECTOR_CONTRACT
    assert config["gamma_passes"] == [0.60, 0.65]
    assert config["parameters"]["adaptiveThreshWinSizeMax"] == 53
    assert config["support_rule"]["maximum_area_ratio"] == 2.0
