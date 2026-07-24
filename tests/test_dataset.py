from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_rig_calibration.config.models import (
    DatasetSettings,
    MovingCameraSettings,
    RigConfig,
    StaticCameraSettings,
)
from camera_rig_calibration.dataset.discovery import inspect_prepared_dataset
from camera_rig_calibration.dataset.validation import validate_dataset

from conftest import make_prepared_dataset


@pytest.mark.parametrize("camera_count", [2, 4, 6])
def test_arbitrary_camera_counts_and_ids(tmp_path: Path, camera_count: int) -> None:
    ids = [f"exterior.camera-{index}" for index in range(camera_count)]
    root = make_prepared_dataset(tmp_path / f"rig_{camera_count}", ids)
    config = RigConfig(
        dataset=DatasetSettings(id=f"rig_{camera_count}", prepared_root=root),
        static_cameras=[StaticCameraSettings(id=value) for value in ids],
        moving_camera=MovingCameraSettings(id="calibration_camera"),
    )
    result = validate_dataset(config, root)
    assert result.valid, result.errors
    assert result.static_camera_count == camera_count
    assert result.moving_frame_count == 1
    assert inspect_prepared_dataset(root)["static_camera_ids"] == ids


def test_moving_camera_id_must_be_unique(tmp_path: Path) -> None:
    root = make_prepared_dataset(tmp_path / "rig", ["roof"])
    with pytest.raises(ValueError, match="moving camera ID"):
        RigConfig(
            dataset=DatasetSettings(id="rig", prepared_root=root),
            static_cameras=[StaticCameraSettings(id="roof")],
            moving_camera=MovingCameraSettings(id="roof"),
        )


def test_moving_intrinsics_resolution_mismatch_fails_preflight(
    tmp_path: Path,
) -> None:
    root = make_prepared_dataset(tmp_path / "rig", ["roof"])
    moving_frame = root / "raw_images/moving/frame_000000.png"
    cv2.imwrite(
        str(moving_frame),
        np.zeros((720, 1280, 3), dtype=np.uint8),
    )
    config = RigConfig(
        dataset=DatasetSettings(id="rig", prepared_root=root),
        static_cameras=[StaticCameraSettings(id="roof")],
        moving_camera=MovingCameraSettings(id="calibration_camera"),
    )

    result = validate_dataset(config, root)

    assert not result.valid
    assert any(
        "intrinsic resolution does not match" in error
        and "rigcal never scales K silently" in error
        for error in result.errors
    )
