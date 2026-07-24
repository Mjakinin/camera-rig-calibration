from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_rig_calibration.config.models import (
    DatasetSettings,
    MethodSettings,
    MovingCameraSettings,
    ProjectSettings,
    RigConfig,
    SceneType,
    StaticCameraSettings,
)


def write_intrinsics(path: Path, camera_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera_name": camera_id,
                "width": 640,
                "height": 480,
                "distortion_model": "plumb_bob",
                "K": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def make_prepared_dataset(root: Path, camera_ids: list[str]) -> Path:
    raw = root / "raw_images"
    (raw / "static").mkdir(parents=True)
    (raw / "moving").mkdir(parents=True)
    (raw / "camera_info").mkdir(parents=True)
    for camera_id in camera_ids:
        (raw / "static" / f"{camera_id}.png").write_bytes(b"fixture")
        write_intrinsics(raw / "camera_info" / f"{camera_id}.json", camera_id)
    (raw / "moving" / "frame_000000.png").write_bytes(b"fixture")
    write_intrinsics(
        raw / "camera_info" / "calibration_camera.json", "calibration_camera"
    )
    return root


@pytest.fixture
def prepared_config(tmp_path: Path) -> RigConfig:
    cameras = ["front-left", "roof.camera"]
    dataset_root = make_prepared_dataset(tmp_path / "prepared", cameras)
    return RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(
            id="generic_fixture",
            scene_type=SceneType.EXTERIOR,
            prepared_root=dataset_root,
        ),
        static_cameras=[StaticCameraSettings(id=value) for value in cameras],
        moving_camera=MovingCameraSettings(id="calibration_camera"),
        methods=MethodSettings(enabled=["ap02"]),
    )
