from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration import experiments
from camera_rig_calibration.ap03_camera_model_sensitivity_policy import (
    CALIBRATED,
    PINHOLE_INTRINSICS_ONLY,
    _methods_with_policy,
    _policy_from_methods,
    install_ap03_camera_model_sensitivity_policy,
)
from camera_rig_calibration.config.models import (
    DatasetSettings,
    MethodSettings,
    RigConfig,
    StaticCameraSettings,
)
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.methods.ap03.pipeline import AP03Method
from camera_rig_calibration.methods.ap03.pinhole_reconstruct_stage import (
    prepare_pinhole_camera_info,
)


def _config(tmp_path: Path, policy: str) -> RigConfig:
    methods = _methods_with_policy(
        MethodSettings(enabled=["ap03"]), policy
    )
    return RigConfig(
        dataset=DatasetSettings(id="ap03_camera_model_test"),
        static_cameras=[
            StaticCameraSettings(id="camera_left"),
            StaticCameraSettings(id="camera_right"),
        ],
        methods=methods,
    )


def test_prepare_pinhole_camera_info_changes_only_moving_copy(
    tmp_path: Path,
) -> None:
    shared_raw = tmp_path / "raw_images"
    camera_info = shared_raw / "camera_info"
    camera_info.mkdir(parents=True)
    moving = {
        "width": 1080,
        "height": 1920,
        "distortion_model": "plumb_bob",
        "K": [1000.0, 0.0, 540.0, 0.0, 1001.0, 960.0, 0.0, 0.0, 1.0],
        "D": [0.2, -1.6, 0.0, 0.001, 3.7],
    }
    static = {
        "width": 1920,
        "height": 1080,
        "distortion_model": "plumb_bob",
        "K": [900.0, 0.0, 960.0, 0.0, 901.0, 540.0, 0.0, 0.0, 1.0],
        "D": [0.1, -0.2, 0.0, 0.0, 0.05],
    }
    (camera_info / "moving_calib_camera.json").write_text(
        json.dumps(moving), encoding="utf-8"
    )
    (camera_info / "camera_left.json").write_text(
        json.dumps(static), encoding="utf-8"
    )

    destination = tmp_path / "diagnostic"
    prepare_pinhole_camera_info(
        shared_raw=shared_raw,
        destination=destination,
        moving_camera_id="moving_calib_camera",
    )

    original_moving = json.loads(
        (camera_info / "moving_calib_camera.json").read_text(encoding="utf-8")
    )
    diagnostic_moving = json.loads(
        (destination / "camera_info/moving_calib_camera.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostic_static = json.loads(
        (destination / "camera_info/camera_left.json").read_text(
            encoding="utf-8"
        )
    )

    assert original_moving["D"] == moving["D"]
    assert diagnostic_moving["K"] == moving["K"]
    assert diagnostic_moving["D"] == []
    assert diagnostic_moving["distortion_model"] == "none"
    assert diagnostic_static == static

    metadata = json.loads(
        (destination / "AP03_CAMERA_MODEL_SENSITIVITY.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["scope"] == "ap03_colmap_moving_camera_only"
    assert metadata["original_files_modified"] is False
    assert metadata["ground_truth_used"] is False


def test_policy_switches_only_ap03_reconstruction_stage(tmp_path: Path) -> None:
    install_ap03_camera_model_sensitivity_policy()
    config = _config(tmp_path, PINHOLE_INTRINSICS_ONLY)
    context = RunContext(
        repository_root=tmp_path,
        config=config,
        dataset_root=tmp_path / "dataset",
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap03_single_scale_marker_id=0,
        resolved_ap03_multi_marker_ids=(0, 1),
    )

    commands = AP03Method().commands(context)
    reconstruction = next(
        command for command in commands if command.stage_id == "ap03_reconstruct"
    )
    assert (
        "camera_rig_calibration.methods.ap03.pinhole_reconstruct_stage"
        in reconstruction.argv
    )
    assert "moving-camera pinhole sensitivity" in reconstruction.display_name

    calibrated = _config(tmp_path, CALIBRATED)
    calibrated_context = RunContext(
        repository_root=tmp_path,
        config=calibrated,
        dataset_root=tmp_path / "dataset",
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run_calibrated",
        resolved_ap03_single_scale_marker_id=0,
        resolved_ap03_multi_marker_ids=(0, 1),
    )
    calibrated_commands = AP03Method().commands(calibrated_context)
    calibrated_reconstruction = next(
        command
        for command in calibrated_commands
        if command.stage_id == "ap03_reconstruct"
    )
    assert (
        "camera_rig_calibration.methods.ap03.reconstruct_stage"
        in calibrated_reconstruction.argv
    )


def test_sensitivity_policy_has_distinct_colmap_artifact_fingerprint(
    tmp_path: Path,
) -> None:
    install_ap03_camera_model_sensitivity_policy()
    calibrated = _config(tmp_path, CALIBRATED)
    pinhole = _config(tmp_path, PINHOLE_INTRINSICS_ONLY)

    assert _policy_from_methods(calibrated.methods) == CALIBRATED
    assert _policy_from_methods(pinhole.methods) == PINHOLE_INTRINSICS_ONLY
    assert experiments.colmap_artifact_fingerprint(
        calibrated, "ap03", "input_x"
    ) != experiments.colmap_artifact_fingerprint(
        pinhole, "ap03", "input_x"
    )
