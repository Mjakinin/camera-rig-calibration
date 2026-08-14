from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration import experiments
from camera_rig_calibration.policies.ap03_camera_model_sensitivity_policy import (
    CALIBRATED,
    UNDISTORTED_PINHOLE,
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
    prepare_undistorted_colmap_dataset,
)


def _config(tmp_path: Path, policy: str) -> RigConfig:
    prepared = tmp_path / "dataset"
    prepared.mkdir(parents=True, exist_ok=True)
    methods = _methods_with_policy(
        MethodSettings(enabled=["ap03"]), policy
    )
    return RigConfig(
        dataset=DatasetSettings(
            id="ap03_camera_model_test",
            prepared_root=prepared,
        ),
        static_cameras=[
            StaticCameraSettings(id="camera_left"),
            StaticCameraSettings(id="camera_right"),
        ],
        methods=methods,
    )


def _moving_info() -> dict:
    return {
        "width": 64,
        "height": 48,
        "distortion_model": "plumb_bob",
        "K": [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0],
        "D": [0.15, -0.08, 0.0, 0.001, 0.02],
    }


def test_prepare_pinhole_camera_info_changes_only_moving_copy(
    tmp_path: Path,
) -> None:
    shared_raw = tmp_path / "raw_images"
    camera_info = shared_raw / "camera_info"
    camera_info.mkdir(parents=True)
    moving = _moving_info()
    static = {
        "width": 64,
        "height": 48,
        "distortion_model": "plumb_bob",
        "K": [51.0, 0.0, 32.0, 0.0, 51.0, 24.0, 0.0, 0.0, 1.0],
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
    assert metadata["policy"] == "moving_undistorted_pinhole_v1"
    assert metadata["scope"] == "ap03_colmap_moving_camera_only"
    assert metadata["distortion_used_for_preprocessing"] is True
    assert metadata["distortion_used_by_colmap"] is False
    assert metadata["original_files_modified"] is False
    assert metadata["ground_truth_used"] is False


def test_prepare_undistorted_dataset_keeps_static_and_rectifies_moving(
    tmp_path: Path,
) -> None:
    import cv2
    import numpy as np

    shared_raw = tmp_path / "raw_images"
    camera_info = shared_raw / "camera_info"
    camera_info.mkdir(parents=True)
    (camera_info / "moving_calib_camera.json").write_text(
        json.dumps(_moving_info()), encoding="utf-8"
    )

    source_dataset = tmp_path / "source_colmap"
    images = source_dataset / "images"
    images.mkdir(parents=True)
    static_image = np.zeros((48, 64, 3), dtype=np.uint8)
    moving_image = np.zeros((48, 64, 3), dtype=np.uint8)
    moving_image[10:38, 20:44] = 255
    assert cv2.imwrite(str(images / "static_camera_left.png"), static_image)
    assert cv2.imwrite(str(images / "moving_frame_000001.png"), moving_image)

    with (source_dataset / "image_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "source_type", "source_id", "source_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_name": "static_camera_left.png",
                "source_type": "static",
                "source_id": "camera_left",
                "source_path": "unused",
            }
        )
        writer.writerow(
            {
                "image_name": "moving_frame_000001.png",
                "source_type": "moving",
                "source_id": "moving_calib_camera",
                "source_path": "unused",
            }
        )

    destination = tmp_path / "undistorted_colmap"
    prepare_undistorted_colmap_dataset(
        source_dataset=source_dataset,
        shared_raw=shared_raw,
        destination=destination,
        moving_camera_id="moving_calib_camera",
    )

    assert (destination / "image_manifest.csv").is_file()
    assert (destination / "images/static_camera_left.png").is_file()
    assert (destination / "images/moving_frame_000001.png").is_file()
    output = cv2.imread(str(destination / "images/moving_frame_000001.png"))
    assert output is not None
    assert output.shape == moving_image.shape
    summary = json.loads(
        (destination / "AP03_UNDISTORTION_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["moving_images_undistorted"] == 1
    assert summary["static_images_linked_unchanged"] == 1
    assert summary["original_files_modified"] is False


def test_policy_switches_reconstruction_and_scale_image_geometry(tmp_path: Path) -> None:
    install_ap03_camera_model_sensitivity_policy()
    config = _config(tmp_path, UNDISTORTED_PINHOLE)
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
    assert "undistorted moving-camera PINHOLE" in reconstruction.display_name

    expected_images = str(
        context.run_directory
        / "04_AP03"
        / "colmap"
        / "undistorted_pinhole_dataset"
        / "images"
    )
    for stage_id in ("ap03_single_scale", "ap03_multi_scale"):
        scale = next(command for command in commands if command.stage_id == stage_id)
        image_index = scale.argv.index("--image-dir")
        assert scale.argv[image_index + 1] == expected_images
        assert "matched to undistorted COLMAP image geometry" in scale.display_name

    calibrated = _config(tmp_path / "calibrated", CALIBRATED)
    calibrated_context = RunContext(
        repository_root=tmp_path,
        config=calibrated,
        dataset_root=tmp_path / "calibrated/dataset",
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
    for stage_id in ("ap03_single_scale", "ap03_multi_scale"):
        scale = next(
            command for command in calibrated_commands if command.stage_id == stage_id
        )
        assert "--image-dir" not in scale.argv


def test_sensitivity_policy_has_distinct_colmap_artifact_fingerprint(
    tmp_path: Path,
) -> None:
    install_ap03_camera_model_sensitivity_policy()
    calibrated = _config(tmp_path / "calibrated", CALIBRATED)
    undistorted = _config(tmp_path / "undistorted", UNDISTORTED_PINHOLE)

    assert _policy_from_methods(calibrated.methods) == CALIBRATED
    assert _policy_from_methods(undistorted.methods) == UNDISTORTED_PINHOLE
    assert experiments.colmap_artifact_fingerprint(
        calibrated, "ap03", "input_x"
    ) != experiments.colmap_artifact_fingerprint(
        undistorted, "ap03", "input_x"
    )
