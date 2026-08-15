from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import camera_rig_calibration.intrinsics_profiles as profile_module
from camera_rig_calibration.input import intrinsics as profile_runner
from camera_rig_calibration.input.video_geometry import VideoGeometry
from camera_rig_calibration.intrinsics_profiles import (
    discover_intrinsic_profiles,
    profile_fingerprint,
)
from conftest import REPOSITORY_ROOT


ENGINE_PATH = (
    REPOSITORY_ROOT
    / "src"
    / "camera_rig_calibration"
    / "input"
    / "intrinsics_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("rigcal_intrinsics_engine", ENGINE_PATH)
assert SPEC is not None and SPEC.loader is not None
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)


def _intrinsics(path: Path, *, width: int = 640, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera_name": "moving_calib_camera",
                "width": width,
                "height": height,
                "distortion_model": "plumb_bob",
                "K": [500, 0, width / 2, 0, 500, height / 2, 0, 0, 1],
                "D": [0, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )


def test_balanced_candidate_passes_are_deterministic_and_interleaved() -> None:
    first = ENGINE.balanced_candidate_indices(100, 30.0, 3.0)
    second = ENGINE.balanced_candidate_indices(
        100, 30.0, 6.0, tested=set(first)
    )

    assert first == list(range(0, 100, 10))
    assert second == list(range(5, 100, 10))
    assert set(first).isdisjoint(second)


def test_balanced_detector_returns_full_resolution_subpixel_corners() -> None:
    tile = 80
    board = np.full((9 * tile, 11 * tile), 210, dtype=np.uint8)
    for row in range(7):
        for column in range(9):
            color = 0 if (row + column) % 2 == 0 else 255
            y0 = (row + 1) * tile
            x0 = (column + 1) * tile
            board[y0 : y0 + tile, x0 : x0 + tile] = color

    found, corners = ENGINE.detect_checkerboard_balanced(
        board, (8, 6), preview_max_dimension=440
    )

    assert found
    assert corners is not None
    assert corners.shape == (48, 1, 2)
    assert float(corners[:, :, 0].max()) > 600


def test_intrinsics_engine_accepts_checkerboard_image_folder(
    tmp_path: Path, monkeypatch
) -> None:
    tile = 50
    board = np.full((8 * tile, 10 * tile), 210, dtype=np.uint8)
    for row in range(7):
        for column in range(9):
            color = 0 if (row + column) % 2 == 0 else 255
            y0 = row * tile
            x0 = column * tile
            board[y0 : y0 + tile, x0 : x0 + tile] = color
    images = tmp_path / "intrinsics_images"
    images.mkdir()
    assert cv2.imwrite(str(images / "view_01.png"), board)
    output = tmp_path / "output"

    monkeypatch.setattr(
        ENGINE,
        "model_comparison",
        lambda *args, **kwargs: (
            False,
            {
                "training_views": 1,
                "holdout_views": 1,
                "standard_holdout_median_rmse_px": 0.1,
                "rational_holdout_median_rmse_px": 0.1,
                "selected_model": "plumb_bob",
            },
        ),
    )

    def fake_calibration(detections, selected, *args, **kwargs):
        return (
            list(selected),
            {
                "K": np.array(
                    [[500.0, 0.0, 250.0], [0.0, 500.0, 200.0], [0.0, 0.0, 1.0]]
                ),
                "D": np.zeros(5),
                "rms": 0.1,
                "per_view": [0.1 for _ in selected],
                "median_view_error": 0.1,
                "maximum_view_error": 0.1,
            },
            [],
        )

    monkeypatch.setattr(ENGINE, "calibrate_with_outlier_filter", fake_calibration)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intrinsics-engine",
            "--images",
            str(images),
            "--out",
            str(output),
            "--minimum-detections",
            "1",
            "--max-views",
            "1",
        ],
    )

    ENGINE.main()

    payload = json.loads(
        (output / "moving_calib_camera.json").read_text(encoding="utf-8")
    )
    assert payload["source_type"] == "image_directory"
    assert payload["source_images"] == str(images.resolve())
    assert payload["successful_checkerboard_detections"] == 1
    assert (output / "selected_frames/frame_000000.png").is_file()


def test_profile_catalog_indexes_managed_profiles_only(tmp_path: Path) -> None:
    catalog = tmp_path / "config/intrinsics"
    new_root = catalog / "iphone_05x_4k/abcdef123456"
    _intrinsics(new_root / "intrinsics.json", width=3840, height=2160)
    (new_root / "profile.yaml").write_text(
        yaml.safe_dump(
            {
                "profile_id": "iphone_05x_4k",
                "fingerprint": "abcdef1234567890",
                "intrinsics": "intrinsics.json",
                "created_at": "2026-07-24T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    _intrinsics(tmp_path / "results/real_vehicle/_intrinsics/legacy.json")

    profiles = discover_intrinsic_profiles(tmp_path)

    assert {profile.profile_id for profile in profiles} == {"iphone_05x_4k"}
    assert next(
        profile for profile in profiles if profile.profile_id == "iphone_05x_4k"
    ).key == "iphone_05x_4k@abcdef1234567890"


def _mock_video_geometry(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_module,
        "probe_video_geometry",
        lambda path: VideoGeometry(
            encoded_width=1920,
            encoded_height=1080,
            display_rotation_degrees=-90,
            output_width=1080,
            output_height=1920,
        ),
    )


def test_profile_fingerprint_changes_with_scan_contract(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_video_geometry(monkeypatch)
    video = tmp_path / "checkerboard.mov"
    video.write_bytes(b"video")
    common = {
        "columns": 8,
        "rows": 6,
        "maximum_views": 80,
        "minimum_frame_gap": 5,
        "minimum_detections": 20,
        "scan_target_hz": 3.0,
        "preview_max_dimension": 1920,
    }

    balanced = profile_fingerprint(video, scan_mode="balanced", **common)
    exhaustive = profile_fingerprint(
        video, scan_mode="full_frame", **common
    )

    assert balanced != exhaustive


def test_profile_fingerprint_supports_checkerboard_image_folder(
    tmp_path: Path,
) -> None:
    images = tmp_path / "intrinsics_images"
    images.mkdir()
    (images / "view_01.png").write_bytes(b"first")
    (images / "view_02.jpg").write_bytes(b"second")
    settings = {
        "columns": 8,
        "rows": 6,
        "maximum_views": 80,
        "minimum_frame_gap": 5,
        "minimum_detections": 20,
        "scan_mode": "balanced",
        "scan_target_hz": 3.0,
        "preview_max_dimension": 1920,
    }

    first = profile_fingerprint(images, **settings)
    second = profile_fingerprint(images, **settings)
    (images / "view_02.jpg").write_bytes(b"changed")
    changed = profile_fingerprint(images, **settings)

    assert first == second
    assert changed != first


def test_profile_runner_passes_checkerboard_image_folder_to_engine(
    tmp_path: Path, monkeypatch
) -> None:
    images = tmp_path / "intrinsics_images"
    images.mkdir()
    (images / "view_01.png").write_bytes(b"fixture")
    script = tmp_path / "engine.py"
    script.write_text("# fixture", encoding="utf-8")
    destination = tmp_path / "installed.json"
    observed_command: list[str] = []

    def fake_run(command, **kwargs):
        observed_command.extend(command)
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True, exist_ok=True)
        _intrinsics(output / "moving_calib_camera.json")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intrinsics",
            "--script",
            str(script),
            "--images",
            str(images),
            "--work-directory",
            str(tmp_path / "work"),
            "--destination",
            str(destination),
            "--camera-id",
            "moving",
        ],
    )

    profile_runner.main()

    assert observed_command[
        observed_command.index("--images") + 1
    ] == str(images.resolve())
    assert destination.is_file()


def test_failed_profile_generation_is_not_published(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_video_geometry(monkeypatch)
    video = tmp_path / "checkerboard.mov"
    script = tmp_path / "engine.py"
    video.write_bytes(b"video")
    script.write_text("# fixture", encoding="utf-8")
    destination = tmp_path / "installed.json"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(2, "engine")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intrinsics",
            "--script",
            str(script),
            "--video",
            str(video),
            "--work-directory",
            str(tmp_path / "work"),
            "--destination",
            str(destination),
            "--camera-id",
            "moving",
            "--repository",
            str(tmp_path),
            "--profile-id",
            "phone_4k",
        ],
    )

    with pytest.raises(subprocess.CalledProcessError):
        profile_runner.main()

    profile_parent = (
        tmp_path / "config/intrinsics/phone_4k"
    )
    assert not list(profile_parent.glob("*/profile.yaml"))
    assert not list(profile_parent.glob(".*.staging-*"))
    assert not destination.exists()


def test_successful_profile_is_published_atomically_with_clean_layout(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_video_geometry(monkeypatch)
    video = tmp_path / "checkerboard.mov"
    script = tmp_path / "engine.py"
    video.write_bytes(b"video")
    script.write_text("# fixture", encoding="utf-8")
    destination = tmp_path / "installed.json"

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True, exist_ok=True)
        _intrinsics(output / "moving_calib_camera.json")
        (output / "selected_frames").mkdir()
        (output / "debug_selected").mkdir()
        (output / "INTRINSICS_REPORT.txt").write_text(
            "fixture\n", encoding="utf-8"
        )
        (output / "checkerboard_detections.csv").write_text(
            "frame_index\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intrinsics",
            "--script",
            str(script),
            "--video",
            str(video),
            "--work-directory",
            str(tmp_path / "work"),
            "--destination",
            str(destination),
            "--camera-id",
            "moving",
            "--repository",
            str(tmp_path),
            "--profile-id",
            "phone_4k",
        ],
    )

    profile_runner.main()

    profiles = discover_intrinsic_profiles(tmp_path)
    assert len(profiles) == 1
    profile = profiles[0]
    assert (profile.root / "intrinsics.json").is_file()
    assert (profile.root / "profile.yaml").is_file()
    assert (profile.root / "timings.json").is_file()
    assert (profile.root / "selected_frames").is_dir()
    assert (
        profile.root / "diagnostics/checkerboard_detections.csv"
    ).is_file()
    installed = json.loads(destination.read_text(encoding="utf-8"))
    assert installed["camera_name"] == "moving"
    assert installed["rigcal_intrinsics_profile"].startswith("phone_4k@")
