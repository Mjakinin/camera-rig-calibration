from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    SimulationExperimentSummary,
    discover_prepared_datasets,
    discover_raw_input_folders,
    discover_simulation_experiments,
    find_matching_simulation,
    format_simulation_parameters,
)
from camera_rig_calibration.dataset.discovery import (
    discover_image_directories,
    media_path_role,
)


def test_local_inputs_and_prepared_datasets_are_discovered_recursively(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "data_local/exterior_01"
    (raw / "videos").mkdir(parents=True)
    (raw / "static").mkdir()
    (raw / "camera_info").mkdir()
    (raw / "videos/moving.mp4").write_bytes(b"video")
    (raw / "static/front-left.png").write_bytes(b"image")
    (raw / "camera_info/front-left.yaml").write_text("camera_name: front-left")

    prepared = tmp_path / "results/real_vehicle/native_rate/existing"
    for name in ("static", "moving", "camera_info"):
        (prepared / "raw_images" / name).mkdir(parents=True, exist_ok=True)
    (prepared / "raw_images/static/front-left.png").write_bytes(b"image")
    (prepared / "raw_images/moving/frame_000000.png").write_bytes(b"frame")
    (prepared / "raw_images/camera_info/front-left.json").write_text("{}")
    (prepared / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": "existing",
                "category": "real_vehicle",
            }
        ),
        encoding="utf-8",
    )

    raw_entries = discover_raw_input_folders(tmp_path)
    prepared_entries = discover_prepared_datasets(tmp_path)

    assert len(raw_entries) == 1
    assert (raw_entries[0].videos, raw_entries[0].images, raw_entries[0].intrinsics) == (
        1,
        1,
        1,
    )
    assert len(prepared_entries) == 1
    assert prepared_entries[0].static_camera_ids == ("front-left",)
    assert prepared_entries[0].moving_frames == 1


def test_prepared_dataset_without_completed_method_is_not_called_a_result(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "results/real_vehicle/3Hz/detector_retry"
    for name in ("static", "moving", "camera_info"):
        (prepared / "raw_images" / name).mkdir(parents=True, exist_ok=True)
    (prepared / "raw_images/static/cam.png").write_bytes(b"image")
    (prepared / "raw_images/moving/frame.png").write_bytes(b"frame")
    (prepared / "raw_images/camera_info/cam.json").write_text("{}")
    (prepared / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": "detector_retry",
                "category": "real_vehicle",
            }
        ),
        encoding="utf-8",
    )
    (prepared / "SUMMARY.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "status": "prepared",
                "available_methods": 0,
                "methods": [],
            }
        ),
        encoding="utf-8",
    )

    entry = next(
        item
        for item in discover_prepared_datasets(tmp_path)
        if item.id == "detector_retry"
    )

    assert entry.has_results is False


def test_loose_data_local_files_absorb_nested_rosbag_without_duplicate_row(
    tmp_path: Path,
) -> None:
    local = tmp_path / "data_local"
    bag = local / "rosbag2_recording"
    bag.mkdir(parents=True)
    (local / "moving.mov").write_bytes(b"video")
    (local / "checkerboard.mov").write_bytes(b"video")
    (bag / "recording.mcap").write_bytes(b"bag")

    entries = discover_raw_input_folders(tmp_path)

    assert len(entries) == 1
    assert entries[0].path == local.resolve()
    assert entries[0].videos == 2
    assert entries[0].recordings == 1


def test_role_folders_directly_below_data_local_form_one_acquisition(
    tmp_path: Path,
) -> None:
    local = tmp_path / "data_local"
    media = {
        local / "static_v2/front.png": b"static",
        local / "moving_frames_v2/frame.png": b"moving",
        local / "iphone_intrinsics_v2/view.png": b"checkerboard",
    }
    for path, content in media.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    entries = discover_raw_input_folders(tmp_path)

    assert len(entries) == 1
    assert entries[0].path == local.resolve()
    assert entries[0].images == 3


def test_named_image_folders_are_classified_by_camera_role(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local/recording"
    paths = {
        "static": root / "static_v2/front_left/images",
        "moving": root / "moving_frames_3hz/images",
        "checkerboard": root / "iphone_intrinsics_v2/images",
    }
    for directory in paths.values():
        directory.mkdir(parents=True)
        (directory / "frame_0001.png").write_bytes(b"image")

    discovered = discover_image_directories(root)

    assert discovered == {
        role: [path.resolve()] for role, path in paths.items()
    }


def test_arbitrary_video_names_inherit_role_from_versioned_folder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data_local/recording"
    paths = {
        "static": root / "static_night_v2/IMG_1001.mov",
        "moving": root / "moving_capture_v3/IMG_1002.mov",
        "checkerboard": root / "iphone_intrinsics_v4/IMG_1003.mov",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    assert {
        role: media_path_role(path, root)
        for role, path in paths.items()
    } == {
        "static": "static",
        "moving": "moving",
        "checkerboard": "checkerboard",
    }


def test_layout_v2_datasets_are_shown_once_and_named_clearly(
    tmp_path: Path,
) -> None:
    simulation = tmp_path / "results/simulation/fov/100deg"
    simulation.mkdir(parents=True)
    (simulation / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": "fov_100deg",
                "category": "simulation",
                "storage": {"factor": "fov", "value": "100deg"},
                "simulation_parameters": {"moving_hfov_deg": 100.0},
            }
        ),
        encoding="utf-8",
    )
    raw = simulation / "raw_images"
    for name in ("static", "moving", "camera_info"):
        (raw / name).mkdir(parents=True)
    (raw / "static/cam1.png").write_bytes(b"image")
    (raw / "moving/frame.png").write_bytes(b"image")
    (raw / "camera_info/cam1.json").write_text("{}")
    real = tmp_path / "results/real_vehicle/1Hz/real_05x_4k_1hz"
    real.mkdir(parents=True)
    (real / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": "real_05x_4k_1hz",
                "category": "real_vehicle",
            }
        ),
        encoding="utf-8",
    )
    real_raw = real / "raw_images"
    for name in ("static", "moving", "camera_info"):
        (real_raw / name).mkdir(parents=True)
    (real_raw / "static/cam1.png").write_bytes(b"image")
    (real_raw / "moving/frame.png").write_bytes(b"image")
    (real_raw / "camera_info/cam1.json").write_text("{}")

    entries = discover_prepared_datasets(tmp_path)
    identifiers = [entry.id for entry in entries]
    real_entries = [
        entry for entry in entries if entry.category == "real_vehicle"
    ]

    assert identifiers.count("fov_100deg") == 1
    assert any(
        entry.display_name == "real_05x_4k_1hz"
        for entry in real_entries
    )
    assert all(entry.id != "00_shared_input" for entry in real_entries)


def test_simulation_catalogue_uses_layout_v2_parameters_not_sdf_files(
    tmp_path: Path,
) -> None:
    variants = (
        ("fov_100deg", "fov", "100deg", {"moving_hfov_deg": 100.0}),
        (
            "moving_blur_k21_strong",
            "motion_blur",
            "kernel_21",
            {"motion_blur_kernel": 21},
        ),
        (
            "ceiling_normal",
            "lighting",
            "normal",
            {"lighting": "normal"},
        ),
    )
    for variant, factor, value, parameters in variants:
        root = tmp_path / "results/simulation" / factor / value
        for name in ("static", "moving", "camera_info"):
            (root / "raw_images" / name).mkdir(parents=True, exist_ok=True)
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "layout_version": 2,
                    "id": variant,
                    "category": "simulation",
                    "storage": {"factor": factor, "value": value},
                    "simulation_parameters": parameters,
                }
            ),
            encoding="utf-8",
        )

    entries = discover_simulation_experiments(tmp_path)
    by_variant = {entry.variant: entry for entry in entries}

    assert "route2" in by_variant
    assert "fov_100deg" in by_variant
    assert "moving_blur_k21_strong" in by_variant
    assert "ceiling_normal" in by_variant
    assert entries[0].variant == "route2"
    assert "fov_69deg_baseline" not in by_variant
    assert "moving_blur_k00_baseline" not in by_variant
    assert "moving_res_1280x720_baseline" not in by_variant
    assert by_variant["fov_100deg"].parameters["moving_hfov_deg"] == 100.0
    assert by_variant["moving_blur_k21_strong"].parameters["motion_blur_kernel"] == 21
    assert by_variant["ceiling_normal"].parameters["lighting"] == "normal"
    assert ".sdf" not in format_simulation_parameters(by_variant["route2"].parameters)


def test_exact_baseline_match_prefers_a_local_canonical_dataset(
    tmp_path: Path,
) -> None:
    parameters = dict(BASELINE_SIMULATION_PARAMETERS)
    entries = [
        SimulationExperimentSummary(
            "route2",
            "baseline",
            "result only",
            189,
            True,
            None,
            parameters,
        ),
        SimulationExperimentSummary(
            "route2",
            "baseline",
            "local input",
            189,
            True,
            tmp_path / "input_route2",
            parameters,
        ),
    ]
    match = find_matching_simulation(entries, dict(BASELINE_SIMULATION_PARAMETERS))

    assert match is not None
    assert match.variant == "route2"
    assert match.dataset_root == tmp_path / "input_route2"
    assert match.has_results


def test_capture_timing_is_part_of_exact_simulation_matching(
    tmp_path: Path,
) -> None:
    entries = discover_simulation_experiments(tmp_path)
    parameters = dict(BASELINE_SIMULATION_PARAMETERS)
    parameters["settle_seconds"] = 0.5

    assert find_matching_simulation(entries, parameters) is None
