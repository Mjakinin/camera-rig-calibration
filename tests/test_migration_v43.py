from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.migration_v43 import (
    apply_simulation_migration,
    inventory_tree,
    migrate_unpublished_flat_experiments,
    plan_simulation_migration,
)


def test_verified_simulation_migration_splits_inputs_from_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results/simulation/fov_100deg"
    input_root = source / "inputs/input_a/raw_images/moving"
    observation = (
        source
        / "observations/input_a/legacy/shared_all_aruco_observations.csv"
    )
    method = source / "legacy_results/input_a/AP02/result.csv"
    input_root.mkdir(parents=True)
    observation.parent.mkdir(parents=True)
    method.parent.mkdir(parents=True)
    (input_root / "frame.png").write_bytes(b"frame")
    observation.write_text("marker_id\n3\n")
    method.write_text("cost\n1\n")
    (source / "legacy_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experiment_id": "fov_100deg",
                "parameters": {
                    "route": "route2",
                    "moving_width": 1280,
                    "moving_height": 720,
                    "moving_hfov_deg": 100.0,
                    "lighting": "baseline",
                    "lighting_scale": 1.0,
                    "motion_blur_kernel": 0,
                    "motion_blur_angle_deg": 0.0,
                    "target_route_frames": 189,
                    "route_sampling_strategy": "original_route_poses",
                },
            }
        )
    )
    before = inventory_tree(source)

    plan = plan_simulation_migration(tmp_path)
    journal = apply_simulation_migration(tmp_path, plan)

    result = tmp_path / "results/simulation/fov/100deg"
    dataset = tmp_path / "datasets/simulation/fov/100deg"
    assert len(plan) == 1
    assert not source.exists()
    assert (result / "legacy_results/input_a/AP02/result.csv").is_file()
    assert (
        dataset / "inputs/input_a/raw_images/moving/frame.png"
    ).is_file()
    assert (
        dataset
        / "inputs/input_a/observations/legacy/"
        "shared_all_aruco_observations.csv"
    ).is_file()
    assert (result / "PUBLISHED.json").is_file()
    assert (dataset / "PUBLISHED.json").is_file()
    dataset_manifest = json.loads(
        (dataset / "inputs/input_a/dataset_manifest.json").read_text()
    )
    assert dataset_manifest["scene_type"] == "simulation"
    assert dataset_manifest["moving_camera"]["image_count"] == 1
    assert json.loads(journal.read_text())["status"] == "completed"
    assert inventory_tree(result).file_count + inventory_tree(
        dataset
    ).file_count >= before.file_count


def test_unpublished_flat_experiment_moves_to_temporary_queue(
    tmp_path: Path,
) -> None:
    result = tmp_path / "results/real_vehicle/unfinished"
    cache = tmp_path / "datasets/unfinished/hash"
    result.mkdir(parents=True)
    cache.mkdir(parents=True)
    (result / "experiment.yaml").write_text("schema_version: 4\n")
    (result / "frame.png").write_bytes(b"result-frame")
    (cache / "frame.png").write_bytes(b"cache-frame")

    moved = migrate_unpublished_flat_experiments(tmp_path)

    assert len(moved) == 1
    transaction = moved[0]
    assert not result.exists()
    assert not (tmp_path / "datasets/unfinished").exists()
    assert (
        transaction / "legacy_experiment/frame.png"
    ).read_bytes() == b"result-frame"
    assert (
        transaction / "legacy_dataset_cache/hash/frame.png"
    ).read_bytes() == b"cache-frame"
    receipt = json.loads(
        (transaction / "queue_transaction.json").read_text()
    )
    assert receipt["status"] == "incomplete"
