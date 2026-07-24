from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from camera_rig_calibration.gallery import build_moving_debug_gallery
from camera_rig_calibration.storage import (
    build_cleanup_plan,
    build_data_local_cleanup_plan,
    execute_cleanup,
)


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((900, 1600, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["observer_type", "observer_id", "frame_id", "marker_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_gallery_contains_every_moving_frame_and_connectivity_summary(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    observations = tmp_path / "observations"
    for index in range(3):
        _image(
            dataset / "raw_images/moving" / f"frame_{index:06d}.png",
            50 + index,
        )
        _image(
            observations
            / "debug_images/moving"
            / f"frame_{index:06d}_detections.png",
            80 + index,
        )
    _write_rows(
        observations / "shared_static_aruco_observations.csv",
        [
            {
                "observer_type": "static",
                "observer_id": "cam_a",
                "frame_id": "static",
                "marker_id": 3,
            },
            {
                "observer_type": "static",
                "observer_id": "cam_b",
                "frame_id": "static",
                "marker_id": 4,
            },
        ],
    )
    _write_rows(
        observations / "shared_moving_aruco_observations.csv",
        [
            {
                "observer_type": "moving",
                "observer_id": "moving_frame_000001",
                "frame_id": 1,
                "marker_id": 3,
            },
            {
                "observer_type": "moving",
                "observer_id": "moving_frame_000001",
                "frame_id": 1,
                "marker_id": 4,
            },
        ],
    )

    summary = build_moving_debug_gallery(
        dataset_root=dataset,
        observations_root=observations,
    )

    previews = sorted((observations / "debug_gallery").glob("*.jpg"))
    assert len(previews) == 3
    assert summary["total_moving_frames"] == 3
    assert summary["frames_without_markers"] == 2
    assert summary["frames_with_multiple_markers"] == 1
    assert summary["ap02_bridge_frames"] == 1
    for preview in previews:
        image = cv2.imread(str(preview))
        assert image is not None
        assert max(image.shape[:2]) <= 1280
    connectivity = json.loads(
        (observations / "connectivity_report.json").read_text()
    )
    assert connectivity["gallery_path"] == str(
        (observations / "debug_gallery").resolve()
    )


def test_cleanup_is_hardlink_aware_and_keeps_scientific_results(
    tmp_path: Path,
) -> None:
    repository = tmp_path
    cache = repository / "datasets/cache_a/raw_images/moving"
    cache.mkdir(parents=True)
    source = cache / "frame_000000.png"
    source.write_bytes(b"x" * 4096)
    experiment = repository / "results/simulation/paper"
    result_raw = experiment / "datasets/input_a/raw_images/moving"
    result_raw.mkdir(parents=True)
    os.link(source, result_raw / source.name)
    observations = (
        experiment / "datasets/input_a/observations/detection_a"
    )
    (observations / "debug_gallery").mkdir(parents=True)
    (observations / "debug_gallery/frame.jpg").write_bytes(b"gallery")
    (observations / "debug_images").mkdir()
    (observations / "debug_images/frame.png").write_bytes(b"debug")
    (observations / "connectivity_report.json").write_text("{}")
    method = experiment / "methods/ap01/baseline/current"
    final = method / "99_FINAL_RESULTS"
    numeric = method / "02_AP01/01_moving_colmap/sparse/0"
    final.mkdir(parents=True)
    numeric.mkdir(parents=True)
    report = final / "SUMMARY.txt"
    report.write_text("scientific result\n")
    numeric_result = numeric / "cameras.bin"
    numeric_result.write_bytes(b"numeric")
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    numeric_hash = hashlib.sha256(numeric_result.read_bytes()).hexdigest()
    local = repository / "data_local/capture.mov"
    local.parent.mkdir()
    local.write_bytes(b"local")

    plan = build_cleanup_plan(repository)

    assert plan.reclaimable_bytes >= 4096
    assert not any(
        target.kind == "user data_local input" for target in plan.targets
    )
    execute_cleanup(plan)
    assert report.is_file()
    assert hashlib.sha256(report.read_bytes()).hexdigest() == report_hash
    assert numeric_result.is_file()
    assert hashlib.sha256(numeric_result.read_bytes()).hexdigest() == numeric_hash
    assert (observations / "connectivity_report.json").is_file()
    assert not (observations / "debug_gallery").exists()
    assert not (observations / "debug_images").exists()
    assert not source.exists()
    assert not (result_raw / source.name).exists()
    assert local.is_file()
    removed = json.loads((experiment / "INPUT_REMOVED.json").read_text())
    assert removed["rerunnable"] is False
    assert removed["results_preserved"] is True

    local_plan = build_data_local_cleanup_plan(repository)
    assert len(local_plan.targets) == 1
    execute_cleanup(local_plan)
    assert not local.exists()


def test_cleanup_protects_resumable_run_inputs(tmp_path: Path) -> None:
    experiment = tmp_path / "results/real_vehicle/incomplete"
    run = experiment / ".staging/run_a"
    raw = experiment / "datasets/input_a/raw_images/moving"
    run.mkdir(parents=True)
    raw.mkdir(parents=True)
    (raw / "frame.png").write_bytes(b"keep")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "failed_preflight",
                "experiment_root": str(experiment),
            }
        )
    )

    plan = build_cleanup_plan(tmp_path)

    assert not any(
        target.experiment_root == experiment for target in plan.targets
    )
    assert raw.is_dir()


def test_cleanup_protects_prepared_input_of_unfinished_queue(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "results/simulation/queued"
    preparation = (
        experiment / "datasets/input_a/preparations/preflight"
    )
    dataset = tmp_path / "datasets/queued/cache"
    (preparation / "00_INPUT").mkdir(parents=True)
    dataset.mkdir(parents=True)
    (dataset / "frame.png").write_bytes(b"resume")
    (preparation / "00_INPUT/dataset_pointer.json").write_text(
        json.dumps({"dataset_root": str(dataset)})
    )
    state = tmp_path / "workspace/queues/queued.state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "queue_id": "queued",
                "entries": {
                    "ap01": {"status": "blocked_by_queue_preflight"},
                    "ap02": {"status": "failed_preflight"},
                },
                "preflight_preparation": str(preparation),
            }
        )
    )

    plan = build_cleanup_plan(tmp_path)

    assert experiment.resolve() in plan.protected_paths
    assert dataset.resolve() in plan.protected_paths
    assert not any(
        target.experiment_root == experiment for target in plan.targets
    )
    assert not any(
        target.path == dataset.resolve() for target in plan.targets
    )


def test_cleanup_unlinks_working_image_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_images"
    external.mkdir()
    kept = external / "frame.png"
    kept.write_bytes(b"outside")
    link = (
        tmp_path
        / "results/simulation/paper/methods/ap03/variant/colmap/images"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    plan = build_cleanup_plan(tmp_path)
    target = next(
        item for item in plan.targets if item.path == link.absolute()
    )
    execute_cleanup(
        type(plan)(
            targets=(target,),
            protected_paths=(),
            file_count=0,
            logical_bytes=0,
            reclaimable_bytes=0,
        )
    )

    assert not link.exists()
    assert kept.is_file()
