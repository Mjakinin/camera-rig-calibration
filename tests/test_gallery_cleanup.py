from __future__ import annotations

import hashlib
import os
from pathlib import Path

from camera_rig_calibration.storage import (
    build_cleanup_plan,
    build_data_local_cleanup_plan,
    execute_cleanup,
)

def test_cleanup_is_hardlink_aware_and_keeps_layout_v2_scientific_data(
    tmp_path: Path,
) -> None:
    repository = tmp_path
    cache = repository / "workspace/cache/colmap"
    cache.mkdir(parents=True)
    source = cache / "frame_000000.png"
    source.write_bytes(b"x" * 4096)
    dataset = repository / "results/simulation/baseline/route2"
    result_raw = dataset / "raw_images/moving"
    result_raw.mkdir(parents=True)
    os.link(source, result_raw / source.name)
    observations = dataset / "observations"
    (observations / "debug_images").mkdir(parents=True)
    (observations / "debug_images/frame.png").write_bytes(b"debug")
    (observations / "connectivity_report.json").write_text("{}")
    method = dataset / "methods/ap01/baseline"
    numeric = method / "diagnostics/method/moving_colmap/sparse/0"
    method.mkdir(parents=True)
    numeric.mkdir(parents=True)
    report = method / "RESULT.txt"
    report.write_text("scientific result\n")
    numeric_result = numeric / "cameras.bin"
    numeric_result.write_bytes(b"numeric")
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    numeric_hash = hashlib.sha256(numeric_result.read_bytes()).hexdigest()
    local = repository / "data_local/capture.mov"
    local.parent.mkdir()
    local.write_bytes(b"local")

    plan = build_cleanup_plan(repository)

    assert plan.logical_bytes >= 4096
    assert not any(
        target.kind == "user data_local input" for target in plan.targets
    )
    execute_cleanup(plan)
    assert report.is_file()
    assert hashlib.sha256(report.read_bytes()).hexdigest() == report_hash
    assert numeric_result.is_file()
    assert hashlib.sha256(numeric_result.read_bytes()).hexdigest() == numeric_hash
    assert (observations / "connectivity_report.json").is_file()
    assert (observations / "debug_images").is_dir()
    assert not source.exists()
    assert (result_raw / source.name).is_file()
    assert local.is_file()

    local_plan = build_data_local_cleanup_plan(repository)
    assert len(local_plan.targets) == 1
    execute_cleanup(local_plan)
    assert not local.exists()


def test_cleanup_protects_temporary_runs_and_canonical_data(
    tmp_path: Path,
) -> None:
    run = tmp_path / "workspace/temporary_runs/queue_a/jobs/ap02"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}")
    raw = tmp_path / "results/real_vehicle/1Hz/paper/raw_images/moving"
    raw.mkdir(parents=True)
    (raw / "frame.png").write_bytes(b"keep")

    plan = build_cleanup_plan(tmp_path)

    assert (tmp_path / "workspace/temporary_runs").resolve() in plan.protected_paths
    assert not plan.targets
    assert raw.is_dir()


def test_cleanup_unlinks_cache_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_images"
    external.mkdir()
    kept = external / "frame.png"
    kept.write_bytes(b"outside")
    link = (
        tmp_path
        / "workspace/cache/colmap/images"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    plan = build_cleanup_plan(tmp_path)
    target = next(
        item
        for item in plan.targets
        if item.path == (tmp_path / "workspace/cache").absolute()
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
