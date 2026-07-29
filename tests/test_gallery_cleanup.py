from __future__ import annotations

import io
from pathlib import Path

import pytest
import typer
from rich.console import Console

import camera_rig_calibration.wizard as wizard_module
from camera_rig_calibration.storage import (
    CleanupPlan,
    CleanupTarget,
    build_dataset_cleanup_plan,
    build_results_cleanup_plan,
    build_temporary_cleanup_plan,
    execute_cleanup,
)


def _write(path: Path, value: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_results_cleanup_removes_every_published_result_but_not_data_local(
    tmp_path: Path,
) -> None:
    result = _write(
        tmp_path
        / "results/real_vehicle/3Hz/test/methods/ap02/baseline/RESULT.txt"
    )
    _write(
        tmp_path
        / "results/simulation/baseline/route2/raw_images/moving/frame.png"
    )
    local = _write(tmp_path / "data_local/capture.mov")
    intrinsic = _write(
        tmp_path / "config/intrinsics/phone/hash/intrinsics.json"
    )

    plan = build_results_cleanup_plan(tmp_path)
    outcome = execute_cleanup(plan)

    assert outcome["verified_removed"] is True
    assert not result.exists()
    assert not any((tmp_path / "results").iterdir())
    assert local.is_file()
    assert intrinsic.is_file()
    assert all("data_local" not in str(target.path) for target in plan.targets)


def test_dataset_cleanup_is_separate_from_temporary_workspace_cleanup(
    tmp_path: Path,
) -> None:
    legacy = _write(tmp_path / "datasets/real_vehicle/test/dataset.json")
    prepared = _write(
        tmp_path / "workspace/preparation_cache/test/hash/dataset.json"
    )
    queue = _write(tmp_path / "workspace/queues/test.yaml")
    local = _write(tmp_path / "data_local/capture.mov")
    result = _write(
        tmp_path / "results/real_vehicle/3Hz/test/RESULTS.txt"
    )

    dataset_plan = build_dataset_cleanup_plan(tmp_path)
    execute_cleanup(dataset_plan)

    assert not legacy.exists()
    assert not prepared.exists()
    assert queue.is_file()
    assert result.is_file()
    assert local.is_file()

    temporary_plan = build_temporary_cleanup_plan(tmp_path)
    execute_cleanup(temporary_plan)

    assert not queue.exists()
    assert result.is_file()
    assert local.is_file()


def test_temporary_cleanup_removes_runs_queues_batches_and_caches(
    tmp_path: Path,
) -> None:
    temporary = _write(
        tmp_path / "workspace/temporary_runs/run/jobs/ap02/log.txt"
    )
    queue = _write(tmp_path / "workspace/experiment/queue/queue.yaml")
    batch = _write(tmp_path / "workspace/batches/batch/batch.yaml")
    cache = _write(tmp_path / "workspace/cache/colmap/database.db")
    prepared = _write(
        tmp_path / "workspace/preparation_cache/test/dataset.json"
    )
    local = _write(tmp_path / "data_local/capture.mov")

    plan = build_temporary_cleanup_plan(tmp_path)
    execute_cleanup(plan)

    assert not temporary.exists()
    assert not queue.exists()
    assert not batch.exists()
    assert not cache.exists()
    assert prepared.is_file()
    assert local.is_file()


def test_cleanup_unlinks_workspace_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_images"
    kept = _write(external / "frame.png", b"outside")
    link = tmp_path / "workspace/cache/colmap/images"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    execute_cleanup(build_temporary_cleanup_plan(tmp_path))

    assert not link.exists()
    assert kept.is_file()


def test_execute_cleanup_refuses_target_outside_reviewed_scope(
    tmp_path: Path,
) -> None:
    local = _write(tmp_path / "data_local/capture.mov")
    plan = CleanupPlan(
        targets=(CleanupTarget(local, "forged"),),
        protected_paths=(),
        scope_roots=(tmp_path / "workspace",),
        file_count=1,
        logical_bytes=local.stat().st_size,
        reclaimable_bytes=local.stat().st_size,
    )

    with pytest.raises(RuntimeError, match="outside its reviewed"):
        execute_cleanup(plan)
    assert local.is_file()


def test_cleanup_wizard_confirms_groups_then_deletes_without_data_local_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(
        tmp_path
        / "results/real_vehicle/3Hz/test/methods/ap02/baseline/RESULT.txt"
    )
    _write(tmp_path / "datasets/legacy/dataset.json")
    _write(tmp_path / "workspace/preparation_cache/test/dataset.json")
    _write(tmp_path / "workspace/temporary_runs/stale/log.txt")
    local = _write(tmp_path / "data_local/capture.mov")
    prompts: list[str] = []

    def confirm(prompt: str, *, default: bool = False) -> bool:
        prompts.append(prompt)
        return True

    def prompt(
        text: str,
        *,
        default: str = "",
        show_default: bool = False,
    ) -> str:
        prompts.append(text)
        return "DELETE"

    monkeypatch.setattr(typer, "confirm", confirm)
    monkeypatch.setattr(typer, "prompt", prompt)
    console = Console(file=io.StringIO(), force_terminal=False)

    wizard_module.cleanup_storage_wizard(tmp_path, console)

    assert not any((tmp_path / "results").iterdir())
    assert not any((tmp_path / "datasets").iterdir())
    assert not any((tmp_path / "workspace").iterdir())
    assert local.is_file()
    assert all("data_local" not in value for value in prompts)
    assert prompts[:3] == [
        (
            "Select all published results, including embedded raw datasets, "
            "for permanent deletion?"
        ),
        (
            "Select all legacy/prepared datasets and dataset caches for "
            "permanent deletion?"
        ),
        (
            "Select all temporary runs, queues, batches and workspace caches "
            "for permanent deletion?"
        ),
    ]


def test_cleanup_wizard_refuses_while_another_run_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _write(
        tmp_path
        / "workspace/temporary_runs/active/jobs/ap02/run_manifest.json",
        b"{}",
    )
    result = _write(tmp_path / "results/real_vehicle/3Hz/test/RESULTS.txt")
    prompted = False

    def fail_confirm(*args, **kwargs) -> bool:
        nonlocal prompted
        prompted = True
        return True

    monkeypatch.setattr(
        wizard_module,
        "_run_process_is_active",
        lambda path: path.name == "active",
    )
    monkeypatch.setattr(typer, "confirm", fail_confirm)

    wizard_module.cleanup_storage_wizard(
        tmp_path,
        Console(file=io.StringIO(), force_terminal=False),
    )

    assert active.is_file()
    assert result.is_file()
    assert prompted is False
