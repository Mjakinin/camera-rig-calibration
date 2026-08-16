from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.publication_services.evaluation import (
    publish_evaluation_tree,
)


def _write_job(
    root: Path,
    *,
    fingerprint: str,
    runtime_seconds: float,
    payload: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison").mkdir()
    status = {
        "anchor_marker_id": 14,
        "success_for_every_method": True,
        "evaluation_job_fingerprint": fingerprint,
        "runtime_seconds": runtime_seconds,
        "method_statuses": {"AP02__baseline": "OK"},
        "output": str(root),
    }
    (root / "COMMON_ANCHOR_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "comparison" / "summary.txt").write_text(
        payload,
        encoding="utf-8",
    )


def _write_selected(source: Path, job: Path) -> None:
    status = json.loads(
        (job / "COMMON_ANCHOR_STATUS.json").read_text(encoding="utf-8")
    )
    (source / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )


def test_different_common_evaluation_jobs_coexist_and_pointer_moves(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published" / "evaluations"
    base_name = "anchor_marker_14_a6b5a4d3"

    first_source = tmp_path / "first"
    first_job = first_source / base_name
    first_fp = "a" * 64
    _write_job(
        first_job,
        fingerprint=first_fp,
        runtime_seconds=10.0,
        payload="ap02-only",
    )
    _write_selected(first_source, first_job)
    publish_evaluation_tree(first_source, destination)

    first_target = destination / f"{base_name}__job_{first_fp[:12]}"
    assert first_target.is_dir()
    first_selected = json.loads(
        (destination / "SELECTED_COMMON_EVALUATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_selected["evaluation_job_fingerprint"] == first_fp
    assert first_selected["evaluation_directory"] == first_target.name
    assert Path(first_selected["output"]) == first_target.resolve()

    second_source = tmp_path / "second"
    second_job = second_source / base_name
    second_fp = "b" * 64
    _write_job(
        second_job,
        fingerprint=second_fp,
        runtime_seconds=20.0,
        payload="ap01-plus-ap02",
    )
    _write_selected(second_source, second_job)
    publish_evaluation_tree(second_source, destination)

    second_target = destination / f"{base_name}__job_{second_fp[:12]}"
    assert first_target.is_dir()
    assert second_target.is_dir()
    assert (first_target / "comparison" / "summary.txt").read_text(
        encoding="utf-8"
    ) == "ap02-only"
    assert (second_target / "comparison" / "summary.txt").read_text(
        encoding="utf-8"
    ) == "ap01-plus-ap02"
    second_selected = json.loads(
        (destination / "SELECTED_COMMON_EVALUATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_selected["evaluation_job_fingerprint"] == second_fp
    assert second_selected["evaluation_directory"] == second_target.name
    assert Path(second_selected["output"]) == second_target.resolve()


def test_repeated_same_evaluation_job_reuses_immutable_publication(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published" / "evaluations"
    base_name = "anchor_marker_14_a6b5a4d3"
    fingerprint = "c" * 64

    first_source = tmp_path / "first"
    first_job = first_source / base_name
    _write_job(
        first_job,
        fingerprint=fingerprint,
        runtime_seconds=12.0,
        payload="original immutable evaluation",
    )
    _write_selected(first_source, first_job)
    publish_evaluation_tree(first_source, destination)

    repeated_source = tmp_path / "repeat"
    repeated_job = repeated_source / base_name
    _write_job(
        repeated_job,
        fingerprint=fingerprint,
        runtime_seconds=99.0,
        payload="runtime-only rerun must not replace canonical job",
    )
    _write_selected(repeated_source, repeated_job)
    publish_evaluation_tree(repeated_source, destination)

    target = destination / f"{base_name}__job_{fingerprint[:12]}"
    assert (target / "comparison" / "summary.txt").read_text(
        encoding="utf-8"
    ) == "original immutable evaluation"
    selected = json.loads(
        (destination / "SELECTED_COMMON_EVALUATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["runtime_seconds"] == 12.0
    assert selected["evaluation_job_fingerprint"] == fingerprint
    assert Path(selected["output"]) == target.resolve()
