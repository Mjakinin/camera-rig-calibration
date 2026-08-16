"""Publication semantics for experiment-level common evaluations.

Calibration methods and evaluation job artifacts remain immutable.  The small
SELECTED_COMMON_EVALUATION.json front-door file is intentionally mutable: it is
an index pointing at the currently selected immutable evaluation job.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .core import _materialize_tree, _read_json, _sha256, _write_json


_SELECTED = "SELECTED_COMMON_EVALUATION.json"
_UNAVAILABLE = "COMMON_EVALUATION_UNAVAILABLE.json"
_POINTER_FILES = {_SELECTED, _UNAVAILABLE}


def _job_fingerprint(root: Path) -> str:
    status = _read_json(root / "COMMON_ANCHOR_STATUS.json")
    return str(status.get("evaluation_job_fingerprint", "")).strip()


def _job_target(source: Path, destination: Path) -> Path:
    fingerprint = _job_fingerprint(source)
    if not fingerprint:
        return destination / source.name
    return destination / f"{source.name}__job_{fingerprint[:12]}"


def _materialize_root_file(source: Path, target: Path) -> None:
    """Publish a non-pointer evaluation root file with immutable semantics."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if (
            target.stat().st_size != source.stat().st_size
            or _sha256(target) != _sha256(source)
        ):
            raise RuntimeError(f"Immutable publication conflict at {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def publish_evaluation_tree(source: Path, destination: Path) -> None:
    """Publish immutable evaluation jobs and atomically refresh their pointer.

    Queue-level common evaluation depends on both the evaluation configuration
    and the exact method-result set.  The queue runner records that complete
    identity as ``evaluation_job_fingerprint``.  Different jobs therefore get
    different canonical directories even when they use the same anchor and
    evaluation settings.

    A repeated publication of the *same* job fingerprint reuses the already
    published immutable job.  This intentionally ignores runtime/log-only
    differences from a repeated evaluator invocation.
    """
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)

    published_by_fingerprint: dict[str, Path] = {}

    for item in sorted(source.iterdir(), key=lambda path: path.name):
        if not item.is_dir():
            continue
        fingerprint = _job_fingerprint(item)
        target = _job_target(item, destination)
        if target.is_dir() and fingerprint:
            existing_fingerprint = _job_fingerprint(target)
            if existing_fingerprint == fingerprint:
                published_by_fingerprint[fingerprint] = target
                continue
        _materialize_tree(item, target)
        if fingerprint:
            published_by_fingerprint[fingerprint] = target

    for item in sorted(source.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.name in _POINTER_FILES:
            continue
        _materialize_root_file(item, destination / item.name)

    selected_source = source / _SELECTED
    unavailable_source = source / _UNAVAILABLE
    if selected_source.is_file():
        selected = _read_json(selected_source)
        fingerprint = str(
            selected.get("evaluation_job_fingerprint", "")
        ).strip()
        target = published_by_fingerprint.get(fingerprint)
        if target is not None:
            # If this exact job was already present, expose the immutable
            # canonical status rather than new runtime-only metadata.
            canonical_status = _read_json(
                target / "COMMON_ANCHOR_STATUS.json"
            )
            if canonical_status:
                selected = canonical_status
            selected["output"] = str(target.resolve())
            selected["evaluation_directory"] = target.name
        _write_json(destination / _SELECTED, selected)
        (destination / _UNAVAILABLE).unlink(missing_ok=True)
    elif unavailable_source.is_file():
        _write_json(
            destination / _UNAVAILABLE,
            _read_json(unavailable_source),
        )
        (destination / _SELECTED).unlink(missing_ok=True)


__all__ = ["publish_evaluation_tree"]
