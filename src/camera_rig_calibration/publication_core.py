"""Focused atomic-publication responsibility."""

from __future__ import annotations

import hashlib
import csv
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .anchor_export import export_method_anchor_poses
from .dataset.discovery import safe_id
from .dataset_identity import (
    build_dataset_identity,
    identities_match,
    write_dataset_identity,
)
from .evaluation.reporting import write_scientific_experiment_reports
from .experiments import (
    experiment_manifest_payload,
    experiment_paths,
    method_result_label,
)
from .filesystem import rename_with_retry
from .storage_layout import storage_manifest




METHOD_DIRECTORIES = {
    "ap01": Path("02_AP01"),
    "ap02": Path("03_AP02"),
    "ap03": Path("04_AP03"),
}
PRIMARY_POSES = {
    "ap01": Path(
        "02_AP01/03_static_extrinsics/"
        "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
    ),
    "ap02": Path(
        "03_AP02/07_graph_ba/with_moving/"
        "optimized_static_camera_poses_ref_marker.csv"
    ),
    "ap03": Path(
        "04_AP03/scale_multi/"
        "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_tree(
    source: Path,
    destination: Path,
    *,
    keep_existing: tuple[str, ...] = (),
) -> None:
    """Copy/hardlink a tree and reject any differing existing file."""
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if relative.as_posix() in keep_existing:
                continue
            if (
                target.stat().st_size != item.stat().st_size
                or _sha256(target) != _sha256(item)
            ):
                raise RuntimeError(
                    f"Immutable publication conflict at {target}"
                )
            continue
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


def _materialize_semantic_tree(source: Path, destination: Path) -> None:
    """Publish method internals with descriptive names, not stage numbers."""
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        semantic = Path(
            *(
                re.sub(r"^\d+_", "", part) or part
                for part in relative.parts
            )
        )
        target = destination / semantic
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(
                    f"Semantic diagnostic-name collision at {target}"
                )
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def _rename_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 8,
) -> None:
    """Backward-compatible alias for the shared filesystem helper."""
    rename_with_retry(source, target, attempts=attempts)


def _atomic_replace(incoming: Path, target: Path) -> None:
    backup = target.with_name(
        f".previous_{target.name}_{os.getpid()}_{time.time_ns()}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _rename_with_retry(target, backup)
    try:
        _rename_with_retry(incoming, target)
    except Exception:
        if backup.exists() and not target.exists():
            _rename_with_retry(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_fingerprint(root: Path) -> str:
    descriptor = _read_json(root / "dataset.json")
    value = descriptor.get("input_fingerprint")
    if value:
        return str(value)
    source = _read_json(root / "metadata" / "source.json")
    return str(source.get("input_id", ""))


def _validate_dataset(root: Path) -> None:
    if not (root / "raw_images").is_dir():
        raise RuntimeError(
            f"Prepared transaction has no raw_images directory: {root}"
        )
    required = (
        "shared_static_aruco_observations.csv",
        "shared_moving_aruco_observations.csv",
        "shared_all_aruco_observations.csv",
        "SELECTION_CANDIDATES.json",
        "SELECTION_CANDIDATES.csv",
        "REFERENCE_SELECTIONS.json",
        "REFERENCE_MARKER_ID.txt",
        "PUBLICATION_COMPLETE.json",
    )
    missing = [
        name for name in required if not (root / "observations" / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "The canonical dataset is not publishable because observation "
            "generation is incomplete: " + ", ".join(missing)
        )
    if not (root / "observations" / "quality").is_dir():
        raise RuntimeError(
            "The canonical dataset is not publishable because observation "
            "quality evidence is missing."
        )
    quality_required = (
        "marker_inventory.csv",
        "marker_inventory.json",
    )
    missing_quality = [
        name
        for name in quality_required
        if not (root / "observations" / "quality" / name).is_file()
    ]
    if missing_quality:
        raise RuntimeError(
            "The canonical dataset is not publishable because marker "
            "inventory evidence is incomplete: "
            + ", ".join(missing_quality)
        )
    if not (root / "observations" / "debug_images").is_dir():
        raise RuntimeError(
            "The canonical dataset is not publishable because the ArUco debug "
            "images are missing."
        )


def _refresh_dataset_descriptor(root: Path, config: Any) -> None:
    """Bind a prepared dataset descriptor to the final experiment config."""
    descriptor_path = root / "dataset.json"
    descriptor = _read_json(descriptor_path)
    if descriptor:
        input_id = str(descriptor.get("input_fingerprint", "")).strip()
        if not input_id:
            raise RuntimeError(
                f"Prepared dataset descriptor has no input fingerprint: "
                f"{descriptor_path}"
            )
        descriptor.update(
            experiment_manifest_payload(
                config,
                experiment_paths(config),
                input_id,
                created_at=(
                    str(descriptor["created_at"])
                    if descriptor.get("created_at")
                    else None
                ),
            )
        )
        _write_json(descriptor_path, descriptor)


def _finalize_dataset_front_door(
    canonical: Path, config: Any
) -> None:
    """Replace transaction-only paths with stable canonical metadata."""
    _refresh_dataset_descriptor(canonical, config)
    source_path = canonical / "metadata" / "source.json"
    source = _read_json(source_path)
    if source:
        source.pop("canonical_source_roots", None)
        source["canonical_dataset_root"] = str(canonical.resolve())
        _write_json(source_path, source)

__all__ = [
    'METHOD_DIRECTORIES',
    'PRIMARY_POSES',
    '_now',
    '_write_json',
    '_sha256',
    '_materialize_tree',
    '_materialize_semantic_tree',
    '_rename_with_retry',
    '_atomic_replace',
    '_read_json',
    '_dataset_fingerprint',
    '_validate_dataset',
    '_refresh_dataset_descriptor',
    '_finalize_dataset_front_door',
]
