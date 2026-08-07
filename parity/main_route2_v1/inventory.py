"""Deterministic input inventory and opaque SHA-256 hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


GROUND_TRUTH_NAMES = {
    "ground_truth.json",
    "anchor_camera_gt.csv",
    "pairwise_gt.csv",
}


def assert_pre_solver_path(path: Path) -> None:
    """Reject semantically named GT artifacts in pre-solver operations."""

    lowered = {part.lower() for part in path.parts}
    if lowered & GROUND_TRUTH_NAMES or "ground_truth" in lowered:
        raise PermissionError(
            f"pre-solver parity is not allowed to read Ground Truth: {path}"
        )


def sha256_file(
    path: Path, *, read_bytes: Callable[[Path], bytes] | None = None
) -> str:
    assert_pre_solver_path(path)
    reader = read_bytes or (lambda item: item.read_bytes())
    return hashlib.sha256(reader(path)).hexdigest()


def _paths(root: Path, pattern: str) -> Iterable[Path]:
    return (path for path in root.glob(pattern) if path.is_file())


def _classified_paths(root: Path) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for category, pattern in (
        ("raw_static_image", "raw_images/static/*"),
        ("raw_moving_image", "raw_images/moving/*"),
        ("camera_info", "raw_images/camera_info/*"),
        ("route_metadata", "metadata/simulation/route_commanded.csv"),
        ("route_metadata", "metadata/simulation/moving_route_capture/route_commanded.csv"),
        ("world_metadata_opaque_hash_only", "metadata/simulation/world_snapshot.sdf"),
        ("simulation_metadata", "metadata/simulation/capture_metadata.json"),
        ("simulation_metadata", "metadata/simulation_capture.json"),
        ("simulation_metadata", "metadata/simulation/*.log"),
        ("simulation_metadata", "metadata/simulation/moving_route_capture/README.txt"),
        ("dataset_descriptor", "dataset.json"),
        ("dataset_descriptor", "metadata/dataset_manifest.json"),
        ("dataset_descriptor", "metadata/dataset_identity.json"),
        ("dataset_descriptor", "metadata/source.json"),
        ("dataset_descriptor", "metadata/preparation.json"),
        ("observation_file", "observations/**/*"),
    ):
        selected.extend((category, path) for path in _paths(root, pattern))
    # A path is emitted once even if future patterns overlap.
    unique = {(path.resolve(), category): (category, path) for category, path in selected}
    return sorted(
        unique.values(),
        key=lambda item: (item[1].relative_to(root).as_posix(), item[0]),
    )


def build_file_inventory(
    dataset_root: Path,
    *,
    read_bytes: Callable[[Path], bytes] | None = None,
) -> list[dict[str, Any]]:
    """Inventory only calibration inputs and observation evidence, never GT."""

    root = dataset_root.resolve()
    rows: list[dict[str, Any]] = []
    for category, path in _classified_paths(root):
        assert_pre_solver_path(path)
        rows.append(
            {
                "dataset_side": "wizard_current",
                "category": category,
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path, read_bytes=read_bytes),
            }
        )
    return rows


def inventory_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    normalized = [
        {
            "dataset_side": row.get("dataset_side"),
            "category": row["category"],
            "path": row["path"],
            "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def category_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))

