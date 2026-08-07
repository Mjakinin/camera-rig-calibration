"""Read-only verification of the detached historical Route-2 worktree."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .evidence import write_json
from .inventory import assert_pre_solver_path


EXPECTED_MAIN_COMMIT = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
HISTORICAL_RAW_RELATIVE = Path(
    "results/bus_real_data/ablation/world/route/route2/raw_images"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hashes(inventory_csv: Path) -> dict[str, str]:
    assert_pre_solver_path(inventory_csv)
    with inventory_csv.open(newline="", encoding="utf-8") as handle:
        return {
            row["path"]: row["sha256"]
            for row in csv.DictReader(handle)
            if row.get("dataset_side") == "main_historical"
        }


def _content_status(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return "lfs_pointer"
    if path.suffix.lower() == ".png":
        return "valid_png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "invalid_png"
    if path.suffix.lower() == ".json":
        try:
            json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "invalid_json"
        return "valid_json"
    if path.suffix.lower() == ".csv":
        try:
            rows = list(csv.reader(data.decode("utf-8-sig").splitlines()))
        except (UnicodeDecodeError, csv.Error):
            return "invalid_csv"
        return "valid_csv" if rows and rows[0] else "invalid_csv"
    return "ordinary_file"


def verify_historical_materialization(
    *,
    worktree: Path,
    inventory_csv: Path,
    output: Path,
) -> dict[str, Any]:
    root = worktree.resolve()
    dataset = (root / HISTORICAL_RAW_RELATIVE).resolve()
    expected = _expected_hashes(inventory_csv.resolve())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    worktree_status = subprocess.check_output(
        ["git", "-c", "core.longpaths=true", "status", "--short"],
        cwd=root,
        text=True,
    ).splitlines()

    files = sorted(path for path in dataset.rglob("*") if path.is_file())
    actual: dict[str, str] = {}
    content_status: dict[str, str] = {}
    for path in files:
        assert_pre_solver_path(path)
        relative = path.relative_to(dataset).as_posix()
        actual[relative] = _sha256(path)
        content_status[relative] = _content_status(path)

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path] != actual[path]
    )
    invalid_content = sorted(
        path
        for path, status in content_status.items()
        if status.startswith("invalid") or status == "lfs_pointer"
    )
    route_metadata = sorted(
        path for path in actual if path.lower().endswith(".csv")
    )
    verified = all(
        (
            commit == EXPECTED_MAIN_COMMIT,
            not worktree_status,
            not missing,
            not unexpected,
            not mismatches,
            not invalid_content,
        )
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified" if verified else "mismatch",
        "materialization_status": (
            "EXACT_HASH_MATCH" if verified else "VERIFICATION_FAILED"
        ),
        "worktree_path": str(root),
        "dataset_root": str(dataset),
        "commit_sha": commit,
        "expected_commit_sha": EXPECTED_MAIN_COMMIT,
        "worktree_clean": not worktree_status,
        "worktree_status": worktree_status,
        "counts": {
            "moving_images": len(list((dataset / "moving").glob("*.png"))),
            "static_images": len(list((dataset / "static").glob("*.png"))),
            "camera_info_files": len(
                list((dataset / "camera_info").glob("*.json"))
            ),
            "route_metadata_files": len(route_metadata),
            "total_files": len(actual),
        },
        "route_metadata_paths": route_metadata,
        "hash_status": "all_match" if not mismatches else "mismatch",
        "expected_inventory_file_count": len(expected),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "hash_mismatches": mismatches,
        "lfs_pointer_files": sorted(
            path for path, status in content_status.items() if status == "lfs_pointer"
        ),
        "invalid_content_files": invalid_content,
        "content_status_counts": {
            status: sum(value == status for value in content_status.values())
            for status in sorted(set(content_status.values()))
        },
        "ground_truth_used": False,
        "historical_worktree_modified": False,
    }
    write_json(output.resolve(), payload)
    return payload
