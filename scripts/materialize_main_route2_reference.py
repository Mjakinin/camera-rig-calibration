#!/usr/bin/env python3
"""Materialize the historical Main Route-2 baseline as a local prepared dataset.

The historical images already live in this repository's Git history.  This
helper avoids duplicating ~200 binary files on the current branch: it extracts
the exact tracked blobs from the frozen Main reference commit into an input-only
layout-v2 dataset that the current Gazebo wizard can discover under
"existing experiments".

No calibration result is copied.  AP01/AP02/AP03 are intentionally rerun by the
current code against these historical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


HISTORICAL_COMMIT = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
HISTORICAL_DATASET = Path(
    "results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
)
HISTORICAL_WORLD = Path(
    "src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"
)
HISTORICAL_ROUTE = Path(
    "src/calib_lab/bus_real_data/config/moving_camera_route2_interpolated_final.json"
)
DEFAULT_TARGET = Path(
    "results/simulation/reference_inputs/main_route2_reference"
)

STATIC_CAMERAS = (
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
)
MOVING_CAMERA = "moving_calib_camera"
EXPECTED_MOVING_FRAMES = 189
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

BASELINE_PARAMETERS = {
    "route": "route2",
    "moving_width": 1280,
    "moving_height": 720,
    "moving_hfov_deg": 69.1,
    "lighting": "baseline",
    "lighting_scale": 1.0,
    "motion_blur_kernel": 0,
    "motion_blur_angle_deg": 0.0,
    "target_route_frames": 189,
    "route_sampling_strategy": "original_route_poses",
    "settle_seconds": 0.35,
    "post_pose_skip": 5,
    "frame_timeout_seconds": 3.0,
    "startup_timeout_seconds": 60.0,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ensure_historical_commit(repo: Path) -> None:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{HISTORICAL_COMMIT}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Historical Main reference commit is not available locally. Run "
            "`git fetch origin main` and retry."
        )


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Historical archive unexpectedly contains a link: {member.name}"
                )
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(
                    f"Historical archive contains an unsafe path: {member.name}"
                )
        handle.extractall(destination)


def _files(directory: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not directory.is_dir():
        return []
    result = [item for item in sorted(directory.iterdir()) if item.is_file()]
    if suffixes is None:
        return result
    return [item for item in result if item.suffix.lower() in suffixes]


def _validate_raw_images(root: Path) -> dict[str, object]:
    raw = root / "raw_images"
    static = _files(raw / "static", IMAGE_SUFFIXES)
    moving = _files(raw / "moving", IMAGE_SUFFIXES)
    camera_info = _files(raw / "camera_info", {".json", ".yaml", ".yml"})

    static_ids = tuple(sorted(path.stem for path in static))
    expected_static = tuple(sorted(STATIC_CAMERAS))
    if static_ids != expected_static:
        raise RuntimeError(
            "Historical static-camera set mismatch: "
            f"expected {expected_static}, found {static_ids}"
        )
    if len(moving) != EXPECTED_MOVING_FRAMES:
        raise RuntimeError(
            "Historical moving-frame count mismatch: "
            f"expected {EXPECTED_MOVING_FRAMES}, found {len(moving)}"
        )
    expected_info = set(STATIC_CAMERAS) | {MOVING_CAMERA}
    info_ids = {path.stem for path in camera_info}
    if info_ids != expected_info:
        raise RuntimeError(
            "Historical CameraInfo set mismatch: "
            f"expected {sorted(expected_info)}, found {sorted(info_ids)}"
        )

    content_digest = hashlib.sha256()
    all_raw_files = sorted(item for item in raw.rglob("*") if item.is_file())
    for path in all_raw_files:
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        content_digest.update(relative.encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(file_hash.encode("ascii"))
        content_digest.update(b"\0")
        content_digest.update(str(path.stat().st_size).encode("ascii"))
        content_digest.update(b"\n")

    return {
        "static_images": len(static),
        "moving_images": len(moving),
        "camera_info_files": len(camera_info),
        "raw_file_count": len(all_raw_files),
        "raw_images_fingerprint_sha256": content_digest.hexdigest(),
    }


def _materialize(repo: Path, target: Path) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.materialize-",
            dir=target.parent,
        )
    )
    try:
        archive = temporary / "historical_reference.tar"
        extracted = temporary / "_extracted"
        extracted.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive}",
                HISTORICAL_COMMIT,
                (HISTORICAL_DATASET / "raw_images").as_posix(),
                HISTORICAL_WORLD.as_posix(),
                HISTORICAL_ROUTE.as_posix(),
            ],
            cwd=repo,
            check=True,
        )
        _safe_extract_tar(archive, extracted)

        historical_root = extracted / HISTORICAL_DATASET
        source_raw = historical_root / "raw_images"
        if not source_raw.is_dir():
            raise RuntimeError(
                f"Historical raw_images directory was not extracted: {source_raw}"
            )
        shutil.move(str(source_raw), str(temporary / "raw_images"))

        simulation_metadata = temporary / "metadata" / "simulation"
        simulation_metadata.mkdir(parents=True, exist_ok=True)
        source_world = extracted / HISTORICAL_WORLD
        source_route = extracted / HISTORICAL_ROUTE
        if not source_world.is_file() or not source_route.is_file():
            raise RuntimeError(
                "Historical world or Route-2 JSON is missing from the archive"
            )
        shutil.copy2(source_world, simulation_metadata / "world_snapshot.sdf")
        shutil.copy2(source_route, simulation_metadata / "route2_reference.json")

        archive.unlink(missing_ok=True)
        shutil.rmtree(extracted)

        validation = _validate_raw_images(temporary)
        world_hash = _sha256(simulation_metadata / "world_snapshot.sdf")
        route_hash = _sha256(simulation_metadata / "route2_reference.json")

        descriptor = {
            "schema_version": 5,
            "layout_version": 2,
            "id": "main_route2_reference",
            "category": "simulation",
            "scene_type": "simulation",
            "source_kind": "prepared",
            "description": (
                "Historical Main Route-2 raw images materialized byte-for-byte "
                f"from commit {HISTORICAL_COMMIT}. Input only; rerun methods with "
                "the current rigcal implementation."
            ),
            "created_at": _now(),
            "static_cameras": [
                {"id": camera_id, "label": None}
                for camera_id in STATIC_CAMERAS
            ],
            "moving_camera": {"id": MOVING_CAMERA},
            "simulation_parameters": dict(BASELINE_PARAMETERS),
            "storage": {
                "schema_version": 5,
                "layout_version": 2,
                "category": "simulation",
                "factor": "reference_input",
                "value": "main_route2_reference",
                "canonical_id": "main_route2_reference",
                "relative_path": "reference_inputs/main_route2_reference",
                "dataset_root": str(target.resolve()),
                "result_root": str(target.resolve()),
            },
        }
        _write_json(temporary / "dataset.json", descriptor)
        _write_json(
            temporary / "metadata" / "reference_source.json",
            {
                "schema_version": 1,
                "kind": "historical_main_route2_reference",
                "materialized_at": _now(),
                "source_commit": HISTORICAL_COMMIT,
                "source_dataset_path": HISTORICAL_DATASET.as_posix(),
                "source_world_path": HISTORICAL_WORLD.as_posix(),
                "source_route_path": HISTORICAL_ROUTE.as_posix(),
                "world_snapshot_sha256": world_hash,
                "route_sha256": route_hash,
                **validation,
            },
        )
        (temporary / "README.txt").write_text(
            "HISTORICAL MAIN ROUTE-2 REFERENCE INPUT\n"
            "=======================================\n\n"
            f"Source commit: {HISTORICAL_COMMIT}\n"
            f"Source dataset: {HISTORICAL_DATASET.as_posix()}\n\n"
            "This directory contains input only. It exists to run the current "
            "AP01/AP02/AP03 implementation against the historical Main images "
            "without changing branches or duplicating those binary blobs in "
            "the current branch.\n",
            encoding="utf-8",
        )

        temporary.rename(target)
        return {
            **validation,
            "world_snapshot_sha256": world_hash,
            "route_sha256": route_hash,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _already_materialized(target: Path) -> dict[str, object] | None:
    source = _read_json(target / "metadata" / "reference_source.json")
    if source.get("source_commit") != HISTORICAL_COMMIT:
        return None
    validation = _validate_raw_images(target)
    expected_fingerprint = source.get("raw_images_fingerprint_sha256")
    if expected_fingerprint != validation["raw_images_fingerprint_sha256"]:
        raise RuntimeError(
            "Existing historical reference input no longer matches its stored "
            "raw-image fingerprint. Refusing to silently repair or overwrite it."
        )
    return {**source, **validation}


def _exclude_local_reference(repo: Path, target: Path) -> None:
    try:
        relative = target.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return
    exclude = repo / ".git" / "info" / "exclude"
    if not exclude.parent.is_dir():
        return
    pattern = f"/{relative}/"
    existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    if pattern not in {line.strip() for line in existing.splitlines()}:
        with exclude.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(pattern + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the frozen historical Main Route-2 input as a local "
            "prepared simulation dataset discoverable by rigcal."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Target dataset directory (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()

    repo = _repo_root()
    target = args.target
    if not target.is_absolute():
        target = (repo / target).resolve()
    _ensure_historical_commit(repo)

    if target.exists():
        existing = _already_materialized(target)
        if existing is None:
            raise RuntimeError(
                f"Target already exists and is not this frozen reference: {target}"
            )
        result = existing
        status = "already materialized"
    else:
        result = _materialize(repo, target)
        status = "materialized"

    _exclude_local_reference(repo, target)

    print(f"[OK] Historical Main Route-2 reference {status}")
    print(f"[OK] target: {target}")
    print(
        "[OK] files: "
        f"{result['static_images']} static + "
        f"{result['moving_images']} moving + "
        f"{result['camera_info_files']} CameraInfo"
    )
    print(
        "[OK] raw fingerprint: "
        f"{result['raw_images_fingerprint_sha256']}"
    )
    print(
        "[OK] world snapshot SHA-256: "
        f"{result['world_snapshot_sha256']}"
    )
    print()
    print("Next: run `rigcal`, choose Gazebo simulation, then")
    print("      'add existing experiments' -> main_route2_reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
