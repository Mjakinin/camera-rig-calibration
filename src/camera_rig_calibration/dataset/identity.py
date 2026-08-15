"""Immutable, method-independent identity for one published dataset.

The identity deliberately contains only acquisition evidence.  Method,
selection, optimizer, COLMAP, evaluation, reporting and visualization
settings belong to their own fingerprints and must never split a shared
queue dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CONTRACT_VERSION = "rigcal_dataset_identity_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path, relatives: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(relatives), key=lambda item: item.as_posix()):
        path = root / relative
        if not path.is_file():
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _manifest_file_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        checksum = str(item.get("sha256", "")).strip().lower()
        if role and len(checksum) == 64:
            result[role] = checksum
    return dict(sorted(result.items()))


def build_dataset_identity(root: Path) -> dict[str, Any]:
    """Hash the authoritative acquisition content below ``root``.

    Derived ground truth, observations and method output are intentionally
    excluded.  The SDF snapshot and commanded route are acquisition inputs,
    while regenerated ``ground_truth.json`` is post-method evaluation data.
    """

    dataset_root = root.resolve()
    manifest = _read_json(
        dataset_root / "metadata" / "dataset_manifest.json"
    )
    raw = dataset_root / "raw_images"
    raw_relatives = (
        path.relative_to(dataset_root)
        for path in raw.rglob("*")
        if path.is_file()
    )
    acquisition_relatives = [
        Path("metadata/simulation/world_snapshot.sdf"),
        Path("metadata/simulation/route_commanded.csv"),
        Path("metadata/simulation/capture_metadata.json"),
        Path("metadata/simulation_capture.json"),
    ]
    content_files = _files(
        dataset_root, [*raw_relatives, *acquisition_relatives]
    )
    static_cameras = sorted(
        str(item.get("id"))
        for item in manifest.get("static_cameras", [])
        if isinstance(item, dict) and item.get("id")
    )
    moving = manifest.get("moving_camera", {})
    moving_camera = (
        str(moving.get("id"))
        if isinstance(moving, dict) and moving.get("id")
        else None
    )
    scientific_contract = {
        "contract_version": CONTRACT_VERSION,
        "scene_type": str(manifest.get("scene_type") or ""),
        "static_camera_ids": static_cameras,
        "moving_camera_id": moving_camera,
        "marker_dictionary": manifest.get("marker_dictionary"),
        "marker_length_m": manifest.get("marker_length_m"),
        "sampling_hz": manifest.get("sampling_hz"),
        "capture_parameters": manifest.get(
            "simulation_parameters"
        ),
        "manifest_source_hashes": _manifest_file_hashes(manifest),
        "content_files": content_files,
    }
    serialized = json.dumps(
        scientific_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        **scientific_contract,
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(content_files),
    }


def identities_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(first.get("fingerprint")) and (
        first.get("fingerprint") == second.get("fingerprint")
    )


def write_dataset_identity(root: Path) -> Path:
    destination = root.resolve() / "metadata" / "dataset_identity.json"
    payload = build_dataset_identity(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
