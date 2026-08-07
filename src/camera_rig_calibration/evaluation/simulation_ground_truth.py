from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..anchor_export.geometry import validate_transform
from ..methods.common.sdf_utils import (
    parse_world_poses,
    sdf_marker_model_to_opencv_frame,
    sdf_model_pose_to_optical,
)


GROUND_TRUTH_CONTRACT = "rigcal_simulation_ground_truth_v2"
GROUND_TRUTH_PARSER_VERSION = 2
CAMERA_CONVENTION = (
    "T_world_camera_optical maps OpenCV camera coordinates to world"
)
MARKER_CONVENTION = (
    "T_world_marker_opencv maps OpenCV ArUco coordinates to world"
)


@dataclass(frozen=True)
class GroundTruthResolution:
    payload: dict[str, Any]
    regenerated: bool
    cache_reason: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    text = _canonical_json(payload)
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_id(entity_name: str) -> int | None:
    if entity_name == "aruco_ref_floor_14":
        return 14
    match = re.fullmatch(r"marker_(\d{3})", entity_name)
    return int(match.group(1)) if match else None


def _expected_cameras(dataset: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(item["id"] if isinstance(item, dict) else item)
            for item in dataset.get("static_cameras", [])
            if (
                isinstance(item, str)
                and item.strip()
                or isinstance(item, dict)
                and item.get("id")
            )
        }
    )


def _matrix_valid(value: Any) -> bool:
    try:
        validate_transform(value)
    except (TypeError, ValueError):
        try:
            import numpy as np

            validate_transform(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError):
            return False
    return True


def _fingerprint(
    *,
    world_sha256: str,
    expected_camera_ids: list[str],
    marker_ids: list[int],
) -> str:
    contract = {
        "contract": GROUND_TRUTH_CONTRACT,
        "parser_version": GROUND_TRUTH_PARSER_VERSION,
        "world_snapshot_sha256": world_sha256,
        "expected_static_camera_ids": expected_camera_ids,
        "available_marker_ids": marker_ids,
        "camera_transform_convention": CAMERA_CONVENTION,
        "marker_transform_convention": MARKER_CONVENTION,
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _cache_reason(
    existing: dict[str, Any],
    *,
    expected_camera_ids: list[str],
    marker_ids: list[int],
    world_sha256: str,
    fingerprint: str,
) -> str | None:
    if existing.get("status") != "available":
        return "cache_status_not_available"
    cameras = existing.get("static_cameras")
    if not isinstance(cameras, dict) or not cameras:
        return "cache_static_cameras_missing_or_empty"
    if set(cameras) != set(expected_camera_ids):
        return "cache_static_camera_set_mismatch"
    if any(not _matrix_valid(value) for value in cameras.values()):
        return "cache_static_camera_transform_invalid"
    markers = existing.get("markers")
    if not isinstance(markers, dict):
        return "cache_markers_missing"
    cached_marker_ids: set[int] = set()
    for marker, value in markers.items():
        try:
            marker_id = int(marker)
        except (TypeError, ValueError):
            return "cache_marker_id_invalid"
        if (
            not isinstance(value, dict)
            or not _matrix_valid(value.get("T_world_marker_opencv"))
        ):
            return "cache_marker_transform_invalid"
        cached_marker_ids.add(marker_id)
    if cached_marker_ids != set(marker_ids):
        return "cache_marker_set_mismatch"
    if existing.get("world_sha256") != world_sha256:
        return "cache_world_hash_mismatch"
    if existing.get("ground_truth_fingerprint") != fingerprint:
        return "cache_fingerprint_mismatch"
    if existing.get("contract") != GROUND_TRUTH_CONTRACT:
        return "cache_contract_mismatch"
    if existing.get("parser_version") != GROUND_TRUTH_PARSER_VERSION:
        return "cache_parser_version_mismatch"
    if existing.get("camera_transform_convention") != CAMERA_CONVENTION:
        return "cache_camera_convention_mismatch"
    if existing.get("marker_transform_convention") != MARKER_CONVENTION:
        return "cache_marker_convention_mismatch"
    return None


def _unavailable(
    *,
    reason: str,
    expected: list[str],
    resolved: list[str] | None = None,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    resolved_ids = sorted(resolved or [])
    return {
        "schema_version": 5,
        "layout_version": 2,
        "contract": GROUND_TRUTH_CONTRACT,
        "parser_version": GROUND_TRUTH_PARSER_VERSION,
        "status": "unavailable",
        "reason": reason,
        "expected_static_cameras": expected,
        "resolved_static_cameras": resolved_ids,
        "missing_static_cameras": sorted(set(expected) - set(resolved_ids)),
        "world_snapshot": snapshot.name if snapshot is not None else None,
    }


def resolve_simulation_ground_truth(
    dataset_root: Path,
    *,
    world_path: Path | None = None,
    backfilled: bool = False,
) -> GroundTruthResolution:
    """Resolve authoritative post-hoc GT without consulting mutable source SDFs."""
    dataset_root = dataset_root.resolve()
    metadata = dataset_root / "metadata" / "simulation"
    destination = metadata / "ground_truth.json"
    snapshot = metadata / "world_snapshot.sdf"
    metadata.mkdir(parents=True, exist_ok=True)

    # Freeze acquisition geometry before relying on dataset.json. During a fresh
    # Gazebo capture the composed world already exists, but dataset.json is only
    # written after capture/finalization. The previous ordering returned early
    # on the missing descriptor and therefore silently lost the authoritative
    # world snapshot.
    if not snapshot.is_file():
        snapshot_source: Path | None = None
        if world_path is not None and world_path.is_file():
            snapshot_source = world_path
        else:
            # Existing layout-v2 experiments may predate the snapshot fix. The
            # composed world below metadata/simulation/generated is already part
            # of the published immutable experiment and is therefore a safe
            # repair source. Never fall back to a mutable source-tree SDF here.
            published_composed_world = metadata / "generated" / "composed_world.sdf"
            if published_composed_world.is_file():
                snapshot_source = published_composed_world
        if snapshot_source is not None:
            shutil.copy2(snapshot_source, snapshot)

    dataset = _read_json(dataset_root / "dataset.json")
    expected = _expected_cameras(dataset)
    if not expected:
        payload = _unavailable(
            reason="dataset.json declares no static cameras",
            expected=expected,
            snapshot=snapshot if snapshot.is_file() else None,
        )
        _write_json_if_changed(destination, payload)
        return GroundTruthResolution(
            payload, True, "dataset_static_camera_list_empty"
        )

    if not snapshot.is_file():
        payload = _unavailable(
            reason=(
                "metadata/simulation/world_snapshot.sdf is missing and no "
                "published immutable composed-world snapshot is available; "
                "a published experiment never falls back to a mutable "
                "source-tree SDF"
            ),
            expected=expected,
        )
        _write_json_if_changed(destination, payload)
        return GroundTruthResolution(
            payload, True, "authoritative_snapshot_missing"
        )

    try:
        poses = parse_world_poses(snapshot)
    except Exception as exc:  # XML/pose parsing errors are evidence, not crashes.
        payload = _unavailable(
            reason=f"world_snapshot.sdf could not be parsed: {exc}",
            expected=expected,
            snapshot=snapshot,
        )
        _write_json_if_changed(destination, payload)
        return GroundTruthResolution(payload, True, "snapshot_parse_failed")

    marker_entities = sorted(
        (
            marker,
            entity_name,
            pose,
        )
        for entity_name, pose in poses.items()
        if (marker := _marker_id(entity_name)) is not None
    )
    marker_ids = [marker for marker, _, _ in marker_entities]
    world_sha256 = _sha256(snapshot)
    fingerprint = _fingerprint(
        world_sha256=world_sha256,
        expected_camera_ids=expected,
        marker_ids=marker_ids,
    )
    existing = _read_json(destination)
    invalid_reason = _cache_reason(
        existing,
        expected_camera_ids=expected,
        marker_ids=marker_ids,
        world_sha256=world_sha256,
        fingerprint=fingerprint,
    )
    if invalid_reason is None:
        return GroundTruthResolution(existing, False, "valid_cache_hit")

    cameras: dict[str, list[list[float]]] = {}
    invalid_transforms: list[str] = []
    for camera in expected:
        pose = poses.get(camera)
        if pose is None:
            continue
        try:
            transform = validate_transform(
                sdf_model_pose_to_optical(pose["T_W_model"])
            )
        except (KeyError, TypeError, ValueError):
            invalid_transforms.append(camera)
            continue
        cameras[camera] = transform.tolist()
    if set(cameras) != set(expected) or invalid_transforms:
        reason = (
            "The authoritative world snapshot does not resolve every declared "
            "static camera to a valid SE(3) optical pose."
        )
        if invalid_transforms:
            reason += " Invalid transforms: " + ", ".join(invalid_transforms)
        payload = _unavailable(
            reason=reason,
            expected=expected,
            resolved=sorted(cameras),
            snapshot=snapshot,
        )
        payload.update(
            {
                "world_sha256": world_sha256,
                "ground_truth_fingerprint": fingerprint,
            }
        )
        _write_json_if_changed(destination, payload)
        return GroundTruthResolution(payload, True, invalid_reason)

    markers: dict[str, Any] = {}
    for marker, entity_name, pose in marker_entities:
        try:
            transform = validate_transform(
                sdf_marker_model_to_opencv_frame(pose["T_W_model"])
            )
        except (KeyError, TypeError, ValueError):
            payload = _unavailable(
                reason=(
                    f"Marker {marker} in world_snapshot.sdf has an invalid "
                    "SE(3) transform."
                ),
                expected=expected,
                resolved=sorted(cameras),
                snapshot=snapshot,
            )
            _write_json_if_changed(destination, payload)
            return GroundTruthResolution(
                payload, True, "snapshot_marker_transform_invalid"
            )
        markers[str(marker)] = {
            "entity_name": entity_name,
            "T_world_marker_opencv": transform.tolist(),
        }

    payload = {
        "schema_version": 5,
        "layout_version": 2,
        "contract": GROUND_TRUTH_CONTRACT,
        "parser_version": GROUND_TRUTH_PARSER_VERSION,
        "status": "available",
        "generated_at": _now(),
        "snapshot_origin": (
            "backfilled_from_published_world_snapshot"
            if backfilled
            else "captured_before_calibration"
        ),
        "world_snapshot": "world_snapshot.sdf",
        "world_sha256": world_sha256,
        "ground_truth_fingerprint": fingerprint,
        "expected_static_cameras": expected,
        "resolved_static_cameras": sorted(cameras),
        "missing_static_cameras": [],
        "available_marker_ids": marker_ids,
        "camera_transform_convention": CAMERA_CONVENTION,
        "marker_transform_convention": MARKER_CONVENTION,
        "static_cameras": cameras,
        "markers": markers,
    }
    _write_json_if_changed(destination, payload)
    return GroundTruthResolution(payload, True, invalid_reason)


def ensure_simulation_ground_truth(
    dataset_root: Path,
    *,
    world_path: Path | None = None,
    backfilled: bool = False,
) -> dict[str, Any]:
    return resolve_simulation_ground_truth(
        dataset_root,
        world_path=world_path,
        backfilled=backfilled,
    ).payload
