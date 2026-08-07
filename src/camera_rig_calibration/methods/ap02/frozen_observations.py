"""Fail-closed access to the no-GT historical AP02 observation stream."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FROZEN_AP02_MANIFEST = (
    "parity/main_route2_v1/frozen/"
    "AP02_FROZEN_OBSERVATIONS_CONTRACT.json"
)


@dataclass(frozen=True)
class FrozenAP02Observations:
    manifest: Path
    observations: Path
    payload: dict[str, Any]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(repository: Path, relative: str) -> Path:
    path = (repository / relative).resolve()
    if not path.is_relative_to(repository.resolve()):
        raise RuntimeError("Frozen AP02 artifact escapes the repository")
    if not path.is_file():
        raise RuntimeError(f"Frozen AP02 artifact is missing: {path}")
    return path


def _ordering_schema(
    path: Path,
) -> tuple[list[str], int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = list(reader)
    payload = {
        "columns": columns,
        "row_keys": [
            [
                row.get("observer_type", ""),
                row.get("observer_id", ""),
                row.get("frame_id", ""),
                row.get("marker_id", ""),
            ]
            for row in rows
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return columns, len(rows), hashlib.sha256(encoded.encode()).hexdigest()


def _intrinsics_hashes(dataset: Path, camera_ids: list[str]) -> dict[str, str]:
    root = dataset / "raw_images" / "camera_info"
    return {
        camera_id: _sha256(root / f"{camera_id}.json")
        for camera_id in camera_ids
    }


def _aggregate_hash(payload: dict[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_frozen_ap02_observations(
    *,
    dataset: Path,
    historical_reproduction: bool,
    method_contract_name: str,
    method_contract_sha256: str,
    reference_marker_policy: str,
    reference_marker_id: int,
    root_pose_policy: str,
    manifest_path: Path | None = None,
    repository_root: Path | None = None,
) -> FrozenAP02Observations:
    """Validate every frozen-input boundary and fail closed on any drift."""

    if not historical_reproduction:
        raise RuntimeError(
            "Frozen AP02 observations require explicit historical reproduction"
        )
    repository = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[4]
    )
    manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else _repository_path(repository, FROZEN_AP02_MANIFEST)
    )
    if not manifest.is_file():
        raise RuntimeError(f"Frozen AP02 manifest is missing: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    method = payload.get("method_contract", {})
    reference = payload.get("reference_marker_semantics", {})
    checks = {
        "schema_version": payload.get("schema_version") == 1,
        "reproduction_mode": payload.get("reproduction_mode")
        == "historical_ap02_observations_v1",
        "ground_truth_used": payload.get("ground_truth_used") is False,
        "ground_truth_values_included": payload.get(
            "ground_truth_values_included"
        )
        is False,
        "method_contract_name": method.get("name")
        == method_contract_name
        == "baseline_v1",
        "method_contract_sha256": method.get("scientific_sha256")
        == method_contract_sha256,
        "reference_marker_policy": reference.get("selection_policy")
        == reference_marker_policy,
        "reference_marker_id": reference.get("reference_marker_id")
        == reference_marker_id,
        "root_pose_policy": reference.get("root_pose_policy")
        == root_pose_policy,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Frozen AP02 manifest mismatch: " + ", ".join(failed)
        )

    identity_path = dataset / "metadata" / "dataset_identity.json"
    descriptor_path = dataset / "dataset.json"
    if not identity_path.is_file() or not descriptor_path.is_file():
        raise RuntimeError("Prepared AP02 dataset identity is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    expected_input = payload.get("acquisition_input", {})
    input_checks = {
        "dataset_identity_fingerprint": identity.get("fingerprint")
        == expected_input.get("dataset_identity_fingerprint"),
        "input_fingerprint": descriptor.get("input_fingerprint")
        == expected_input.get("input_fingerprint"),
        "experiment_fingerprint": descriptor.get("experiment_fingerprint")
        == expected_input.get("experiment_fingerprint"),
    }
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Frozen AP02 acquisition/input fingerprint mismatch: "
            + ", ".join(failed)
        )

    expected_intrinsics = payload.get("intrinsics", {})
    expected_files = expected_intrinsics.get("files", {})
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError("Frozen AP02 intrinsics contract is empty")
    try:
        actual_files = _intrinsics_hashes(dataset, sorted(expected_files))
    except FileNotFoundError as exc:
        raise RuntimeError("Frozen AP02 intrinsics file is missing") from exc
    if actual_files != expected_files or _aggregate_hash(actual_files) != (
        expected_intrinsics.get("aggregate_sha256")
    ):
        raise RuntimeError("Frozen AP02 intrinsics fingerprint mismatch")

    artifact_record = payload.get("artifact", {})
    artifact = _repository_path(
        repository, str(artifact_record.get("relative_path", ""))
    )
    if (
        _sha256(artifact) != artifact_record.get("sha256")
        or artifact.stat().st_size != artifact_record.get("size_bytes")
    ):
        raise RuntimeError("Frozen AP02 observation artifact hash mismatch")
    columns, row_count, ordering_hash = _ordering_schema(artifact)
    forbidden = [
        field
        for field in columns
        if "ground_truth" in field.lower() or field.lower().startswith("gt_")
    ]
    if forbidden:
        raise RuntimeError(
            "Frozen AP02 artifact contains Ground Truth fields: "
            + ", ".join(forbidden)
        )
    if (
        columns != artifact_record.get("columns")
        or row_count != artifact_record.get("row_count")
        or ordering_hash != artifact_record.get("ordering_schema_sha256")
        or artifact_record.get("ordering_schema_algorithm")
        != "sha256_canonical_json_columns_and_ordered_observer_keys_v1"
    ):
        raise RuntimeError("Frozen AP02 ordering/schema hash mismatch")

    provenance = {
        "schema_version": 1,
        "observation_input_mode": "frozen_historical_ap02_reproduction",
        "historical_reproduction": True,
        "ground_truth_used": False,
        "source_manifest": str(manifest),
        "source_artifact": str(artifact),
        "source_artifact_sha256": artifact_record["sha256"],
        "source_observation_count": row_count,
        "ordering_schema_sha256": ordering_hash,
        "dataset_identity_fingerprint": identity["fingerprint"],
        "input_fingerprint": descriptor["input_fingerprint"],
        "intrinsics_sha256": expected_intrinsics["aggregate_sha256"],
        "method_contract": method_contract_name,
        "method_contract_sha256": method_contract_sha256,
        "reference_marker_id": reference_marker_id,
        "validation_status": "passed",
    }
    return FrozenAP02Observations(
        manifest=manifest,
        observations=artifact,
        payload=payload,
        provenance=provenance,
    )


def resolve_ap02_observation_input(
    *,
    observations_root: Path,
    dataset: Path | None,
    historical_reproduction: bool,
    method_contract_name: str,
    method_contract_sha256: str,
    reference_marker_policy: str,
    reference_marker_id: int,
    root_pose_policy: str,
) -> tuple[Path, FrozenAP02Observations | None]:
    """Keep normal baseline on fresh observations; gate frozen input explicitly."""

    normal = observations_root / "shared_all_aruco_observations.csv"
    if not historical_reproduction:
        return normal, None
    if dataset is None:
        raise RuntimeError(
            "Historical AP02 reproduction requires the prepared dataset root"
        )
    frozen = validate_frozen_ap02_observations(
        dataset=dataset,
        historical_reproduction=True,
        method_contract_name=method_contract_name,
        method_contract_sha256=method_contract_sha256,
        reference_marker_policy=reference_marker_policy,
        reference_marker_id=reference_marker_id,
        root_pose_policy=root_pose_policy,
    )
    return frozen.observations, frozen
