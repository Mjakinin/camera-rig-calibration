"""Strict no-GT AP01 historical-SfM intermediate validation and materialization."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import AP01MethodContract


@dataclass(frozen=True)
class FrozenAP01Intermediate:
    manifest: Path
    moving_poses: Path
    metric_scale: Path
    payload: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(repository_root: Path, relative: str) -> Path:
    path = (repository_root / relative).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise RuntimeError("Frozen AP01 artifact escapes the repository")
    if not path.is_file():
        raise RuntimeError(f"Frozen AP01 artifact is missing: {path}")
    return path


def validate_frozen_intermediate(
    *,
    dataset: Path,
    moving_camera_id: str,
    contract: AP01MethodContract,
) -> FrozenAP01Intermediate:
    """Fail closed on schema, input, intrinsics, contract, or artifact drift."""

    if contract.sfm_execution_policy != "frozen_historical_reproduction":
        raise RuntimeError("AP01 contract does not authorize frozen SfM reuse")
    manifest_relative = contract.sfm_frozen_intermediate_manifest
    if manifest_relative is None:
        raise RuntimeError("AP01 frozen-SfM manifest is not configured")
    repository = Path(__file__).resolve().parents[4]
    manifest = _repository_path(repository, manifest_relative)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    expected_schema = contract.sfm_frozen_intermediate_schema_version
    checks = {
        "schema_version": payload.get("schema_version") == expected_schema,
        "sfm_mode": payload.get("sfm_mode")
        == "frozen_historical_reproduction",
        "method_contract": payload.get("method_contract") == contract.name,
        "method_contract_sha256": payload.get("method_contract_sha256")
        == contract.scientific_fingerprint(),
        "input_fingerprint": payload.get("input_fingerprint")
        == contract.sfm_frozen_input_fingerprint,
        "ground_truth_used": payload.get("ground_truth_used") is False,
        "recommended_reuse_forbidden": payload.get(
            "recommended_wizard_reuse_allowed"
        )
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Frozen AP01 intermediate manifest mismatch: "
            + ", ".join(failed)
        )

    identity_path = dataset / "metadata" / "dataset_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("Prepared AP01 dataset identity is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    actual_input = str(identity.get("fingerprint", ""))
    if actual_input != contract.sfm_frozen_input_fingerprint:
        raise RuntimeError(
            "Frozen AP01 input fingerprint mismatch: "
            f"expected {contract.sfm_frozen_input_fingerprint}, got {actual_input}"
        )

    intrinsics = (
        dataset
        / "raw_images"
        / "camera_info"
        / f"{moving_camera_id}.json"
    )
    actual_intrinsics = _sha256(intrinsics)
    manifest_intrinsics = payload.get("intrinsics", {})
    expected_intrinsics = contract.sfm_frozen_intrinsics_sha256
    if not (
        expected_intrinsics
        and actual_intrinsics == expected_intrinsics
        and manifest_intrinsics.get("sha256") == expected_intrinsics
        and manifest_intrinsics.get("camera_id") == moving_camera_id
    ):
        raise RuntimeError(
            "Frozen AP01 intrinsics fingerprint mismatch: "
            f"expected {expected_intrinsics}, got {actual_intrinsics}"
        )

    artifacts = payload.get("artifacts", {})
    moving_record = artifacts.get("moving_poses", {})
    scale_record = artifacts.get("metric_scale", {})
    moving_poses = _repository_path(
        repository, str(moving_record.get("relative_path", ""))
    )
    metric_scale = _repository_path(
        repository, str(scale_record.get("relative_path", ""))
    )
    moving_hash = _sha256(moving_poses)
    scale_hash = _sha256(metric_scale)
    if not (
        moving_hash == contract.sfm_frozen_images_sha256
        == moving_record.get("sha256")
    ):
        raise RuntimeError("Frozen AP01 moving-pose artifact hash mismatch")
    if not (
        scale_hash == contract.scale_frozen_metric_sha256
        == scale_record.get("sha256")
    ):
        raise RuntimeError("Frozen AP01 metric-scale artifact hash mismatch")
    return FrozenAP01Intermediate(
        manifest=manifest,
        moving_poses=moving_poses,
        metric_scale=metric_scale,
        payload=payload,
    )


def materialize_frozen_sfm(
    *,
    dataset: Path,
    moving_camera_id: str,
    stage_root: Path,
    contract: AP01MethodContract,
) -> tuple[Path, dict[str, Any]]:
    frozen = validate_frozen_intermediate(
        dataset=dataset,
        moving_camera_id=moving_camera_id,
        contract=contract,
    )
    images = stage_root / "sparse_txt_best" / "images.txt"
    images.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen.moving_poses, images)
    provenance = {
        "schema_version": 1,
        "sfm_mode": "frozen_historical_reproduction",
        "ground_truth_used": False,
        "source_manifest": str(frozen.manifest),
        "source_manifest_contract_sha256": frozen.payload[
            "method_contract_sha256"
        ],
        "input_fingerprint": frozen.payload["input_fingerprint"],
        "intrinsics_sha256": frozen.payload["intrinsics"]["sha256"],
        "moving_poses_sha256": frozen.payload["artifacts"]["moving_poses"][
            "sha256"
        ],
    }
    provenance_path = stage_root / "FROZEN_SFM_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = stage_root / "COLMAP_REPORT.txt"
    report.write_text(
        "AP01 MOVING-CAMERA HISTORICAL REPRODUCTION\n"
        "=" * 72
        + "\n\n"
        "sfm_mode: frozen_historical_reproduction\n"
        "COLMAP invoked: no\n"
        "Ground Truth used: no\n"
        f"Source: {frozen.moving_poses}\n"
        f"Contract SHA-256: {contract.scientific_fingerprint()}\n",
        encoding="utf-8",
    )
    return images, provenance
