from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from camera_rig_calibration.methods.ap03.report import run as run_ap03_report
from camera_rig_calibration.policies.submission_quality_policy import (
    _merge_method_status,
    _repair_anchor_exports,
    ap03_quality_semantics,
)


def _metadata(
    *,
    relative_std: float,
    status: str = "OK_FULL",
    missing: list[str] | None = None,
) -> dict:
    missing = [] if missing is None else missing
    return {
        "status": status,
        "scale_m_per_colmap_unit": 0.65,
        "used_rel_std_scale": relative_std,
        "available_static_cameras": ["camera_a", "camera_b"],
        "missing_static_cameras": missing,
    }


def test_ap03_weak_scale_is_completed_but_not_deployable() -> None:
    semantics = ap03_quality_semantics(
        _metadata(relative_std=0.356, status="SCALE_WEAK_CHECK_REQUIRED")
    )
    assert semantics["quality_status"] == "poor_scale_dispersion"
    assert semantics["calibration_status"] == "rejected_by_quality_gate"
    assert semantics["deployment_eligible"] is False
    assert semantics["deployment_eligible_cameras"] == []


def test_ap03_partial_coverage_is_not_deployable_even_with_good_scale() -> None:
    semantics = ap03_quality_semantics(
        _metadata(
            relative_std=0.02,
            status="PARTIAL_2_OF_3",
            missing=["camera_c"],
        )
    )
    assert semantics["quality_status"] == "good"
    assert semantics["calibration_status"] == "partial_coverage"
    assert semantics["deployment_eligible"] is False


def test_ap03_full_good_scale_is_deployable() -> None:
    semantics = ap03_quality_semantics(_metadata(relative_std=0.02))
    assert semantics["quality_status"] == "good"
    assert semantics["calibration_status"] == "available"
    assert semantics["deployment_eligible"] is True
    assert semantics["deployment_eligible_cameras"] == [
        "camera_a",
        "camera_b",
    ]


def test_publication_merge_preserves_partial_ap02_semantics() -> None:
    result = {
        "artifact_status": "available",
        "calibration_status": "available",
    }
    status = {
        "execution_status": "completed",
        "calibration_status": "partial_coverage",
        "quality_status": "warning_high_reprojection",
        "full_rig_result_available": False,
    }
    merged = _merge_method_status(result, status)
    assert merged["artifact_status"] == "available"
    assert merged["calibration_status"] == "partial_coverage"
    assert merged["deployment_eligible"] is False
    assert merged["deployment_eligible_cameras"] == []


def test_anchor_export_becomes_diagnostic_when_calibration_rejected(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "methods/ap03_multi/baseline"
    method_root.mkdir(parents=True)
    (method_root / "RESULT.json").write_text(
        json.dumps(
            {
                "artifact_status": "available",
                "calibration_status": "rejected_by_quality_gate",
                "quality_status": "poor_scale_dispersion",
                "deployment_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "calibration_status": "available",
        "cameras": [
            {
                "camera_id": "camera_a",
                "status": "available",
                "quality_status": "accepted",
                "deployment_eligible": True,
            }
        ],
    }
    (method_root / "camera_extrinsics_anchor.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (method_root / "camera_extrinsics_anchor.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    fields = [
        "camera_id",
        "status",
        "quality_status",
        "deployment_eligible",
    ]
    with (method_root / "camera_extrinsics_anchor.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "camera_id": "camera_a",
                "status": "available",
                "quality_status": "accepted",
                "deployment_eligible": "True",
            }
        )

    _repair_anchor_exports(method_root)

    repaired = json.loads(
        (method_root / "camera_extrinsics_anchor.json").read_text(
            encoding="utf-8"
        )
    )
    assert repaired["calibration_status"] == "rejected_by_quality_gate"
    camera = repaired["cameras"][0]
    assert camera["deployment_eligible"] is False
    assert camera["status"] == "available_diagnostic_only"
    assert camera["quality_status"] == "poor_scale_dispersion"


def test_ap03_report_uses_primary_multi_status_and_quality_gate(
    tmp_path: Path,
) -> None:
    single_root = tmp_path / "scale_single"
    multi_root = tmp_path / "scale_multi"
    inspection_root = tmp_path / "colmap/inspection"
    for root in (single_root, multi_root, inspection_root):
        root.mkdir(parents=True, exist_ok=True)

    (single_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json").write_text(
        json.dumps(_metadata(relative_std=0.02)), encoding="utf-8"
    )
    (multi_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json").write_text(
        json.dumps(
            _metadata(
                relative_std=0.35,
                status="SCALE_WEAK_CHECK_REQUIRED",
            )
        ),
        encoding="utf-8",
    )
    (inspection_root / "AP03_RECONSTRUCTION_DIAGNOSTICS.json").write_text(
        json.dumps(
            {
                "quality_status": "warning_weak_reconstruction_support",
                "registered_static_camera_count": 2,
                "registered_moving_frame_count": 10,
                "sparse_point_count": 100,
            }
        ),
        encoding="utf-8",
    )

    run_ap03_report(output_root=tmp_path)
    status = json.loads((tmp_path / "METHOD_STATUS.json").read_text())
    assert status["status"] == "SCALE_WEAK_CHECK_REQUIRED"
    assert status["success"] is True
    assert status["execution_status"] == "completed"
    assert status["calibration_status"] == "rejected_by_quality_gate"
    assert status["quality_status"] == "poor_scale_dispersion"
    assert status["deployment_eligible"] is False
