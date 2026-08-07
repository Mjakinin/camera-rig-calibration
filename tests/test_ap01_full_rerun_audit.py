from __future__ import annotations

import json
from pathlib import Path


def _evidence_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "parity/main_route2_v1/ap01/full_rerun"
    )


def _read(name: str) -> dict:
    return json.loads((_evidence_root() / name).read_text(encoding="utf-8"))


def test_controlled_ap01_rerun_was_rejected_before_execution() -> None:
    manifest = _read("AP01_FULL_RERUN_MANIFEST.json")
    validation = _read("AP01_FULL_RERUN_VALIDATION.json")
    stages = _read("AP01_FULL_RERUN_STAGE_COUNTS.json")

    assert manifest["status"] == "PRE_RUN_REJECTED"
    assert manifest["command_executed"] is None
    assert manifest["command_execution_count"] == 0
    assert manifest["queue_id"] is None
    assert manifest["run_id"] is None
    assert manifest["ap01_contract_name"] == "main_route2_parity_v1"
    assert validation["checks"]["requested_contract_is_main_route2_parity_v1"]
    assert not validation["checks"][
        "target_dataset_matches_locked_historical_input"
    ]
    assert not validation["checks"]["accepted_observation_count_is_554"]
    assert stages["observations"]["actual"]["accepted_total"] == 512
    assert stages["stage_execution"]["candidate_construction"] == "not_started"
    assert stages["stage_execution"]["publication"] == "not_started"


def test_pre_run_rejection_preserved_all_published_methods() -> None:
    publication = _read("AP01_FULL_RERUN_PUBLICATION.json")

    assert publication["publication_status"] == "NOT_ATTEMPTED"
    assert publication["reconciliation_status"] == "NOT_ATTEMPTED"
    assert publication["old_public_ap01"]["still_present"] is True
    assert publication["old_public_ap01"]["unchanged"] is True
    assert publication["attempts"]["unchanged"] is True
    assert publication["ap02"]["unchanged"] is True
    assert publication["ap03"]["unchanged"] is True
    assert publication["shared_experiment_metadata"]["unchanged"] is True


def test_production_parity_is_honestly_unavailable_without_alignment() -> None:
    parity = _read("AP01_PRODUCTION_PARITY.json")

    assert parity["classification"] == "UNAVAILABLE"
    assert parity["production_result"] is None
    assert parity["alignment_used"] is False
    assert parity["best_fit_alignment_used"] is False
    assert parity["ground_truth_used"] is False
    assert parity["method_execution_invoked"] is False
    assert parity["colmap_invoked"] is False
    assert parity["publication_invoked"] is False
    assert parity["reconciliation_invoked"] is False
