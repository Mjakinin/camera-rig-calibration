from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE = (
    REPOSITORY
    / "parity/main_route2_v1/ap01/historical_prepared_run"
)


def _read(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_historical_prepared_input_identity_is_exact() -> None:
    prepared = _read("HISTORICAL_PREPARED_EXPERIMENT.json")
    identity = _read("HISTORICAL_PREPARED_INPUT_PARITY.json")

    assert prepared["status"] == "READY_FOR_AP01"
    assert prepared["input_parity"] == "EXACT_HASH_MATCH"
    assert identity["classification"] == "EXACT_HASH_MATCH"
    assert identity["locked_file_count"] == identity["matched_file_count"]
    assert identity["mismatch_count"] == 0
    assert identity["prepared_input_fingerprint"] == (
        prepared["prepared_input_fingerprint"]
    )


def test_historical_prepared_observations_reproduce_lock() -> None:
    observations = _read("HISTORICAL_PREPARED_OBSERVATION_PARITY.json")

    assert observations["classification"] == "EXACT"
    assert observations["prepared_counts"] == observations["locked_counts"]
    assert observations["row_keys_exact"] is True
    assert observations["original_order_exact"] is True
    assert observations["semantic_fields_exact"] is True
    assert observations["difference_count"] == 0


def test_production_attempt_resolved_parity_contract_and_locked_selection() -> None:
    production = _read("AP01_PRODUCTION_RUN.json")
    locked_candidates = json.loads(
        (
            REPOSITORY
            / "parity/main_route2_v1/ap01/post_fix/AP01_CANDIDATE_PARITY.json"
        ).read_text(encoding="utf-8")
    )
    locked_poses = json.loads(
        (
            REPOSITORY
            / "parity/main_route2_v1/ap01/final_pose/wizard/AP01_FINAL_CAMERA_POSES.json"
        ).read_text(encoding="utf-8")
    )
    locked_selection = {
        row["camera_id"]: row["source_selected_candidate_type"]
        for row in locked_poses["camera_records"]
    }

    assert production["enabled_methods"] == ["ap01"]
    assert production["method_contract"] == "main_route2_parity_v1"
    assert production["observation_counts"] == {
        "total": 554,
        "static": 30,
        "moving": 524,
    }
    assert production["candidate_count_including_root_gauge"] == (
        locked_candidates["candidate_multiplicity"]["wizard_count"]
    )
    assert production["candidate_breakdown"] == (
        locked_candidates["candidate_counts"]["wizard"]
    )
    assert production["selection"] == locked_selection


def test_failed_attempt_isolated_and_not_published_or_reconciled() -> None:
    parity = _read("AP01_PRODUCTION_PARITY.json")
    publication = _read("AP01_PRODUCTION_PUBLICATION.json")

    assert parity["classification"] == "UNAVAILABLE"
    assert parity["partial_scientific_classification"] == (
        "DIFFERENT_FINAL_POSES"
    )
    assert parity["publication_succeeded"] is False
    assert parity["reconciliation_performed"] is False
    assert publication["status"] == "NOT_PUBLISHED_METHOD_FAILED"
    assert publication["authoritative_method_result_present"] is False
    assert publication["failed_attempt_preserved"] is True
    assert "route2_main_parity_v1" in publication["attempt"]
    assert "route2_cpu_ref14_50x50" not in publication["attempt"]


def test_protected_current_experiment_is_byte_unchanged() -> None:
    protected = _read("CURRENT_EXPERIMENT_AFTER.json")

    assert protected["status"] == "UNCHANGED"
    assert protected["unchanged"] is True
    assert protected["before_tree_sha256"] == protected["after_tree_sha256"]
    assert protected["before_file_count"] == protected["after_file_count"]


def test_ap01_end_to_end_frozen_reproduction_is_published_and_reconciled() -> None:
    status = json.loads(
        (
            REPOSITORY
            / "parity/main_route2_v1/ap01/AP01_END_TO_END_STATUS.json"
        ).read_text(encoding="utf-8")
    )
    published = (
        REPOSITORY
        / "results/simulation/baseline/route2_main_parity_v1/methods/ap01/baseline"
    )

    assert status["status"] == (
        "END_TO_END_EXACT_WITH_FROZEN_HISTORICAL_SFM"
    )
    assert status["fresh_production"]["classification"] == (
        "FRESH_COLMAP_EXACT_REPRODUCTION_NOT_PORTABLE"
    )
    assert status["fresh_production"]["published"] is False
    assert status["frozen_production"]["published"] is True
    assert status["frozen_production"]["reconciled"] is True
    assert status["frozen_production"]["published_camera_count"] == 4
    assert status["frozen_production"]["published_pairwise_count"] == 6
    validation_path = (
        published
        / "diagnostics/method/reproduction_validation/"
        "AP01_REPRODUCTION_VALIDATION.json"
    )
    if not validation_path.is_file():
        # The full historical result tree is intentionally local/ignored. The
        # compact, tracked status above remains portable evidence; validate the
        # detailed publication only when that local result is materialized.
        assert status["published_result"] == (
            "results/simulation/baseline/route2_main_parity_v1/"
            "methods/ap01/baseline"
        )
        return
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "END_TO_END_EQUIVALENT"
    assert validation["checks"] == {
        "registered_images": True,
        "total_moving_images": True,
        "metric_scale": True,
        "candidate_counts": True,
        "selections": True,
        "camera_inventory": True,
        "final_poses": True,
        "locked_reference_no_ground_truth": True,
    }
    assert validation["ground_truth_used"] is False
