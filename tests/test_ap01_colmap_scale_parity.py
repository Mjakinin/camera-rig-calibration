from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "parity/main_route2_v1/ap01/colmap_scale_parity"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "ap01_colmap_scale_parity_audit", EVIDENCE / "audit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(name: str):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_umeyama_recovers_known_similarity() -> None:
    audit = load_audit_module()
    source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.5 * (source @ rotation.T) + np.asarray([4.0, -3.0, 2.0])

    scale, recovered_rotation, translation = audit.umeyama(source, target)

    assert np.isclose(scale, 2.5)
    assert np.allclose(recovered_rotation, rotation)
    assert np.allclose(translation, [4.0, -3.0, 2.0])


def test_robust_scale_rules_are_isolated_on_identical_samples() -> None:
    audit = load_audit_module()
    ratios = [1.0] * 20 + [5.0]

    legacy = audit.robust_statistics(ratios, wizard=False)
    wizard = audit.robust_statistics(ratios, wizard=True)

    assert legacy["final_median"] == wizard["final_median"] == 1.0
    assert legacy["raw_count"] == wizard["raw_count"] == 21


def test_required_evidence_set_exists() -> None:
    required = {
        "COLMAP_CONFIGURATION_PARITY.json",
        "COLMAP_REGISTRATION_PARITY.json",
        "COLMAP_REGISTRATION_DIFF.csv",
        "RAW_TRAJECTORY_PARITY.json",
        "RAW_TRAJECTORY_DIFF.csv",
        "FEATURE_MATCH_PARITY.json",
        "FEATURE_MATCH_DIFF.csv",
        "SCALE_PARITY.json",
        "SCALE_OBSERVATION_DIFF.csv",
        "FIRST_CAUSAL_DIVERGENCE.json",
        "COLMAP_SCALE_PARITY_REPORT.txt",
    }
    assert required <= {path.name for path in EVIDENCE.iterdir()}


def test_preserved_colmap_evidence_identifies_first_divergence() -> None:
    configuration = evidence("COLMAP_CONFIGURATION_PARITY.json")
    registration = evidence("COLMAP_REGISTRATION_PARITY.json")
    features = evidence("FEATURE_MATCH_PARITY.json")

    assert configuration["overall_classification"] == "DIFFERENT_CONFIGURATION"
    assert registration["registered_images"] == {
        "legacy": 175,
        "wizard": 189,
        "common": 175,
    }
    assert registration["legacy_unregistered"] == [
        f"frame_{frame:04d}.png" for frame in range(175, 189)
    ]
    assert features["count_parity"]["legacy_total_features"] == 151_125
    assert features["byte_parity"]["keypoint_blobs_equal"] == 189
    assert features["first_observed_numeric_divergence"]["image"] == "frame_0003.png"
    assert features["semantic_graph_parity"]["matches"]["same_pair_inventory"] is True
    assert features["semantic_graph_parity"]["matches"]["different_records"] == 5_887


def test_trajectory_and_scale_differences_are_not_gauge_only() -> None:
    trajectory = evidence("RAW_TRAJECTORY_PARITY.json")
    scale = evidence("SCALE_PARITY.json")

    assert trajectory["classification"] == "DIFFERENT_REGISTERED_IMAGES"
    sim3 = trajectory["diagnostic_similarity_legacy_to_wizard"]
    assert sim3["common_image_count"] == 175
    assert np.isclose(sim3["scale_wizard_units_per_legacy_unit"], 0.9362444155871841)
    assert np.isclose(sim3["translation_rmse_wizard_units"], 0.006476286738542124)
    assert scale["legacy"]["raw_samples"] == 1_869
    assert scale["wizard"]["raw_samples"] == 4_425
    assert scale["inventory_comparison"]["common_samples"] == 1_718
    assert scale["cause_assessment"]["A_different_raw_gauge_only"] is False
    assert scale["gauge_only_check"]["relative_residual"] > 0.017


def test_artifact_hashes_are_sensitive_to_legacy_and_fresh_inputs() -> None:
    configuration = evidence("COLMAP_CONFIGURATION_PARITY.json")
    hashes = configuration["evidence"]
    registration = evidence("COLMAP_REGISTRATION_PARITY.json")

    assert hashes["legacy_database_sha256"] != hashes["wizard_database_sha256"]
    assert (
        registration["evidence"]["legacy_images_sha256"]
        != registration["evidence"]["wizard_images_sha256"]
    )
