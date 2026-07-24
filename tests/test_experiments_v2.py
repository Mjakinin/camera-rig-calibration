from __future__ import annotations

import pytest

from camera_rig_calibration.config.models import MethodSettings
from camera_rig_calibration.experiments import (
    colmap_artifact_fingerprint,
    first_invalidated_stage,
    method_fingerprint,
    method_variant_name,
    result_category,
    experiment_paths,
    write_experiment_manifest,
)
from camera_rig_calibration.observations import ResolvedSelections


def _resolved(reference: int = 7) -> ResolvedSelections:
    return ResolvedSelections(
        root_camera="front-left",
        ap02_reference_marker_id=reference,
        ap03_single_scale_marker_id=9,
        ap03_multi_marker_ids=(7, 9),
        evaluation_anchor_marker_id=None,
        marker_ids=(7, 9),
        payload={},
    )


def test_method_variant_names_expose_scientific_anchors(prepared_config) -> None:
    ap01 = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap01"])}, deep=True
    )
    ap02 = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap02"])}, deep=True
    )

    assert method_variant_name(ap01, "ap01", _resolved()).startswith(
        "root_front-left__matcher_exhaustive_"
    )
    assert method_variant_name(ap02, "ap02", _resolved()).startswith(
        "ref_marker_7__baseline_"
    )
    assert method_fingerprint(ap02, "ap02", _resolved(7)) != method_fingerprint(
        ap02, "ap02", _resolved(9)
    )
    ap03 = prepared_config.model_copy(
        update={
            "methods": MethodSettings(
                enabled=["ap03"],
                ap03={"multi": {"marker_ids": [7, 9]}},
            )
        },
        deep=True,
    )
    assert method_variant_name(
        ap03, "ap03", _resolved()
    ).startswith("single_marker_9__multi_7-9__matcher_exhaustive_")


def test_stage_invalidation_keeps_evaluation_changes_cheap() -> None:
    assert first_invalidated_stage(["evaluation.anchor_marker_id"]) == "evaluation"
    assert (
        first_invalidated_stage(["methods.ap03.single.scale_marker_id"])
        == "method_estimation"
    )
    assert first_invalidated_stage(["simulation.moving_hfov_deg"]) == "capture_import"


def test_colmap_artifact_identity_excludes_downstream_selections(
    prepared_config,
) -> None:
    ap01_first = prepared_config.model_copy(
        update={
            "methods": MethodSettings(
                enabled=["ap01"],
                ap01={"root_camera": "front-left"},
            )
        },
        deep=True,
    )
    ap01_second = ap01_first.model_copy(
        update={
            "methods": ap01_first.methods.model_copy(
                update={
                    "ap01": ap01_first.methods.ap01.model_copy(
                        update={"root_camera": "roof.camera"}
                    )
                }
            )
        },
        deep=True,
    )
    assert colmap_artifact_fingerprint(
        ap01_first, "ap01", "input_123"
    ) == colmap_artifact_fingerprint(
        ap01_second, "ap01", "input_123"
    )

    ap03_first = prepared_config.model_copy(
        update={
            "methods": MethodSettings(
                enabled=["ap03"],
                ap03={
                    "single": {"scale_marker_id": 7},
                    "multi": {"marker_ids": [7, 9]},
                },
            )
        },
        deep=True,
    )
    ap03_second = prepared_config.model_copy(
        update={
            "methods": MethodSettings(
                enabled=["ap03"],
                ap03={
                    "single": {"scale_marker_id": 9},
                    "multi": {"marker_ids": [9]},
                },
            )
        },
        deep=True,
    )
    assert colmap_artifact_fingerprint(
        ap03_first, "ap03", "input_123"
    ) == colmap_artifact_fingerprint(
        ap03_second, "ap03", "input_123"
    )


def test_non_simulation_results_are_categorized_as_real_vehicle(
    prepared_config,
) -> None:
    assert result_category(prepared_config) == "real_vehicle"


def test_experiment_manifest_refuses_to_mix_different_rigs(
    prepared_config,
) -> None:
    paths = experiment_paths(prepared_config)
    write_experiment_manifest(prepared_config, paths, "input_first")
    changed = prepared_config.model_copy(
        update={
            "static_cameras": [
                prepared_config.static_cameras[0],
                prepared_config.static_cameras[1].model_copy(
                    update={"id": "different-camera"}
                ),
            ]
        },
        deep=True,
    )

    with pytest.raises(RuntimeError, match="different rig"):
        write_experiment_manifest(
            changed, experiment_paths(changed), "input_second"
        )
