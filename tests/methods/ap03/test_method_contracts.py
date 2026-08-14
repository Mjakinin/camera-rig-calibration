from __future__ import annotations

from pathlib import Path

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import AP03Settings
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.experiments import method_fingerprint
from camera_rig_calibration.methods.ap03.contracts import (
    resolve_ap03_method_contract,
)
from camera_rig_calibration.methods.ap03.scale_common import (
    parse_best_model_name,
)
from camera_rig_calibration.observations import ResolvedSelections
from camera_rig_calibration.registry import calibration_methods


def _resolved() -> ResolvedSelections:
    return ResolvedSelections(
        root_camera="front-left",
        ap02_reference_marker_id=14,
        ap03_single_scale_marker_id=14,
        ap03_multi_marker_ids=tuple(range(15)),
        evaluation_anchor_marker_id=14,
        marker_ids=tuple(range(15)),
        payload={},
    )


def _commands(config, tmp_path: Path):
    register_builtin_components()
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap03_single_scale_marker_id=14,
        resolved_ap03_multi_marker_ids=tuple(range(15)),
    )
    return calibration_methods.get("ap03").commands(context)


def test_ap03_defaults_resolve_baseline_contract() -> None:
    settings = AP03Settings()
    contract = resolve_ap03_method_contract(
        settings.method_contract,
        feature_limit_policy=settings.feature_limit_policy,
        scale_input_policy=settings.scale_input_policy,
        scale_marker_ids=settings.multi.marker_ids,
        minimum_marker_area_px2=settings.minimum_marker_area_px2,
    )
    assert settings.method_contract == "baseline_v1"
    assert settings.single.scale_marker_id == 14
    assert settings.multi.marker_ids == list(range(15))
    assert contract.sift_maximum_image_size is None
    assert contract.sift_maximum_features is None
    assert contract.refine_focal_length is False
    assert contract.refine_principal_point is False
    assert contract.refine_extra_parameters is False
    assert contract.scale_minimum_marker_area_px2 == 100.0
    assert contract.reference_frame_convention == (
        "native_colmap_gauge_translation_scaled_only_v1"
    )
    assert contract.ground_truth_policy == (
        "forbidden_during_reconstruction_and_calibration"
    )


def test_ap03_baseline_commands_use_fixed_feature_and_scale_inputs(
    prepared_config, tmp_path: Path
) -> None:
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap03"]
    config.methods.ap03 = AP03Settings()
    commands = _commands(config, tmp_path)
    reconstruct = commands[1].argv
    assert "--max-image-size" not in reconstruct
    assert "--max-features" not in reconstruct
    assert "--loop-detection" not in reconstruct
    assert reconstruct[reconstruct.index("--matcher") + 1] == "exhaustive"
    assert reconstruct[reconstruct.index("--use-gpu") + 1] == "0"
    assert reconstruct[reconstruct.index("--mapper-min-matches") + 1] == "8"
    multi = commands[4].argv
    assert multi[multi.index("--marker-ids") + 1] == ",".join(
        str(value) for value in range(15)
    )
    assert multi[multi.index("--scale-input-policy") + 1] == (
        "registered_image_redetection_v1"
    )
    assert multi[multi.index("--minimum-marker-area-px2") + 1] == "100.0"
    flattened = " ".join(token for command in commands for token in command.argv)
    assert "ground_truth" not in flattened.lower()
    assert "ground-truth" not in flattened.lower()


def test_ap03_advanced_wizard_policies_remain_explicit(
    prepared_config, tmp_path: Path
) -> None:
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap03"]
    config.methods.ap03 = AP03Settings(
        feature_limit_policy="wizard_explicit_limits_v1",
        scale_input_policy="wizard_filtered_observations_v1",
    )
    config.colmap.matcher = "sequential"
    commands = _commands(config, tmp_path)
    reconstruct = commands[1].argv
    assert reconstruct[reconstruct.index("--max-image-size") + 1] == "2400"
    assert reconstruct[reconstruct.index("--max-features") + 1] == "8192"
    assert reconstruct[reconstruct.index("--loop-detection") + 1] == "1"
    multi = commands[4].argv
    assert multi[multi.index("--scale-input-policy") + 1] == (
        "wizard_filtered_observations_v1"
    )


def test_ap03_model_selection_prefers_most_registered_images(tmp_path: Path) -> None:
    summary = tmp_path / "colmap_model_summary.csv"
    summary.write_text(
        "model,registered_images,registered_static_cameras,num_3d_points\n"
        "0,193,3,9000\n"
        "1,180,4,5000\n"
        "2,180,4,6000\n",
        encoding="utf-8",
    )
    assert parse_best_model_name(summary) == "2"


def test_every_ap03_advanced_policy_changes_method_fingerprint(
    prepared_config,
) -> None:
    baseline = prepared_config.model_copy(deep=True)
    baseline.methods.enabled = ["ap03"]
    baseline.methods.ap03 = AP03Settings()
    baseline_fp = method_fingerprint(baseline, "ap03", _resolved())
    changes = (
        {"feature_limit_policy": "wizard_explicit_limits_v1"},
        {"scale_input_policy": "wizard_filtered_observations_v1"},
        {"minimum_marker_area_px2": 125.0},
    )
    for update in changes:
        changed = baseline.model_copy(deep=True)
        changed.methods.ap03 = changed.methods.ap03.model_copy(update=update)
        assert method_fingerprint(changed, "ap03", _resolved()) != baseline_fp
