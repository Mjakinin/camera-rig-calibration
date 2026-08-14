from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from camera_rig_calibration import cli
from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config import (
    load_config,
    save_config,
    save_user_config,
)
from camera_rig_calibration.config.models import MethodSettings
from camera_rig_calibration.experiments import method_fingerprint
from camera_rig_calibration.methods.ap01.contracts import (
    ap01_execution_contract_name,
    resolve_ap01_method_contract,
)
from camera_rig_calibration.methods.ap02.contracts import (
    resolve_ap02_method_contract,
)
from camera_rig_calibration.methods.ap03.contracts import (
    resolve_ap03_method_contract,
)
from camera_rig_calibration.observations import ResolvedSelections
from camera_rig_calibration.registry import calibration_methods
from camera_rig_calibration.rerun import _resolved_rerun_config
from camera_rig_calibration.wizard import (
    _is_internal_evidence_result,
    _load_saved_setup_config,
    _method_job_summary,
    _new_method_job,
    _setting_rows,
    show_summary,
)


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


def _write_rerun_source(
    experiment: Path, method: str, config
) -> None:
    destination = (
        experiment
        / "methods"
        / method
        / "baseline"
        / "provenance"
        / "resolved_config.yaml"
    )
    save_config(config, destination)


def test_public_method_names_and_default_entry_points(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    names = {
        method.id: method.display_name for method in calibration_methods
    }
    assert {"ap01": "AP01", "ap02": "AP02", "ap03": "AP03"}.items() <= (
        names.items()
    )

    defaults = MethodSettings()
    for method_id in ("ap01", "ap02", "ap03"):
        assert getattr(defaults, method_id).method_contract == "baseline_v1"
        job = _new_method_job(
            method_id, prompt_for_single_marker=False
        )
        assert (
            getattr(job.methods, method_id).method_contract
            == "baseline_v1"
        )

    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap01"]
    generated = save_user_config(config, tmp_path / "generated.yaml")
    loaded = load_config(generated, resolve_paths=False)
    for method_id in ("ap01", "ap02", "ap03"):
        assert (
            getattr(loaded.methods, method_id).method_contract
            == "baseline_v1"
        )


def test_ordinary_baseline_rerun_resets_advanced_state(
    prepared_config, tmp_path: Path
) -> None:
    experiment = tmp_path / "experiment"
    ap01 = prepared_config.model_copy(deep=True)
    ap01.methods.enabled = ["ap01"]
    ap01.methods.ap01.method_contract = "recommended_wizard_v1"
    _write_rerun_source(experiment, "ap01", ap01)
    _, rerun = _resolved_rerun_config(
        tmp_path, experiment, "ap01", "baseline"
    )
    assert rerun.methods.ap01.method_contract == "baseline_v1"

    ap02 = prepared_config.model_copy(deep=True)
    ap02.methods.enabled = ["ap02"]
    ap02.methods.ap02.frame_selection_strategy = (
        "wizard_graph_preserving_v1"
    )
    _write_rerun_source(experiment, "ap02", ap02)
    _, rerun = _resolved_rerun_config(
        tmp_path, experiment, "ap02", "baseline"
    )
    assert rerun.methods.ap02.method_contract == "baseline_v1"
    assert rerun.methods.ap02.frame_selection_strategy == "smart_v1"

    ap03 = prepared_config.model_copy(deep=True)
    ap03.methods.enabled = ["ap03"]
    ap03.methods.ap03.feature_limit_policy = (
        "wizard_explicit_limits_v1"
    )
    _write_rerun_source(experiment, "ap03", ap03)
    _, rerun = _resolved_rerun_config(
        tmp_path, experiment, "ap03", "baseline"
    )
    assert rerun.methods.ap03.method_contract == "baseline_v1"
    assert (
        rerun.methods.ap03.feature_limit_policy
        == "colmap_defaults_v1"
    )


def test_wizard_generated_baseline_config_is_clean_and_round_trips(
    prepared_config, tmp_path: Path
) -> None:
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap01"]
    destination = save_user_config(config, tmp_path / "rigcal.yaml")
    text = destination.read_text(encoding="utf-8").lower()
    assert "method_contract: baseline_v1" in text
    loaded = load_config(destination, resolve_paths=False)
    assert loaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_wizard_round_trip_preserves_representative_advanced_settings(
    prepared_config, tmp_path: Path
) -> None:
    ap01 = prepared_config.model_copy(deep=True)
    ap01.methods.enabled = ["ap01"]
    ap01.methods.ap01.method_contract = "recommended_wizard_v1"
    ap01.methods.ap01.top_moving_per_marker = 5
    ap01.methods.ap01.direct_quality_gate.minimum_inlier_ratio = 0.8
    loaded = load_config(
        save_user_config(ap01, tmp_path / "ap01.yaml"),
        resolve_paths=False,
    )
    assert loaded.methods.ap01.top_moving_per_marker == 5
    execution = ap01_execution_contract_name(
        loaded.methods.ap01.method_contract
    )
    assert resolve_ap01_method_contract(execution).name == (
        "recommended_wizard_v1"
    )

    ap02 = prepared_config.model_copy(deep=True)
    ap02.methods.enabled = ["ap02"]
    ap02.methods.ap02.combined_ba_max_function_evaluations = 37
    ap02.methods.ap02.ba_robust_loss = "huber"
    loaded = load_config(
        save_user_config(ap02, tmp_path / "ap02.yaml"),
        resolve_paths=False,
    )
    contract = resolve_ap02_method_contract(
        loaded.methods.ap02.method_contract,
        combined_maximum_function_evaluations=(
            loaded.methods.ap02.combined_ba_max_function_evaluations
        ),
        robust_loss=loaded.methods.ap02.ba_robust_loss,
    )
    assert contract.combined_maximum_function_evaluations == 37
    assert contract.robust_loss == "huber"

    ap03 = prepared_config.model_copy(deep=True)
    ap03.methods.enabled = ["ap03"]
    ap03.methods.ap03.feature_limit_policy = (
        "wizard_explicit_limits_v1"
    )
    ap03.methods.ap03.scale_input_policy = (
        "wizard_filtered_observations_v1"
    )
    ap03.methods.ap03.minimum_marker_area_px2 = 125.0
    ap03.colmap.ap03_maximum_image_size = 1234
    ap03.colmap.ap03_maximum_features = 3456
    loaded = load_config(
        save_user_config(ap03, tmp_path / "ap03.yaml"),
        resolve_paths=False,
    )
    contract = resolve_ap03_method_contract(
        loaded.methods.ap03.method_contract,
        feature_limit_policy=loaded.methods.ap03.feature_limit_policy,
        scale_input_policy=loaded.methods.ap03.scale_input_policy,
        minimum_marker_area_px2=(
            loaded.methods.ap03.minimum_marker_area_px2
        ),
        colmap_maximum_image_size=(
            loaded.colmap.ap03_maximum_image_size or 1600
        ),
        colmap_maximum_features=(
            loaded.colmap.ap03_maximum_features or 4096
        ),
    )
    assert contract.sift_maximum_image_size == 1234
    assert contract.sift_maximum_features == 3456
    assert contract.scale_minimum_marker_area_px2 == 125.0


def test_wizard_controls_are_policy_aware_and_product_named() -> None:
    forbidden = ("historical",)
    for method_id in ("ap01", "ap02", "ap03"):
        job = _new_method_job(method_id, prompt_for_single_marker=False)
        rows = _setting_rows(job, None)
        visible = (_method_job_summary(job) + repr(rows)).lower()
        assert not any(term in visible for term in forbidden)

    ap01 = _new_method_job("ap01", prompt_for_single_marker=False)
    baseline_keys = {row[0] for row in _setting_rows(ap01, None)}
    assert "ap01_direct_inlier_ratio" not in baseline_keys
    assert {"matcher", "maximum_image_size", "maximum_features"} <= baseline_keys
    assert "ap01_method_contract" in baseline_keys
    ap01.methods.ap01.method_contract = "recommended_wizard_v1"
    robust_keys = {row[0] for row in _setting_rows(ap01, None)}
    assert {"ap01_direct_inlier_ratio", "matcher"} <= robust_keys

    ap03 = _new_method_job("ap03", prompt_for_single_marker=False)
    baseline_keys = {row[0] for row in _setting_rows(ap03, None)}
    assert "ap03_image_size" not in baseline_keys
    assert "scale_max_observations" not in baseline_keys
    ap03.methods.ap03.feature_limit_policy = "wizard_explicit_limits_v1"
    ap03.methods.ap03.scale_input_policy = (
        "wizard_filtered_observations_v1"
    )
    advanced_keys = {row[0] for row in _setting_rows(ap03, None)}
    assert {"ap03_image_size", "scale_max_observations"} <= advanced_keys


def test_method_fingerprints_are_sensitive_and_isolated(
    prepared_config,
) -> None:
    selections = _resolved()
    baseline = {
        method: method_fingerprint(prepared_config, method, selections)
        for method in ("ap01", "ap02", "ap03")
    }

    ap01 = prepared_config.model_copy(deep=True)
    ap01.methods.ap01.direct_target_camera = "roof.camera"
    assert method_fingerprint(ap01, "ap01", selections) != baseline["ap01"]
    assert method_fingerprint(ap01, "ap02", selections) == baseline["ap02"]
    assert method_fingerprint(ap01, "ap03", selections) == baseline["ap03"]

    ap02 = prepared_config.model_copy(deep=True)
    ap02.methods.ap02.combined_ba_max_function_evaluations = 37
    assert method_fingerprint(ap02, "ap02", selections) != baseline["ap02"]
    assert method_fingerprint(ap02, "ap01", selections) == baseline["ap01"]
    assert method_fingerprint(ap02, "ap03", selections) == baseline["ap03"]

    ap03 = prepared_config.model_copy(deep=True)
    ap03.methods.ap03.feature_limit_policy = "wizard_explicit_limits_v1"
    ap03.colmap.ap03_maximum_features = 3456
    assert method_fingerprint(ap03, "ap03", selections) != baseline["ap03"]
    assert method_fingerprint(ap03, "ap01", selections) == baseline["ap01"]
    assert method_fingerprint(ap03, "ap02", selections) == baseline["ap02"]

    ui_only = prepared_config.model_copy(deep=True)
    ui_only.project.run_label = "renamed_in_wizard"
    for method in ("ap01", "ap02", "ap03"):
        assert (
            method_fingerprint(ui_only, method, selections)
            == baseline[method]
        )


def test_method_queue_switching_keeps_per_method_state() -> None:
    jobs = {
        method: _new_method_job(method, prompt_for_single_marker=False)
        for method in ("ap01", "ap02", "ap03")
    }
    jobs["ap01"].methods.ap01.direct_target_camera = "camera_a"
    jobs["ap02"].methods.ap02.combined_ba_max_function_evaluations = 37
    jobs["ap03"].methods.ap03.minimum_marker_area_px2 = 125.0

    assert jobs["ap01"].methods.ap01.direct_target_camera == "camera_a"
    assert jobs["ap02"].methods.ap02.combined_ba_max_function_evaluations == 37
    assert jobs["ap03"].methods.ap03.minimum_marker_area_px2 == 125.0
    assert jobs["ap01"].methods.ap01.direct_target_camera == "camera_a"


def test_removed_historical_options_are_absent_from_cli_help(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["rigcal", "rerun-method", "--help"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out.lower()
    assert "recommended_wizard_v1" not in help_text
    assert "historical-reproduction" not in help_text


def test_published_saved_setup_rebinds_to_immutable_experiment(
    prepared_config, tmp_path: Path
) -> None:
    experiment = tmp_path / "results" / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "dataset.json").write_text("{}\n", encoding="utf-8")
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap03"]
    config.dataset.prepared_root = tmp_path / "deleted_temporary_dataset"
    path = (
        experiment
        / "methods"
        / "ap03"
        / "baseline"
        / "provenance"
        / "resolved_config.yaml"
    )
    save_config(config, path)

    loaded = _load_saved_setup_config(path)

    assert loaded.dataset.prepared_root == experiment.resolve()
    assert loaded.dataset.input_root == experiment.resolve()
    assert loaded.methods.ap03.method_contract == "baseline_v1"


def test_internal_evidence_is_filtered_only_from_normal_results() -> None:
    internal = SimpleNamespace(
        experiment_id="route2_pre_fix",
        dataset_id="route2_pre_fix",
        path=Path("results/simulation/baseline/route2_pre_fix"),
    )
    public = SimpleNamespace(
        experiment_id="vehicle_day_01",
        dataset_id="vehicle_day_01",
        path=Path("results/real_vehicle/1Hz/vehicle_day_01"),
    )
    assert _is_internal_evidence_result(internal) is True
    assert _is_internal_evidence_result(public) is False


def test_final_summary_shows_only_enabled_method_selections(
    prepared_config, tmp_path: Path
) -> None:
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = ["ap03"]
    output = StringIO()
    show_summary(
        config,
        tmp_path / "rigcal.yaml",
        Console(file=output, force_terminal=False, width=180),
    )
    rendered = output.getvalue()
    assert "AP03 baseline_v1" in rendered
    assert "AP03 single=" in rendered
    assert "AP01 root=" not in rendered
    assert "AP02 ref=" not in rendered
