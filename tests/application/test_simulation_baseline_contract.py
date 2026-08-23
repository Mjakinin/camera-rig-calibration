from __future__ import annotations

from pathlib import Path
import yaml

from camera_rig_calibration.application.bootstrap import _resolve_wizard_policy_target
from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import RigConfig
from camera_rig_calibration.evaluation import reporting
from camera_rig_calibration.policies.marker_preference_policy import (
    install_marker_preference_policy,
)
from camera_rig_calibration.policies.product_policy import (
    _DATASET_CONTEXT,
    install_product_policy,
)
from camera_rig_calibration.policies.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.policies.submission_policy import (
    install_submission_policy,
)
from camera_rig_calibration.policies.submission_bindings import (
    install_submission_bindings,
)
from camera_rig_calibration import wizard


_resolve_wizard_policy_target()
install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_submission_bindings()

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_simulation_baseline_yaml_satisfies_baseline_contract() -> None:
    register_builtin_components()

    yaml_path = REPOSITORY_ROOT / "examples" / "simulation_baseline.yaml"
    raw_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    config = RigConfig.model_validate(raw_payload)

    assert config.methods.ap02.reference_marker_selection_mode == "baseline"
    assert config.methods.ap02.reference_marker_id == 14
    assert config.methods.ap02.static_only_ba_max_function_evaluations == 80
    assert config.methods.ap02.combined_ba_max_function_evaluations == 80
    assert config.methods.ap02.initialization_strategy == "maximum_frontier_v1"
    assert config.evaluation.anchor_marker_id == 14
    assert config.evaluation.anchor_selection_mode == "explicit"

    contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": config.methods.ap02.reference_marker_id,
                    "resolved_reference_marker_id": 14,
                    "reference_marker_selection_mode": (
                        config.methods.ap02.reference_marker_selection_mode
                    ),
                    "static_max_nfev": (
                        config.methods.ap02.static_only_ba_max_function_evaluations
                    ),
                    "combined_max_nfev": (
                        config.methods.ap02.combined_ba_max_function_evaluations
                    ),
                    "initialization_algorithm": "maximum_bottleneck",
                },
            }
        ],
        evaluation_anchor={"selected": 14},
    )

    assert contract["passes"] is True
    checks = contract["variants"][0]["checks"]
    assert checks["reference_mode_baseline"] is True
    assert checks["reference_marker_14"] is True
    assert checks["static_nfev_80"] is True
    assert checks["combined_nfev_80"] is True


def test_simulation_wizard_ap02_defaults_satisfy_baseline_contract() -> None:
    register_builtin_components()

    token = _DATASET_CONTEXT.set("simulation")
    try:
        job = wizard._new_method_job("ap02", prompt_for_single_marker=False)
        assert job.methods.ap02.reference_marker_selection_mode == "baseline"
        assert job.methods.ap02.reference_marker_id == 14
        assert job.methods.ap02.static_only_ba_max_function_evaluations == 80
        assert job.methods.ap02.combined_ba_max_function_evaluations == 80
        assert job.evaluation.anchor_marker_id == 14
        assert job.evaluation.anchor_selection_mode == "explicit"
        assert job.label == "baseline"

        contract = reporting._baseline_contract(
            category="simulation",
            method_payloads=[
                {
                    "method": "ap02",
                    "label": "baseline",
                    "config_summary": {
                        "reference_marker_id": job.methods.ap02.reference_marker_id,
                        "resolved_reference_marker_id": 14,
                        "reference_marker_selection_mode": (
                            job.methods.ap02.reference_marker_selection_mode
                        ),
                        "static_max_nfev": (
                            job.methods.ap02.static_only_ba_max_function_evaluations
                        ),
                        "combined_max_nfev": (
                            job.methods.ap02.combined_ba_max_function_evaluations
                        ),
                        "initialization_algorithm": "maximum_bottleneck",
                    },
                }
            ],
            evaluation_anchor={"selected": 14},
        )
        assert contract["passes"] is True
        assert contract["variants"][0]["checks"]["reference_mode_baseline"] is True
    finally:
        _DATASET_CONTEXT.reset(token)


def test_auto_reference_mode_does_not_satisfy_strict_baseline_contract() -> None:
    contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": 14,
                    "resolved_reference_marker_id": 14,
                    "reference_marker_selection_mode": "auto",
                    "static_max_nfev": 80,
                    "combined_max_nfev": 80,
                    "initialization_algorithm": "maximum_bottleneck",
                },
            }
        ],
        evaluation_anchor={"selected": 14},
    )
    assert contract["passes"] is False
    assert contract["variants"][0]["checks"]["reference_mode_baseline"] is False


def test_simulation_baseline_missing_marker_14_fails_and_does_not_fall_back_silently(
    tmp_path: Path,
) -> None:
    import csv
    import pytest
    from camera_rig_calibration import observations
    from camera_rig_calibration.config.models import (
        InputSourceKind,
        SimulationSettings,
        StaticCameraSettings,
    )

    yaml_path = REPOSITORY_ROOT / "examples" / "simulation_baseline.yaml"
    raw_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    config = RigConfig.model_validate(raw_payload)
    config = config.model_copy(
        update={
            "simulation": SimulationSettings(enabled=False),
            "methods": config.methods.model_copy(update={"enabled": ["ap02"]}),
            "dataset": config.dataset.model_copy(
                update={
                    "source_kind": InputSourceKind.PREPARED,
                    "prepared_root": tmp_path,
                    "input_root": tmp_path,
                }
            ),
        },
        deep=True,
    )

    obs_root = tmp_path / "observations"
    obs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    camera_ids = [c.id for c in config.static_cameras]
    # Observations contain only marker 5 across all static cameras, marker 14 is absent
    for camera_index, camera in enumerate(camera_ids):
        rows.append(
            {
                "observer_type": "static",
                "observer_id": camera,
                "camera_name": camera,
                "frame_id": "",
                "marker_id": 5,
                "pnp_success": "true",
                "selection_score": 100.0 - camera_index,
                "pnp_reprojection_rmse_px": 0.3 + 0.05 * camera_index,
                "marker_area_ratio": 0.02,
            }
        )
    for frame in (0, 1, 2):
        rows.append(
            {
                "observer_type": "moving",
                "observer_id": "moving_calib_camera",
                "camera_name": "moving_calib_camera",
                "frame_id": frame,
                "marker_id": 5,
                "pnp_success": "true",
                "selection_score": 80.0 - frame * 0.01,
                "pnp_reprojection_rmse_px": 0.4,
                "marker_area_ratio": 0.015,
            }
        )
    fields = [
        "observer_type",
        "observer_id",
        "camera_name",
        "frame_id",
        "marker_id",
        "pnp_success",
        "selection_score",
        "pnp_reprojection_rmse_px",
        "marker_area_ratio",
    ]
    with (obs_root / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 1. Untouched simulation baseline (reference_marker_selection_mode="baseline", reference_marker_id=14)
    # Must raise RuntimeError and NOT silently fall back to marker 5
    with pytest.raises(RuntimeError, match="marker 14"):
        observations.resolve_selections(config, obs_root)

    # 2. If user deliberately switches AP02 to "auto" mode, resolution succeeds with fallback marker 5
    auto_config = config.model_copy(
        update={
            "methods": config.methods.model_copy(
                update={
                    "ap02": config.methods.ap02.model_copy(
                        update={"reference_marker_selection_mode": "auto"}
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )
    resolved_auto = observations.resolve_selections(auto_config, obs_root)
    assert resolved_auto.ap02_reference_marker_id == 5

    # But this run no longer satisfies the strict Route-2 baseline contract
    auto_contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": auto_config.methods.ap02.reference_marker_id,
                    "resolved_reference_marker_id": resolved_auto.ap02_reference_marker_id,
                    "reference_marker_selection_mode": (
                        auto_config.methods.ap02.reference_marker_selection_mode
                    ),
                    "static_max_nfev": 80,
                    "combined_max_nfev": 80,
                    "initialization_algorithm": "maximum_bottleneck",
                },
            }
        ],
        evaluation_anchor={"selected": 14},
    )
    assert auto_contract["passes"] is False
    assert auto_contract["variants"][0]["checks"]["reference_mode_baseline"] is False

    # 3. If user deliberately switches AP02 to manual/explicit marker 5, resolution succeeds with marker 5
    explicit_config = config.model_copy(
        update={
            "methods": config.methods.model_copy(
                update={
                    "ap02": config.methods.ap02.model_copy(
                        update={
                            "reference_marker_selection_mode": "manual",
                            "reference_marker_id": 5,
                        }
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )
    resolved_explicit = observations.resolve_selections(explicit_config, obs_root)
    assert resolved_explicit.ap02_reference_marker_id == 5

    explicit_contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": 5,
                    "resolved_reference_marker_id": 5,
                    "reference_marker_selection_mode": "manual",
                    "static_max_nfev": 80,
                    "combined_max_nfev": 80,
                    "initialization_algorithm": "maximum_bottleneck",
                },
            }
        ],
        evaluation_anchor={"selected": 14},
    )
    assert explicit_contract["passes"] is False
    assert explicit_contract["variants"][0]["checks"]["reference_mode_baseline"] is False
