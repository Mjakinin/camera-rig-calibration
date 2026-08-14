from __future__ import annotations

import csv
from pathlib import Path

from camera_rig_calibration.ap01_auto_direct import automatic_ap01_direct_target
from camera_rig_calibration.config.models import (
    AP02Settings,
    AP03Settings,
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    InputSourceKind,
    MethodSettings,
    RigConfig,
    StaticCameraSettings,
)
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.methods.ap02.pipeline import AP02Method
from camera_rig_calibration.methods.ap03.pipeline import AP03Method
from camera_rig_calibration.policies.product_policy import _DATASET_CONTEXT, install_product_policy
from camera_rig_calibration.policies.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.policies.submission_bindings import install_submission_bindings
from camera_rig_calibration.policies.submission_policy import install_submission_policy
from camera_rig_calibration.policies.ui_display_policy import install_ui_display_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_submission_bindings()
install_ui_display_policy()

from camera_rig_calibration import (  # noqa: E402
    observations,
    preflight,
    queueing,
    runtime,
    wizard,
)
from camera_rig_calibration.policies import submission_policy  # noqa: E402


def _value(argv: tuple[str, ...], option: str) -> str:
    return argv[argv.index(option) + 1]


def _prepared_config(
    tmp_path: Path,
    *,
    methods: MethodSettings,
    colmap: ColmapSettings | None = None,
) -> RigConfig:
    return RigConfig(
        dataset=DatasetSettings(
            id="parameter_plumbing",
            category=DatasetCategory.SIMULATION,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=tmp_path,
            input_root=tmp_path,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_a"),
            StaticCameraSettings(id="cam_b"),
            StaticCameraSettings(id="cam_c"),
        ],
        methods=methods,
        colmap=colmap or ColmapSettings(),
    )


def test_submission_selection_bindings_cover_every_execution_path() -> None:
    assert preflight.resolve_selections is observations.resolve_selections
    assert wizard.resolve_selections is observations.resolve_selections
    assert runtime.resolve_selections is observations.resolve_selections
    assert queueing.freeze_selections is observations.freeze_selections
    assert runtime.freeze_selections is observations.freeze_selections
    assert (
        submission_policy._automatic_ap01_direct_target
        is automatic_ap01_direct_target
    )


def test_ap01_direct_target_is_not_operator_editable() -> None:
    token = _DATASET_CONTEXT.set("simulation")
    try:
        job = wizard._new_method_job("ap01", prompt_for_single_marker=False)
        rows = wizard._setting_rows(job)
    finally:
        _DATASET_CONTEXT.reset(token)
    assert job.methods.ap01.direct_target_camera == "auto"
    assert all(key != "ap01_direct_target" for key, *_ in rows)
    text = "\n".join(
        f"{label} {current} {baseline} {description}"
        for _, _, label, current, baseline, description in rows
    ).lower()
    assert "direct target camera" not in text


def _write_ap01_observations(path: Path, rows: list[dict[str, object]]) -> None:
    """Write minimal but scientifically valid AP01 static PnP observations.

    The auto-Direct selector intentionally reuses AP01's real baseline candidate
    construction.  Test rows therefore include finite PnP poses and marker
    geometry instead of mocking only the high-level selection-score columns.
    """

    path.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, object] = {
        "pnp_success": "true",
        "distance_m": 2.0,
        "center_u": 640.0,
        "center_v": 360.0,
        "corner0_u": 620.0,
        "corner0_v": 340.0,
        "corner1_u": 660.0,
        "corner1_v": 340.0,
        "corner2_u": 660.0,
        "corner2_v": 380.0,
        "corner3_u": 620.0,
        "corner3_v": 380.0,
        "rvec_x": 0.0,
        "rvec_y": 0.0,
        "rvec_z": 0.0,
        "tvec_x_m": 0.0,
        "tvec_y_m": 0.0,
        "tvec_z_m": 2.0,
    }
    materialized = [{**defaults, **row} for row in rows]
    fields = [
        "observer_type",
        "observer_id",
        "marker_id",
        "selection_score",
        "pnp_reprojection_rmse_px",
        "marker_area_ratio",
        "pnp_success",
        "distance_m",
        "center_u",
        "center_v",
        "corner0_u",
        "corner0_v",
        "corner1_u",
        "corner1_v",
        "corner2_u",
        "corner2_v",
        "corner3_u",
        "corner3_v",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x_m",
        "tvec_y_m",
        "tvec_z_m",
    ]
    with (path / "shared_all_aruco_observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def test_ap01_direct_target_is_selected_from_filtered_overlap(tmp_path: Path) -> None:
    observations_root = tmp_path / "observations"
    _write_ap01_observations(
        observations_root,
        [
            {"observer_type": "static", "observer_id": "cam_root", "marker_id": 14, "selection_score": 20, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.02},
            {"observer_type": "static", "observer_id": "cam_root", "marker_id": 7, "selection_score": 18, "pnp_reprojection_rmse_px": 0.6, "marker_area_ratio": 0.018},
            {"observer_type": "static", "observer_id": "cam_direct", "marker_id": 14, "selection_score": 19, "pnp_reprojection_rmse_px": 0.5, "marker_area_ratio": 0.019},
            {"observer_type": "static", "observer_id": "cam_direct", "marker_id": 7, "selection_score": 17, "pnp_reprojection_rmse_px": 0.7, "marker_area_ratio": 0.017},
            {"observer_type": "static", "observer_id": "cam_relay", "marker_id": 14, "selection_score": 30, "pnp_reprojection_rmse_px": 0.2, "marker_area_ratio": 0.03},
        ],
    )
    config = RigConfig(
        dataset=DatasetSettings(
            id="ap01_selection",
            category=DatasetCategory.SIMULATION,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=tmp_path,
            input_root=tmp_path,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_root"),
            StaticCameraSettings(id="cam_direct"),
            StaticCameraSettings(id="cam_relay"),
        ],
    )
    selected, candidates = automatic_ap01_direct_target(
        config, observations_root, "cam_root"
    )
    assert selected == "cam_direct"
    direct = next(item for item in candidates if item["id"] == "cam_direct")
    relay = next(item for item in candidates if item["id"] == "cam_relay")
    assert direct["independent_shared_markers"] == 2
    assert direct["quality_filtered_markers"] == 2
    assert direct["independent_inlier_markers"] == 2
    assert direct["quality_filter_fallback_used"] is False
    assert direct["compatible"] is True
    assert relay["independent_shared_markers"] == 1
    assert relay["compatible"] is False


def test_ap01_direct_target_falls_back_to_relay_only_without_two_markers(
    tmp_path: Path,
) -> None:
    observations_root = tmp_path / "observations"
    _write_ap01_observations(
        observations_root,
        [
            {"observer_type": "static", "observer_id": "cam_root", "marker_id": 14, "selection_score": 20, "pnp_reprojection_rmse_px": 0.4, "marker_area_ratio": 0.02},
            {"observer_type": "static", "observer_id": "cam_other", "marker_id": 14, "selection_score": 19, "pnp_reprojection_rmse_px": 0.5, "marker_area_ratio": 0.019},
        ],
    )
    config = RigConfig(
        dataset=DatasetSettings(
            id="ap01_relay_only",
            category=DatasetCategory.SIMULATION,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=tmp_path,
            input_root=tmp_path,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_root"),
            StaticCameraSettings(id="cam_other"),
        ],
    )
    selected, candidates = automatic_ap01_direct_target(
        config, observations_root, "cam_root"
    )
    assert selected is None
    assert candidates[0]["independent_shared_markers"] == 1
    assert candidates[0]["compatible"] is False


def test_ap02_nonbaseline_parameters_reach_stage_commands(tmp_path: Path) -> None:
    ap02 = AP02Settings(
        reference_marker_selection_mode="explicit",
        reference_marker_id=11,
        frame_selection_strategy="wizard_graph_preserving_v1",
        initialization_strategy="wizard_maximum_bottleneck_v2",
        graph_edge_weight_strategy="wizard_selection_score_v2",
        reprojection_model="distortion_aware_v1",
        reference_marker_maximum_frames=3,
        top_per_marker=5,
        top_per_marker_pair=2,
        maximum_total_frames=20,
        static_only_ba_max_function_evaluations=17,
        combined_ba_max_function_evaluations=19,
        ba_robust_loss="huber",
        ba_robust_loss_scale_px=2.5,
    )
    config = _prepared_config(
        tmp_path,
        methods=MethodSettings(enabled=["ap02"], ap02=ap02),
    )
    context = RunContext(
        repository_root=tmp_path,
        config=config,
        dataset_root=tmp_path,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap02_reference_marker_id=11,
    )
    commands = {item.stage_id: item.argv for item in AP02Method().commands(context)}

    build = commands["ap02_build_graph"]
    assert _value(build, "--graph-observation-policy") == "wizard_graph_preserving_preselection_v1"
    assert _value(build, "--reference-marker-maximum-frames") == "3"
    assert _value(build, "--top-per-marker") == "5"
    assert _value(build, "--top-per-marker-pair") == "2"
    assert _value(build, "--maximum-total-frames") == "20"

    initialize = commands["ap02_combined_initialization"]
    assert _value(initialize, "--initialization-algorithm") == "wizard_maximum_bottleneck_v2"
    assert _value(initialize, "--edge-weight-policy") == "wizard_selection_score_v2"

    optimize = commands["ap02_combined_ba"]
    assert _value(optimize, "--max-nfev") == "19"
    assert _value(optimize, "--robust-loss") == "huber"
    assert _value(optimize, "--robust-loss-scale-px") == "2.5"
    assert _value(optimize, "--reprojection-model") == "distortion_aware_v1"
    assert _value(optimize, "--moving-frame-selection-policy") == "all_graph_preselected_frames_v1"


def test_ap03_nonbaseline_parameters_reach_colmap_and_scale_commands(
    tmp_path: Path,
) -> None:
    ap03 = AP03Settings(
        feature_limit_policy="wizard_explicit_limits_v1",
        scale_input_policy="wizard_filtered_observations_v1",
        minimum_marker_area_px2=321.0,
        scale={
            "reprojection_threshold_px": 2.25,
            "ransac_iterations": 123,
            "minimum_inliers": 6,
            "maximum_observations_per_marker": 9,
        },
        multi={"marker_ids": [3, 5, 8]},
        single={"scale_marker_id": 3},
    )
    colmap = ColmapSettings(
        matcher="sequential",
        compute_mode="cpu_baseline",
        sequential_overlap=13,
        mapper_minimum_matches=9,
        ap03_maximum_image_size=1777,
        ap03_maximum_features=5555,
        ap03_loop_detection=False,
    )
    config = _prepared_config(
        tmp_path,
        methods=MethodSettings(enabled=["ap03"], ap03=ap03),
        colmap=colmap,
    )
    context = RunContext(
        repository_root=tmp_path,
        config=config,
        dataset_root=tmp_path,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap03_single_scale_marker_id=3,
        resolved_ap03_multi_marker_ids=(3, 5, 8),
    )
    commands = {item.stage_id: item.argv for item in AP03Method().commands(context)}

    reconstruct = commands["ap03_reconstruct"]
    assert _value(reconstruct, "--matcher") == "sequential"
    assert _value(reconstruct, "--max-image-size") == "1777"
    assert _value(reconstruct, "--max-features") == "5555"
    assert _value(reconstruct, "--sequential-overlap") == "13"
    assert _value(reconstruct, "--loop-detection") == "0"
    assert _value(reconstruct, "--mapper-min-matches") == "9"

    scale = commands["ap03_multi_scale"]
    assert _value(scale, "--marker-ids") == "3,5,8"
    assert _value(scale, "--reprojection-threshold-px") == "2.25"
    assert _value(scale, "--ransac-iterations") == "123"
    assert _value(scale, "--minimum-inliers") == "6"
    assert _value(scale, "--maximum-observations-per-marker") == "9"
    assert _value(scale, "--scale-input-policy") == "wizard_filtered_observations_v1"
    assert _value(scale, "--minimum-marker-area-px2") == "321.0"
    # estimate_scale receives the filtered policy here and attaches the accepted
    # observation table to its scale_core subprocess during execution.
