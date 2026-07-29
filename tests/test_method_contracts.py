from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.registry import calibration_methods, evaluators


def test_ap02_command_uses_generic_cameras_and_recommended_defaults(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    run = tmp_path / "run"
    run.mkdir()
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=prepared_config,
        dataset_root=prepared_config.dataset.prepared_root,
        observations_root=run / "01_OBSERVATIONS",
        run_directory=run,
        resolved_ap02_reference_marker_id=7,
        resolved_ap03_single_scale_marker_id=9,
        resolved_ap03_multi_marker_ids=(7, 9),
        resolved_evaluation_anchor_marker_id=7,
        resolved_marker_ids=(7, 9),
    )
    commands = calibration_methods.get("ap02").commands(context)
    assert [command.stage_id for command in commands] == [
        "ap02_build_graph",
        "ap02_component_diagnostics",
        "ap02_static_initialization",
        "ap02_static_ba",
        "ap02_combined_initialization",
        "ap02_combined_ba",
        "ap02_report",
    ]
    graph = commands[0].argv
    assert graph[graph.index("--cameras") + 1] == "front-left,roof.camera"
    static_ba = commands[3].argv
    combined_ba = commands[5].argv
    assert static_ba[static_ba.index("--max-nfev") + 1] == "50"
    assert combined_ba[combined_ba.index("--max-nfev") + 1] == "50"
    flattened = " ".join(
        token for command in commands for token in command.argv
    )
    assert "moving-selection" not in flattened
    assert "max-moving-frames" not in flattened
    assert "--top-per-marker 8" in flattened
    assert "--top-per-marker-pair 4" in flattened
    assert commands[1].diagnostic is True
    assert commands[2].diagnostic is True
    assert commands[5].depends_on == ("ap02_combined_initialization",)


def test_ap03_combines_single_and_multi_with_one_colmap_run(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    run = tmp_path / "run"
    run.mkdir()
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=prepared_config,
        dataset_root=prepared_config.dataset.prepared_root,
        observations_root=run / "01_OBSERVATIONS",
        run_directory=run,
        resolved_ap02_reference_marker_id=7,
        resolved_ap03_single_scale_marker_id=9,
        resolved_ap03_multi_marker_ids=(7, 9),
        resolved_evaluation_anchor_marker_id=7,
        resolved_marker_ids=(7, 9),
    )
    commands = calibration_methods.get("ap03").commands(context)
    assert [command.stage_id for command in commands] == [
        "ap03_prepare_colmap",
        "ap03_reconstruct",
        "ap03_inspect",
        "ap03_single_scale",
        "ap03_multi_scale",
        "ap03_report",
    ]
    reconstruct = commands[1].argv
    assert reconstruct[reconstruct.index("--max-image-size") + 1] == "2400"
    assert reconstruct[reconstruct.index("--max-features") + 1] == "8192"
    assert reconstruct[reconstruct.index("--loop-detection") + 1] == "1"
    single = commands[3].argv
    multi = commands[4].argv
    assert single[single.index("--marker-ids") + 1] == "9"
    assert multi[multi.index("--marker-ids") + 1] == "7,9"
    for flag in (
        "--reprojection-threshold-px",
        "--ransac-iterations",
        "--minimum-inliers",
    ):
        assert single[single.index(flag) + 1] == multi[multi.index(flag) + 1]
    flattened = " ".join(
        token for command in commands for token in command.argv
    )
    assert "min-area" not in flattened
    assert "single-ransac" not in flattened
    assert "multi-ransac" not in flattened
    assert commands[3].diagnostic is True
    assert commands[5].depends_on == (
        "ap03_single_scale",
        "ap03_multi_scale",
    )


def test_ap01_uses_configured_quality_ranked_caps(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    config = prepared_config.model_copy(
        update={
            "methods": prepared_config.methods.model_copy(
                update={"enabled": ["ap01"]}, deep=True
            )
        },
        deep=True,
    )
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "accepted",
        run_directory=tmp_path / "run",
        resolved_root_camera="front-left",
        resolved_marker_ids=(7, 9),
    )

    commands = calibration_methods.get("ap01").commands(context)
    flattened = " ".join(
        token for command in commands for token in command.argv
    )
    assert "--top-moving-per-marker 8" in flattened
    assert "--scale-top-per-marker 30" in flattened
    assert "--direct-minimum-independent-markers 3" in flattened
    assert "--direct-minimum-inlier-ratio 0.7" in flattened
    assert "--direct-maximum-translation-dispersion-m 0.12" in flattened
    assert "--direct-maximum-rotation-dispersion-deg 4.0" in flattened
    assert "--relay-minimum-inlier-ratio 0.7" in flattened
    assert "--relay-maximum-translation-dispersion-m 0.3" in flattened
    assert "--relay-maximum-rotation-dispersion-deg 7.0" in flattened
    assert "--maximum-path-translation-disagreement-m 0.12" in flattened
    assert "--maximum-path-rotation-disagreement-deg 4.0" in flattened
    assert commands[-1].depends_on == ("ap01_solve_extrinsics",)


def test_nondefault_gui_parameters_reach_ap01_ap02_ap03_commands(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    methods = prepared_config.methods.model_copy(
        update={
            "enabled": ["ap01", "ap02", "ap03"],
            "ap01": prepared_config.methods.ap01.model_copy(
                update={"root_camera": "roof.camera"}
            ),
            "ap02": prepared_config.methods.ap02.model_copy(
                update={
                    "reference_marker_id": 9,
                    "static_only_ba_max_function_evaluations": 31,
                    "combined_ba_max_function_evaluations": 47,
                    "ba_robust_loss": "huber",
                    "ba_robust_loss_scale_px": 1.25,
                }
            ),
            "ap03": prepared_config.methods.ap03.model_copy(
                update={
                    "single": prepared_config.methods.ap03.single.model_copy(
                        update={"scale_marker_id": 9}
                    ),
                    "multi": prepared_config.methods.ap03.multi.model_copy(
                        update={"marker_ids": [7, 9]}
                    ),
                    "scale": prepared_config.methods.ap03.scale.model_copy(
                        update={
                            "reprojection_threshold_px": 2.5,
                            "ransac_iterations": 321,
                            "minimum_inliers": 6,
                            "maximum_observations_per_marker": 12,
                        }
                    ),
                },
                deep=True,
            ),
        },
        deep=True,
    )
    config = prepared_config.model_copy(
        update={
            "methods": methods,
            "colmap": prepared_config.colmap.model_copy(
                update={
                    "matcher": "sequential",
                    "gpu_mode": "false",
                    "maximum_image_size": 1800,
                    "maximum_features": 4096,
                    "sequential_overlap": 13,
                    "loop_detection": False,
                    "mapper_minimum_matches": 11,
                }
            ),
        },
        deep=True,
    )
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_root_camera="roof.camera",
        resolved_ap02_reference_marker_id=9,
        resolved_ap03_single_scale_marker_id=9,
        resolved_ap03_multi_marker_ids=(7, 9),
        resolved_evaluation_anchor_marker_id=7,
    )

    ap01 = calibration_methods.get("ap01").commands(context)[0].argv
    assert ap01[ap01.index("--root-camera") + 1] == "roof.camera"
    assert ap01[ap01.index("--matcher") + 1] == "sequential"
    assert ap01[ap01.index("--use-gpu") + 1] == "0"
    assert ap01[ap01.index("--max-image-size") + 1] == "1800"
    assert ap01[ap01.index("--max-features") + 1] == "4096"
    assert ap01[ap01.index("--sequential-overlap") + 1] == "13"
    assert ap01[ap01.index("--loop-detection") + 1] == "0"
    assert ap01[ap01.index("--mapper-min-matches") + 1] == "11"

    ap02 = calibration_methods.get("ap02").commands(context)
    static_ba, combined_ba = ap02[3].argv, ap02[5].argv
    assert static_ba[static_ba.index("--max-nfev") + 1] == "31"
    assert combined_ba[combined_ba.index("--max-nfev") + 1] == "47"
    for command in (static_ba, combined_ba):
        assert command[command.index("--robust-loss") + 1] == "huber"
        assert command[command.index("--robust-loss-scale-px") + 1] == "1.25"

    ap03 = calibration_methods.get("ap03").commands(context)
    reconstruct, single, multi = ap03[1].argv, ap03[3].argv, ap03[4].argv
    assert reconstruct[reconstruct.index("--matcher") + 1] == "sequential"
    assert single[single.index("--marker-ids") + 1] == "9"
    assert multi[multi.index("--marker-ids") + 1] == "7,9"
    for command in (single, multi):
        assert command[command.index("--reprojection-threshold-px") + 1] == "2.5"
        assert command[command.index("--ransac-iterations") + 1] == "321"
        assert command[command.index("--minimum-inliers") + 1] == "6"
        assert (
            command[
                command.index("--maximum-observations-per-marker") + 1
            ]
            == "12"
        )


def test_common_evaluation_parameters_reach_evaluator_command(
    prepared_config, tmp_path: Path
) -> None:
    register_builtin_components()
    config = prepared_config.model_copy(
        update={
            "evaluation": prepared_config.evaluation.model_copy(
                update={
                    "anchor_marker_id": 7,
                    "reprojection_threshold_px": 2.25,
                    "minimum_inliers": 6,
                    "ransac_iterations": 456,
                    "minimum_triangulation_angle_deg": 1.5,
                    "maximum_moving_observations_per_marker": 42,
                }
            )
        },
        deep=True,
    )
    context = RunContext(
        repository_root=Path(__file__).resolve().parents[1],
        config=config,
        dataset_root=config.dataset.prepared_root,
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
        resolved_ap02_reference_marker_id=7,
        resolved_evaluation_anchor_marker_id=7,
    )
    argv = evaluators.get("marker_consistency").commands(context)[0].argv
    expected = {
        "--anchor-marker-id": "7",
        "--reprojection-threshold-px": "2.25",
        "--min-inliers": "6",
        "--ransac-iters": "456",
        "--min-triangulation-angle-deg": "1.5",
        "--max-moving-observations-per-marker": "42",
    }
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
