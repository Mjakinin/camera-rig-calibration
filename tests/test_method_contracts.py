from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.contracts import RunContext
from camera_rig_calibration.registry import calibration_methods


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
        "ap02_static_initialization",
        "ap02_static_ba",
        "ap02_combined_initialization",
        "ap02_combined_ba",
        "ap02_report",
    ]
    graph = commands[0].argv
    assert graph[graph.index("--cameras") + 1] == "front-left,roof.camera"
    static_ba = commands[2].argv
    combined_ba = commands[4].argv
    assert static_ba[static_ba.index("--max-nfev") + 1] == "100"
    assert combined_ba[combined_ba.index("--max-nfev") + 1] == "120"
    flattened = " ".join(
        token for command in commands for token in command.argv
    )
    assert "moving-selection" not in flattened
    assert "max-moving-frames" not in flattened
    assert "top-per-marker" not in flattened
    assert commands[1].diagnostic is True
    assert commands[2].diagnostic is True
    assert commands[4].depends_on == ("ap02_combined_initialization",)


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


def test_ap01_uses_every_quality_accepted_observation(
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
    assert "top-moving" not in flattened
    assert "top-per-marker" not in flattened
    assert commands[-1].depends_on == ("ap01_solve_extrinsics",)
