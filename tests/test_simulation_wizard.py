from __future__ import annotations

from io import StringIO
from pathlib import Path

import typer
import pytest
from rich.console import Console

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.inventory import discover_simulation_experiments
from camera_rig_calibration.wizard import (
    METHOD_JOB_GROUPS,
    WizardBack,
    _clone_method_job,
    _edit_method_job,
    _edit_simulation_parameters,
    _method_queue,
    _new_method_job,
    _prompt_path,
    _simulation_experiment_id,
    _setting_rows,
    _simulation_input,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_baseline_simulation_shows_catalogue_and_reuses_existing_capture(
    monkeypatch,
) -> None:
    prompts = iter(["1"])
    confirmations = iter([True, False])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(
        typer, "confirm", lambda *args, **kwargs: next(confirmations)
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=240)

    cameras, moving, simulation, _, reused = _simulation_input(REPOSITORY, console)

    output = stream.getvalue()
    assert "Existing simulation experiments" in output
    assert "Complete parameter vector" in output
    assert "fov_100deg" in output
    assert "moving_blur_k21_strong" in output
    assert "route=route2" in output
    baseline = next(
        entry
        for entry in discover_simulation_experiments(REPOSITORY)
        if entry.variant == "route2"
    )
    assert reused == baseline.dataset_root
    assert reused is not None
    assert reused.is_relative_to(REPOSITORY / "datasets/simulation")
    assert simulation.moving_width == 1280
    assert simulation.moving_height == 720
    assert simulation.moving_hfov_deg == 69.1
    assert simulation.target_route_frames == 189
    assert len(cameras) == 4
    assert moving.id == "moving_calib_camera"


def test_existing_ablation_can_be_selected_directly_by_number(monkeypatch) -> None:
    entries = discover_simulation_experiments(REPOSITORY)
    fov_number = next(
        index for index, entry in enumerate(entries, 1) if entry.variant == "fov_100deg"
    )
    prompts = iter(["2", str(fov_number)])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(prompts))
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=240)

    _, _, simulation, suggested_id, reused = _simulation_input(REPOSITORY, console)

    assert simulation.moving_hfov_deg == 100.0
    assert suggested_id == "fov_100deg"
    selected = entries[fov_number - 1]
    assert reused == selected.dataset_root
    assert reused is not None
    assert reused.is_relative_to(REPOSITORY / "datasets/simulation")
    assert "Selected existing simulation" in stream.getvalue()


def test_new_combination_always_gets_a_new_capture_id(monkeypatch) -> None:
    prompts = iter(["3", ""])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(prompts)
    )
    monkeypatch.setattr(
        typer,
        "confirm",
        lambda *args, **kwargs: pytest.fail(
            "new combinations must not offer captured-input reuse"
        ),
    )
    stream = StringIO()

    _, _, simulation, _, reused = _simulation_input(
        REPOSITORY,
        Console(file=stream, force_terminal=False, width=240),
    )

    assert reused is None
    assert simulation.preset == "bus_composed"
    assert simulation.capture_id is not None
    assert simulation.capture_id.startswith("capture_")
    assert "will record a new input capture" in stream.getvalue()


def test_zero_at_a_path_prompt_goes_back(monkeypatch) -> None:
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: "0")

    with pytest.raises(WizardBack):
        _prompt_path("Gazebo SDF world", directory=False)


def test_simulation_experiment_ids_are_stable_and_describe_combinations() -> None:
    assert _simulation_experiment_id(
        {
            "route": "route2",
            "moving_width": 1280,
            "moving_height": 720,
            "moving_hfov_deg": 69.1,
            "lighting": "baseline",
            "lighting_scale": 1.0,
            "motion_blur_kernel": 0,
            "motion_blur_angle_deg": 0.0,
            "target_route_frames": 189,
        }
    ) == "route2"
    assert _simulation_experiment_id(
        {
            "route": "route2",
            "moving_width": 640,
            "moving_height": 360,
            "moving_hfov_deg": 100.0,
            "lighting": "bright",
            "lighting_scale": 1.0,
            "motion_blur_kernel": 9,
            "motion_blur_angle_deg": 15.0,
            "target_route_frames": 120,
        }
    ).startswith(
        "res_640x360__fov_100deg__light_bright_1x__blur_k9_15deg"
    )


def test_sequential_only_settings_are_contextual() -> None:
    register_builtin_components()
    job = _new_method_job("ap01", prompt_for_single_marker=False)
    exhaustive_keys = {row[0] for row in _setting_rows(job)}
    assert "sequential_overlap" not in exhaustive_keys
    assert "loop_detection" not in exhaustive_keys

    job.colmap = job.colmap.model_copy(update={"matcher": "sequential"})
    sequential_keys = {row[0] for row in _setting_rows(job)}
    assert "sequential_overlap" in sequential_keys
    assert "loop_detection" in sequential_keys


def test_method_editor_shows_matcher_choice_and_accepts_unique_prefix(
    monkeypatch,
) -> None:
    register_builtin_components()
    job = _new_method_job("ap03", prompt_for_single_marker=False)
    matcher_row = next(
        index
        for index, row in enumerate(
            _setting_rows(job, METHOD_JOB_GROUPS), 1
        )
        if row[0] == "matcher"
    )
    responses = iter([str(matcher_row), "se"])
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    stream = StringIO()

    edited = _edit_method_job(
        Console(file=stream, force_terminal=False, width=220),
        job,
    )

    assert edited.colmap.matcher == "sequential"


def test_method_editor_rows_exclude_queue_wide_aruco_and_evaluation() -> None:
    register_builtin_components()
    job = _new_method_job("ap03", prompt_for_single_marker=False)

    groups = {
        row[1] for row in _setting_rows(job, METHOD_JOB_GROUPS)
    }

    assert groups == {
        "OBSERVATION QUALITY",
        "METHOD-SPECIFIC SETTINGS",
        "COLMAP SETTINGS",
    }
    assert "ARUCO INPUT" not in groups
    assert "COMMON EVALUATION" not in groups


def test_default_method_selection_builds_three_independent_baseline_jobs(
    monkeypatch,
) -> None:
    register_builtin_components()
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: kwargs.get("default", ""),
    )
    console = Console(file=StringIO(), force_terminal=False, width=180)

    jobs = _method_queue(console)

    assert [job.method_id for job in jobs] == ["ap01", "ap02", "ap03"]
    assert len({job.label for job in jobs}) == 3
    assert all(
        job.observation_quality.maximum_pnp_reprojection_error_px == 25.0
        for job in jobs
    )


def test_queue_wide_aruco_menu_updates_every_job(monkeypatch) -> None:
    register_builtin_components()
    responses = iter(["1,2", "6", "3", "0.2", "1"])
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)

    jobs = _method_queue(
        Console(file=StringIO(), force_terminal=False, width=180)
    )

    assert [job.markers.length_m for job in jobs] == [0.2, 0.2]


def test_simulation_multiselect_reprompts_only_the_invalid_blur_value(
    monkeypatch,
) -> None:
    responses = iter(["10", 2, 3, ""])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    route = (
        REPOSITORY
        / "src/calib_lab/bus_real_data/config/"
        "moving_camera_route2_interpolated_final.json"
    )
    parameters = {
        "route": "route2",
        "moving_width": 1280,
        "moving_height": 720,
        "moving_hfov_deg": 69.1,
        "lighting": "baseline",
        "lighting_scale": 1.0,
        "motion_blur_kernel": 0,
        "motion_blur_angle_deg": 0.0,
        "target_route_frames": 189,
        "route_sampling_strategy": "original_route_poses",
    }
    console = Console(file=StringIO(), force_terminal=False, width=220)

    resolved, _, _ = _edit_simulation_parameters(
        REPOSITORY, console, parameters, route
    )

    assert resolved["motion_blur_kernel"] == 3


def test_simulation_parameter_back_keeps_values_and_redraws_table(
    monkeypatch,
) -> None:
    responses = iter(["10", "back", ""])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    route = (
        REPOSITORY
        / "src/calib_lab/bus_real_data/config/"
        "moving_camera_route2_interpolated_final.json"
    )
    parameters = {
        "route": "route2",
        "moving_width": 1280,
        "moving_height": 720,
        "moving_hfov_deg": 69.1,
        "lighting": "baseline",
        "lighting_scale": 1.0,
        "motion_blur_kernel": 0,
        "motion_blur_angle_deg": 0.0,
        "target_route_frames": 189,
        "route_sampling_strategy": "original_route_poses",
    }

    resolved, _, _ = _edit_simulation_parameters(
        REPOSITORY,
        Console(file=StringIO(), force_terminal=False, width=220),
        parameters,
        route,
    )

    assert resolved["motion_blur_kernel"] == 0


def test_method_job_duplicate_is_a_deep_snapshot() -> None:
    register_builtin_components()
    original = _new_method_job("ap03", prompt_for_single_marker=False)
    duplicate = _clone_method_job(original, "ap03_variant")
    duplicate.methods = duplicate.methods.model_copy(
        update={
            "ap03": duplicate.methods.ap03.model_copy(
                update={
                    "multi": duplicate.methods.ap03.multi.model_copy(
                        update={"marker_ids": [7, 9]}
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )

    assert original.methods.ap03.multi.marker_ids == "auto"
    assert duplicate.methods.ap03.multi.marker_ids == [7, 9]


def test_remove_jobs_accepts_comma_separated_rows(monkeypatch) -> None:
    register_builtin_components()
    responses = iter(["1,2,3", "8", "1,3", "1"])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    console = Console(file=StringIO(), force_terminal=False, width=180)

    jobs = _method_queue(console)

    assert [job.method_id for job in jobs] == ["ap02"]
