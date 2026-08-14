from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import typer
import pytest
from rich.console import Console

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import (
    DatasetSettings,
    RigConfig,
    MovingCameraSettings,
    SimulationSettings,
    StaticCameraSettings,
)
from camera_rig_calibration.inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    SimulationExperimentSummary,
    discover_simulation_experiments,
)
from camera_rig_calibration.queueing import (
    SelectionReviewJob,
    load_batch,
    load_queue,
)
from camera_rig_calibration.observations import ResolvedSelections
import camera_rig_calibration.wizard as wizard_module
from camera_rig_calibration.wizard import (
    METHOD_JOB_GROUPS,
    SelectionDatasetContext,
    SimulationQueueJob,
    _build_simulation_batch_outcome,
    _clone_method_job,
    _edit_method_job,
    _edit_simulation_parameters,
    _method_queue,
    _new_method_job,
    _show_method_queue,
    _simulation_experiment_id,
    _simulation_job_from_parameters,
    _setting_rows,
    _simulation_input,
    review_queue_selection_candidates,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _prepared_simulation_dataset(root: Path) -> Path:
    raw = root / "raw_images"
    (raw / "static").mkdir(parents=True)
    (raw / "moving").mkdir()
    (raw / "camera_info").mkdir()
    (raw / "static/cam1.png").write_bytes(b"fixture")
    (raw / "moving/frame_000001.png").write_bytes(b"fixture")
    intrinsic = {
        "width": 640,
        "height": 480,
        "K": [500, 0, 320, 0, 500, 240, 0, 0, 1],
        "D": [0, 0, 0, 0, 0],
    }
    (raw / "camera_info/cam1.json").write_text(
        json.dumps({**intrinsic, "camera_name": "cam1"}),
        encoding="utf-8",
    )
    (raw / "camera_info/moving_calib_camera.json").write_text(
        json.dumps(
            {**intrinsic, "camera_name": "moving_calib_camera"}
        ),
        encoding="utf-8",
    )
    return root


def test_baseline_simulation_shows_catalogue_without_claiming_missing_input(
    monkeypatch,
) -> None:
    prompts = iter(["1"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=240)

    cameras, moving, simulation, _, reused = _simulation_input(REPOSITORY, console)

    output = stream.getvalue()
    assert "Existing simulation experiments" in output
    assert "Complete parameter vector" in output
    assert "route2" in output.lower()
    assert "route=route2" in output
    assert reused is None
    assert simulation.capture_id is not None
    assert simulation.moving_width == 1280
    assert simulation.moving_height == 720
    assert simulation.moving_hfov_deg == 69.1
    assert simulation.target_route_frames == 189
    assert len(cameras) == 4
    assert moving.id == "moving_calib_camera"


def test_custom_simulation_world_is_visible_but_unavailable(monkeypatch) -> None:
    prompts = iter(["4", "1"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    stream = StringIO()

    _, _, simulation, _, _ = _simulation_input(
        REPOSITORY,
        Console(file=stream, force_terminal=False, width=240),
    )

    output = stream.getvalue()
    assert "Change simulation world — Coming soon" in output
    assert "no SDF file or path has been changed" in output
    assert simulation.world_id == "bus"


def test_existing_ablation_with_local_input_can_be_reused(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = _prepared_simulation_dataset(tmp_path / "prepared")
    parameters = {
        **BASELINE_SIMULATION_PARAMETERS,
        "moving_hfov_deg": 100.0,
    }
    entry = SimulationExperimentSummary(
        variant="fov_100deg",
        factor="moving-camera horizontal FOV",
        value="100 deg",
        moving_frames=1,
        has_results=True,
        dataset_root=prepared,
        parameters=parameters,
    )
    monkeypatch.setattr(
        wizard_module,
        "discover_simulation_experiments",
        lambda *args, **kwargs: [entry],
    )
    prompts = iter(["2", "1"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(prompts))
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=240)

    _, _, simulation, suggested_id, reused = _simulation_input(REPOSITORY, console)

    assert simulation.moving_hfov_deg == 100.0
    assert suggested_id == "fov_100deg"
    assert reused == prepared
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

    job.methods.ap01.method_contract = "recommended_wizard_v1"
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


def test_new_input_manual_reference_is_deferred_without_free_id_prompt(
    monkeypatch,
) -> None:
    register_builtin_components()
    job = _new_method_job("ap02", prompt_for_single_marker=False)
    reference_row = next(
        index
        for index, row in enumerate(
            _setting_rows(job, METHOD_JOB_GROUPS), 1
        )
        if row[0] == "ap02_reference_mode"
    )
    responses = iter([str(reference_row), "3"])
    prompts: list[str] = []

    def prompt(label, *args, **kwargs):
        prompts.append(label)
        return next(responses)

    monkeypatch.setattr(typer, "prompt", prompt)
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)

    edited = _edit_method_job(
        Console(file=StringIO(), force_terminal=False, width=220),
        job,
    )

    assert edited.selection.mode == "review_once"
    assert edited.methods.ap02.reference_marker_id == "auto"
    assert not any(
        "Reference marker (b = back)" in label for label in prompts
    )


def test_queue_distinguishes_auto_and_manual_reference_jobs() -> None:
    register_builtin_components()
    automatic = _new_method_job(
        "ap02", prompt_for_single_marker=False
    )
    automatic.methods = automatic.methods.model_copy(
        update={
            "ap02": automatic.methods.ap02.model_copy(
                update={
                    "combined_ba_max_function_evaluations": 60
                }
            )
        },
        deep=True,
    )
    manual = _clone_method_job(automatic, automatic.label)
    manual.deferred_selection_keys.add("ap02_reference")
    manual.selection = manual.selection.model_copy(
        update={"mode": "review_once"}
    )
    identical_manual = _clone_method_job(manual, manual.label)
    stream = StringIO()

    _show_method_queue(
        Console(file=stream, force_terminal=False, width=300),
        [automatic, manual, identical_manual],
    )

    output = stream.getvalue()
    assert automatic.label.startswith("combined_nfev_60")
    assert manual.label == wizard_module._method_job_label(
        manual, "new_dataset"
    )
    assert manual.label.endswith("__ref_manual")
    assert "ref=manual after preflight" in output
    assert output.count("independent") == 2
    assert "exact duplicate of row 2; skipped after first" in output


def test_ap03_deferred_single_choice_survives_multi_auto_choice(
    monkeypatch,
) -> None:
    register_builtin_components()
    job = _new_method_job("ap03", prompt_for_single_marker=False)
    job.deferred_selection_keys.add("single_marker")
    job.selection = job.selection.model_copy(update={"mode": "review_once"})
    rows = _setting_rows(job, METHOD_JOB_GROUPS)
    multi_row = next(
        index
        for index, row in enumerate(rows, 1)
        if row[0] == "multi_markers"
    )
    response_values = [str(multi_row), "1"]
    if getattr(wizard_module, "_AP03_CAMERA_MODEL_SENSITIVITY_POLICY", False):
        response_values.insert(0, "1")
    responses = iter(response_values)
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    console = Console(file=StringIO(), force_terminal=False, width=220)

    _edit_method_job(console, job)

    assert job.selection.mode == "review_once"
    assert job.deferred_selection_keys == {"single_marker"}
    assert job.methods.ap03.multi.marker_ids == "auto"


def test_prepared_reference_marker_uses_filtered_numbered_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    register_builtin_components()
    job = _new_method_job("ap02", prompt_for_single_marker=False)
    reference_row = next(
        index
        for index, row in enumerate(
            _setting_rows(job, METHOD_JOB_GROUPS), 1
        )
        if row[0] == "ap02_reference_mode"
    )
    context = SelectionDatasetContext(
        key="prepared",
        display_name="prepared",
        dataset_root=tmp_path / "prepared",
        static_cameras=(
            StaticCameraSettings(id="front"),
            StaticCameraSettings(id="rear"),
        ),
    )
    resolved = ResolvedSelections(
        root_camera="front",
        ap02_reference_marker_id=7,
        ap03_single_scale_marker_id=7,
        ap03_multi_marker_ids=(7,),
        evaluation_anchor_marker_id=None,
        marker_ids=(3, 7),
        payload={
            "ap02_reference_marker": {
                "candidates": [
                    {
                        "id": 3,
                        "compatible": False,
                        "recommended": False,
                        "combined_graph_reachable_static_count": 1,
                        "moving_frames": 1,
                        "accepted_observations": 2,
                        "median_pnp_reprojection_rmse_px": 0.4,
                    },
                    {
                        "id": 7,
                        "compatible": True,
                        "recommended": True,
                        "combined_graph_reachable_static_count": 2,
                        "moving_frames": 4,
                        "accepted_observations": 10,
                        "median_pnp_reprojection_rmse_px": 0.2,
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(
        wizard_module,
        "_preview_prepared_selections",
        lambda *args, **kwargs: resolved,
    )
    responses = iter(
        [
            str(reference_row),
            "3",  # manual
            "1",  # incompatible row; must be rejected locally
            "2",  # compatible marker 7
        ]
    )
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)

    edited = _edit_method_job(
        Console(file=StringIO(), force_terminal=False, width=220),
        job,
        selection_contexts=(context,),
    )

    assert edited.selection.mode == "explicit"
    assert edited.methods.ap02.reference_marker_id == 7
    assert edited.context_methods["prepared"].ap02.reference_marker_id == 7


def test_queue_review_returns_distinct_decisions_for_each_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolved = ResolvedSelections(
        root_camera="front",
        ap02_reference_marker_id=7,
        ap03_single_scale_marker_id=7,
        ap03_multi_marker_ids=(7,),
        evaluation_anchor_marker_id=None,
        marker_ids=(7,),
        payload={},
    )
    reviews = (
        SelectionReviewJob(
            entry_id="ap02_strict",
            config=RigConfig(
                dataset=DatasetSettings(
                    id="prepared", prepared_root=tmp_path
                ),
                static_cameras=[StaticCameraSettings(id="front")],
                methods=_new_method_job(
                    "ap02", prompt_for_single_marker=False
                ).methods
            ),
            selections=resolved,
            output_directory=tmp_path / "strict",
        ),
        SelectionReviewJob(
            entry_id="ap02_relaxed",
            config=RigConfig(
                dataset=DatasetSettings(
                    id="prepared", prepared_root=tmp_path
                ),
                static_cameras=[StaticCameraSettings(id="front")],
                methods=_new_method_job(
                    "ap02", prompt_for_single_marker=False
                ).methods
            ),
            selections=resolved,
            output_directory=tmp_path / "relaxed",
        ),
    )
    calls: list[str] = []

    def fake_review(config, selections, run_directory, console):
        calls.append(str(run_directory))
        marker = 7 if len(calls) == 1 else 9
        return {"ap02_reference_marker_id": marker}

    monkeypatch.setattr(
        wizard_module, "review_selection_candidates", fake_review
    )

    decisions = review_queue_selection_candidates(
        reviews,
        tmp_path / "preflight",
        Console(file=StringIO(), force_terminal=False, width=220),
    )

    assert decisions == {
        "ap02_strict": {"ap02_reference_marker_id": 7},
        "ap02_relaxed": {"ap02_reference_marker_id": 9},
    }
    assert len(calls) == 2


def test_method_editor_rows_exclude_queue_wide_aruco_and_evaluation() -> None:
    register_builtin_components()
    job = _new_method_job("ap03", prompt_for_single_marker=False)

    groups = {
        row[1] for row in _setting_rows(job, METHOD_JOB_GROUPS)
    }

    assert groups == {
        "OBSERVATION QUALITY OVERRIDE",
        "METHOD-SPECIFIC SETTINGS",
        "COLMAP SETTINGS",
    }
    assert "ARUCO INPUT" not in groups
    assert "COMMON EVALUATION" not in groups


def test_default_method_selection_builds_three_independent_product_jobs(
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
    assert all(
        job.label == wizard_module._method_job_label(job) for job in jobs
    )
    assert all(
        job.observation_quality.maximum_pnp_reprojection_error_px == 25.0
        for job in jobs
    )


def test_method_multiselect_preserves_duplicate_rows(monkeypatch) -> None:
    register_builtin_components()
    responses = iter(["1,2,3,3", "1"])
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: next(responses),
    )

    jobs = _method_queue(
        Console(file=StringIO(), force_terminal=False, width=180)
    )

    assert [job.method_id for job in jobs] == [
        "ap01",
        "ap02",
        "ap03",
        "ap03",
    ]
    assert all(
        job.label == wizard_module._method_job_label(job) for job in jobs
    )
    assert jobs[2].label == jobs[3].label


def test_historical_simulation_without_local_input_requires_capture() -> None:
    job = _simulation_job_from_parameters(
        REPOSITORY,
        BASELINE_SIMULATION_PARAMETERS,
        experiment_id="historical_missing",
        prepared_root=None,
        source="historical result",
    )

    assert job.input_mode == "new capture required"
    assert job.simulation.enabled
    assert job.simulation.capture_id is not None


def test_simulation_batch_is_experiments_times_method_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    register_builtin_components()
    prepared_roots = []
    for name in ("route2", "fov_100deg"):
        root = tmp_path / "prepared" / name
        root.mkdir(parents=True)
        prepared_roots.append(root)
    camera = StaticCameraSettings(id="cam_edge_0")
    moving = MovingCameraSettings(id="moving_calib_camera")
    jobs = [
        SimulationQueueJob(
            experiment_id=name,
            parameters={
                **BASELINE_SIMULATION_PARAMETERS,
                "moving_hfov_deg": fov,
            },
            cameras=(camera,),
            moving_camera=moving,
            simulation=SimulationSettings(
                enabled=False,
                world_id="bus",
                moving_hfov_deg=fov,
            ),
            prepared_root=prepared,
            source=f"existing {name}",
        )
        for name, fov, prepared in (
            ("route2", 69.1, prepared_roots[0]),
            ("fov_100deg", 100.0, prepared_roots[1]),
        )
    ]
    method_jobs = [
        _new_method_job("ap02", prompt_for_single_marker=False),
        _new_method_job("ap03", prompt_for_single_marker=False),
    ]
    monkeypatch.setattr(
        wizard_module,
        "_method_queue",
        lambda console, selection_contexts=(): method_jobs,
    )
    responses = iter(["1", "paper_batch"])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )

    outcome = _build_simulation_batch_outcome(
        tmp_path,
        Console(file=StringIO(), force_terminal=False, width=200),
        jobs,
    )

    assert outcome.batch_path is not None
    assert len(outcome.runs) == 4
    batch = load_batch(outcome.batch_path)
    assert [entry.experiment_id for entry in batch.queues] == [
        "route2",
        "fov_100deg",
    ]
    queue_ids = [
        [entry.id for entry in load_queue(item.queue).entries]
        for item in batch.queues
    ]
    assert all(len(entries) == 2 for entries in queue_ids)
    assert all(
        entries[0].endswith(f"ap02__{method_jobs[0].label}__01")
        and entries[1].endswith(f"ap03__{method_jobs[1].label}__02")
        for entries in queue_ids
    )


def test_queue_wide_aruco_menu_updates_every_job(monkeypatch) -> None:
    register_builtin_components()
    responses = iter(["1,2", "5", "1,4", "3", "0.2", "1"])
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
    assert [job.markers.detection_mode for job in jobs] == [
        "high_sensitivity",
        "high_sensitivity",
    ]


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
    original_marker_ids = original.methods.ap03.multi.marker_ids
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

    assert original.methods.ap03.multi.marker_ids == original_marker_ids
    assert duplicate.methods.ap03.multi.marker_ids == [7, 9]


def test_remove_jobs_accepts_comma_separated_rows(monkeypatch) -> None:
    register_builtin_components()
    responses = iter(["1,2,3", "7", "1,3", "1"])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    console = Console(file=StringIO(), force_terminal=False, width=180)

    jobs = _method_queue(console)

    assert [job.method_id for job in jobs] == ["ap02"]
