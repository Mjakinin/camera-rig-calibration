from __future__ import annotations

import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import typer
from rich.console import Console

from camera_rig_calibration.inventory import BASELINE_SIMULATION_PARAMETERS
from camera_rig_calibration.policies.simulation_ui_consistency_policy import (
    _fresh_capture_experiment_id,
    _simulation_summary_parameters,
)
from camera_rig_calibration.ui.wizard_simulation_parameters import (
    _edit_simulation_parameters,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_fresh_capture_id_keeps_existing_immutable_experiment_distinct() -> None:
    assert _fresh_capture_experiment_id(
        "fov_120deg",
        "capture_20260815_210352_123456",
        {"fov_120deg"},
    ) == "fov_120deg__capture_20260815_210352_123456"
    assert _fresh_capture_experiment_id(
        "fov_120deg",
        "capture_20260815_210352_123456",
        {
            "fov_120deg",
            "fov_120deg__capture_20260815_210352_123456",
        },
    ) == "fov_120deg__capture_20260815_210352_123456_2"


def test_summary_parameter_helper_includes_capture_timings() -> None:
    simulation = SimpleNamespace(
        settle_seconds=0.35,
        post_pose_skip=5,
        frame_timeout_seconds=3.0,
        startup_timeout_seconds=60.0,
    )

    resolved = _simulation_summary_parameters(
        {"route": "route2"}, simulation
    )

    assert resolved["settle_seconds"] == 0.35
    assert resolved["post_pose_skip"] == 5
    assert resolved["frame_timeout_seconds"] == 3.0
    assert resolved["startup_timeout_seconds"] == 60.0


def test_parameter_editor_preserves_existing_capture_values_without_duplicates(
    monkeypatch,
) -> None:
    parameters = {
        **BASELINE_SIMULATION_PARAMETERS,
        "settle_seconds": 0.8,
        "post_pose_skip": 7,
        "frame_timeout_seconds": 4.0,
        "startup_timeout_seconds": 90.0,
    }
    route = (
        REPOSITORY
        / "src/calib_lab/bus_real_data/config/"
        "moving_camera_route2_interpolated_final.json"
    )
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: "",
    )
    stream = StringIO()

    resolved, _, capture = _edit_simulation_parameters(
        REPOSITORY,
        Console(file=stream, force_terminal=False, width=220),
        parameters,
        route,
    )

    assert capture == {
        "settle_seconds": 0.8,
        "post_pose_skip": 7,
        "frame_timeout_seconds": 4.0,
        "startup_timeout_seconds": 90.0,
    }
    assert all(
        key not in resolved
        for key in (
            "settle_seconds",
            "post_pose_skip",
            "frame_timeout_seconds",
            "startup_timeout_seconds",
        )
    )
    final_vector = stream.getvalue().split(
        "Complete resolved simulation parameter vector", 1
    )[-1]
    for key in capture:
        assert final_vector.count(key) == 1


def test_product_stack_applies_consistency_to_real_wizard_paths() -> None:
    script = textwrap.dedent(
        """
        from io import StringIO
        from pathlib import Path
        from types import SimpleNamespace

        import typer
        from rich.console import Console

        from camera_rig_calibration.application.bootstrap import install_product_stack
        from camera_rig_calibration.config.models import SceneType
        from camera_rig_calibration.inventory import (
            BASELINE_SIMULATION_PARAMETERS,
            SimulationExperimentSummary,
        )

        install_product_stack()

        import camera_rig_calibration.wizard as wizard
        import camera_rig_calibration.ui.wizard_simulation as simulation_ui

        seen = {}
        def prompt(*args, **kwargs):
            seen.update(kwargs)
            return "b"
        typer.prompt = prompt
        assert wizard._prompt_index(
            "Base number (0/b = back)", default=1, maximum=3
        ) is None
        assert seen["type"] is str
        assert seen["default"] == "1"

        existing = SimulationExperimentSummary(
            variant="route2",
            factor="baseline",
            value="baseline",
            moving_frames=189,
            has_results=True,
            dataset_root=None,
            parameters=dict(BASELINE_SIMULATION_PARAMETERS),
        )
        class Hooks:
            @staticmethod
            def discover_simulation_experiments(repository_root):
                return [existing]
        simulation_ui.current_wizard_bindings = lambda: Hooks()
        job = simulation_ui._simulation_job_from_parameters(
            Path.cwd(),
            dict(BASELINE_SIMULATION_PARAMETERS),
            experiment_id="route2",
            prepared_root=None,
            source="new capture",
        )
        assert job.experiment_id.startswith("route2__capture_")
        assert job.prepared_root is None

        config = SimpleNamespace(
            dataset=SimpleNamespace(
                id="sim", scene_type=SceneType.SIMULATION
            ),
            static_cameras=[SimpleNamespace(id="cam_edge_0")],
            moving_camera=SimpleNamespace(
                id="moving_calib_camera",
                intrinsic_calibration_video=None,
                intrinsic_calibration_images=None,
                intrinsics_profile=None,
                intrinsic_scan=SimpleNamespace(mode="balanced"),
                intrinsics=None,
            ),
            sampling=SimpleNamespace(target_hz=None),
            methods=SimpleNamespace(enabled=[]),
            project=SimpleNamespace(execution_mode="prepare_only"),
            simulation=SimpleNamespace(
                enabled=True,
                route_name="route2",
                moving_width=1280,
                moving_height=720,
                moving_hfov_deg=69.1,
                lighting="baseline",
                lighting_scale=1.0,
                motion_blur_kernel=0,
                motion_blur_angle_deg=0.0,
                target_route_frames=189,
                route_sampling_strategy="original_route_poses",
                settle_seconds=0.35,
                post_pose_skip=5,
                frame_timeout_seconds=3.0,
                startup_timeout_seconds=60.0,
            ),
            evaluation=SimpleNamespace(anchor_marker_id=14),
        )
        output = StringIO()
        wizard.show_summary(
            config,
            Path("sim.yaml"),
            Console(file=output, force_terminal=False, width=220),
        )
        rendered = output.getvalue()
        assert "settle=0.35 s" in rendered
        assert "skip=5" in rendered
        assert "timeouts=3.0/60.0 s" in rendered
        assert "settle=?" not in rendered
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        check=True,
    )
