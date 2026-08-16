from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def test_product_stack_separates_capture_reliability_from_experiment_factors() -> None:
    script = textwrap.dedent(
        """
        from io import StringIO
        from pathlib import Path

        import typer
        from rich.console import Console

        from camera_rig_calibration.application.bootstrap import install_product_stack
        from camera_rig_calibration.inventory import BASELINE_SIMULATION_PARAMETERS

        install_product_stack()

        import camera_rig_calibration.ui.wizard_simulation as simulation_ui

        typer.prompt = lambda *args, **kwargs: ""
        confirm_labels = []
        def confirm(label, *args, **kwargs):
            confirm_labels.append(label)
            return False
        typer.confirm = confirm

        repository = Path.cwd()
        route = (
            repository
            / "src/calib_lab/bus_real_data/config/"
            "moving_camera_route2_interpolated_final.json"
        )
        stream = StringIO()
        resolved, _, capture = simulation_ui._edit_simulation_parameters(
            repository,
            Console(file=stream, force_terminal=False, width=240),
            dict(BASELINE_SIMULATION_PARAMETERS),
            route,
        )

        rendered = stream.getvalue()
        editable_table = rendered.split(
            "Complete resolved simulation parameter vector", 1
        )[0]
        capture_keys = (
            "settle_seconds",
            "post_pose_skip",
            "frame_timeout_seconds",
            "startup_timeout_seconds",
        )
        for key in capture_keys:
            assert key not in editable_table
            assert key not in resolved

        assert capture == {
            "settle_seconds": 0.35,
            "post_pose_skip": 5,
            "frame_timeout_seconds": 3.0,
            "startup_timeout_seconds": 60.0,
        }
        assert "Advanced capture reliability settings" in rendered
        assert "not experiment factors" in rendered
        assert "Every planned route pose still produces one saved image" in rendered
        assert confirm_labels == ["Open advanced capture reliability settings?"]
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        check=True,
    )
