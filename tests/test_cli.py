from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from rich.console import Console

from camera_rig_calibration.config import save_config
from camera_rig_calibration.wizard import _choice, show_summary


REPOSITORY = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(REPOSITORY / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_main_menu_is_the_default_interface() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "camera_rig_calibration"],
        cwd=REPOSITORY,
        env=_environment(),
        input="0\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "CAMERA RIG CALIBRATION" in process.stdout
    assert "Start a new calibration" in process.stdout
    assert "View results" in process.stdout
    assert "Manage incomplete runs" in process.stdout
    assert "Check installation" in process.stdout
    assert "Cleanup storage" in process.stdout
    assert "Manage intrinsics profiles" in process.stdout
    assert "Repeat a saved setup" not in process.stdout
    assert "Advanced experiments" not in process.stdout


def test_noninteractive_config_dry_run(prepared_config, tmp_path: Path) -> None:
    path = save_config(prepared_config, tmp_path / "paper_run.yaml")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "camera_rig_calibration",
            "--config",
            str(path),
            "--dry-run",
        ],
        cwd=REPOSITORY,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    assert "Calibration pipeline plan" in process.stdout
    assert "Dry run complete" in process.stdout
    assert not prepared_config.project.output_root.exists()


def test_numbered_choices_are_visible_before_prompt(monkeypatch, capsys) -> None:
    import typer

    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: "4")
    selected = _choice(
        "Input type",
        {
            "1": "existing prepared dataset (reuse frames; no capture)",
            "4": "new Gazebo simulation capture",
        },
        "1",
    )
    output = capsys.readouterr().out
    assert selected == "4"
    assert "1. existing prepared dataset" in output
    assert "4. new Gazebo simulation capture" in output


def test_normal_summary_does_not_expose_scene_metadata(
    prepared_config, tmp_path: Path
) -> None:
    stream = StringIO()

    show_summary(
        prepared_config,
        tmp_path / "config.yaml",
        Console(file=stream, force_terminal=False, width=180),
    )

    assert "Scene metadata" not in stream.getvalue()
    assert "interior" not in stream.getvalue()
    assert "exterior" not in stream.getvalue()
