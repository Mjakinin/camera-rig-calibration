from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Sequence

import pytest
import typer
from pydantic import BaseModel, ConfigDict
from rich.console import Console

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.contracts import CommandSpec, RequirementResult, RunContext
from camera_rig_calibration.registry import (
    ComponentRegistry,
    calibration_methods,
    reset_registries,
)
from camera_rig_calibration.wizard import (
    METHOD_JOB_GROUPS,
    _edit_method_job,
    _method_queue,
    _new_method_job,
    _setting_rows,
)


class DummyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 1


@dataclass(frozen=True)
class DummyMethod:
    id: str = "paper_dummy"
    display_name: str = "Paper dummy method"
    config_model: type[BaseModel] = DummyOptions

    def requirements(self, context: RunContext) -> RequirementResult:
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()

    def collect(self, context: RunContext) -> dict:
        return {"status": "DUMMY"}


@pytest.fixture(autouse=True)
def _restore_builtin_registries():
    yield
    reset_registries()
    register_builtin_components()


def test_registry_rejects_duplicate_ids() -> None:
    registry = ComponentRegistry[DummyMethod]("dummy")
    registry.register(DummyMethod())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(DummyMethod())


def test_new_method_is_discoverable_without_menu_changes() -> None:
    register_builtin_components()
    if "paper_dummy" not in calibration_methods:
        calibration_methods.register(DummyMethod())
    assert calibration_methods.get("paper_dummy").display_name == "Paper dummy method"


def test_registered_method_appears_in_queue_and_has_validated_yaml_editor(
    monkeypatch,
) -> None:
    register_builtin_components()
    if "paper_dummy" not in calibration_methods:
        calibration_methods.register(DummyMethod())
    choices: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda value="", **kwargs: choices.append(str(value)))
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: kwargs.get("default", ""),
    )

    jobs = _method_queue(
        Console(file=StringIO(), force_terminal=False, width=180)
    )

    assert [job.method_id for job in jobs] == ["ap01", "ap02", "ap03"]
    assert any("PAPER_DUMMY" in value for value in choices)

    extension = _new_method_job(
        "paper_dummy", prompt_for_single_marker=False
    )
    rows = _setting_rows(extension, METHOD_JOB_GROUPS)
    extension_row = next(
        index
        for index, row in enumerate(rows, 1)
        if row[0] == "extension"
    )
    responses = iter([str(extension_row), "{seed: 7}"])
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)

    edited = _edit_method_job(
        Console(file=StringIO(), force_terminal=False, width=180),
        extension,
    )

    assert edited.methods.extensions["paper_dummy"] == {"seed": 7}


def test_schema_v5_has_no_active_frame_selection_registry() -> None:
    import camera_rig_calibration.registry as registry

    assert not hasattr(registry, "frame_selection_policies")


def test_gui_exposes_every_supported_scientific_parameter_group() -> None:
    register_builtin_components()
    common = {
        "quality_reprojection",
        "quality_area",
        "quality_distance",
    }
    ap01 = {
        row[0]
        for row in _setting_rows(
            _new_method_job("ap01", prompt_for_single_marker=False),
            METHOD_JOB_GROUPS,
        )
    }
    ap02 = {
        row[0]
        for row in _setting_rows(
            _new_method_job("ap02", prompt_for_single_marker=False),
            METHOD_JOB_GROUPS,
        )
    }
    ap03 = {
        row[0]
        for row in _setting_rows(
            _new_method_job("ap03", prompt_for_single_marker=False),
            METHOD_JOB_GROUPS,
        )
    }

    assert common | {
        "root_camera",
        "matcher",
        "gpu_mode",
        "mapper_matches",
        "maximum_image_size",
        "maximum_features",
    } <= ap01
    assert common | {
        "ap02_reference",
        "max_nfev_static",
        "max_nfev_moving",
        "ba_loss",
        "ba_loss_scale",
    } <= ap02
    assert common | {
        "single_marker",
        "multi_markers",
        "scale_reprojection",
        "scale_ransac",
        "scale_inliers",
        "matcher",
        "gpu_mode",
        "mapper_matches",
        "maximum_image_size",
        "maximum_features",
    } <= ap03
    assert "label" not in ap01 | ap02 | ap03


def test_active_package_never_imports_historical_run_scripts() -> None:
    repository = Path(__file__).resolve().parents[1]
    package = repository / "src/camera_rig_calibration"
    for source in package.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "run.real_vehicle_data" not in text
        assert "run.bus_real_data" not in text
        assert "run/real_vehicle_data" not in text
        assert "run/bus_real_data" not in text
    assert sorted(
        path.relative_to(repository / "run").as_posix()
        for path in (repository / "run").rglob("*.py")
    ) == ["rigcal.py"]
