from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest
from pydantic import BaseModel, ConfigDict

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.contracts import CommandSpec, RequirementResult, RunContext
from camera_rig_calibration.registry import (
    ComponentRegistry,
    calibration_methods,
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


def test_schema_v4_has_no_active_frame_selection_registry() -> None:
    import camera_rig_calibration.registry as registry

    assert not hasattr(registry, "frame_selection_policies")
