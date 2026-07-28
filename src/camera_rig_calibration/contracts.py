from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pydantic import BaseModel

from .config.models import RigConfig


@dataclass(frozen=True)
class RequirementResult:
    compatible: bool
    reasons: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> "RequirementResult":
        return cls(True)

    @classmethod
    def unavailable(cls, *reasons: str) -> "RequirementResult":
        return cls(False, tuple(reasons))


@dataclass(frozen=True)
class CommandSpec:
    stage_id: str
    display_name: str
    argv: tuple[str, ...]
    cwd: Path
    output_directory: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    diagnostic: bool = False

    def shell_display(self) -> str:
        import shlex

        return shlex.join(self.argv)


@dataclass(frozen=True)
class RunContext:
    repository_root: Path
    config: RigConfig
    dataset_root: Path
    observations_root: Path
    run_directory: Path
    resolved_root_camera: str | None = None
    resolved_ap02_reference_marker_id: int | None = None
    resolved_ap03_single_scale_marker_id: int | None = None
    resolved_ap03_multi_marker_ids: tuple[int, ...] = ()
    resolved_evaluation_anchor_marker_id: int | None = None
    reuse_colmap_artifact: bool = False
    resolved_marker_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.resolved_ap03_multi_marker_ids and self.resolved_marker_ids:
            object.__setattr__(
                self,
                "resolved_ap03_multi_marker_ids",
                self.resolved_marker_ids,
            )


class InputAdapter(Protocol):
    id: str
    display_name: str

    def matches(self, config: RigConfig) -> bool: ...

    def requirements(self, config: RigConfig) -> RequirementResult: ...

    def commands(self, context: RunContext) -> Sequence[CommandSpec]: ...


class CalibrationMethod(Protocol):
    id: str
    display_name: str
    config_model: type[BaseModel]

    def requirements(self, context: RunContext) -> RequirementResult: ...

    def commands(self, context: RunContext) -> Sequence[CommandSpec]: ...

    def collect(self, context: RunContext) -> dict[str, Any]: ...


class Evaluator(Protocol):
    id: str
    display_name: str

    def requirements(self, context: RunContext) -> RequirementResult: ...

    def commands(self, context: RunContext) -> Sequence[CommandSpec]: ...


class ExperimentProvider(Protocol):
    id: str
    display_name: str
    description: str

    def variants(self, config: RigConfig) -> Sequence[tuple[str, RigConfig]]: ...


CompatibilityCheck = Callable[[RunContext], RequirementResult]
