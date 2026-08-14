"""Copyable SDK example: import a solver's canonical 6DOF JSON result.

The example is intentionally not registered as a built-in calibration method.
It provides an executable integration reference without presenting imported
poses as a new scientific algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import CommandSpec, RequirementResult, RunContext
from .contracts import MethodInputRequirements
from .results import CanonicalMethodResult


class PoseImportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_json: Path = Field(
        description="Canonical 6DOF JSON produced by the external solver."
    )


@dataclass(frozen=True)
class CanonicalPoseImportMethod:
    id: str = "canonical_pose_import"
    display_name: str = "Canonical pose import (SDK example)"
    algorithm_version: str = "canonical_pose_import_v1"
    artifact_directory: str = "canonical_pose_import"
    primary_pose_path: str | None = None
    result_contract_required: bool = True
    input_requirements: MethodInputRequirements = MethodInputRequirements()
    config_model: type[BaseModel] = PoseImportOptions

    def _options(self, context: RunContext) -> PoseImportOptions:
        return PoseImportOptions.model_validate(
            context.config.methods.extensions.get(self.id, {})
        )

    def requirements(self, context: RunContext) -> RequirementResult:
        path = self._options(context).result_json.expanduser()
        if not path.is_absolute():
            path = context.repository_root / path
        return (
            RequirementResult.ok()
            if path.is_file()
            else RequirementResult.unavailable(
                f"canonical result JSON does not exist: {path}"
            )
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        del context
        return ()

    def collect(self, context: RunContext) -> dict[str, Any]:
        return {
            "status": "IMPORTED",
            "success": True,
            "quality_status": "externally_supplied",
            "available_static_cameras": [
                camera.id for camera in context.config.static_cameras
            ],
        }

    def canonical_result(
        self, context: RunContext, status: dict[str, Any]
    ) -> CanonicalMethodResult:
        path = self._options(context).result_json.expanduser()
        if not path.is_absolute():
            path = context.repository_root / path
        source = CanonicalMethodResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        return source.model_copy(
            update={
                "method_id": self.id,
                "algorithm_version": self.algorithm_version,
                "quality_status": str(
                    status.get("quality_status", source.quality_status)
                ),
                "native_artifacts": {
                    **source.native_artifacts,
                    "imported_result": str(path.resolve()),
                },
            },
            deep=True,
        )


__all__ = ["CanonicalPoseImportMethod", "PoseImportOptions"]
