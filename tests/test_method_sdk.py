from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config import save_config
from camera_rig_calibration.contracts import (
    CommandSpec,
    RequirementResult,
    RunContext,
)
from camera_rig_calibration.method_sdk import (
    CanonicalCameraPose,
    CanonicalMethodResult,
    MethodInputRequirements,
    materialize_method_result,
    resolved_method_metadata,
)
from camera_rig_calibration.publication import _publish_success
from camera_rig_calibration.registry import calibration_methods, reset_registries
from camera_rig_calibration.ui.auto_form import (
    auto_form_fields,
    prompt_initial_options,
    update_auto_form_value,
)
from camera_rig_calibration.visualization.scene import (
    ensure_visualization_artifacts,
)


class NestedOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=1.5, gt=0, description="Solver threshold.")


class SdkOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=4, ge=1, description="Iteration budget.")
    mode: Literal["fast", "robust"] = Field(
        default="fast", description="Solver mode."
    )
    nested: NestedOptions = Field(default_factory=NestedOptions)


@dataclass(frozen=True)
class SdkFixtureMethod:
    id: str = "sdk_fixture"
    display_name: str = "SDK fixture"
    algorithm_version: str = "sdk_fixture_v2"
    artifact_directory: str = "sdk_fixture_artifacts"
    primary_pose_path: str | None = None
    result_contract_required: bool = True
    input_requirements: MethodInputRequirements = MethodInputRequirements()
    config_model: type[BaseModel] = SdkOptions

    def requirements(self, context: RunContext) -> RequirementResult:
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()

    def collect(self, context: RunContext) -> dict:
        return {
            "status": "OK",
            "success": True,
            "quality_status": "good",
            "available_static_cameras": [
                camera.id for camera in context.config.static_cameras
            ],
        }

    def canonical_result(
        self, context: RunContext, status: dict
    ) -> CanonicalMethodResult:
        return CanonicalMethodResult(
            method_id=self.id,
            algorithm_version=self.algorithm_version,
            status="available",
            reference_frame="fixture_reference",
            quality_status=status["quality_status"],
            camera_poses=[
                CanonicalCameraPose.from_transform(
                    camera_id=camera.id,
                    reference_frame="fixture_reference",
                    transform=np.eye(4),
                    source="fixture",
                )
                for camera in context.config.static_cameras
            ],
        )


@dataclass(frozen=True)
class IncompleteSdkMethod(SdkFixtureMethod):
    id: str = "sdk_incomplete"

    def canonical_result(
        self, context: RunContext, status: dict
    ) -> CanonicalMethodResult:
        del context, status
        return CanonicalMethodResult(
            method_id=self.id,
            algorithm_version=self.algorithm_version,
            status="incomplete",
            reference_frame="fixture_reference",
        )


@pytest.fixture(autouse=True)
def _restore_registry():
    yield
    reset_registries()
    register_builtin_components()


def test_pydantic_auto_form_flattens_and_validates_nested_fields() -> None:
    fields = {field.key: field for field in auto_form_fields(SdkOptions, {})}

    assert set(fields) == {
        "extension.iterations",
        "extension.mode",
        "extension.nested.threshold",
    }
    assert fields["extension.mode"].choices == ("fast", "robust")
    assert fields["extension.iterations"].description == "Iteration budget."
    updated = update_auto_form_value(
        SdkOptions, {}, ("nested", "threshold"), "2.75"
    )
    assert updated["nested"]["threshold"] == 2.75
    with pytest.raises(ValueError):
        update_auto_form_value(SdkOptions, {}, ("iterations",), "0")


def test_required_extension_fields_are_prompted_individually() -> None:
    class RequiredOptions(BaseModel):
        model_config = ConfigDict(extra="forbid")

        input_path: Path = Field(description="Required solver input.")
        attempts: int = 3

    prompts: list[str] = []
    payload = prompt_initial_options(
        RequiredOptions,
        lambda label: prompts.append(label) or "/tmp/result.json",
    )

    assert prompts == ["Input Path"]
    assert payload == {"input_path": Path("/tmp/result.json"), "attempts": 3}


def test_canonical_pose_rejects_non_rigid_matrices() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        CanonicalCameraPose.from_transform(
            camera_id="camera",
            reference_frame="reference",
            transform=matrix,
        )


def test_successful_sdk_calibration_requires_available_6dof_result(
    prepared_config, tmp_path: Path
) -> None:
    method = IncompleteSdkMethod()
    context = RunContext(
        repository_root=tmp_path,
        config=prepared_config,
        dataset_root=tmp_path / "dataset",
        observations_root=tmp_path / "observations",
        run_directory=tmp_path / "run",
    )

    with pytest.raises(RuntimeError, match="without 6DOF camera poses"):
        materialize_method_result(method, context, method.collect(context))


def test_builtin_run_manifest_versions_stay_compatible() -> None:
    assert (
        resolved_method_metadata("ap01").run_manifest_algorithm_version
        == "ap01_baseline_hierarchical_v1"
    )
    assert (
        resolved_method_metadata("ap03").run_manifest_algorithm_version
        == "ap03_shared_colmap_single_multi_v1"
    )


def test_sdk_method_materializes_and_publishes_generic_6dof_result(
    prepared_config, tmp_path: Path
) -> None:
    method = SdkFixtureMethod()
    calibration_methods.register(method)
    config = prepared_config.model_copy(deep=True)
    config.methods.enabled = [method.id]
    config.methods.extensions = {method.id: SdkOptions().model_dump(mode="python")}
    config.evaluation.enabled = False
    source = tmp_path / "completed"
    source.mkdir()
    context = RunContext(
        repository_root=tmp_path,
        config=config,
        dataset_root=tmp_path / "dataset",
        observations_root=tmp_path / "observations",
        run_directory=source,
    )
    status = method.collect(context)
    canonical = materialize_method_result(method, context, status)

    assert canonical is not None
    assert len(canonical.camera_poses) == len(config.static_cameras)
    assert (source / method.artifact_directory / "canonical_method_result.json").is_file()
    assert (source / method.artifact_directory / "camera_poses_6dof.csv").is_file()
    (source / method.artifact_directory / "METHOD_STATUS.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    save_config(config, source / "resolved_config.yaml")
    save_config(config, source / "requested_config.yaml")
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "method_id": method.id,
                "variant": "baseline",
                "method_fingerprint": "sdk-fixture-fingerprint",
                "input_id": "input-fixture",
                "algorithm_version": method.algorithm_version,
            }
        ),
        encoding="utf-8",
    )
    target, outcome = _publish_success(
        source,
        config=config,
        canonical_root=tmp_path / "published",
        queue_id="sdk-test",
    )

    assert outcome == "completed"
    result = json.loads((target / "RESULT.json").read_text(encoding="utf-8"))
    assert result["canonical_result_status"] == "available"
    assert result["canonical_pose_count"] == len(config.static_cameras)
    assert (target / "camera_extrinsics.csv").is_file()
    assert (target / "camera_poses_6dof.csv").is_file()
    visualization = ensure_visualization_artifacts(tmp_path / "published")
    assert visualization["status"] == "OK_NATIVE_CANONICAL_6DOF"
    assert visualization["point_count"] == 0
    assert visualization["variants"] == [
        {"method": method.id, "label": "baseline"}
    ]
