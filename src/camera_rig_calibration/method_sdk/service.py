"""Runtime and publication helpers for SDK-aware method components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import RunContext
from ..registry import calibration_methods
from .contracts import MethodMetadata, method_metadata
from .results import CanonicalMethodResult, write_canonical_result


_BUILTIN_ARTIFACTS = {
    "ap01": Path("02_AP01"),
    "ap02": Path("03_AP02"),
    "ap03": Path("04_AP03"),
}
_BUILTIN_PRIMARY_POSES = {
    "ap01": Path(
        "03_static_extrinsics/AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
    ),
    "ap02": Path(
        "07_graph_ba/with_moving/optimized_static_camera_poses_ref_marker.csv"
    ),
    "ap03": Path(
        "scale_multi/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
    ),
}


def registered_method(method_id: str) -> Any | None:
    try:
        return calibration_methods.get(method_id)
    except KeyError:
        return None


def resolved_method_metadata(method_id: str) -> MethodMetadata:
    method = registered_method(method_id)
    if method is not None:
        return method_metadata(method)
    algorithm_version = {
        "ap01": "ap01_explicit_method_contract_v2",
        "ap02": "ap02_maximum_frontier_v1",
        "ap03": "ap03_baseline_method_contract_v1",
    }.get(method_id, "extension_v1")
    return MethodMetadata(
        algorithm_version=algorithm_version,
        run_manifest_algorithm_version={
            "ap01": "ap01_baseline_hierarchical_v1",
            "ap02": "ap02_maximum_frontier_v1",
            "ap03": "ap03_shared_colmap_single_multi_v1",
        }.get(method_id, algorithm_version),
        artifact_directory=_BUILTIN_ARTIFACTS.get(
            method_id, Path(method_id)
        ),
        primary_pose_path=_BUILTIN_PRIMARY_POSES.get(method_id),
        input_requirements=method_metadata_fallback_inputs(),
        result_contract_required=method_id in _BUILTIN_ARTIFACTS,
        config_editor=None,
    )


def method_metadata_fallback_inputs():
    from .contracts import MethodInputRequirements

    return MethodInputRequirements()


def method_artifact_root(root: Path, method_id: str) -> Path:
    return root / resolved_method_metadata(method_id).artifact_directory


def materialize_method_result(
    method: Any,
    context: RunContext,
    status: dict[str, Any],
) -> CanonicalMethodResult | None:
    """Run a method adapter and enforce the opt-in SDK result contract."""

    metadata = method_metadata(method)
    builder = getattr(method, "canonical_result", None)
    if not callable(builder):
        if metadata.result_contract_required and status.get("success", False):
            raise RuntimeError(
                f"Method '{method.id}' completed without a canonical result adapter"
            )
        return None
    result = builder(context, status)
    if not isinstance(result, CanonicalMethodResult):
        raise TypeError(
            f"Method '{method.id}' canonical_result returned "
            f"{type(result).__name__}, expected CanonicalMethodResult"
        )
    if result.method_id != method.id:
        raise ValueError(
            f"Canonical result method_id '{result.method_id}' does not match "
            f"registered method '{method.id}'"
        )
    calibration_status = status.get("calibration_status")
    calibration_succeeded = (
        calibration_status == "available"
        or calibration_status is None
        and bool(status.get("success", False))
    )
    if (
        metadata.result_contract_required
        and calibration_succeeded
        and result.status != "available"
    ):
        raise RuntimeError(
            f"Method '{method.id}' reported success without 6DOF camera poses"
        )
    write_canonical_result(
        context.run_directory / metadata.artifact_directory,
        result,
    )
    return result


__all__ = [
    "materialize_method_result",
    "method_artifact_root",
    "registered_method",
    "resolved_method_metadata",
]
