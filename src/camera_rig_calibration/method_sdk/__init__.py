"""Public SDK for trusted in-project calibration-method extensions."""

from .contracts import (
    CanonicalResultBuilder,
    MethodConfigEditor,
    MethodInputRequirements,
    MethodMetadata,
    method_metadata,
)
from .results import (
    CANONICAL_RESULT_CONTRACT,
    TRANSFORM_CONVENTION,
    CanonicalCameraPose,
    CanonicalMethodResult,
    load_canonical_result,
    write_canonical_result,
    write_native_camera_extrinsics,
)
from .service import (
    materialize_method_result,
    method_artifact_root,
    resolved_method_metadata,
)
from .example_method import CanonicalPoseImportMethod, PoseImportOptions

__all__ = [
    "CANONICAL_RESULT_CONTRACT",
    "TRANSFORM_CONVENTION",
    "CanonicalCameraPose",
    "CanonicalMethodResult",
    "CanonicalResultBuilder",
    "CanonicalPoseImportMethod",
    "MethodConfigEditor",
    "MethodInputRequirements",
    "MethodMetadata",
    "PoseImportOptions",
    "load_canonical_result",
    "materialize_method_result",
    "method_artifact_root",
    "method_metadata",
    "resolved_method_metadata",
    "write_canonical_result",
    "write_native_camera_extrinsics",
]
