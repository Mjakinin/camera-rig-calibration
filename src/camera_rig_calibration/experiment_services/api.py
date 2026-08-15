"""Compatibility facade for focused experiment identity services.

Policy installers replace the stable _method_payload and
colmap_artifact_fingerprint on this module. The small wrappers below keep
those hooks live while implementation stays separated by responsibility.
"""

from __future__ import annotations

from .config.models import RigConfig
from .experiment_services.identity import (
    ExperimentPaths,
    colmap_artifact_fingerprint,
    digest as _digest,
    experiment_fingerprint,
    experiment_paths,
    input_fingerprint,
    result_category,
)
from .experiment_services.manifests import (
    PARAMETER_INVALIDATION,
    STAGE_ORDER,
    experiment_manifest_payload,
    first_invalidated_stage,
    write_experiment_manifest,
)
from .experiment_services.method_identity import (
    automatic_method_label,
    build_method_payload,
    evaluation_fingerprint,
    method_config_diff,
    method_result_label,
    method_variant_name as _method_variant_name,
)
from .observations import ResolvedSelections


# Deliberately replaceable compatibility hook used by the installed AP03
# camera-model sensitivity policy.
_method_payload = build_method_payload


def method_fingerprint(
    config: RigConfig,
    method_id: str,
    selections: ResolvedSelections,
) -> str:
    return _digest(_method_payload(config, method_id, selections), 64)


def method_variant_name(
    config: RigConfig,
    method_id: str,
    selections: ResolvedSelections,
) -> str:
    return _method_variant_name(
        config,
        method_id,
        selections,
        fingerprint_value=method_fingerprint(
            config, method_id, selections
        ),
    )


__all__ = [
    "ExperimentPaths",
    "PARAMETER_INVALIDATION",
    "STAGE_ORDER",
    "_method_payload",
    "automatic_method_label",
    "colmap_artifact_fingerprint",
    "evaluation_fingerprint",
    "experiment_fingerprint",
    "experiment_manifest_payload",
    "experiment_paths",
    "first_invalidated_stage",
    "input_fingerprint",
    "method_config_diff",
    "method_fingerprint",
    "method_result_label",
    "method_variant_name",
    "result_category",
    "write_experiment_manifest",
]
