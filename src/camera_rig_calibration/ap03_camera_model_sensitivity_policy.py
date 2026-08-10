from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any


_INSTALLED = False
_EXTENSION_NAMESPACE = "ap03"
_EXTENSION_KEY = "colmap_camera_model_policy"
CALIBRATED = "calibrated"
PINHOLE_INTRINSICS_ONLY = "pinhole_intrinsics_only"
_ALLOWED = {CALIBRATED, PINHOLE_INTRINSICS_ONLY}


def _policy_from_methods(methods: Any) -> str:
    extensions = getattr(methods, "extensions", {}) or {}
    namespace = extensions.get(_EXTENSION_NAMESPACE, {})
    if not isinstance(namespace, dict):
        return CALIBRATED
    value = str(namespace.get(_EXTENSION_KEY, CALIBRATED)).strip()
    if value not in _ALLOWED:
        raise ValueError(
            "AP03 COLMAP camera-model policy must be one of: "
            + ", ".join(sorted(_ALLOWED))
        )
    return value


def _methods_with_policy(methods: Any, policy: str) -> Any:
    if policy not in _ALLOWED:
        raise ValueError(f"Unsupported AP03 camera-model policy: {policy}")
    extensions = {
        str(key): dict(value) if isinstance(value, dict) else value
        for key, value in (getattr(methods, "extensions", {}) or {}).items()
    }
    namespace = dict(extensions.get(_EXTENSION_NAMESPACE, {}) or {})
    namespace[_EXTENSION_KEY] = policy
    extensions[_EXTENSION_NAMESPACE] = namespace
    return methods.model_copy(update={"extensions": extensions}, deep=True)


def _install_wizard_policy() -> None:
    from . import wizard
    from .product_policy import _DATASET_CONTEXT

    if getattr(wizard, "_AP03_CAMERA_MODEL_SENSITIVITY_POLICY", False):
        return

    original_method_job_label = wizard._method_job_label

    def method_job_label(job, context_key=None):
        label = original_method_job_label(job, context_key)
        if job.method_id != "ap03":
            return label
        methods = wizard._job_methods(job, context_key)
        if _policy_from_methods(methods) == PINHOLE_INTRINSICS_ONLY:
            return wizard.safe_id(f"{label}__moving_pinhole_diag")
        return label

    wizard._method_job_label = method_job_label

    original_method_queue = wizard._method_queue

    def method_queue(*args, **kwargs):
        jobs = original_method_queue(*args, **kwargs)
        if _DATASET_CONTEXT.get() != "real_vehicle":
            return jobs
        for job in jobs:
            if job.method_id != "ap03":
                continue
            current = _policy_from_methods(job.methods)
            selected = wizard._prompt_enum_choice(
                "AP03 COLMAP moving-camera model",
                current,
                (
                    (
                        CALIBRATED,
                        "scientific baseline: preserve calibrated distortion mapping",
                    ),
                    (
                        PINHOLE_INTRINSICS_ONLY,
                        "diagnostic sensitivity: preserve fx/fy/cx/cy but ignore moving-camera distortion only inside AP03 COLMAP",
                    ),
                ),
            )
            job.methods = _methods_with_policy(job.methods, selected)
            for context_key, contextual in list(job.context_methods.items()):
                job.context_methods[context_key] = _methods_with_policy(
                    contextual, selected
                )
            wizard._refresh_method_job_label(job)
        return jobs

    wizard._method_queue = method_queue
    wizard._AP03_CAMERA_MODEL_SENSITIVITY_POLICY = True


def _install_ap03_command_policy() -> None:
    from .methods.ap03 import pipeline

    original = pipeline.AP03Method.commands
    if getattr(original, "_rigcal_ap03_camera_model_sensitivity", False):
        return

    def commands(self, context):
        specs = tuple(original(self, context))
        policy = _policy_from_methods(context.config.methods)
        if policy == CALIBRATED:
            return specs
        if policy != PINHOLE_INTRINSICS_ONLY:
            raise ValueError(f"Unsupported AP03 camera-model policy: {policy}")

        updated = []
        for spec in specs:
            if spec.stage_id != "ap03_reconstruct":
                updated.append(spec)
                continue
            argv = list(spec.argv)
            try:
                module_index = argv.index(
                    "camera_rig_calibration.methods.ap03.reconstruct_stage"
                )
            except ValueError as exc:
                raise RuntimeError(
                    "AP03 sensitivity policy could not locate the reconstruction stage"
                ) from exc
            argv[module_index] = (
                "camera_rig_calibration.methods.ap03.pinhole_reconstruct_stage"
            )
            updated.append(
                replace(
                    spec,
                    argv=tuple(argv),
                    display_name=(
                        "AP03: grouped COLMAP reconstruction "
                        "(moving-camera pinhole sensitivity)"
                    ),
                )
            )
        return tuple(updated)

    commands._rigcal_ap03_camera_model_sensitivity = True  # type: ignore[attr-defined]
    pipeline.AP03Method.commands = commands


def _install_fingerprint_policy() -> None:
    from . import experiments

    original_payload = experiments._method_payload
    if not getattr(
        original_payload, "_rigcal_ap03_camera_model_sensitivity", False
    ):

        def method_payload(config, method_id, selections):
            payload = original_payload(config, method_id, selections)
            if method_id != "ap03":
                return payload
            policy = _policy_from_methods(config.methods)
            policy_id = (
                "calibration_distortion_mapping_v1"
                if policy == CALIBRATED
                else "moving_pinhole_intrinsics_only_v1"
            )
            payload["ap03_camera_model_sensitivity"] = {
                "policy": policy,
                "resolved_policy_id": policy_id,
                "scope": "moving_camera_only",
                "original_camera_info_modified": False,
                "ground_truth_used": False,
            }
            contract = payload.get("resolved_method_contract")
            if isinstance(contract, dict):
                contract = dict(contract)
                contract["camera_model_policy"] = policy_id
                payload["resolved_method_contract"] = contract
            return payload

        method_payload._rigcal_ap03_camera_model_sensitivity = True  # type: ignore[attr-defined]
        experiments._method_payload = method_payload

    original_colmap_fingerprint = experiments.colmap_artifact_fingerprint
    if not getattr(
        original_colmap_fingerprint,
        "_rigcal_ap03_camera_model_sensitivity",
        False,
    ):

        def colmap_artifact_fingerprint(config, method_id, input_id):
            fingerprint = original_colmap_fingerprint(
                config, method_id, input_id
            )
            if method_id != "ap03":
                return fingerprint
            policy = _policy_from_methods(config.methods)
            if policy == CALIBRATED:
                return fingerprint
            return hashlib.sha256(
                f"{fingerprint}|{_EXTENSION_KEY}={policy}".encode("utf-8")
            ).hexdigest()

        colmap_artifact_fingerprint._rigcal_ap03_camera_model_sensitivity = True  # type: ignore[attr-defined]
        experiments.colmap_artifact_fingerprint = colmap_artifact_fingerprint


def install_ap03_camera_model_sensitivity_policy() -> None:
    """Expose an auditable AP03 camera-model sensitivity variant.

    The baseline remains unchanged. The diagnostic mode creates a private copy
    of camera-info metadata for AP03 COLMAP, preserves fx/fy/cx/cy, removes only
    moving-camera distortion in that copy, and uses a distinct result/fingerprint
    identity so baseline artifacts can never be silently reused.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _install_wizard_policy()
    _install_ap03_command_policy()
    _install_fingerprint_policy()
    _INSTALLED = True


__all__ = [
    "CALIBRATED",
    "PINHOLE_INTRINSICS_ONLY",
    "install_ap03_camera_model_sensitivity_policy",
]
