from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any


_INSTALLED = False
_EXTENSION_NAMESPACE = "ap03"
_EXTENSION_KEY = "colmap_camera_model_policy"
CALIBRATED = "calibrated"
UNDISTORTED_PINHOLE = "undistorted_pinhole"
_ALLOWED = {CALIBRATED, UNDISTORTED_PINHOLE}


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
    from .. import wizard
    from .product_policy import _DATASET_CONTEXT

    if getattr(wizard, "_AP03_CAMERA_MODEL_SENSITIVITY_POLICY", False):
        return

    original_method_job_label = wizard._method_job_label

    def method_job_label(job, context_key=None):
        label = original_method_job_label(job, context_key)
        if job.method_id != "ap03":
            return label
        methods = wizard._job_methods(job, context_key)
        if _policy_from_methods(methods) == UNDISTORTED_PINHOLE:
            return wizard.safe_id(f"{label}__moving_undistorted_pinhole_diag")
        return label

    wizard._method_job_label = method_job_label

    original_edit_method_job = wizard._edit_method_job

    def edit_method_job(
        console,
        job,
        *,
        groups=wizard.METHOD_JOB_GROUPS,
        title=None,
        selection_contexts=(),
    ):
        active_groups = set(groups)
        if (
            _DATASET_CONTEXT.get() == "real_vehicle"
            and job.method_id == "ap03"
            and "METHOD-SPECIFIC SETTINGS" in active_groups
        ):
            current = _policy_from_methods(job.methods)
            selected = wizard._prompt_enum_choice(
                "AP03 COLMAP moving-camera model",
                current,
                (
                    (
                        CALIBRATED,
                        "baseline: raw moving frames with calibrated distortion model",
                    ),
                    (
                        UNDISTORTED_PINHOLE,
                        "diagnostic: undistort moving frames first, then use the same fx/fy/cx/cy as PINHOLE",
                    ),
                ),
            )
            job.methods = _methods_with_policy(job.methods, selected)
            for context_key, contextual in list(job.context_methods.items()):
                job.context_methods[context_key] = _methods_with_policy(
                    contextual, selected
                )
            wizard._refresh_method_job_label(job)

        return original_edit_method_job(
            console,
            job,
            groups=groups,
            title=title,
            selection_contexts=selection_contexts,
        )

    wizard._edit_method_job = edit_method_job
    wizard._AP03_CAMERA_MODEL_SENSITIVITY_POLICY = True


def _install_ap03_command_policy() -> None:
    from ..methods.ap03 import pipeline

    original = pipeline.AP03Method.commands
    if getattr(original, "_rigcal_ap03_camera_model_sensitivity", False):
        return

    def commands(self, context):
        specs = tuple(original(self, context))
        policy = _policy_from_methods(context.config.methods)
        if policy == CALIBRATED:
            return specs
        if policy != UNDISTORTED_PINHOLE:
            raise ValueError(f"Unsupported AP03 camera-model policy: {policy}")

        updated = []
        for spec in specs:
            if spec.stage_id == "ap03_reconstruct":
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
                            "(undistorted moving-camera PINHOLE sensitivity)"
                        ),
                    )
                )
                continue

            if spec.stage_id in {"ap03_single_scale", "ap03_multi_scale"}:
                argv = list(spec.argv)
                if "--image-dir" not in argv:
                    if spec.output_directory is None:
                        raise RuntimeError(
                            "AP03 sensitivity scale stage has no output directory"
                        )
                    image_dir = (
                        spec.output_directory.parent
                        / "colmap"
                        / "undistorted_pinhole_dataset"
                        / "images"
                    )
                    argv.extend(["--image-dir", str(image_dir)])
                updated.append(
                    replace(
                        spec,
                        argv=tuple(argv),
                        display_name=(
                            spec.display_name
                            + " (matched to undistorted COLMAP image geometry)"
                        ),
                    )
                )
                continue

            updated.append(spec)
        return tuple(updated)

    commands._rigcal_ap03_camera_model_sensitivity = True  # type: ignore[attr-defined]
    pipeline.AP03Method.commands = commands


def _install_fingerprint_policy() -> None:
    from .. import experiments

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
                else "moving_undistorted_pinhole_v1"
            )
            payload["ap03_camera_model_sensitivity"] = {
                "policy": policy,
                "resolved_policy_id": policy_id,
                "scope": "moving_camera_only",
                "moving_image_preprocessing": (
                    "none"
                    if policy == CALIBRATED
                    else "opencv_undistort_same_intrinsic_matrix"
                ),
                "scale_redetection_image_geometry": (
                    "original_colmap_dataset"
                    if policy == CALIBRATED
                    else "same_undistorted_images_as_reconstruction"
                ),
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

    # runtime.py imports the artifact-fingerprint function directly during the
    # Wizard import. Rebind that local reference after wrapping experiments so a
    # sensitivity run can never reuse the calibrated AP03 COLMAP cache.
    from .. import runtime

    runtime.colmap_artifact_fingerprint = experiments.colmap_artifact_fingerprint


def install_ap03_camera_model_sensitivity_policy() -> None:
    """Expose an auditable AP03 camera-model sensitivity variant.

    The baseline remains unchanged. The diagnostic mode rectifies only the
    moving-camera images with their calibrated distortion, then reconstructs
    those rectified images with the same fx/fy/cx/cy as a PINHOLE camera. Static
    images and camera models remain unchanged. Marker-scale redetection uses the
    exact same rectified image geometry as the COLMAP reconstruction, so detected
    marker pixels and sparse-model camera rays remain consistent. A distinct
    fingerprint prevents reuse of baseline AP03 COLMAP artifacts.
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
    "UNDISTORTED_PINHOLE",
    "install_ap03_camera_model_sensitivity_policy",
]
