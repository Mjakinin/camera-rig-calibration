from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_DATASET_CONTEXT: ContextVar[str] = ContextVar(
    "rigcal_product_dataset_context", default="real_vehicle"
)
_INSTALLED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_derived_tree(source: Path, destination: Path) -> None:
    """Refresh derived evaluation files without relaxing native immutability."""
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == item.stat().st_size:
            if _sha256(target) == _sha256(item):
                continue
        temporary = target.with_name(
            f".incoming_{target.name}_{os.getpid()}_{time.time_ns()}"
        )
        try:
            shutil.copy2(item, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _install_publication_policy() -> None:
    from . import publication

    original = publication._materialize_tree
    if getattr(original, "_rigcal_product_policy", False):
        return

    def materialize_tree(
        source: Path,
        destination: Path,
        *,
        keep_existing: tuple[str, ...] = (),
    ) -> None:
        # Calibration outputs and prepared scientific inputs remain immutable.
        # Only queue-derived evaluation/front-door artifacts are refreshable.
        if source.name == "evaluations" and destination.name == "evaluations":
            _refresh_derived_tree(source, destination)
            return
        original(source, destination, keep_existing=keep_existing)

    materialize_tree._rigcal_product_policy = True  # type: ignore[attr-defined]
    publication._materialize_tree = materialize_tree


def _install_reporting_policy() -> None:
    from .evaluation import reporting

    original = reporting._baseline_contract
    if getattr(original, "_rigcal_product_policy", False):
        return

    def baseline_contract(
        *,
        category: str,
        method_payloads: list[dict[str, Any]],
        evaluation_anchor: dict[str, Any],
    ) -> dict[str, Any]:
        contract = original(
            category=category,
            method_payloads=method_payloads,
            evaluation_anchor=evaluation_anchor,
        )
        # The reconstructed Main AP02 implementation uses 80/80. The old
        # 50/50 report name/check was stale reporting metadata, not method
        # semantics. Preserve every other contract check unchanged.
        contract["contract"] = "route2_cpu_ref14_80x80_v1"
        payloads = {
            (str(item.get("method", "")), str(item.get("label", ""))): item
            for item in method_payloads
        }
        for variant in contract.get("variants", []):
            if str(variant.get("method")) != "ap02":
                continue
            payload = payloads.get(
                (str(variant.get("method", "")), str(variant.get("label", ""))),
                {},
            )
            config = payload.get("config_summary", {})
            checks = variant.get("checks", {})
            checks.pop("static_nfev_50", None)
            checks.pop("combined_nfev_50", None)
            try:
                static_nfev = int(config.get("static_max_nfev") or 0)
            except (TypeError, ValueError):
                static_nfev = 0
            try:
                combined_nfev = int(config.get("combined_max_nfev") or 0)
            except (TypeError, ValueError):
                combined_nfev = 0
            checks["static_nfev_80"] = static_nfev == 80
            checks["combined_nfev_80"] = combined_nfev == 80
            variant["passes"] = all(bool(value) for value in checks.values())
        contract["passes"] = bool(contract.get("variants")) and all(
            bool(item.get("passes")) for item in contract.get("variants", [])
        )
        return contract

    baseline_contract._rigcal_product_policy = True  # type: ignore[attr-defined]
    reporting._baseline_contract = baseline_contract


def _install_ap03_anchor_policy() -> None:
    """Make the frozen common evaluation anchor authoritative for AP03 derivation."""
    from .evaluation import ap03_derived

    original = ap03_derived._selection_anchor
    if getattr(original, "_rigcal_product_policy", False):
        return

    def selection_anchor(experiment_root: Path) -> int | None:
        selected = (
            experiment_root
            / "evaluations"
            / "SELECTED_COMMON_EVALUATION.json"
        )
        try:
            payload = json.loads(selected.read_text(encoding="utf-8"))
            value = int(payload["anchor_marker_id"])
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            value = -1
        if value >= 0:
            return value
        return original(experiment_root)

    selection_anchor._rigcal_product_policy = True  # type: ignore[attr-defined]
    ap03_derived._selection_anchor = selection_anchor


def _install_selection_policy() -> None:
    """Keep automatic AP01 selection compatible with its Direct target."""
    from . import observations

    original = observations.resolve_selections
    if getattr(original, "_rigcal_product_policy", False):
        return

    def resolve_selections(*args, **kwargs):
        resolved = original(*args, **kwargs)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is None or "ap01" not in set(config.methods.enabled):
            return resolved
        ap01 = config.methods.ap01
        if not (
            ap01.root_camera == "auto"
            and ap01.method_contract == "baseline_v1"
            and not ap01.historical_reproduction
            and ap01.advanced_strategy == "legacy_main_v1"
            and resolved.root_camera == ap01.direct_target_camera
        ):
            return resolved

        root_payload = resolved.payload.get("ap01_root_camera", {})
        candidates = {
            str(candidate.get("id")): candidate
            for candidate in root_payload.get("candidates", [])
            if candidate.get("compatible")
            and str(candidate.get("id")) != ap01.direct_target_camera
        }
        if not candidates:
            # A rig with only the configured Direct target has no alternative
            # root; keep the original deterministic result and let AP01 report
            # its actual available Relay/Direct evidence.
            return resolved
        alternative = str(
            observations._best_candidate(candidates, observations._root_rank)
        )
        payload = copy.deepcopy(resolved.payload)
        payload["ap01_root_camera"]["selected"] = alternative
        payload["ap01_root_camera"]["reason"] = (
            "automatic AP01 root ranking with the configured Direct target "
            "reserved as a non-root camera"
        )
        return observations.ResolvedSelections(
            root_camera=alternative,
            ap02_reference_marker_id=resolved.ap02_reference_marker_id,
            ap03_single_scale_marker_id=resolved.ap03_single_scale_marker_id,
            ap03_multi_marker_ids=resolved.ap03_multi_marker_ids,
            evaluation_anchor_marker_id=resolved.evaluation_anchor_marker_id,
            marker_ids=resolved.marker_ids,
            payload=payload,
        )

    resolve_selections._rigcal_product_policy = True  # type: ignore[attr-defined]
    observations.resolve_selections = resolve_selections


def _replace_product_wording(text: str) -> str:
    replacements = {
        "smart frame budgets": "explicit per-marker / marker-pair BA frame limits",
        "Smart frame budgets": "Explicit per-marker / marker-pair BA frame limits",
        "smart selection at the BA boundary": (
            "explicit per-marker / marker-pair frame limits at the BA boundary"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _install_wizard_policy() -> None:
    from . import wizard

    if getattr(wizard, "_FINAL_PRODUCT_POLICY_INSTALLED", False):
        return

    wizard._PUBLIC_POLICY_NAMES["legacy_smart_v1"] = (
        "explicit per-marker / marker-pair BA frame limits"
    )

    original_prompt_enum_choice = wizard._prompt_enum_choice

    def prompt_enum_choice(
        label: str,
        current: str,
        choices: tuple[tuple[str, str], ...],
    ) -> str:
        return original_prompt_enum_choice(
            label,
            current,
            tuple(
                (value, _replace_product_wording(description))
                for value, description in choices
            ),
        )

    wizard._prompt_enum_choice = prompt_enum_choice

    original_setting_rows = wizard._setting_rows

    def setting_rows(job, groups=None):
        rows = original_setting_rows(job, groups)
        rendered = []
        for key, group, label, current, baseline, description in rows:
            description = _replace_product_wording(str(description))
            if key == "ap02_frame_strategy":
                label = "BA frame-limit application"
                description = (
                    "Baseline applies the explicit limits shown below at the "
                    "bundle-adjustment boundary: top frames per marker, top "
                    "frames per marker pair, and optional total frame cap. "
                    "Graph-preserving preselection remains an advanced option."
                )
            elif key == "evaluation_anchor":
                if _DATASET_CONTEXT.get() == "simulation":
                    baseline = "marker 14"
                    description = (
                        "Simulation baseline is marker 14. Edit this row to use "
                        "automatic selection or one manual post-preflight marker "
                        "choice for an ablation."
                    )
                else:
                    baseline = "auto"
                    description = (
                        "Real-vehicle default is automatic common-marker "
                        "selection. Manual post-preflight selection remains "
                        "available."
                    )
            elif key == "ap02_reference_mode":
                description = (
                    "Simulation baseline uses explicit marker 14. Real-vehicle "
                    "jobs default to automatic reference-marker selection. "
                    "Manual post-preflight selection remains available."
                )
            elif key == "root_camera" and _DATASET_CONTEXT.get() == "simulation":
                baseline = "cam_edge_3"
                description = (
                    "Final simulation AP01 baseline uses cam_edge_3, matching the "
                    "validated Main Direct/Relay geometry. Change only for an "
                    "explicit parameter study."
                )
            rendered.append(
                (key, group, label, current, baseline, description)
            )
        return rendered

    wizard._setting_rows = setting_rows

    original_new_method_job = wizard._new_method_job

    def new_method_job(*args, **kwargs):
        job = original_new_method_job(*args, **kwargs)
        methods = job.methods
        if _DATASET_CONTEXT.get() == "simulation":
            ap01 = methods.ap01.model_copy(
                update={
                    "method_contract": "baseline_v1",
                    "historical_reproduction": False,
                    "advanced_strategy": "legacy_main_v1",
                    "root_camera": "cam_edge_3",
                }
            )
            ap02 = methods.ap02.model_copy(
                update={
                    "method_contract": "baseline_v1",
                    "reference_marker_selection_mode": "baseline",
                    "reference_marker_id": 14,
                }
            )
            ap03 = methods.ap03.model_copy(
                update={
                    "method_contract": "baseline_v1",
                    "feature_limit_policy": "legacy_colmap_defaults_v1",
                    "scale_input_policy": (
                        "legacy_registered_image_redetection_v1"
                    ),
                }
            )
            methods = methods.model_copy(
                update={"ap01": ap01, "ap02": ap02, "ap03": ap03},
                deep=True,
            )
            job.evaluation = job.evaluation.model_copy(
                update={
                    "anchor_marker_id": 14,
                    "anchor_selection_mode": "explicit",
                }
            )
        else:
            # Real-vehicle data uses the same scientific method cores, but the
            # rig has no simulation GT contract that justifies hard-coding a
            # particular reference marker or common evaluation anchor.
            ap01 = methods.ap01.model_copy(
                update={
                    "method_contract": "baseline_v1",
                    "historical_reproduction": False,
                    "advanced_strategy": "legacy_main_v1",
                    "root_camera": "auto",
                }
            )
            ap02 = methods.ap02.model_copy(
                update={
                    "method_contract": "baseline_v1",
                    "reference_marker_selection_mode": "auto",
                    "reference_marker_id": "auto",
                }
            )
            methods = methods.model_copy(
                update={"ap01": ap01, "ap02": ap02}, deep=True
            )
            job.evaluation = job.evaluation.model_copy(
                update={
                    "anchor_marker_id": "auto",
                    "anchor_selection_mode": "auto",
                }
            )
        job.methods = methods
        wizard._refresh_method_job_label(job)
        return job

    wizard._new_method_job = new_method_job

    original_build_simulation_batch_outcome = wizard._build_simulation_batch_outcome

    def build_simulation_batch_outcome(*args, **kwargs):
        token = _DATASET_CONTEXT.set("simulation")
        try:
            return original_build_simulation_batch_outcome(*args, **kwargs)
        finally:
            _DATASET_CONTEXT.reset(token)

    wizard._build_simulation_batch_outcome = build_simulation_batch_outcome
    wizard._FINAL_PRODUCT_POLICY_INSTALLED = True


def install_product_policy() -> None:
    """Install final product defaults without changing scientific method cores."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_publication_policy()
    _install_reporting_policy()
    _install_ap03_anchor_policy()
    _install_selection_policy()
    _install_wizard_policy()
    _INSTALLED = True
