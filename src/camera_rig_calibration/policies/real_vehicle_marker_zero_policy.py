from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any


_INSTALLED = False


def _category(config: Any) -> str:
    value = getattr(config.dataset, "category", "real_vehicle")
    return str(getattr(value, "value", value))


def _candidate(payload: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    for section in ("ap02_reference_marker", "ap03_single_scale_marker"):
        for item in payload.get(section, {}).get("candidates", []):
            try:
                if int(item.get("id")) == marker_id:
                    return item
            except (TypeError, ValueError):
                continue
    return None


def _write_selection(root: Path, payload: dict[str, Any], observations: Any, ap02_ref: int) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
        (root / name).write_text(text, encoding="utf-8")
    (root / "REFERENCE_MARKER_ID.txt").write_text(f"{ap02_ref}\n", encoding="utf-8")
    observations.write_selection_candidates_csv(root, payload)


def _install_selection_policy() -> None:
    from .. import observations

    original = observations.resolve_selections
    if getattr(original, "_rigcal_real_vehicle_marker_zero", False):
        return

    def resolve_selections(config, observations_root):
        resolved = original(config, observations_root)
        if _category(config) != "real_vehicle":
            return resolved

        ap02_default = (
            config.methods.ap02.reference_marker_selection_mode == "auto"
            and config.methods.ap02.reference_marker_id == 0
        )
        ap03_default = config.methods.ap03.single.scale_marker_id == 0
        eval_default = (
            config.evaluation.enabled
            and config.evaluation.anchor_selection_mode == "auto"
            and config.evaluation.anchor_marker_id == 0
        )
        if not (ap02_default or ap03_default or eval_default):
            return resolved

        payload = copy.deepcopy(resolved.payload)
        marker_zero = _candidate(payload, 0)
        zero_observed = marker_zero is not None

        # Real Vehicle uses marker 0 as the canonical reference whenever marker 0
        # survives observation filtering at all. We deliberately do NOT switch to
        # a different marker merely because it ranks better or has stronger graph
        # support. If marker 0 is observed but later proves insufficient, preflight
        # must fail visibly instead of silently changing the scientific reference.
        ap02_selected = resolved.ap02_reference_marker_id
        ap03_selected = resolved.ap03_single_scale_marker_id
        eval_selected = resolved.evaluation_anchor_marker_id

        if ap02_default:
            if zero_observed:
                ap02_selected = 0
            section = payload.setdefault("ap02_reference_marker", {})
            section.update(
                {
                    "configured": 0,
                    "selected": int(ap02_selected),
                    "selection_mode": "required_if_observed_else_auto_fallback",
                    "preferred_marker_id": 0,
                    "fallback_used": not zero_observed,
                    "reason": (
                        "Real Vehicle canonical reference marker 0 was observed; marker 0 is required"
                        if zero_observed
                        else "Real Vehicle canonical marker 0 has zero accepted observations; deterministic auto fallback is allowed"
                    ),
                }
            )

        if ap03_default:
            if zero_observed:
                ap03_selected = 0
            section = payload.setdefault("ap03_single_scale_marker", {})
            section.update(
                {
                    "configured": 0,
                    "selected": int(ap03_selected),
                    "selection_mode": "required_if_observed_else_auto_fallback",
                    "preferred_marker_id": 0,
                    "fallback_used": not zero_observed,
                    "reason": (
                        "Real Vehicle canonical marker 0 was observed; AP03 Single keeps marker 0"
                        if zero_observed
                        else "Real Vehicle canonical marker 0 has zero accepted observations; AP03 Single auto fallback is allowed"
                    ),
                }
            )

        if eval_default:
            if zero_observed:
                eval_selected = 0
            section = payload.setdefault("evaluation_anchor", {})
            section.update(
                {
                    "configured": 0,
                    "selected": eval_selected,
                    "selection_mode": "required_if_observed_else_auto_fallback",
                    "preferred_marker_id": 0,
                    "fallback_used": not zero_observed,
                    "reason": (
                        "Real Vehicle canonical evaluation/export anchor marker 0 was observed; marker 0 is required"
                        if zero_observed
                        else "Real Vehicle canonical marker 0 has zero accepted observations; common-anchor auto fallback is allowed"
                    ),
                }
            )

        payload["real_vehicle_marker_zero_policy"] = {
            "canonical_marker_id": 0,
            "marker_zero_observed": zero_observed,
            "fallback_allowed": not zero_observed,
            "rule": "marker_0_required_if_observed_else_auto_fallback",
            "ground_truth_used": False,
        }
        category = payload.setdefault("category_marker_preference", {})
        category.update(
            {
                "dataset_category": "real_vehicle",
                "category_default_marker_id": 0,
                "ground_truth_used": False,
            }
        )
        if ap02_default:
            category["ap02"] = {
                "preferred": 0,
                "selected": int(ap02_selected),
                "fallback_used": not zero_observed,
            }
        if ap03_default:
            category["ap03_single"] = {
                "preferred": 0,
                "selected": int(ap03_selected),
                "fallback_used": not zero_observed,
            }
        if eval_default:
            category["evaluation_anchor"] = {
                "preferred": 0,
                "selected": eval_selected,
                "fallback_used": not zero_observed,
            }

        final = replace(
            resolved,
            ap02_reference_marker_id=int(ap02_selected),
            ap03_single_scale_marker_id=int(ap03_selected),
            evaluation_anchor_marker_id=(
                int(eval_selected) if eval_selected is not None else None
            ),
            payload=payload,
        )
        root = Path(observations_root)
        _write_selection(root, payload, observations, int(ap02_selected))
        return final

    resolve_selections._rigcal_real_vehicle_marker_zero = True  # type: ignore[attr-defined]
    observations.resolve_selections = resolve_selections


def _install_wizard_wording() -> None:
    from .. import wizard
    from .product_policy import _DATASET_CONTEXT

    if getattr(wizard, "_REAL_VEHICLE_MARKER_ZERO_WORDING", False):
        return
    original = wizard._setting_rows

    def setting_rows(job, groups=None):
        rows = original(job, groups)
        if _DATASET_CONTEXT.get() != "real_vehicle":
            return rows
        rendered = []
        for key, group, label, current, baseline, description in rows:
            if key == "evaluation_anchor":
                current = "marker 0 required if observed; auto fallback only if absent"
                baseline = current
                description = (
                    "Real Vehicle canonical evaluation/export origin is ArUco marker 0. "
                    "Preflight may choose another marker only when marker 0 has zero accepted observations."
                )
            elif key == "ap02_reference_mode":
                baseline = "auto with canonical marker 0"
                description = (
                    "Real Vehicle AP02 uses marker 0 whenever it is observed. "
                    "If marker 0 is observed but unusable, preflight fails instead of silently changing reference."
                )
            elif key == "ap02_reference_display":
                current = "marker 0 required if observed"
                baseline = current
            elif key == "single_marker":
                current = "marker 0 required if observed"
                baseline = current
                description = (
                    "Real Vehicle AP03 Single uses marker 0 whenever marker 0 survives observation filtering; "
                    "fallback is allowed only when marker 0 is absent."
                )
            rendered.append((key, group, label, current, baseline, description))
        return rendered

    wizard._setting_rows = setting_rows
    wizard._REAL_VEHICLE_MARKER_ZERO_WORDING = True


def install_real_vehicle_marker_zero_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_selection_policy()
    _install_wizard_wording()
    _INSTALLED = True
