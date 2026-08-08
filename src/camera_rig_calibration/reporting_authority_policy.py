from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


_INSTALLED = False


def _install_effective_anchor_read() -> None:
    """Prefer the published common evaluation over stale preflight recommendations."""
    from .evaluation import reporting

    original = reporting._read_json
    if getattr(original, "_rigcal_reporting_authority", False):
        return

    def read_json(path: Path) -> dict[str, Any]:
        payload = original(path)
        if path.name != "SELECTION_CANDIDATES.json" or path.parent.name != "observations":
            return payload

        selected = original(
            path.parent.parent
            / "evaluations"
            / "SELECTED_COMMON_EVALUATION.json"
        )
        try:
            anchor_marker_id = int(selected["anchor_marker_id"])
        except (KeyError, TypeError, ValueError):
            return payload

        merged = copy.deepcopy(payload)
        anchor = dict(merged.get("evaluation_anchor", {}))
        anchor.update(
            {
                "selected": anchor_marker_id,
                "configured": anchor_marker_id,
                "selection_mode": "published_common_evaluation",
                "resolution_stage": "published_common_evaluation",
                "reason": (
                    "published common evaluation is authoritative for final "
                    "reporting; the original preflight recommendation remains "
                    "preserved in the source selection artifact"
                ),
            }
        )
        merged["evaluation_anchor"] = anchor
        return merged

    read_json._rigcal_reporting_authority = True  # type: ignore[attr-defined]
    reporting._read_json = read_json


def _install_baseline_contract_scope() -> None:
    """Evaluate the baseline contract only on variants explicitly named baseline."""
    from .evaluation import reporting

    original = reporting._baseline_contract
    if getattr(original, "_rigcal_reporting_authority", False):
        return

    def baseline_contract(
        *,
        category: str,
        method_payloads: list[dict[str, Any]],
        evaluation_anchor: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_payloads = [
            item
            for item in method_payloads
            if str(item.get("label", "")) == "baseline"
        ]
        return original(
            category=category,
            method_payloads=baseline_payloads or method_payloads,
            evaluation_anchor=evaluation_anchor,
        )

    baseline_contract._rigcal_reporting_authority = True  # type: ignore[attr-defined]
    reporting._baseline_contract = baseline_contract


def _install_direct_anchor_guard() -> None:
    """Never compare a method anchor pose against GT for a different marker frame."""
    from .evaluation import reporting

    original = reporting._anchor_camera_gt_rows
    if getattr(original, "_rigcal_reporting_authority", False):
        return

    def anchor_camera_gt_rows(
        method: str,
        label: str,
        anchor_payload: dict[str, Any],
        *,
        anchor_marker_id: int,
        gt_cameras: dict[str, Any],
        gt_markers: dict[int, Any],
    ) -> list[dict[str, Any]]:
        try:
            payload_anchor = int(anchor_payload.get("anchor_marker_id"))
        except (TypeError, ValueError):
            return []
        if payload_anchor != int(anchor_marker_id):
            return []
        return original(
            method,
            label,
            anchor_payload,
            anchor_marker_id=anchor_marker_id,
            gt_cameras=gt_cameras,
            gt_markers=gt_markers,
        )

    anchor_camera_gt_rows._rigcal_reporting_authority = True  # type: ignore[attr-defined]
    reporting._anchor_camera_gt_rows = anchor_camera_gt_rows


def install_reporting_authority_policy() -> None:
    """Install final reporting rules without changing calibration estimates."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_effective_anchor_read()
    _install_baseline_contract_scope()
    _install_direct_anchor_guard()
    _INSTALLED = True
