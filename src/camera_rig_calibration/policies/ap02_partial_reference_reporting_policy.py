from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


_INSTALLED = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _primary_reference(payload: dict[str, Any]) -> int | None:
    config = payload.get("config_summary", {})
    if not isinstance(config, dict):
        return None
    for key in ("resolved_reference_marker_id", "reference_marker_id"):
        value = config.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def install_ap02_partial_reference_reporting_policy() -> None:
    """Keep AP02 partial-report gauges consistent with the actual primary solve.

    Component diagnostics may select their own convenient local anchor for a
    disconnected component. The primary component, however, is not re-solved by
    that diagnostic stage: its published poses remain in the AP02 reference marker
    that was frozen before calibration. Reporting must therefore show that actual
    primary reference rather than the component-diagnostic candidate anchor.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from . import real_partial_evaluation_policy as partial

    original_summary = partial._component_summary_text
    if not getattr(original_summary, "_rigcal_primary_reference_gauge", False):
        def component_summary_text(payload: dict[str, Any]) -> str:
            reference = _primary_reference(payload)
            if reference is None:
                return original_summary(payload)
            updated = copy.deepcopy(payload)
            metrics = updated.get("metrics", {})
            results = (
                metrics.get("ap02_component_results", {})
                if isinstance(metrics, dict)
                else {}
            )
            if isinstance(results, dict):
                primary = str(results.get("primary_component_id") or "")
                for component in results.get("components", []):
                    if (
                        isinstance(component, dict)
                        and str(component.get("component_id")) == primary
                    ):
                        component["local_reference_marker_id"] = reference
                        component["reported_reference_source"] = (
                            "primary_ap02_frozen_reference"
                        )
            return original_summary(updated)

        component_summary_text._rigcal_primary_reference_gauge = True  # type: ignore[attr-defined]
        partial._component_summary_text = component_summary_text

    original_detail = partial._component_pose_detail
    if not getattr(original_detail, "_rigcal_primary_reference_gauge", False):
        def component_pose_detail(result_root: Path) -> str:
            text = original_detail(result_root)
            payload = _read_json(Path(result_root) / "RESULT.json")
            reference = _primary_reference(payload)
            if not text or reference is None:
                return text
            summary = _read_json(
                Path(result_root)
                / "diagnostics"
                / "method"
                / "component_diagnostics"
                / "AP02_COMPONENT_RESULTS.json"
            )
            primary = str(summary.get("primary_component_id") or "")
            if not primary:
                return text
            pattern = rf"^{re.escape(primary)} \| execution=([^|]+)\| local frame=marker_[^\n]+$"
            replacement = (
                f"{primary} | execution=\\1| local frame=marker_{reference}"
            )
            return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)

        component_pose_detail._rigcal_primary_reference_gauge = True  # type: ignore[attr-defined]
        partial._component_pose_detail = component_pose_detail

    _INSTALLED = True


__all__ = ["install_ap02_partial_reference_reporting_policy"]
