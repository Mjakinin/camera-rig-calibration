from __future__ import annotations

from typing import Any

from .ap02_convergence_reporting_policy import (
    _SECTION_TITLE,
    convergence_report_text,
)


_INSTALLED = False


def _append_if_needed(
    text: str,
    payload: dict[str, Any],
    method_payloads: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    ap02 = next(
        (
            item
            for item in method_payloads
            if str(item.get("method")) == "ap02"
        ),
        None,
    )
    if not isinstance(ap02, dict):
        return text, payload
    metrics = ap02.get("metrics", {})
    stages = metrics.get("ap02_convergence_stages", {}) if isinstance(metrics, dict) else {}
    if not isinstance(stages, dict) or not stages:
        return text, payload
    payload["ap02_optimization_convergence"] = {
        "method": "ap02",
        "stages": stages,
        "ground_truth_used": False,
        "method_rerun": False,
    }
    if _SECTION_TITLE in text:
        return text, payload
    report = convergence_report_text({"stages": stages})
    return text.rstrip() + "\n\n" + report, payload


def install_ap02_convergence_frontdoor_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from ..evaluation import reporting

    original_real = reporting._real_results_text
    if not getattr(original_real, "_rigcal_ap02_convergence_frontdoor", False):
        def real_results_text(experiment_root, method_payloads, dataset_root=None):
            text, payload = original_real(
                experiment_root, method_payloads, dataset_root
            )
            return _append_if_needed(text, payload, method_payloads)

        real_results_text._rigcal_ap02_convergence_frontdoor = True  # type: ignore[attr-defined]
        reporting._real_results_text = real_results_text

    original_sim = reporting._simulation_results
    if not getattr(original_sim, "_rigcal_ap02_convergence_frontdoor", False):
        def simulation_results(experiment_root, dataset_root, method_payloads):
            text, payload = original_sim(
                experiment_root, dataset_root, method_payloads
            )
            return _append_if_needed(text, payload, method_payloads)

        simulation_results._rigcal_ap02_convergence_frontdoor = True  # type: ignore[attr-defined]
        reporting._simulation_results = simulation_results

    _INSTALLED = True
