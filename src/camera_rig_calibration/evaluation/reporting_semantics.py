"""Small compatibility layer for reporting semantics that must track config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config.models import AP02Settings


def configuration_summary(result_root: Path, method: str) -> dict[str, Any]:
    """Return the legacy summary with AP02 initialization semantics corrected."""
    from .reporting_configuration import _configuration_summary as legacy

    summary = dict(legacy(result_root, method))
    if method != "ap02":
        return summary

    configured = AP02Settings().initialization_strategy
    path = result_root / "provenance" / "resolved_config.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        method_config = payload.get("methods", {}).get("ap02", {})
        configured = method_config.get("initialization_strategy", configured)
    except (OSError, AttributeError, yaml.YAMLError):
        pass

    # Historical reports incorrectly labelled the alternate maximum-bottleneck
    # diagnostic as the productive initializer.  The runner selects the tree
    # named by AP02Settings.initialization_strategy; maximum_frontier_v1 is the
    # baseline default.  Remove the stale fields rather than preserving a lie.
    summary.pop("initialization_algorithm", None)
    summary.pop("initialization_diagnostic", None)
    summary["initialization_strategy"] = configured
    return summary


def baseline_contract(
    *,
    category: str,
    method_payloads: list[dict[str, Any]],
    evaluation_anchor: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the existing contract, but against the real AP02 initializer."""
    from .reporting_configuration import _baseline_contract as legacy

    contract = legacy(
        category=category,
        method_payloads=method_payloads,
        evaluation_anchor=evaluation_anchor,
    )
    defaults = AP02Settings()
    payload_by_key = {
        (str(item.get("method")), str(item.get("label"))): item
        for item in method_payloads
    }
    for variant in contract.get("variants", []):
        if variant.get("method") != "ap02":
            continue
        checks = dict(variant.get("checks", {}))
        checks.pop("maximum_bottleneck_initialization", None)
        payload = payload_by_key.get(
            (str(variant.get("method")), str(variant.get("label"))), {}
        )
        config = payload.get("config_summary", {})
        checks["initialization_strategy_default"] = (
            config.get("initialization_strategy")
            == defaults.initialization_strategy
        )
        variant["checks"] = checks
        variant["passes"] = all(checks.values())
    contract["contract"] = "route2_cpu_ref14_ap02_defaults_v3"
    variants = contract.get("variants", [])
    contract["passes"] = bool(variants) and all(
        bool(item.get("passes")) for item in variants
    )
    return contract


def refresh_method_reports(experiment_root: Path) -> list[dict[str, Any]]:
    """Run the canonical refresher with the corrected config-summary hook.

    ``reporting_method`` imported the historical summary function directly.
    Temporarily replacing that module-local hook keeps this compatibility fix
    narrow while all scientific estimates remain untouched.
    """
    from . import reporting_method

    previous = reporting_method._configuration_summary
    reporting_method._configuration_summary = configuration_summary
    try:
        return reporting_method.refresh_method_reports(experiment_root)
    finally:
        reporting_method._configuration_summary = previous


__all__ = [
    "configuration_summary",
    "baseline_contract",
    "refresh_method_reports",
]
