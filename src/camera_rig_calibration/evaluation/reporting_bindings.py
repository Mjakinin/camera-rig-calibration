"""Late-bound reporting hooks installed by the product policy stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Hook = Callable[..., Any]


@dataclass(frozen=True)
class ReportingBindings:
    read_json: Hook
    write_json: Hook
    baseline_contract: Hook
    anchor_camera_gt_rows: Hook
    method_report_text: Hook
    refresh_method_reports: Hook
    real_results_text: Hook
    simulation_results: Hook
    repository_root: Hook
    ensure_ap03_derived_results: Hook
    ensure_visualization_artifacts: Hook


def current_reporting_bindings() -> ReportingBindings:
    from . import reporting

    return ReportingBindings(
        read_json=reporting._read_json,
        write_json=reporting._write_json,
        baseline_contract=reporting._baseline_contract,
        anchor_camera_gt_rows=reporting._anchor_camera_gt_rows,
        method_report_text=reporting._method_report_text,
        refresh_method_reports=reporting.refresh_method_reports,
        real_results_text=reporting._real_results_text,
        simulation_results=reporting._simulation_results,
        repository_root=reporting._repository_root,
        ensure_ap03_derived_results=reporting.ensure_ap03_derived_results,
        ensure_visualization_artifacts=reporting.ensure_visualization_artifacts,
    )


__all__ = ["ReportingBindings", "current_reporting_bindings"]
