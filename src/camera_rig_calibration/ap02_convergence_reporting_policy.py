from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


_INSTALLED = False
_SECTION_TITLE = "AP02 OPTIMIZATION CONVERGENCE"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None:
        return None
    return int(round(number))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _representative_points(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return one conservative point per solver_nfev.

    AP02's trace contains residual/Jacobian evaluations in addition to solver
    state evaluations.  We therefore do not label an arbitrary CSV row as the
    exact solver iterate.  For each reported solver_nfev we retain the lowest
    finite robust cost observed at that nfev, then publish a monotone
    best-observed-so-far envelope.  This is sufficient for budget diagnostics
    such as 70->80 without pretending to expose SciPy internals that were not
    recorded explicitly.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        nfev = _integer(row.get("solver_nfev"))
        cost = _finite(row.get("robust_cost"))
        if nfev is None or nfev < 0 or cost is None:
            continue
        item = {
            "solver_nfev": nfev,
            "residual_evaluation_index": _integer(
                row.get("residual_evaluation_index")
            ),
            "source": str(row.get("source") or ""),
            "solver_iteration": _integer(row.get("solver_iteration")),
            "elapsed_seconds": _finite(row.get("elapsed_seconds")),
            "robust_cost": cost,
            "reprojection_rmse_px": _finite(row.get("reprojection_rmse_px")),
            "mean_reprojection_error_px": _finite(
                row.get("mean_reprojection_error_px")
            ),
            "maximum_reprojection_error_px": _finite(
                row.get("maximum_reprojection_error_px")
            ),
            "parameter_step_norm": _finite(row.get("parameter_step_norm")),
        }
        grouped.setdefault(nfev, []).append(item)

    representatives: list[dict[str, Any]] = []
    best_cost = float("inf")
    best_point: dict[str, Any] | None = None
    for nfev in sorted(grouped):
        candidate = min(
            grouped[nfev],
            key=lambda item: (
                float(item["robust_cost"]),
                int(item.get("residual_evaluation_index") or -1),
            ),
        )
        if float(candidate["robust_cost"]) < best_cost:
            best_cost = float(candidate["robust_cost"])
            best_point = candidate
        point = dict(candidate)
        point["observations_at_nfev"] = len(grouped[nfev])
        point["best_observed_cost"] = best_cost
        point["best_observed_reprojection_rmse_px"] = (
            best_point.get("reprojection_rmse_px") if best_point else None
        )
        point["best_observed_elapsed_seconds"] = (
            best_point.get("elapsed_seconds") if best_point else None
        )
        representatives.append(point)
    return representatives


def _point_at_or_before(
    points: list[dict[str, Any]], target_nfev: int
) -> dict[str, Any] | None:
    eligible = [point for point in points if int(point["solver_nfev"]) <= target_nfev]
    return eligible[-1] if eligible else None


def _tail_window(
    points: list[dict[str, Any]], final_nfev: int, width: int
) -> dict[str, Any] | None:
    end = _point_at_or_before(points, final_nfev)
    start = _point_at_or_before(points, max(0, final_nfev - width))
    first = points[0] if points else None
    if start is None or end is None or first is None:
        return None
    start_cost = _finite(start.get("best_observed_cost"))
    end_cost = _finite(end.get("best_observed_cost"))
    initial_cost = _finite(first.get("best_observed_cost"))
    if start_cost is None or end_cost is None or initial_cost is None:
        return None
    improvement = max(0.0, start_cost - end_cost)
    total_improvement = max(0.0, initial_cost - end_cost)
    start_rmse = _finite(start.get("best_observed_reprojection_rmse_px"))
    end_rmse = _finite(end.get("best_observed_reprojection_rmse_px"))
    return {
        "requested_window_nfev": width,
        "start_nfev": int(start["solver_nfev"]),
        "end_nfev": int(end["solver_nfev"]),
        "start_best_observed_cost": start_cost,
        "end_best_observed_cost": end_cost,
        "absolute_cost_improvement": improvement,
        "relative_cost_improvement_from_window_start": (
            improvement / start_cost if start_cost > 0.0 else None
        ),
        "fraction_of_total_observed_cost_improvement_in_window": (
            improvement / total_improvement if total_improvement > 0.0 else None
        ),
        "start_best_observed_reprojection_rmse_px": start_rmse,
        "end_best_observed_reprojection_rmse_px": end_rmse,
        "reprojection_rmse_improvement_px": (
            max(0.0, start_rmse - end_rmse)
            if start_rmse is not None and end_rmse is not None
            else None
        ),
    }


def _stage_summary(stage_root: Path, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history_path = stage_root / "ap02_optimization_history.csv"
    optimizer_path = stage_root / "ap02_optimization_summary.json"
    if not optimizer_path.is_file():
        optimizer_path = stage_root / "optimizer_report.json"
    optimizer = _read_json(optimizer_path)
    rows = _read_csv(history_path)
    points = _representative_points(rows)

    final_nfev = _integer(optimizer.get("nfev"))
    maximum_nfev = _integer(optimizer.get("maximum_function_evaluations"))
    success = bool(optimizer.get("solver_success", optimizer.get("success", False)))
    message = str(
        optimizer.get("solver_message", optimizer.get("message", "")) or ""
    ).strip()
    limit_reached = bool(
        not success
        and (
            "maximum number" in message.lower()
            or (
                final_nfev is not None
                and maximum_nfev is not None
                and final_nfev >= maximum_nfev
            )
        )
    )
    if success:
        termination = "converged"
    elif limit_reached:
        termination = "function_evaluation_limit_reached"
    else:
        termination = "stopped_without_reported_convergence"

    windows = {
        str(width): value
        for width in (5, 10, 20)
        if final_nfev is not None
        and (value := _tail_window(points, final_nfev, width)) is not None
    }
    final_trace = _point_at_or_before(points, final_nfev) if final_nfev is not None else None
    final_summary_cost = _finite(optimizer.get("final_cost"))
    trace_best_cost = _finite(final_trace.get("best_observed_cost")) if final_trace else None
    summary = {
        "stage": stage,
        "history_available": history_path.is_file() and bool(points),
        "history_path": str(history_path),
        "optimizer_summary_path": str(optimizer_path),
        "solver_success": success,
        "solver_message": message,
        "termination": termination,
        "nfev": final_nfev,
        "maximum_function_evaluations": maximum_nfev,
        "optimality": _finite(optimizer.get("optimality")),
        "initial_cost": _finite(optimizer.get("initial_cost")),
        "final_cost": final_summary_cost,
        "initial_reprojection_rmse_px": _finite(
            optimizer.get("initial_reprojection_rmse_px")
        ),
        "final_reprojection_rmse_px": _finite(
            optimizer.get("final_reprojection_rmse_px")
        ),
        "trace_distinct_nfev_count": len(points),
        "trace_first_nfev": int(points[0]["solver_nfev"]) if points else None,
        "trace_last_nfev": int(points[-1]["solver_nfev"]) if points else None,
        "trace_best_observed_cost_at_final_nfev": trace_best_cost,
        "trace_vs_optimizer_final_cost_absolute_difference": (
            abs(trace_best_cost - final_summary_cost)
            if trace_best_cost is not None and final_summary_cost is not None
            else None
        ),
        "tail_windows": windows,
        "trace_semantics": (
            "one representative per recorded solver_nfev; representative is the "
            "lowest robust cost observed within that nfev; tail comparisons use "
            "the monotone best-observed-so-far cost envelope"
        ),
    }
    return summary, points


def analyze_ap02_convergence(result_root: Path) -> dict[str, Any]:
    method_root = Path(result_root) / "diagnostics" / "method" / "graph_ba"
    stages: dict[str, Any] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for stage, directory in (
        ("static_only", method_root / "static_only"),
        ("with_moving", method_root / "with_moving"),
    ):
        summary, points = _stage_summary(directory, stage)
        stages[stage] = summary
        for point in points:
            checkpoint_rows.append({"stage": stage, **point})
    combined = stages.get("with_moving", {})
    return {
        "schema_version": 1,
        "method": "ap02",
        "stages": stages,
        "combined": combined,
        "checkpoint_rows": checkpoint_rows,
        "ground_truth_used": False,
        "method_rerun": False,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _finite(value)
    return "-" if number is None else f"{100.0 * number:.3f}%"


def _stage_text(stage: dict[str, Any]) -> list[str]:
    name = str(stage.get("stage", "-"))
    lines = [
        (
            f"{name}: termination={stage.get('termination', '-')}, "
            f"nfev={stage.get('nfev', '-')}/{stage.get('maximum_function_evaluations', '-')}, "
            f"cost={_fmt(stage.get('initial_cost'))} -> {_fmt(stage.get('final_cost'))}, "
            f"RMSE={_fmt(stage.get('initial_reprojection_rmse_px'))} -> "
            f"{_fmt(stage.get('final_reprojection_rmse_px'))} px, "
            f"optimality={_fmt(stage.get('optimality'))}"
        )
    ]
    for width in (20, 10, 5):
        window = stage.get("tail_windows", {}).get(str(width))
        if not isinstance(window, dict):
            continue
        lines.append(
            f"  last ~{width} nfev ({window.get('start_nfev')}->{window.get('end_nfev')}): "
            f"best-observed cost {_fmt(window.get('start_best_observed_cost'))} -> "
            f"{_fmt(window.get('end_best_observed_cost'))}; "
            f"improvement={_pct(window.get('relative_cost_improvement_from_window_start'))} "
            f"of window-start cost, "
            f"{_pct(window.get('fraction_of_total_observed_cost_improvement_in_window'))} "
            "of total observed improvement; "
            f"best-observed RMSE {_fmt(window.get('start_best_observed_reprojection_rmse_px'))} -> "
            f"{_fmt(window.get('end_best_observed_reprojection_rmse_px'))} px"
        )
    if stage.get("solver_message"):
        lines.append(f"  solver message: {stage['solver_message']}")
    return lines


def convergence_report_text(analysis: dict[str, Any]) -> str:
    lines = [
        _SECTION_TITLE,
        "-" * 118,
        (
            "Trace diagnostic only; no calibration is rerun. Tail comparisons "
            "use best-observed robust cost up to the recorded solver_nfev."
        ),
    ]
    for stage_name in ("static_only", "with_moving"):
        stage = analysis.get("stages", {}).get(stage_name)
        if isinstance(stage, dict):
            lines.extend(_stage_text(stage))
    lines.extend(
        [
            "Detailed machine-readable diagnostics:",
            "- diagnostics/reporting/AP02_CONVERGENCE_SUMMARY.json",
            "- diagnostics/reporting/AP02_CONVERGENCE_CHECKPOINTS.csv",
            "Ground truth used: no",
            "",
        ]
    )
    return "\n".join(lines)


def _strip_existing_section(text: str) -> str:
    marker = "\n" + _SECTION_TITLE + "\n"
    index = text.find(marker)
    if index < 0 and text.startswith(_SECTION_TITLE + "\n"):
        index = 0
    return text[:index].rstrip() if index >= 0 else text.rstrip()


def _publish_for_result(result_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_ap02_convergence(result_root)
    reporting_root = Path(result_root) / "diagnostics" / "reporting"
    summary_path = reporting_root / "AP02_CONVERGENCE_SUMMARY.json"
    checkpoints_path = reporting_root / "AP02_CONVERGENCE_CHECKPOINTS.csv"
    _write_json(
        summary_path,
        {key: value for key, value in analysis.items() if key != "checkpoint_rows"},
    )
    _write_csv(checkpoints_path, list(analysis.get("checkpoint_rows", [])))

    metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {}
    metrics["ap02_convergence"] = analysis.get("combined", {})
    metrics["ap02_convergence_stages"] = analysis.get("stages", {})
    payload["metrics"] = metrics
    details = list(payload.get("detail_artifacts", []))
    for relative in (
        "diagnostics/reporting/AP02_CONVERGENCE_SUMMARY.json",
        "diagnostics/reporting/AP02_CONVERGENCE_CHECKPOINTS.csv",
    ):
        if relative not in details:
            details.append(relative)
    payload["detail_artifacts"] = details
    _write_json(Path(result_root) / "RESULT.json", payload)

    result_text_path = Path(result_root) / "RESULT.txt"
    try:
        current = result_text_path.read_text(encoding="utf-8")
    except OSError:
        current = ""
    report = convergence_report_text(analysis)
    result_text_path.write_text(
        _strip_existing_section(current) + "\n\n" + report,
        encoding="utf-8",
    )
    return payload


def install_ap02_convergence_reporting_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from .evaluation import reporting

    original = reporting.refresh_method_reports
    if getattr(original, "_rigcal_ap02_convergence_reporting", False):
        _INSTALLED = True
        return

    def refresh_method_reports(experiment_root: Path):
        payloads = original(experiment_root)
        updated: list[dict[str, Any]] = []
        root = Path(experiment_root)
        for payload in payloads:
            if str(payload.get("method")) != "ap02":
                updated.append(payload)
                continue
            result_root = root / "methods" / "ap02" / str(payload.get("label"))
            updated.append(_publish_for_result(result_root, dict(payload)))
        return updated

    refresh_method_reports._rigcal_ap02_convergence_reporting = True  # type: ignore[attr-defined]
    reporting.refresh_method_reports = refresh_method_reports
    _INSTALLED = True
