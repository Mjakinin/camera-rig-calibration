"""Interactive browsing of published calibration results."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..publication import reconcile_existing_experiment
from ..results import index_results
from ..visualization import launch_isolated_rviz


def _is_internal_evidence_result(entry: object) -> bool:
    """Keep reproducibility experiments out of the ordinary result catalogue."""

    text = " ".join(
        str(value).lower()
        for value in (
            getattr(entry, "experiment_id", ""),
            getattr(entry, "dataset_id", ""),
            getattr(entry, "path", ""),
        )
    )
    return any(
        token in text
        for token in (
            "route2_main_parity_v1",
            "main_route2_parity",
            "main_parity",
            "pre_fix",
            "post_fix",
        )
    )


def show_results(repository_root: Path, console: Console) -> None:
    entries = [
        entry
        for entry in index_results(repository_root / "results")
        if not _is_internal_evidence_result(entry)
    ]
    if not entries:
        console.print(
            "No layout-v2 calibration result was found. Completed experiments "
            "appear here after SUMMARY.json has been published."
        )
        return
    table = Table(title="Calibration results", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Experiment", ratio=2, overflow="fold")
    table.add_column("Dataset / input", ratio=2, overflow="fold")
    table.add_column("Result status", ratio=1, overflow="fold")
    table.add_column("Methods", ratio=1, overflow="fold")
    for index, entry in enumerate(entries, 1):
        summary_payload = json.loads(
            (entry.path / "SUMMARY.json").read_text(encoding="utf-8")
        )
        input_label = (
            f"{summary_payload.get('sampling_rate', 'native_rate')}\n"
            f"{entry.dataset_state}"
        )
        result_label = ", ".join(entry.method_statuses) or "-"
        table.add_row(
            str(index),
            entry.experiment_id or entry.dataset_id,
            input_label,
            entry.status,
            result_label,
        )
    console.print(table)
    console.print(
        "[dim]Simulation scope: route, density, resolution, FOV and blur affect "
        "the moving camera; lighting affects the whole world. pct = percent.[/dim]"
    )
    selected = typer.prompt("Result number to inspect (0 = back)", default=0, type=int)
    if selected == 0:
        return
    if selected < 1 or selected > len(entries):
        raise typer.BadParameter("Invalid result number")
    entry = entries[selected - 1]
    # Layout-v2 scientific products are derived idempotently from published
    # artifacts on first access. Calibration methods are never rerun here.
    reconcile_existing_experiment(
        entry.path,
        dataset_root=entry.path,
        category=entry.category,
    )
    results_txt = entry.path / "RESULTS.txt"
    console.print(
        Panel(
            f"Human-readable results: {results_txt}\n"
            f"Result folder: {entry.path}",
            title=f"{entry.dataset_id} — result folder",
        )
    )
    comparison_payload = json.loads(
        (entry.path / "COMPARISON.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in comparison_payload.get("methods", [])
        if isinstance(row, dict)
    ]
    methods = Table(title="Method variants and scientific status")
    methods.add_column("#", justify="right")
    methods.add_column("Method / variant")
    methods.add_column("Artifact")
    methods.add_column("Quality")
    methods.add_column("Anchor export")
    methods.add_column("Runtime")
    methods.add_column("Coverage")
    methods.add_column("Primary")
    methods.add_column("Result path", overflow="fold")
    methods.add_column("Configuration / warning", overflow="fold")
    readable: list[Path] = []
    for index, row in enumerate(rows, 1):
        result_path = entry.path / str(row.get("result_path", ""))
        if (
            row.get("artifact_status") == "available"
            or row.get("status") == "available"
        ):
            readable.append(result_path)
        runtime = row.get("runtime_seconds")
        methods.add_row(
            str(index),
            f"{row.get('method', '-')}/{row.get('label', '-')}",
            str(row.get("artifact_status") or row.get("status", "-")),
            str(row.get("quality_status") or "-"),
            str(row.get("anchor_export_status") or "-"),
            f"{float(runtime):.1f}s" if runtime is not None else "-",
            str(row.get("static_camera_count") or "-"),
            str(row.get("primary_result") or "-"),
            str(result_path),
            "; ".join(
                value
                for value in (
                    ", ".join(
                        f"{key}={item}"
                        for key, item in (
                            row.get("config_summary") or {}
                        ).items()
                        if item is not None
                    ),
                    str(row.get("warning") or ""),
                )
                if value
            ),
        )
    console.print(methods)
    console.print(
        f"Human-readable experiment results: {results_txt}\n"
        f"Machine comparison: {entry.path / 'COMPARISON.json'}\n"
        f"Common evaluations: {entry.path / 'evaluations'}"
    )
    report_options = {"1": results_txt}
    camera_map = entry.path / "SECONDARY_CAMERA_MAP_RESULTS.txt"
    marker_map = entry.path / "SECONDARY_AP02_MARKER_MAP_RESULTS.txt"
    option_labels = ["1 = overall RESULTS.txt", "2 = one method RESULT.txt"]
    if camera_map.is_file():
        report_options["3"] = camera_map
        option_labels.append("3 = secondary camera-map GT")
    if marker_map.is_file():
        report_options["4"] = marker_map
        option_labels.append("4 = secondary AP02 marker-map GT")
    option_labels.append("5 = open isolated RViz visualization")
    choice = str(
        typer.prompt(
            "Read " + ", ".join(option_labels) + ", 0 = back",
            default="1",
        )
    ).strip()
    if choice == "0":
        return
    if choice == "5":
        try:
            session = launch_isolated_rviz(entry.path, repository_root)
        except RuntimeError as exc:
            console.print(
                Panel(
                    str(exc),
                    title="RViz visualization unavailable",
                    border_style="yellow",
                )
            )
            return
        console.print(
            Panel(
                (
                    f"Session: {session['session_id']}\n"
                    f"ROS_DOMAIN_ID: {session['ros_domain_id']}\n"
                    f"PID: {session['pid']}\n"
                    f"Log: {session['log']}\n\n"
                    "RViz runs independently in the background. You can open "
                    "another result without closing this window."
                ),
                title="RViz session started",
                border_style="green",
            )
        )
        return
    if choice in report_options:
        selected_report = report_options[choice]
        if not selected_report.is_file():
            console.print(
                "The requested report is unavailable; see RESULTS.json for "
                "the recorded reason."
            )
            return
        console.print(
            selected_report.read_text(encoding="utf-8"), markup=False
        )
        return
    if choice != "2" or not readable:
        raise typer.BadParameter("Invalid result view")
    inspect = typer.prompt("Method number", type=int)
    if inspect < 1 or inspect > len(rows):
        raise typer.BadParameter("Invalid method number")
    selected_row = rows[inspect - 1]
    selected_path = entry.path / str(selected_row.get("result_path", ""))
    result_txt = selected_path / "RESULT.txt"
    if not result_txt.is_file():
        console.print("This row is a failed attempt; see FAILURE.txt there.")
        return
    console.print(result_txt.read_text(encoding="utf-8"), markup=False)
    console.print(
        f"Diagnostics: {selected_path / 'diagnostics'}\n"
        f"Complete logs: {selected_path / 'logs'}\n"
        f"Provenance: {selected_path / 'provenance'}"
    )



__all__ = ["_is_internal_evidence_result", "show_results"]

