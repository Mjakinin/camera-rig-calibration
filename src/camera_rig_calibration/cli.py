from __future__ import annotations

import json
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .assets import ensure_bus_mesh
from .components import register_builtin_components
from .config import load_config
from .runtime import PipelineOrchestrator
from .queueing import (
    BatchConfig,
    ObservationReviewDecision,
    QueueConfig,
    QueueEntry,
    QueueRunner,
    is_batch_config,
    is_queue_config,
    load_batch,
    load_queue,
    load_queue_partitions,
)
from .wizard import (
    cleanup_storage_wizard,
    find_incomplete_transaction,
    incomplete_runs,
    incomplete_resume_source,
    manage_intrinsics_profiles,
    manage_incomplete_runs,
    new_calibration_wizard,
    show_doctor,
    show_results,
    show_queue_summary,
    show_summary,
    review_queue_selection_candidates,
    review_selection_candidates,
    WizardBack,
)


console = Console()


def _review_queue_selections(jobs, run_directory):
    return review_queue_selection_candidates(
        jobs, run_directory, console
    )


def _review_single_selection(config, resolved, run_directory):
    return review_selection_candidates(
        config, resolved, run_directory, console
    )


def _review_observation_coverage(preflight, run_directory):
    missing = ", ".join(preflight.missing_required_cameras) or "none"
    console.print(
        "\n[bold yellow]Observation review required[/bold yellow]\n"
        f"Required cameras without detections: {missing}\n"
        f"Evidence: {run_directory}"
    )
    for job in preflight.jobs:
        diagnosis = job.ap02_graph_diagnosis
        if diagnosis is None or diagnosis.complete:
            continue
        console.print(
            f"\n[bold]AP02 {job.job_id}:[/bold] Combined "
            f"{len(diagnosis.reached_static_cameras)}/"
            f"{len(diagnosis.expected_static_cameras)} cameras, "
            f"{len(diagnosis.components)} connected components; "
            f"reference component "
            f"{diagnosis.reference_component_id or 'not available'}"
        )
        component_table = Table(show_header=True)
        component_table.add_column("Component")
        component_table.add_column("Role")
        component_table.add_column("Cameras")
        component_table.add_column("Markers")
        component_table.add_column("Moving", justify="right")
        component_table.add_column("Bridging", justify="right")
        component_table.add_column("Runnable")
        for component in diagnosis.components:
            component_table.add_row(
                component.component_id,
                (
                    "primary"
                    if component.component_id
                    == diagnosis.reference_component_id
                    else "diagnostic"
                ),
                ",".join(component.static_cameras) or "-",
                ",".join(map(str, component.marker_ids)),
                str(len(component.moving_frames)),
                str(len(component.connecting_moving_frames)),
                "diagnostic" if component.calibratable else "no",
            )
        console.print(component_table)
        console.print(
            "Not reachable from reference component: "
            + (
                ", ".join(diagnosis.missing_static_cameras)
                if diagnosis.missing_static_cameras
                else "none"
            )
            + "\n"
            "Cause: "
            + ", ".join(diagnosis.cause_codes)
            + "\n"
            + diagnosis.explanation
        )

    console.print(
        "\n  1. retry ArUco detection on the same normalized frames\n"
        "  2. continue explicitly with diagnostic partial coverage\n"
        "  3. save this queue and return to the main menu"
    )
    action = typer.prompt("Selection", default="1").strip().lower()
    while action not in {"1", "2", "3"}:
        action = typer.prompt("Selection", default="1").strip().lower()
    if action == "2":
        return ObservationReviewDecision("continue_partial")
    if action == "3":
        return ObservationReviewDecision("pause")

    summary = json.loads(
        (run_directory / "queue_preflight_summary.json").read_text(
            encoding="utf-8"
        )
    )
    current = str(summary.get("detection_mode", "baseline"))
    modes = [
        mode
        for mode in ("high_sensitivity", "subpixel_refined", "baseline")
        if mode != current
    ]
    descriptions = {
        "high_sensitivity": (
            "recommended for dark, small or previously missed markers"
        ),
        "subpixel_refined": (
            "baseline candidates with subpixel corner refinement"
        ),
        "baseline": "original OpenCV detector behavior",
    }
    mode_table = Table(title=f"Detector retry — current: {current}")
    mode_table.add_column("#")
    mode_table.add_column("Mode")
    mode_table.add_column("Meaning")
    for index, mode in enumerate(modes, 1):
        mode_table.add_row(str(index), mode, descriptions[mode])
    console.print(mode_table)
    selected = typer.prompt("Detector number", default="1").strip()
    while not selected.isdigit() or not 1 <= int(selected) <= len(modes):
        selected = typer.prompt("Detector number", default="1").strip()
    return ObservationReviewDecision(
        "retry_detector", modes[int(selected) - 1]
    )


def _run_pipeline(orchestrator: PipelineOrchestrator, **kwargs) -> Path | None:
    try:
        return orchestrator.run(**kwargs)
    except KeyboardInterrupt:
        orchestrator.mark_interrupted()
        console.print(
            "\n[yellow]Run interrupted safely. Open 'Manage incomplete runs' to "
            "resume it or delete its files.[/yellow]"
        )
        return None


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    cwd = Path.cwd().resolve()
    if (cwd / ".git").exists():
        return cwd
    raise RuntimeError("Could not locate the camera-rig-calibration repository")


def _short_runtime(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_left = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_left:02d}s"
    if minutes:
        return f"{minutes}m{seconds_left:02d}s"
    return f"{seconds_left}s"


def _run_batch(
    batch: BatchConfig,
    *,
    root: Path,
    dry_run: bool,
    assume_yes: bool,
) -> bool:
    queues = [load_queue(entry.queue) for entry in batch.queues]
    total_jobs = sum(len(queue.entries) for queue in queues)
    if not dry_run and not assume_yes and not typer.confirm(
        f"Start {len(queues)} experiment queues with {total_jobs} method jobs?",
        default=True,
    ):
        console.print("Batch cancelled; all saved queue files were kept.")
        return False
    batch_started = time.monotonic()
    successful = True
    batch_results: dict[str, object] = {}
    state_path = (
        root
        / "workspace"
        / "batches"
        / batch.id
        / "batch_state.json"
    )
    def save_batch_state(status: str) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "rigcal_batch_state",
            "batch_id": batch.id,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "elapsed_seconds": time.monotonic() - batch_started,
            "experiments": batch_results,
        }
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(state_path)

    save_batch_state("running")
    for batch_index, (entry, queue) in enumerate(
        zip(batch.queues, queues, strict=True), 1
    ):
        experiment_started = time.monotonic()
        console.print(
            f"\n[bold]EXPERIMENT {batch_index}/{len(queues)} — "
            f"{entry.experiment_id}[/bold]"
        )
        runner = QueueRunner(
            root,
            console,
            selection_reviewer=(
                _review_queue_selections if sys.stdin.isatty() else None
            ),
            observation_reviewer=(
                _review_observation_coverage if sys.stdin.isatty() else None
            ),
        )
        runner.show(queue)
        try:
            results = runner.run(
                queue,
                dry_run=dry_run,
                batch_started_monotonic=batch_started,
            )
        except KeyboardInterrupt:
            save_batch_state("interrupted")
            raise
        except Exception as exc:
            results = {
                "_experiment": {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            }
            console.print(
                f"[red]Experiment queue failed: "
                f"{entry.experiment_id}: {exc}[/red]"
            )
        experiment_terminal = all(
            row.get("status")
            in {
                "completed",
                "duplicate_skipped",
                "published",
                "failed_published",
                "dry_run",
            }
            for row in results.values()
        )
        successful = successful and experiment_terminal
        batch_results[entry.experiment_id] = {
            "queue": str(entry.queue),
            "status": (
                "terminal"
                if experiment_terminal
                else "failed_or_incomplete"
            ),
            "elapsed_seconds": time.monotonic() - experiment_started,
            "entries": results,
        }
        save_batch_state("running")
        if not experiment_terminal and not batch.continue_independent:
            break
    save_batch_state("completed" if successful else "completed_with_failures")
    if not dry_run:
        table = Table(
            title="Calibration batch completed",
            caption=(
                f"Batch time: {time.monotonic() - batch_started:.1f} s | "
                f"{len(batch_results)} experiment(s)"
            ),
            expand=True,
        )
        table.add_column("Experiment")
        table.add_column("Method variant")
        table.add_column("Status")
        table.add_column("Method time")
        table.add_column("Experiment time")
        table.add_column("Canonical result", overflow="fold")
        table.add_column("Experiment reports", overflow="fold")
        for experiment_id, experiment_payload in batch_results.items():
            entries = (
                experiment_payload.get("entries", {})
                if isinstance(experiment_payload, dict)
                else {}
            )
            for entry_id, row in entries.items():
                if not isinstance(row, dict):
                    continue
                result = Path(str(row.get("result", "")))
                result_payload: dict[str, object] = {}
                result_json = result / "RESULT.json"
                if result_json.is_file():
                    try:
                        candidate = json.loads(
                            result_json.read_text(encoding="utf-8")
                        )
                        if isinstance(candidate, dict):
                            result_payload = candidate
                    except (OSError, json.JSONDecodeError):
                        pass
                experiment_root = next(
                    (
                        parent
                        for parent in (result, *result.parents)
                        if (parent / "SUMMARY.json").is_file()
                    ),
                    None,
                )
                table.add_row(
                    experiment_id,
                    (
                        f"{result_payload.get('method')}/"
                        f"{result_payload.get('label')}"
                        if result_payload
                        else entry_id
                    ),
                    (
                        "available"
                        if row.get("status")
                        in {"completed", "duplicate_skipped"}
                        else str(
                            row.get("failure", {}).get(
                                "cause_code",
                                row.get("status", "unknown"),
                            )
                        )
                    ),
                    _short_runtime(result_payload.get("runtime_seconds")),
                    _short_runtime(
                        experiment_payload.get("elapsed_seconds")
                        if isinstance(experiment_payload, dict)
                        else None
                    ),
                    str(result),
                    (
                        (
                            f"{experiment_root / 'RESULTS.txt'} | "
                            f"{experiment_root / 'COMPARISON.json'}"
                        )
                        if experiment_root is not None
                        else "-"
                    ),
                )
        console.print(table)
    return successful


def _execute(
    config_path: Path,
    *,
    dry_run: bool,
    assume_yes: bool,
) -> None:
    root = repository_root()
    if is_batch_config(config_path):
        _run_batch(
            load_batch(config_path),
            root=root,
            dry_run=dry_run,
            assume_yes=assume_yes,
        )
        return
    if is_queue_config(config_path):
        partitions = load_queue_partitions(config_path)
        total = sum(len(queue.entries) for queue in partitions)
        if len(partitions) > 1:
            console.print(
                f"[yellow]Legacy multi-dataset queue partitioned into "
                f"{len(partitions)} ordered dataset subqueues.[/yellow]"
            )
        if not dry_run and not assume_yes and not typer.confirm(
            f"Start all {total} queued experiment jobs?",
            default=True,
        ):
            console.print(f"Cancelled. Queue kept at {config_path}")
            return
        for queue in partitions:
            runner = QueueRunner(
                root,
                console,
                selection_reviewer=(
                    _review_queue_selections
                    if sys.stdin.isatty()
                    else None
                ),
                observation_reviewer=(
                    _review_observation_coverage
                    if sys.stdin.isatty()
                    else None
                ),
            )
            runner.show(queue)
            runner.run(queue, dry_run=dry_run)
        return
    config = load_config(config_path)
    if config.project.execution_mode == "complete" and len(config.methods.enabled) > 1:
        raise RuntimeError(
            "This config contains multiple method jobs. Save a "
            "rigcal_queue file with one method variant per entry."
        )
    show_summary(config, config_path, console)
    if dry_run:
        QueueRunner(root, console).run(
            QueueConfig(
                id=(
                    f"{config.dataset.id}_"
                    f"{config.project.run_label}_queue"
                ),
                entries=[
                    QueueEntry(
                        id=config.project.run_label,
                        config=config_path.resolve(),
                    )
                ],
            ),
            dry_run=True,
        )
        return
    PipelineOrchestrator(root, console).validate_ready(config)
    if not assume_yes and not typer.confirm(
        "Start this reproducible pipeline?", default=True
    ):
        console.print(f"Cancelled. Reproducible configuration kept at {config_path}")
        return
    QueueRunner(
        root,
        console,
        selection_reviewer=(
            _review_queue_selections if sys.stdin.isatty() else None
        ),
        observation_reviewer=(
            _review_observation_coverage if sys.stdin.isatty() else None
        ),
    ).run(
        QueueConfig(
            id=(
                f"{config.dataset.id}_"
                f"{config.project.run_label}_queue"
            ),
            entries=[
                QueueEntry(
                    id=config.project.run_label,
                    config=config_path.resolve(),
                )
            ],
        )
    )


def _execute_wizard_outcome(
    outcome,
    *,
    root: Path,
    dry_run: bool,
    assume_yes: bool,
) -> bool:
    """Show, validate and execute an ordered interactive run queue."""
    if outcome.batch_path is not None:
        show_queue_summary(outcome, console)
        return _run_batch(
            load_batch(outcome.batch_path),
            root=root,
            dry_run=dry_run,
            assume_yes=assume_yes,
        )
    show_queue_summary(outcome, console)
    if dry_run:
        for index, queued in enumerate(outcome.runs, 1):
            console.print(
                f"\n[bold]Queue row {index}/{len(outcome.runs)}: "
                f"{queued.config.project.run_label}[/bold]"
            )
            PipelineOrchestrator(root, console).show_dry_run(queued.config)
        return False
    # Validate every row before the confirmation and before the first result
    # directory is created. A missing COLMAP must not fail after preprocessing.
    for queued in outcome.runs:
        PipelineOrchestrator(root, console).validate_ready(queued.config)
    question = (
        f"Start all {len(outcome.runs)} queued reproducible runs?"
        if len(outcome.runs) > 1
        else "Start this reproducible pipeline?"
    )
    if not assume_yes and not typer.confirm(question, default=True):
        paths = ", ".join(str(run.path) for run in outcome.runs)
        console.print(f"Cancelled. Reproducible configuration(s) kept: {paths}")
        return False
    queue_path = outcome.path.parent / "queue.yaml"
    if queue_path.is_file():
        queue = load_queue(queue_path)
    else:
        queue = QueueConfig(
            id=(
                f"{outcome.config.dataset.id}_"
                f"{outcome.config.project.run_label}_queue"
            ),
            entries=[
                QueueEntry(
                    id=queued.config.project.run_label,
                    config=queued.path.resolve(),
                )
                for queued in outcome.runs
            ],
        )
    runner = QueueRunner(
        root,
        console,
        selection_reviewer=_review_queue_selections,
        observation_reviewer=_review_observation_coverage,
    )
    try:
        results = runner.run(queue)
    except WizardBack as exc:
        console.clear()
        console.print(
            f"[yellow]{exc}. No calibration method was started; use "
            "'Manage incomplete runs' to resume the prepared selection "
            "checkpoint.[/yellow]"
        )
        return False
    return all(
        row.get("status") in {"completed", "duplicate_skipped"}
        for row in results.values()
    )


def calibration_label(config) -> str:
    method = ", ".join(config.methods.enabled)
    return f"{config.project.run_label} — {method}"


def _interactive(*, dry_run: bool, assume_yes: bool) -> None:
    root = repository_root()
    register_builtin_components()
    while True:
        pending = incomplete_runs(root)
        pending_text = f" ({len(pending)} found)" if pending else ""
        console.print(
            Panel.fit(
                "1. Start a new calibration\n"
                "2. View results\n"
                f"3. Manage incomplete runs{pending_text}\n"
                "4. Check installation\n"
                "5. Cleanup storage\n"
                "6. Manage intrinsics profiles\n"
                "0. Exit",
                title="CAMERA RIG CALIBRATION",
            )
        )
        choice = typer.prompt("Choose an action", default="1").strip()
        if choice == "0":
            return
        if choice == "1":
            outcome = new_calibration_wizard(root, console)
            if outcome is None:
                continue
            if _execute_wizard_outcome(
                outcome,
                root=root,
                dry_run=dry_run,
                assume_yes=assume_yes,
            ):
                return
            continue
        if choice == "2":
            show_results(root, console)
            continue
        if choice == "3":
            resume = manage_incomplete_runs(root, console)
            if resume is not None:
                kind, source = incomplete_resume_source(resume)
                if kind == "queue":
                    try:
                        QueueRunner(
                            root,
                            console,
                            selection_reviewer=_review_queue_selections,
                            observation_reviewer=_review_observation_coverage,
                        ).run(load_queue(source))
                    except KeyboardInterrupt:
                        console.print(
                            "\n[yellow]Queue interrupted safely. Its capture, "
                            "observations and completed jobs remain under "
                            "workspace/temporary_runs.[/yellow]"
                        )
                else:
                    _run_pipeline(
                        PipelineOrchestrator(
                            root,
                            console,
                            selection_reviewer=_review_single_selection,
                        ),
                        resume_directory=source,
                    )
                return
            continue
        if choice == "4":
            show_doctor(root, console)
            continue
        if choice == "5":
            cleanup_storage_wizard(root, console)
            continue
        if choice == "6":
            manage_intrinsics_profiles(root, console)
            continue
        console.print("Choose 0, 1, 2, 3, 4, 5 or 6.")


def entry(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Run the saved reproducible YAML configuration.",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and show the plan without running stages."
    ),
    resume: str | None = typer.Option(
        None, "--resume", metavar="RUN_ID", help="Resume an incomplete run."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the final confirmation."),
) -> None:
    """A guided, reproducible camera-rig calibration pipeline."""

    try:
        root = repository_root()
        ensure_bus_mesh(root, announce=console.print)
        if config is not None and resume is not None:
            raise typer.BadParameter("Use either --config or --resume, not both")
        if resume is not None:
            if dry_run:
                raise typer.BadParameter("--dry-run cannot be combined with --resume")
            transaction = find_incomplete_transaction(root, resume)
            kind, source = incomplete_resume_source(transaction)
            if kind == "queue":
                try:
                    QueueRunner(
                        root,
                        console,
                        selection_reviewer=(
                            _review_queue_selections
                            if sys.stdin.isatty()
                            else None
                        ),
                        observation_reviewer=(
                            _review_observation_coverage
                            if sys.stdin.isatty()
                            else None
                        ),
                    ).run(load_queue(source))
                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Queue interrupted safely. Resume it later "
                        "from Manage incomplete runs.[/yellow]"
                    )
            else:
                _run_pipeline(
                    PipelineOrchestrator(
                        root,
                        console,
                        selection_reviewer=(
                            _review_single_selection
                            if sys.stdin.isatty()
                            else None
                        ),
                    ),
                    resume_directory=source,
                )
            return
        if config is not None:
            _execute(config, dry_run=dry_run, assume_yes=yes)
            return
        _interactive(dry_run=dry_run, assume_yes=yes)
    except WizardBack as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    except (ValidationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {
        "rerun-method",
        "reconcile",
    }:
        command = sys.argv[1]
        parser = argparse.ArgumentParser(prog=f"rigcal {command}")
        parser.add_argument(
            "--experiment", type=Path, required=True
        )
        if command == "rerun-method":
            parser.add_argument(
                "--method",
                required=True,
                choices=("ap01", "ap02", "ap03"),
            )
            parser.add_argument("--variant", required=True)
            parser.add_argument(
                "--reuse-prepared-input", action="store_true"
            )
            parser.add_argument(
                "--reuse-matching-intermediates", action="store_true"
            )
            parser.add_argument(
                "--reconcile-after", action="store_true"
            )
        arguments = parser.parse_args(sys.argv[2:])
        try:
            from .rerun import (
                reconcile_experiment,
                run_single_method_rerun,
            )

            experiment = arguments.experiment.resolve()
            if command == "reconcile":
                payload = reconcile_experiment(experiment)
                console.print(
                    "[green]Derived reports reconciled without running a "
                    f"calibration method:[/green] {experiment}"
                )
                console.print_json(data=payload)
                return
            results = run_single_method_rerun(
                repository_root=repository_root(),
                experiment=experiment,
                method=arguments.method,
                variant=arguments.variant,
                reuse_prepared_input=arguments.reuse_prepared_input,
                reuse_matching_intermediates=(
                    arguments.reuse_matching_intermediates
                ),
                reconcile_after=arguments.reconcile_after,
                console=console,
            )
            failed = [
                key
                for key, value in results.items()
                if value.get("status")
                not in {"completed", "duplicate_skipped"}
            ]
            if failed:
                raise RuntimeError(
                    "Single-method rerun did not complete: "
                    + ", ".join(failed)
                )
            return
        except (
            ValidationError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise SystemExit(1) from exc
    typer.run(entry)


if __name__ == "__main__":
    main()
