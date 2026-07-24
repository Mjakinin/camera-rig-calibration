from __future__ import annotations

import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from .assets import ensure_bus_mesh
from .components import register_builtin_components
from .config import load_config
from .runtime import PipelineOrchestrator
from .queueing import (
    QueueConfig,
    QueueEntry,
    QueueRunner,
    is_queue_config,
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
    review_selection_candidates,
    WizardBack,
)


console = Console()


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


def _execute(
    config_path: Path,
    *,
    dry_run: bool,
    assume_yes: bool,
) -> None:
    root = repository_root()
    if is_queue_config(config_path):
        partitions = load_queue_partitions(config_path)
        selection_cache: dict[tuple[object, ...], dict[str, object]] = {}

        def reviewer(config, resolved, run_directory):
            key = (
                config.dataset.id,
                tuple(camera.id for camera in config.static_cameras),
                tuple(resolved.marker_ids),
                str(config.dataset.prepared_root or config.dataset.input_root),
            )
            if key not in selection_cache:
                selection_cache[key] = review_selection_candidates(
                    config, resolved, run_directory, console
                )
            return selection_cache[key]

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
                selection_reviewer=reviewer if sys.stdin.isatty() else None,
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
    reviewer = None
    if sys.stdin.isatty():
        def reviewer(config, resolved, run_directory):
            return review_selection_candidates(
                config, resolved, run_directory, console
            )
    QueueRunner(
        root, console, selection_reviewer=reviewer
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
    selection_cache: dict[tuple[object, ...], dict[str, object]] = {}

    def reviewer(config, resolved, run_directory):
        key = (
            config.dataset.id,
            tuple(camera.id for camera in config.static_cameras),
            tuple(resolved.marker_ids),
            str(config.dataset.prepared_root or config.dataset.input_root),
        )
        if key not in selection_cache:
            selection_cache[key] = review_selection_candidates(
                config, resolved, run_directory, console
            )
        else:
            console.print(
                "[dim]Reusing the selections already confirmed for this queue "
                "input; no additional prompt is required.[/dim]"
            )
        return selection_cache[key]

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
        root, console, selection_reviewer=reviewer
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
                def resume_reviewer(config, resolved, run_directory):
                    return review_selection_candidates(
                        config, resolved, run_directory, console
                    )

                kind, source = incomplete_resume_source(resume)
                if kind == "queue":
                    try:
                        QueueRunner(
                            root,
                            console,
                            selection_reviewer=resume_reviewer,
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
                            selection_reviewer=resume_reviewer,
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
            reviewer = None
            if sys.stdin.isatty():
                def reviewer(config, resolved, run_directory):
                    return review_selection_candidates(
                        config, resolved, run_directory, console
                    )
            kind, source = incomplete_resume_source(transaction)
            if kind == "queue":
                try:
                    QueueRunner(
                        root, console, selection_reviewer=reviewer
                    ).run(load_queue(source))
                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Queue interrupted safely. Resume it later "
                        "from Manage incomplete runs.[/yellow]"
                    )
            else:
                _run_pipeline(
                    PipelineOrchestrator(
                        root, console, selection_reviewer=reviewer
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
    typer.run(entry)


if __name__ == "__main__":
    main()
