"""Interactive, reviewed cleanup of application-owned storage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..results import index_results
from ..storage import (
    CleanupPlan,
    build_preparation_cache_cleanup_plan,
    build_results_cleanup_plan,
    build_temporary_cleanup_plan,
    combine_cleanup_plans,
    execute_cleanup,
)


def human_size(value: int) -> str:
    """Format a byte count for product-facing tables and summaries."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def run_storage_cleanup(
    repository_root: Path,
    console: Console,
    *,
    run_is_active: Callable[[Path], bool],
    show_error: Callable[[str], None],
) -> None:
    """Review and execute cleanup while keeping wizard policy hooks injectable."""
    temporary_root = (
        repository_root.resolve() / "workspace" / "temporary_runs"
    )
    active_runs = [
        path
        for path in (
            sorted(temporary_root.iterdir())
            if temporary_root.is_dir()
            else ()
        )
        if path.is_dir() and run_is_active(path)
    ]
    if active_runs:
        console.print(
            Panel(
                "Cleanup was not started because another rigcal calibration "
                "is still active:\n"
                + "\n".join(f"- {path}" for path in active_runs)
                + "\n\nStop the active run first, then open Cleanup storage "
                "again.",
                title="Cleanup blocked",
                border_style="red",
            )
        )
        return

    console.print(
        Panel(
            "Choose three storage groups independently. Every selected group "
            "is shown first and deleted only after one final confirmation.\n\n"
            "Published results include their embedded canonical raw images, "
            "observations and diagnostics. Preparation-cache cleanup covers "
            "only reusable normalized inputs under workspace/. "
            "Temporary cleanup covers runs, queues, batches and caches under "
            "workspace/.\n\n"
            "Always kept: data_local/, config/intrinsics/, source code and "
            "simulation assets. rigcal never asks to delete data_local here.",
            title="Cleanup storage",
        )
    )

    def review(
        plan: CleanupPlan,
        *,
        title: str,
        prompt: str,
    ) -> CleanupPlan | None:
        if not plan.targets:
            console.print(f"{title}: already empty.")
            return None
        table = Table(title=title)
        table.add_column("Selected path", overflow="fold")
        table.add_column("Kind")
        for target in plan.targets:
            try:
                display = target.path.relative_to(repository_root.resolve())
            except ValueError:
                display = target.path
            table.add_row(str(display), target.kind)
        table.add_row(
            f"{plan.file_count} files",
            (
                f"{human_size(plan.logical_bytes)} logical; "
                f"{human_size(plan.reclaimable_bytes)} reclaimable"
            ),
        )
        console.print(table)
        return plan if typer.confirm(prompt, default=False) else None

    selected: list[tuple[str, CleanupPlan]] = []
    for name, plan, title, prompt in (
        (
            "results",
            build_results_cleanup_plan(repository_root),
            "1. Published results",
            (
                "Select all published results, including embedded raw "
                "datasets, for permanent deletion?"
            ),
        ),
        (
            "preparation cache",
            build_preparation_cache_cleanup_plan(repository_root),
            "2. Reusable preparation cache",
            (
                "Select all reusable prepared-input caches for "
                "permanent deletion?"
            ),
        ),
        (
            "temporary workspace",
            build_temporary_cleanup_plan(repository_root),
            "3. Temporary workspace data",
            (
                "Select all temporary runs, queues, batches and workspace "
                "caches for permanent deletion?"
            ),
        ),
    ):
        reviewed = review(plan, title=title, prompt=prompt)
        if reviewed is not None:
            selected.append((name, reviewed))

    if not selected:
        console.print("Nothing was selected. Storage was left unchanged.")
        return

    final_plan = combine_cleanup_plans(*(plan for _, plan in selected))
    console.print(
        Panel(
            "Selected groups: "
            + ", ".join(name for name, _ in selected)
            + f"\nTargets: {len(final_plan.targets)}"
            + f"\nFiles: {final_plan.file_count}"
            + f"\nReclaimable: {human_size(final_plan.reclaimable_bytes)}"
            + "\n\ndata_local/ is not selected and will not be touched.",
            title="Final permanent-deletion confirmation",
            border_style="red",
        )
    )
    confirmation = typer.prompt(
        "Type DELETE to permanently remove exactly the selected storage",
        default="",
        show_default=False,
    ).strip()
    if confirmation != "DELETE":
        console.print("Confirmation did not match. Storage was left unchanged.")
        return

    try:
        result = execute_cleanup(final_plan)
    except (OSError, RuntimeError) as exc:
        show_error(
            "Cleanup stopped because deletion or verification failed: "
            f"{exc}"
        )
        return

    (repository_root / "results").mkdir(parents=True, exist_ok=True)
    (repository_root / "workspace").mkdir(parents=True, exist_ok=True)
    selected_names = {name for name, _ in selected}
    if "results" in selected_names and index_results(
        repository_root / "results"
    ):
        show_error(
            "Cleanup verification failed: View results still indexes a "
            "published experiment."
        )
        return
    console.print(
        "[green]Cleanup completed and verified. "
        f"{len(result['removed_targets'])} selected path(s) and "
        f"{result['file_count']} file(s) were removed. "
        "Estimated reclaimed space: "
        f"{human_size(int(result['reclaimable_bytes_estimate']))}. "
        "data_local was not touched.[/green]"
    )


__all__ = ["human_size", "run_storage_cleanup"]
