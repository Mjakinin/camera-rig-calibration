"""Safe discovery and lifecycle management for incomplete rigcal runs."""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..results import ResultEntry


def _choice(label: str, choices: dict[str, str], default: str) -> str:
    """Prompt for a menu choice used by the run-management UI."""
    typer.echo(f"\n{label}:")
    for key, description in choices.items():
        typer.echo(f"  {key}. {description}")
    while True:
        value = typer.prompt("Selection", default=default).strip()
        if value.lower() in {"b", "back"} and "0" in choices:
            return "0"
        if value in choices:
            return value
        typer.echo("Error: Choose one of: " + ", ".join(choices))


def _manifest_process_is_active(manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        pid = int(payload.get("runner_pid") or 0)
    except Exception:
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        # Windows has no /proc command line. A live PID recorded by a rigcal
        # manifest is treated conservatively as active instead of risking
        # deletion underneath another process.
        return os.name == "nt"
    try:
        command = command_path.read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"rigcal" in command or b"camera_rig_calibration" in command


def _transaction_payload(path: Path, name: str) -> dict:
    candidate = path / name
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def incomplete_runs(repository_root: Path) -> list[ResultEntry]:
    entries: list[ResultEntry] = []
    temporary_root = (
        repository_root.resolve() / "workspace" / "temporary_runs"
    )
    if not temporary_root.is_dir():
        return entries
    for transaction in sorted(
        path for path in temporary_root.iterdir() if path.is_dir()
    ):
        state = _transaction_payload(transaction, "queue_state.json")
        journal = _transaction_payload(
            transaction, "queue_transaction.json"
        )
        manifests: list[dict] = []
        for manifest_path in transaction.rglob("run_manifest.json"):
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                manifests.append(payload)
        if not state and not journal and not manifests:
            continue
        statuses = [
            str(row.get("status", "pending"))
            for row in state.get("entries", {}).values()
            if isinstance(row, dict)
        ]
        statuses.extend(
            str(payload.get("status", "unknown")) for payload in manifests
        )
        journal_status = str(journal.get("status", ""))
        if "running" in statuses:
            status = "running"
        elif journal_status == "publication_failed":
            status = "publication_failed"
        elif "failed_preflight" in statuses:
            status = "failed_preflight"
        elif "failed" in statuses:
            status = "failed"
        elif "interrupted" in statuses:
            status = "interrupted"
        elif "waiting_for_selection" in statuses:
            status = "waiting_for_selection"
        elif "waiting_for_observation_review" in statuses:
            status = "waiting_for_observation_review"
        else:
            status = journal_status or "incomplete"
        first = manifests[0] if manifests else {}
        methods = tuple(
            dict.fromkeys(
                method
                for payload in manifests
                for method in payload.get("enabled_methods", [])
            )
        )
        entries.append(
            ResultEntry(
                dataset_id=str(
                    first.get("dataset_id")
                    or state.get("dataset_id")
                    or transaction.name
                ),
                run_id=str(
                    journal.get("queue_id")
                    or state.get("queue_id")
                    or transaction.name
                ),
                status=status,
                path=transaction,
                methods=methods,
                category=str(first.get("result_category", "")),
                experiment_id=str(first.get("experiment_id", "")),
            )
        )
    return entries


def _active_run_stage(run: Path) -> str:
    for manifest_path in sorted(run.rglob("run_manifest.json")):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        for stage in manifest.get("stages", []):
            if stage.get("status") in {
                "running",
                "failed",
                "interrupted",
            }:
                return str(stage.get("id", "unknown"))
        for stage in manifest.get("stages", []):
            if stage.get("status") == "pending":
                return str(stage.get("id", "unknown"))
    return "unknown"


def _run_process_is_active(run: Path) -> bool:
    return any(
        _manifest_process_is_active(path)
        for path in run.rglob("run_manifest.json")
    )


def _validated_incomplete_run(repository_root: Path, entry: ResultEntry) -> Path:
    temporary_root = (
        repository_root / "workspace" / "temporary_runs"
    ).resolve()
    run = entry.path.resolve()
    try:
        relative = run.relative_to(temporary_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to modify a run outside workspace/temporary_runs/: {run}"
        ) from exc
    if len(relative.parts) != 1:
        raise RuntimeError(f"Refusing unexpected run path: {run}")
    if not (
        (run / "queue_transaction.json").is_file()
        or (run / "queue_state.json").is_file()
        or any(run.rglob("run_manifest.json"))
    ):
        raise RuntimeError(f"Temporary queue metadata is missing: {run}")
    if _run_process_is_active(run):
        raise RuntimeError(
            "This run still has an active rigcal process. Press Ctrl+C in its original "
            "terminal first, then open Manage incomplete runs again."
        )
    return run


def _interrupt_incomplete_run(run: Path) -> None:
    manifest_paths = list(run.rglob("run_manifest.json"))
    active_pids: set[int] = set()
    for manifest_path in manifest_paths:
        if not _manifest_process_is_active(manifest_path):
            continue
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            active_pids.add(int(payload.get("runner_pid") or 0))
        except Exception:
            continue
    for pid in sorted(value for value in active_pids if value > 0):
        os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _run_process_is_active(run):
        time.sleep(0.1)
    if _run_process_is_active(run):
        for manifest_path in manifest_paths:
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                pid = int(payload.get("runner_pid") or 0)
            except Exception:
                continue
            if _manifest_process_is_active(manifest_path):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _run_process_is_active(run):
            time.sleep(0.1)
    if _run_process_is_active(run):
        raise RuntimeError(
            "Could not stop every active rigcal process; temporary files "
            "were kept unchanged"
        )
    interrupted_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") not in {"completed", "duplicate_skipped"}:
            manifest["status"] = "interrupted"
        manifest["runner_pid"] = None
        manifest["interrupted_at"] = interrupted_at
        for stage in manifest.get("stages", []):
            if stage.get("status") == "running":
                stage["status"] = "interrupted"
                stage["error"] = "Interrupted from Manage incomplete runs"
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)
    state_path = run / "queue_state.json"
    if state_path.is_file():
        state = _transaction_payload(run, "queue_state.json")
        for row in state.get("entries", {}).values():
            if (
                isinstance(row, dict)
                and row.get("status")
                not in {"completed", "duplicate_skipped"}
            ):
                row["status"] = "interrupted"
        state["updated_at"] = interrupted_at
        temporary = state_path.with_name(state_path.name + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(state_path)


def _delete_incomplete_run(
    repository_root: Path,
    entry: ResultEntry,
    *,
    delete_private_inputs: bool,
) -> list[Path]:
    run = _validated_incomplete_run(repository_root, entry)
    if _run_process_is_active(run):
        raise RuntimeError(
            "This temporary queue is active. Stop it before deletion."
        )
    shutil.rmtree(run)
    return [run]


def incomplete_resume_source(transaction: Path) -> tuple[str, Path]:
    """Return ``("queue"|"run", path)`` for an incomplete transaction."""
    queue = transaction / "requested_queue.yaml"
    if queue.is_file():
        return "queue", queue
    resolved_queue = transaction / "resolved" / "queue.yaml"
    if resolved_queue.is_file():
        return "queue", resolved_queue
    manifests = sorted(transaction.rglob("run_manifest.json"))
    if not manifests:
        raise RuntimeError(
            f"Temporary transaction has no resumable queue or run: {transaction}"
        )
    incomplete = []
    for path in manifests:
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get(
                "status"
            )
        except Exception:
            status = None
        if status not in {"completed", "duplicate_skipped"}:
            incomplete.append(path.parent)
    return "run", (incomplete or [manifests[-1].parent])[0]


def find_incomplete_transaction(
    repository_root: Path, run_id_or_path: str
) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        temporary = (
            repository_root.resolve() / "workspace" / "temporary_runs"
        )
        if not resolved.is_relative_to(temporary):
            raise RuntimeError(
                "--resume accepts only workspace/temporary_runs transactions"
            )
        return resolved
    matches = [
        entry.path
        for entry in incomplete_runs(repository_root)
        if entry.run_id == run_id_or_path
        or entry.path.name == run_id_or_path
    ]
    if not matches:
        raise FileNotFoundError(
            f"Incomplete queue not found: {run_id_or_path}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Incomplete queue ID is ambiguous: {run_id_or_path}"
        )
    return matches[0]


def _remove_failed_queue_jobs(transaction: Path, console: Console) -> None:
    queue_path = transaction / "requested_queue.yaml"
    state_path = transaction / "queue_state.json"
    if not queue_path.is_file() or not state_path.is_file():
        raise RuntimeError(
            "This transaction has no editable schema-v5 queue manifest"
        )
    queue_payload = yaml.safe_load(
        queue_path.read_text(encoding="utf-8")
    ) or {}
    state = _transaction_payload(transaction, "queue_state.json")
    entry_state = state.get("entries", {})
    removable_statuses = {
        "failed",
        "failed_preflight",
        "interrupted",
        "waiting_for_selection",
        "waiting_for_observation_review",
        "skipped_dependency",
    }
    candidates = [
        item
        for item in queue_payload.get("entries", [])
        if str(entry_state.get(item.get("id"), {}).get("status"))
        in removable_statuses
    ]
    if not candidates:
        console.print("No failed or blocked queue jobs can be removed.")
        return
    table = Table(title="Failed/blocked jobs")
    table.add_column("#", justify="right")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Config", overflow="fold")
    for index, item in enumerate(candidates, 1):
        row = entry_state.get(item["id"], {})
        table.add_row(
            str(index),
            str(item["id"]),
            str(row.get("status", "unknown")),
            str(item.get("config", "")),
        )
    console.print(table)
    raw = typer.prompt(
        "Job number(s), comma-separated, all, or 0 = back",
        default="0",
    ).strip().lower()
    if raw == "0":
        return
    if raw == "all":
        selected = list(range(1, len(candidates) + 1))
    else:
        try:
            selected = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in raw.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(
                "Use comma-separated job numbers or 'all'"
            ) from exc
    if (
        not selected
        or min(selected) < 1
        or max(selected) > len(candidates)
    ):
        raise typer.BadParameter("Invalid failed job number")
    removed_ids = {candidates[index - 1]["id"] for index in selected}
    remaining = [
        item
        for item in queue_payload.get("entries", [])
        if item.get("id") not in removed_ids
    ]
    if not remaining:
        raise RuntimeError(
            "Removing every job would leave an empty queue; delete the "
            "temporary queue instead"
        )
    blocked = [
        str(item.get("id"))
        for item in remaining
        if removed_ids.intersection(item.get("depends_on", []))
    ]
    if blocked:
        raise RuntimeError(
            "Also remove dependent jobs before their prerequisites: "
            + ", ".join(blocked)
        )
    if not typer.confirm(
        "Remove these jobs from the queue? Their temporary job outputs are "
        "deleted; shared capture and observations remain.",
        default=False,
    ):
        return
    queue_payload["entries"] = remaining
    temporary = queue_path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(
            queue_payload, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    temporary.replace(queue_path)
    for job_id in removed_ids:
        entry_state.pop(job_id, None)
        state.get("source_fingerprints", {}).pop(job_id, None)
        state.get("resolved_configs", {}).pop(job_id, None)
        shutil.rmtree(transaction / "jobs" / job_id, ignore_errors=True)
        for path in (transaction / "resolved").glob(f"*_{job_id}_resolved.yaml"):
            path.unlink(missing_ok=True)
    state["updated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    temporary_state = state_path.with_name(state_path.name + ".tmp")
    temporary_state.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    temporary_state.replace(state_path)
    console.print(
        "Removed queue job(s): " + ", ".join(sorted(removed_ids))
    )


def manage_incomplete_runs(
    repository_root: Path, console: Console
) -> Path | None:
    entries = incomplete_runs(repository_root)
    if not entries:
        console.print("No incomplete runs exist.")
        return None
    table = Table(title="Incomplete runs")
    table.add_column("#", justify="right")
    table.add_column("Dataset")
    table.add_column("Run")
    table.add_column("Status")
    table.add_column("Current/next stage")
    for index, entry in enumerate(entries, 1):
        table.add_row(
            str(index),
            entry.dataset_id,
            entry.run_id,
            entry.status,
            _active_run_stage(entry.path),
        )
    console.print(table)
    raw_selection = typer.prompt(
        "Run number(s), comma-separated, all, or 0 = back",
        default="1",
    ).strip().lower()
    if raw_selection == "0":
        return None
    if raw_selection == "all":
        selected_numbers = list(range(1, len(entries) + 1))
    else:
        try:
            selected_numbers = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in raw_selection.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(
                "Use one number, comma-separated numbers, or 'all'"
            ) from exc
    if (
        not selected_numbers
        or min(selected_numbers) < 1
        or max(selected_numbers) > len(entries)
    ):
        raise typer.BadParameter("Invalid incomplete run number")
    selected_entries = [entries[number - 1] for number in selected_numbers]
    if len(selected_entries) > 1:
        targets = "\n".join(f"- {entry.path}" for entry in selected_entries)
        console.print(
            Panel(
                f"{targets}\n\nOnly incomplete run folders will be deleted. "
                "Shared/content-addressed inputs are protected.",
                title="Bulk-delete incomplete runs",
            )
        )
        if not typer.confirm(
            f"Delete these {len(selected_entries)} incomplete runs?",
            default=False,
        ):
            console.print("Deletion cancelled.")
            return None
        removed: list[Path] = []
        for selected_entry in selected_entries:
            if _run_process_is_active(selected_entry.path):
                _interrupt_incomplete_run(selected_entry.path)
            removed.extend(
                _delete_incomplete_run(
                    repository_root,
                    selected_entry,
                    delete_private_inputs=False,
                )
            )
        console.print(
            "Deleted incomplete run folders:\n"
            + "\n".join(f"- {path}" for path in removed)
        )
        return None
    entry = selected_entries[0]
    action = _choice(
        "Incomplete run action",
        {
            "1": "resume; completed stages are skipped",
            "2": "stop/abort the active run but keep all files for a later resume",
            "3": "remove failed/blocked jobs from this queue",
            "4": "delete this complete temporary queue including capture and work data",
            "0": "back to main menu",
        },
        "1",
    )
    if action == "0":
        return None
    if action == "1":
        if _run_process_is_active(entry.path):
            raise RuntimeError(
                "This run is already active in another terminal. Do not start it twice."
            )
        return entry.path
    if action == "2":
        _interrupt_incomplete_run(entry.path)
        console.print(
            "Run stopped and marked interrupted. Its files remain resumable."
        )
        return None
    if action == "3":
        _remove_failed_queue_jobs(entry.path, console)
        return None
    console.print(
        Panel(
            f"Temporary queue: {entry.path}\n"
            "This removes its capture, prepared frames, observations, job "
            "outputs and logs. Published results and data_local are untouched.\n"
            "This deletion cannot be undone.",
            title="Confirm deletion",
        )
    )
    if not typer.confirm("Delete the selected incomplete run?", default=False):
        console.print("Deletion cancelled.")
        return None
    if _run_process_is_active(entry.path):
        _interrupt_incomplete_run(entry.path)
    removed = _delete_incomplete_run(
        repository_root, entry, delete_private_inputs=True
    )
    console.print("Deleted:\n" + "\n".join(f"- {path}" for path in removed))
    return None


__all__ = [
    "_active_run_stage",
    "_delete_incomplete_run",
    "_interrupt_incomplete_run",
    "_manifest_process_is_active",
    "_remove_failed_queue_jobs",
    "_run_process_is_active",
    "_transaction_payload",
    "_validated_incomplete_run",
    "find_incomplete_transaction",
    "incomplete_resume_source",
    "incomplete_runs",
    "manage_incomplete_runs",
]
