from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    status: str
    output_directory: Path
    outputs: dict[str, str]
    runtime_seconds: float


@dataclass(frozen=True)
class StageContract:
    stage_id: str
    depends_on: tuple[str, ...] = ()
    diagnostic: bool = False


def validate_stage_dag(contracts: Iterable[StageContract]) -> None:
    """Reject duplicate, missing and cyclic stage dependencies."""
    stages = tuple(contracts)
    by_id = {stage.stage_id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise ValueError("Stage IDs must be unique within one method pipeline")
    for stage in stages:
        missing = sorted(set(stage.depends_on) - set(by_id))
        if missing:
            raise ValueError(
                f"Stage '{stage.stage_id}' has unknown dependencies: "
                + ", ".join(missing)
            )

    visiting: set[str] = set()
    completed: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in completed:
            return
        if stage_id in visiting:
            raise ValueError(f"Stage dependency cycle includes '{stage_id}'")
        visiting.add(stage_id)
        for dependency in by_id[stage_id].depends_on:
            visit(dependency)
        visiting.remove(stage_id)
        completed.add(stage_id)

    for stage_id in by_id:
        visit(stage_id)


def run_stage(
    stage_id: str,
    output_directory: Path,
    action: Callable[[], Mapping[str, Path | str | int | float | bool | None]],
    *,
    inputs: Mapping[str, Path | str | int | float | bool | None] | None = None,
    parameters: Mapping[str, Any] | None = None,
    failure_is_diagnostic: bool = False,
) -> StageResult:
    """Execute one resumable method stage and atomically record its contract."""
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "stage_manifest.json"
    started_wall = _now()
    started = time.monotonic()
    base: dict[str, Any] = {
        "schema_version": 5,
        "stage_id": stage_id,
        "status": "RUNNING",
        "started_at": started_wall,
        "inputs": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in (inputs or {}).items()
        },
        "parameters": dict(parameters or {}),
    }
    _write_json(manifest_path, base)
    print(f"RIGCAL_STAGE_START {stage_id}", flush=True)
    try:
        produced = dict(action())
        rendered = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in produced.items()
        }
        elapsed = time.monotonic() - started
        _write_json(
            manifest_path,
            {
                **base,
                "status": "COMPLETED",
                "completed_at": _now(),
                "runtime_seconds": elapsed,
                "outputs": rendered,
            },
        )
        print(
            f"RIGCAL_STAGE_END {stage_id} elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        return StageResult(stage_id, "COMPLETED", output, rendered, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - started
        _write_json(
            manifest_path,
            {
                **base,
                "status": "FAILED",
                "completed_at": _now(),
                "runtime_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"RIGCAL_STAGE_FAILED {stage_id} elapsed_seconds={elapsed:.3f} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        if failure_is_diagnostic:
            return StageResult(stage_id, "FAILED_DIAGNOSTIC", output, {}, elapsed)
        raise
