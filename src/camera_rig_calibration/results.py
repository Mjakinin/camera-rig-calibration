from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultEntry:
    dataset_id: str
    run_id: str
    status: str
    path: Path
    methods: tuple[str, ...] = ()
    method_statuses: tuple[str, ...] = ()
    category: str = ""
    experiment_id: str = ""
    input_id: str = ""
    dataset_state: str = "unknown"
    variant: str = ""
    legacy: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def index_results(output_root: Path) -> list[ResultEntry]:
    """Index only layout-v2 experiment front doors.

    A result is intentionally undiscoverable until its root ``SUMMARY.json``
    exists. This avoids guessing through method internals or stale workspaces.
    """
    root = output_root.resolve()
    entries: list[ResultEntry] = []
    if not root.is_dir():
        return entries
    for summary_path in sorted(root.rglob("SUMMARY.json")):
        relative_parts = summary_path.parent.relative_to(root).parts
        if (
            len(relative_parts) < 3
            or relative_parts[0] not in {"real_vehicle", "simulation"}
            or any(
                name in relative_parts
                for name in {"methods", "evaluations", "attempts"}
            )
        ):
            continue
        payload = _read_json(summary_path)
        if payload.get("layout_version") != 2:
            continue
        experiment_root = summary_path.parent
        methods_value = payload.get("methods", [])
        rows = [
            row for row in methods_value if isinstance(row, dict)
        ] if isinstance(methods_value, list) else []
        method_names = tuple(
            dict.fromkeys(str(row.get("method", "")) for row in rows if row.get("method"))
        )
        method_statuses = tuple(
            f"{row.get('method')}/{row.get('label')}: {row.get('status')}"
            for row in rows
        )
        dataset_path = Path(str(payload.get("dataset_path", "")))
        dataset_state = (
            "available"
            if dataset_path.is_dir()
            and (dataset_path / "dataset.json").is_file()
            else "not local"
        )
        experiment = str(payload.get("experiment") or experiment_root.name)
        entries.append(
            ResultEntry(
                dataset_id=experiment,
                run_id=str(payload.get("queue_id") or "published"),
                status=str(payload.get("status") or "unknown"),
                path=experiment_root,
                methods=method_names,
                method_statuses=method_statuses,
                category=str(payload.get("category") or summary_path.parents[1].name),
                experiment_id=experiment,
                input_id="",
                dataset_state=dataset_state,
            )
        )
    return sorted(
        entries,
        key=lambda entry: (
            entry.category,
            entry.path.as_posix().lower(),
        ),
        reverse=True,
    )


def write_comparison(
    run_directory: Path, method_results: dict[str, dict[str, Any]]
) -> None:
    """Write the temporary per-job report consumed by layout-v2 publication."""
    comparison = run_directory / "07_COMPARISON"
    final = run_directory / "99_FINAL_RESULTS"
    comparison.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method_id, payload in method_results.items():
        rows.append(
            {
                "method": method_id,
                "status": payload.get("status", "MISSING"),
                "success": payload.get("success", False),
                "static_camera_count": len(
                    payload.get("available_static_cameras", [])
                ),
                "runtime_seconds": payload.get("runtime_seconds"),
                "warning": payload.get("error", ""),
            }
        )
    with (comparison / "method_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(rows[0]) if rows else ["method"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (comparison / "method_status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 5,
        "temporary": True,
        "methods": rows,
    }
    (final / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (final / "SUMMARY.txt").write_text(
        "Temporary method-job summary. The canonical layout-v2 publisher "
        "creates RESULT.*, SUMMARY.* and COMPARISON.* at the experiment front "
        "door after validation.\n",
        encoding="utf-8",
    )
