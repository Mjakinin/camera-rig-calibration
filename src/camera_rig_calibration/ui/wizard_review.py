"""Focused wizard responsibilities extracted from the compatibility facade."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_user_config
from ..config.models import (
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    McapSettings,
    MethodSettings,
    MarkerSettings,
    MovingCameraSettings,
    IntrinsicScanSettings,
    InputSourceKind,
    ObservationQualitySettings,
    ProjectSettings,
    RigConfig,
    SamplingSettings,
    SceneType,
    SelectionSettings,
    SimulationSettings,
    StaticCameraSettings,
    effective_observation_quality,
)
from ..dataset.discovery import (
    IMAGE_SUFFIXES,
    discover_image_directories,
    discover_inputs,
    inspect_prepared_dataset,
    media_path_role,
    safe_id,
)
from ..doctor import run_checks
from ..experiments import automatic_method_label
from ..input.topics import McapTopic, list_mcap_topics
from ..input.video_geometry import probe_video_geometry
from ..intrinsics_profiles import (
    IntrinsicProfile,
    discover_intrinsic_profiles,
    intrinsic_dimensions,
)
from ..inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    PreparedDatasetSummary,
    RawInputSummary,
    SimulationExperimentSummary,
    discover_prepared_datasets,
    discover_raw_input_folders,
    discover_simulation_experiments,
    find_matching_simulation,
    format_simulation_parameters,
)
from ..registry import (
    calibration_methods,
    experiment_providers,
    input_adapters,
)
from ..runtime import PipelineOrchestrator
from ..observation_quality import filter_observations
from ..observations import ResolvedSelections, resolve_selections
from ..queueing import SelectionReviewJob, save_batch
# Compatibility hooks wrapped by the product policy stack. The concrete result
# browser lives under ui/, but these names remain stable until the wrappers are
# converted to explicit composition.
from ..publication import reconcile_existing_experiment
from ..visualization import launch_isolated_rviz



from .wizard_method_jobs import (
    _method_job_summary,
)
from .wizard_models import (
    MethodQueueJob,
    WizardBack,
    WizardOutcome,
)
from .wizard_bindings import current_wizard_bindings
from .wizard_prompts import (
    _clear_terminal,
)
from .wizard_saved_flow import (
    show_summary,
)

def _review_common_anchor(
    selections: list[ResolvedSelections],
    console: Console,
) -> tuple[int, bool]:
    aggregate: dict[int, dict[str, object]] = {}
    for resolved in selections:
        inventory = resolved.payload.get("raw_marker_inventory") or [
            {
                "id": marker_id,
                "raw_observations": 0,
                "static_cameras": [],
                "moving_frames": 0,
                "accepted_observations": 0,
                "compatible": marker_id
                in resolved.payload["evaluation_anchor"].get(
                    "observation_candidates", []
                ),
                "automatic_candidate": marker_id
                in resolved.payload["evaluation_anchor"].get(
                    "automatic_observation_candidates", []
                ),
                "issues": [],
            }
            for marker_id in resolved.detected_marker_ids
        ]
        for item in inventory:
            marker_id = int(item["id"])
            row = aggregate.setdefault(
                marker_id,
                {
                    "id": marker_id,
                    "raw_observations": 0,
                    "accepted_observations": 0,
                    "static_cameras": set(),
                    "moving_frames": 0,
                    "compatible_jobs": 0,
                    "automatic_jobs": 0,
                    "scores": [],
                    "rmse": [],
                    "areas": [],
                    "issues": set(),
                },
            )
            row["raw_observations"] = max(
                int(row["raw_observations"]),
                int(item.get("raw_observations") or 0),
            )
            row["accepted_observations"] = int(
                row["accepted_observations"]
            ) + int(item.get("accepted_observations") or 0)
            row["static_cameras"].update(item.get("static_cameras") or [])
            row["moving_frames"] = max(
                int(row["moving_frames"]),
                int(item.get("moving_frames") or 0),
            )
            row["compatible_jobs"] = int(row["compatible_jobs"]) + int(
                bool(item.get("compatible"))
            )
            row["automatic_jobs"] = int(row["automatic_jobs"]) + int(
                bool(item.get("automatic_candidate"))
            )
            for source, key in (
                ("median_selection_score", "scores"),
                ("median_pnp_reprojection_rmse_px", "rmse"),
                ("median_marker_area_ratio", "areas"),
            ):
                value = item.get(source)
                if value is not None:
                    row[key].append(float(value))
            row["issues"].update(str(value) for value in item.get("issues", []))
    if not aggregate:
        raise RuntimeError(
            "Manual common-anchor review found no actually detected marker ID."
        )
    job_count = len(selections)
    compatible = [
        row
        for row in aggregate.values()
        if int(row["automatic_jobs"]) == job_count
    ]

    def rank(row: dict[str, object]) -> tuple[object, ...]:
        scores = list(row["scores"])
        rmse = list(row["rmse"])
        areas = list(row["areas"])
        return (
            int(row["compatible_jobs"]),
            len(row["static_cameras"]),
            int(row["moving_frames"]),
            min(scores) if scores else 0.0,
            int(row["accepted_observations"]),
            -(max(rmse) if rmse else float("inf")),
            min(areas) if areas else 0.0,
            -int(row["id"]),
        )

    recommended = max(compatible or list(aggregate.values()), key=rank)
    ordered = sorted(aggregate.values(), key=lambda row: int(row["id"]))
    table = Table(title="Common evaluation and export anchor")
    table.add_column("#", justify="right")
    table.add_column("Marker ID")
    table.add_column("Raw")
    table.add_column("Accepted")
    table.add_column("Static cameras")
    table.add_column("Moving frames")
    table.add_column("Compatible jobs")
    table.add_column("Assessment", overflow="fold")
    for index, row in enumerate(ordered, 1):
        fully_compatible = int(row["compatible_jobs"]) == job_count
        assessment = (
            "recommended"
            if row is recommended
            else "compatible"
            if fully_compatible
            else "problematic: "
            + (
                "; ".join(sorted(row["issues"]))
                or "not supported by every method variant"
            )
        )
        table.add_row(
            str(index),
            str(row["id"]),
            str(row["raw_observations"]),
            str(row["accepted_observations"]),
            ",".join(sorted(row["static_cameras"])) or "-",
            str(row["moving_frames"]),
            f"{row['compatible_jobs']}/{job_count}",
            assessment,
        )
    console.print(table)
    default = ordered.index(recommended) + 1
    while True:
        raw = str(
            typer.prompt(
                "Common anchor marker table number (b = back)",
                default=default,
            )
        ).strip().lower()
        if raw in {"b", "back", "0"}:
            raise WizardBack(
                "Common anchor review paused; prepared observations remain reusable."
            )
        if raw.isdigit() and 1 <= int(raw) <= len(ordered):
            chosen = ordered[int(raw) - 1]
            break
        typer.echo(f"Choose a table number from 1 to {len(ordered)}.")
    warned = int(chosen["compatible_jobs"]) != job_count
    if warned and not typer.confirm(
        "This marker is not compatible with every method variant. Continue "
        "without any fallback anchor? Unsupported exports will be unavailable.",
        default=False,
    ):
        return _review_common_anchor(selections, console)
    return int(chosen["id"]), warned


def review_selection_candidates(
    config: RigConfig,
    resolved: ResolvedSelections,
    run_directory: Path,
    console: Console,
    *,
    review_evaluation_anchor: bool = True,
) -> dict[str, object]:
    """One attended checkpoint after all static and moving observations exist."""
    payload = resolved.payload
    console.print(
        Panel(
            "Intrinsics validate the camera models but do not select a coordinate "
            "origin. The recommendations below use the actual static and moving "
            "ArUco observation graph. These values are frozen before any AP method "
            "starts. The common evaluation/export anchor is also frozen here "
            "when manual review was requested.",
            title="One-time scientific selection review",
        )
    )
    labels = {
        camera.id: camera.label or ""
        for camera in config.static_cameras
    }
    selected_methods = set(config.methods.enabled)
    roots = payload["ap01_root_camera"]["candidates"]
    root_table = Table(title="AP01 root-camera candidates")
    root_table.add_column("#", justify="right")
    root_table.add_column("Camera ID")
    root_table.add_column("Label")
    root_table.add_column("Reachable")
    root_table.add_column("Direct")
    root_table.add_column("Moving bridges")
    root_table.add_column("Markers")
    root_table.add_column("Accepted")
    root_table.add_column("Median PnP RMSE")
    root_table.add_column("Median area")
    root_table.add_column("Status")
    for index, item in enumerate(roots, 1):
        status = (
            "recommended"
            if item.get("recommended")
            else "compatible"
            if item.get("compatible")
            else "not compatible"
        )
        rmse = item.get("median_pnp_reprojection_rmse_px")
        area = item.get("median_marker_area_px2")
        root_table.add_row(
            str(index),
            str(item["id"]),
            labels.get(str(item["id"]), ""),
            f"{len(item['reachable_cameras'])}/{len(config.static_cameras)}",
            str(len(item["direct_connections"])),
            str(len(item["moving_bridges"])),
            str(item["distinct_markers"]),
            str(item["observations"]),
            f"{float(rmse):.2f} px" if rmse is not None else "unknown",
            f"{float(area):.0f} px²" if area is not None else "unknown",
            status,
        )
    if "ap01" in selected_methods:
        console.print(root_table)

    marker_rows = payload["ap03_single_scale_marker"]["candidates"]
    marker_table = Table(title="Method-marker candidates")
    marker_table.add_column("#", justify="right")
    marker_table.add_column("Marker")
    marker_table.add_column("Static-only reached")
    marker_table.add_column("Combined reached")
    marker_table.add_column("Moving frames")
    marker_table.add_column("Static direct")
    marker_table.add_column("Accepted")
    marker_table.add_column("Median PnP RMSE")
    marker_table.add_column("Median area")
    if "ap02" in selected_methods:
        marker_table.add_column("AP02 reference")
    if "ap03" in selected_methods:
        marker_table.add_column("AP03 single scale")
        marker_table.add_column("AP03 multi scale")
    ap02_recommended = {
        int(item["id"])
        for item in payload["ap02_reference_marker"]["candidates"]
        if item.get("recommended")
    }
    for index, item in enumerate(marker_rows, 1):
        marker_id = int(item["id"])
        cells = [
            str(index),
            str(marker_id),
            f"{item['static_graph_reachable_count']}/{len(config.static_cameras)}",
            (
                f"{item['combined_graph_reachable_static_count']}/"
                f"{len(config.static_cameras)}"
            ),
            str(item["moving_frames"]),
            str(item["static_camera_count"]),
            str(item["accepted_observations"]),
            (
                f"{float(item['median_pnp_reprojection_rmse_px']):.2f} px"
                if item.get("median_pnp_reprojection_rmse_px") is not None
                else "unknown"
            ),
            (
                f"{float(item['median_marker_area_px2']):.0f} px²"
                if item.get("median_marker_area_px2") is not None
                else "unknown"
            ),
        ]
        if "ap02" in selected_methods:
            ap02_details = next(
                (
                    candidate
                    for candidate in payload[
                        "ap02_reference_marker"
                    ]["candidates"]
                    if int(candidate["id"]) == marker_id
                ),
                item,
            )
            cells.append(
                (
                    "recommended (diagnostic partial)"
                    if ap02_details.get("diagnostic_partial")
                    else "recommended"
                )
                if marker_id in ap02_recommended
                else (
                    "diagnostic partial"
                    if ap02_details.get("diagnostic_partial")
                    else "compatible"
                )
                if ap02_details.get("compatible")
                else "not compatible"
            )
        if "ap03" in selected_methods:
            cells.append(
                "recommended"
                if item.get("recommended")
                else "compatible"
                if item["ap03_compatible"]
                else "not compatible"
            )
            cells.append(
                "included by auto"
                if marker_id in resolved.ap03_multi_marker_ids
                else "not compatible"
            )
        marker_table.add_row(*cells)
    if selected_methods & {"ap02", "ap03"}:
        console.print(marker_table)

    root_default = next(
        index
        for index, item in enumerate(roots, 1)
        if str(item["id"]) == resolved.root_camera
    )
    root = resolved.root_camera
    ap02 = resolved.ap02_reference_marker_id
    single = resolved.ap03_single_scale_marker_id
    multi: str | list[int] = list(
        resolved.ap03_multi_marker_ids
    )
    automatic = Table(title="Automatic selections")
    automatic.add_column("Role")
    automatic.add_column("Resolution")
    if "ap01" in selected_methods:
        automatic.add_row(
            "AP01 root camera",
            f"{config.methods.ap01.root_camera} → {root} (recommended)",
        )
    if "ap02" in selected_methods:
        automatic.add_row(
            "AP02 reference marker",
            (
                f"{config.methods.ap02.reference_marker_id} → marker {ap02} "
                "(recommended)"
            ),
        )
    if "ap03" in selected_methods:
        automatic.add_row(
            "AP03 single scale marker",
            (
                f"{config.methods.ap03.single.scale_marker_id} → marker "
                f"{single} (recommended)"
            ),
        )
        automatic.add_row(
            "AP03 multi marker set",
            (
                f"{config.methods.ap03.multi.marker_ids} → "
                f"{len(resolved.ap03_multi_marker_ids)} compatible markers"
            ),
        )
    console.print(automatic)
    typer.echo("\nEnter  Accept automatic selections")
    typer.echo("O      Override")
    typer.echo("B      Back")
    while True:
        action = typer.prompt(
            "Selection review action",
            default="",
            show_default=False,
        ).strip().lower()
        if action in {"", "accept", "a"}:
            break
        if action in {"b", "back"}:
            _clear_terminal()
            raise WizardBack(
                "Selection review paused; prepared observations remain "
                f"resumable at {run_directory}"
            )
        if action not in {"o", "override"}:
            typer.echo("Press Enter to accept, O to override, or B to go back.")
            continue
        if "ap01" in selected_methods:
            while True:
                root_raw = typer.prompt(
                    "AP01 root camera table number (b = back)",
                    default=str(root_default),
                ).strip()
                if root_raw.lower() in {"b", "back"}:
                    break
                if (
                    root_raw.isdigit()
                    and 1 <= int(root_raw) <= len(roots)
                    and roots[int(root_raw) - 1].get("compatible", True)
                ):
                    root = str(roots[int(root_raw) - 1]["id"])
                    break
                typer.echo(
                    "Choose the table number of a compatible root camera."
                )
            if root_raw.lower() in {"b", "back"}:
                continue
        if "ap02" in selected_methods:
            ap02_default = next(
                index
                for index, item in enumerate(marker_rows, 1)
                if int(item["id"]) == ap02
            )
            while True:
                raw = typer.prompt(
                    "AP02 reference marker table number (b = back)",
                    default=str(ap02_default),
                ).strip()
                if raw.lower() in {"b", "back"}:
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(marker_rows):
                    selected = marker_rows[int(raw) - 1]
                    ap02_candidate = next(
                        (
                            candidate
                            for candidate in payload[
                                "ap02_reference_marker"
                            ]["candidates"]
                            if int(candidate["id"]) == int(selected["id"])
                        ),
                        selected,
                    )
                    if (
                        ap02_candidate.get("compatible")
                        or config.methods.ap02.reference_marker_selection_mode
                        == "manual"
                    ):
                        if (
                            not ap02_candidate.get("compatible")
                            and not typer.confirm(
                                "This marker is detected but does not connect "
                                "a complete AP02 graph under the current "
                                "filters. Freeze it as a documented partial/"
                                "diagnostic reference?",
                                default=False,
                            )
                        ):
                            continue
                        ap02 = int(selected["id"])
                        break
                typer.echo(
                    "Choose a detected marker table number."
                )
            if raw.lower() in {"b", "back"}:
                continue
        if "ap03" in selected_methods:
            single_default = next(
                index
                for index, item in enumerate(marker_rows, 1)
                if int(item["id"]) == single
            )
            while True:
                raw = typer.prompt(
                    "AP03 Single scale marker table number (b = back)",
                    default=str(single_default),
                ).strip()
                if raw.lower() in {"b", "back"}:
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(marker_rows):
                    selected = marker_rows[int(raw) - 1]
                    if selected.get("ap03_compatible"):
                        single = int(selected["id"])
                        break
                typer.echo(
                    "Choose the table number of an AP03-compatible marker."
                )
            if raw.lower() in {"b", "back"}:
                continue
            compatible_rows = {
                index: int(item["id"])
                for index, item in enumerate(marker_rows, 1)
                if item.get("ap03_compatible")
            }
            multi_default = ",".join(
                str(index)
                for index, marker_id in compatible_rows.items()
                if marker_id in resolved.ap03_multi_marker_ids
            )
            while True:
                raw = typer.prompt(
                    "AP03 Multi marker table numbers, comma-separated, "
                    "or all (b = back)",
                    default=multi_default,
                ).strip().lower()
                if raw in {"b", "back"}:
                    break
                if raw == "all":
                    multi = list(compatible_rows.values())
                    break
                try:
                    numbers = list(
                        dict.fromkeys(
                            int(item.strip())
                            for item in raw.split(",")
                            if item.strip()
                        )
                    )
                except ValueError:
                    numbers = []
                if numbers and all(
                    number in compatible_rows for number in numbers
                ):
                    multi = [compatible_rows[number] for number in numbers]
                    break
                typer.echo(
                    "Choose compatible table numbers or enter 'all'."
                )
            if raw in {"b", "back"}:
                continue
        break
    if multi == "auto":
        multi = list(resolved.ap03_multi_marker_ids)
    choices: dict[str, object] = {
        "root_camera": root,
        "ap02_reference_marker_id": ap02,
        "ap03_single_scale_marker_id": single,
        "ap03_multi_marker_ids": multi,
    }
    evaluation_anchor = resolved.evaluation_anchor_marker_id
    if (
        review_evaluation_anchor
        and config.evaluation.enabled
        and config.evaluation.anchor_selection_mode == "review_once"
    ):
        evaluation_anchor, _ = _review_common_anchor([resolved], console)
        choices["evaluation_anchor_marker_id"] = evaluation_anchor
    summary = Table(title="Selections to freeze before the method queue")
    summary.add_column("Role")
    summary.add_column("Selected")
    if "ap01" in selected_methods:
        summary.add_row("AP01 root camera", root)
    if "ap02" in selected_methods:
        summary.add_row("AP02 reference marker", str(ap02))
    if "ap03" in selected_methods:
        summary.add_row("AP03 Single scale marker", str(single))
    if "ap03" in selected_methods:
        summary.add_row(
            "AP03 Multi marker set",
            ",".join(str(value) for value in multi),
        )
    summary.add_row(
        "Common evaluation and export anchor",
        (
            "pending queue-wide marker review"
            if (
                config.evaluation.anchor_selection_mode == "review_once"
                and not review_evaluation_anchor
            )
            else (
                f"marker {evaluation_anchor} "
                + (
                    "(manual post-preflight selection)"
                    if config.evaluation.anchor_selection_mode
                    == "review_once"
                    else "(automatic preflight recommendation)"
                )
            )
        ),
    )
    console.print(summary)
    return choices


def review_queue_selection_candidates(
    jobs: tuple[SelectionReviewJob, ...],
    run_directory: Path,
    console: Console,
) -> dict[str, dict[str, object]]:
    """Review every manual queue variant at one post-preflight checkpoint."""

    review_selection_candidates = (
        current_wizard_bindings().review_selection_candidates
    )
    console.print(
        Panel(
            f"{len(jobs)} manually configured method variant(s) are ready. "
            "Each section below uses that variant's own ArUco and observation-"
            "quality filters. Automatic variants need no confirmation.",
            title="Queue selection review",
        )
    )
    decisions: dict[str, dict[str, object]] = {}
    for index, job in enumerate(jobs, 1):
        console.print(
            Panel(
                f"Variant {index}/{len(jobs)}\n"
                f"Queue entry: {job.entry_id}\n"
                f"Method: {job.config.methods.enabled[0]}\n"
                f"Preflight evidence: {job.output_directory}",
                title="Manual selection",
            )
        )
        decisions[job.entry_id] = {}
        anchor_only_review = (
            job.config.selection.mode != "review_once"
            and job.config.evaluation.enabled
            and job.config.evaluation.anchor_selection_mode == "review_once"
        )
        if not anchor_only_review:
            if (
                job.config.evaluation.enabled
                and job.config.evaluation.anchor_selection_mode == "review_once"
            ):
                decisions[job.entry_id] = review_selection_candidates(
                    job.config,
                    job.selections,
                    run_directory,
                    console,
                    review_evaluation_anchor=False,
                )
            else:
                # Keep the long-standing four-argument reviewer contract for
                # method-only reviews.  The common anchor has its own
                # queue-wide review below.
                decisions[job.entry_id] = review_selection_candidates(
                    job.config,
                    job.selections,
                    run_directory,
                    console,
                )
    anchor_jobs = [
        job
        for job in jobs
        if job.config.evaluation.enabled
        and job.config.evaluation.anchor_selection_mode == "review_once"
    ]
    if anchor_jobs:
        anchor, warned = _review_common_anchor(
            [job.selections for job in anchor_jobs],
            console,
        )
        for job in anchor_jobs:
            decisions[job.entry_id]["evaluation_anchor_marker_id"] = anchor
            decisions[job.entry_id][
                "evaluation_anchor_warning_confirmed"
            ] = warned
    return decisions


def show_queue_summary(outcome: WizardOutcome, console: Console) -> None:
    if len(outcome.runs) == 1:
        show_summary(outcome.config, outcome.path, console)
        return
    table = Table(title="Final calibration queue")
    table.add_column("#", justify="right")
    if outcome.batch_path is not None:
        table.add_column("Experiment")
        table.add_column("Input")
    table.add_column("Run label")
    table.add_column("Method")
    table.add_column("Key configuration", overflow="fold")
    table.add_column("Saved config", overflow="fold")
    for index, queued in enumerate(outcome.runs, 1):
        prepare_only = (
            queued.config.project.execution_mode == "prepare_only"
        )
        method_id = queued.config.methods.enabled[0]
        job = MethodQueueJob(
            method_id=method_id,
            label=queued.config.project.run_label,
            methods=queued.config.methods,
            markers=queued.config.markers,
            observation_quality=queued.config.observation_quality,
            colmap=queued.config.colmap,
            evaluation=queued.config.evaluation,
        )
        row = [str(index)]
        if outcome.batch_path is not None:
            row.extend(
                [
                    queued.config.dataset.id,
                    (
                        "reuse local"
                        if queued.config.dataset.prepared_root is not None
                        else "new capture"
                    ),
                ]
            )
        row.extend(
            [
                queued.config.project.run_label,
                (
                    "prepare input only"
                    if prepare_only
                    else calibration_methods.get(method_id).display_name
                ),
                (
                    "capture/prepare/validate once; no AP method"
                    if prepare_only
                    else _method_job_summary(job)
                ),
                str(queued.path),
            ]
        )
        table.add_row(*row)
    console.print(table)
    if outcome.batch_path is not None:
        console.print(
            f"Batch: {outcome.batch_path} | "
            f"{len(outcome.queue_paths)} experiments | "
            f"{len(outcome.runs)} total queue rows"
        )
        return
    console.print(
        f"Dataset: {outcome.config.dataset.id} | {len(outcome.runs)} independent runs | "
        "shared immutable input under "
        "results/<category>/<rate-or-factor>/<experiment>/"
    )


__all__ = [
    '_review_common_anchor',
    'review_selection_candidates',
    'review_queue_selection_candidates',
    'show_queue_summary',
]
