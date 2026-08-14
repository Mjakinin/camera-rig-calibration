"""Queue behavior grouped by one cohesive responsibility."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import Field, model_validator
from rich.console import Console
from rich.table import Table

from ..config import config_fingerprint, load_config, save_config
from ..config.models import RigConfig, StrictModel
from ..config.models import (
    DatasetSettings,
    EvaluationSettings,
    MarkerSettings,
    ObservationQualitySettings,
)
from ..dataset.discovery import safe_id
from ..dataset_identity import build_dataset_identity
from ..experiments import (
    automatic_method_label,
    evaluation_fingerprint,
    experiment_paths,
)
from ..filesystem import promote_directory, rename_with_retry
from ..methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from ..observations import (
    ResolvedSelections,
    freeze_selections,
    write_selection_candidates_csv,
)
from ..runtime import PipelineOrchestrator, observation_id
from ..preflight import (
    PreflightJob,
    QueuePreflightResult,
    run_queue_preflight,
)
from ..observation_quality import filter_observations
from ..publication import (
    publish_preparation_transaction,
    publish_queue_transaction,
)
from ..storage_layout import queue_temporary_root


@dataclass(frozen=True)
class SelectionReviewJob:
    """One independently filtered method job awaiting an attended selection."""

    entry_id: str
    config: RigConfig
    selections: ResolvedSelections
    output_directory: Path


QueueSelectionReviewer = Callable[
    [tuple[SelectionReviewJob, ...], Path],
    dict[str, dict[str, Any]],
]


from .common import (
    _write_json,
)
from .models import (
    QueueConfig,
)
from .bindings import current_queue_bindings


class QueueEvaluationMixin:
    def _run_common_evaluations(
        self,
        queue: QueueConfig,
        results: dict[str, dict[str, Any]],
        requested_configs: list[RigConfig],
    ) -> None:
        """Evaluate each experiment with one anchor shared by every method.

        This stage never re-runs a calibration method. Candidate anchors are
        tried in deterministic observation-quality order; unsuccessful
        evaluation artifacts remain visible for diagnosis.
        """
        groups: dict[
            tuple[str, str],
            list[tuple[Path, dict[str, Any], RigConfig]],
        ] = {}
        for entry, requested_config in zip(
            queue.entries, requested_configs, strict=True
        ):
            row = results.get(entry.id, {})
            if row.get("status") not in {
                "completed",
                "published",
                "duplicate_skipped",
            }:
                continue
            result_path = Path(str(row.get("result", "")))
            manifest_path = result_path / "provenance" / "run_manifest.json"
            config_path = result_path / "provenance" / "resolved_config.yaml"
            if not manifest_path.is_file() or not config_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_root = str(experiment_paths(requested_config).root)
            input_id = str(manifest.get("input_id", ""))
            if not experiment_root or not input_id:
                continue
            groups.setdefault((experiment_root, input_id), []).append(
                (result_path, manifest, requested_config)
            )

        label_by_method = {
            "ap01": "AP01",
            "ap02": "AP02",
            "ap03": "AP03_MULTI",
        }
        directory_by_method = {
            "ap01": "diagnostics/method",
            "ap02": "diagnostics/method",
            "ap03": "diagnostics/method",
        }
        for (experiment_text, input_id), group in groups.items():
            experiment = Path(experiment_text)
            first_path, first_manifest, first_config = group[0]
            enabled_group = [item for item in group if item[2].evaluation.enabled]
            if not enabled_group:
                continue
            group = enabled_group
            first_path, first_manifest, first_config = group[0]
            transaction_dataset = (
                queue_temporary_root(first_config, queue.id) / "dataset"
            )
            dataset_root = (
                transaction_dataset
                if transaction_dataset.is_dir()
                else experiment_paths(first_config).dataset_root
            )
            observations = dataset_root / "observations"
            candidate_path = observations / "SELECTION_CANDIDATES.json"
            if not candidate_path.is_file():
                unavailable = (
                    queue_temporary_root(first_config, queue.id)
                    / "results"
                    / "evaluations"
                    / "COMMON_EVALUATION_UNAVAILABLE.json"
                )
                _write_json(
                    unavailable,
                    {
                        "schema_version": 5,
                        "layout_version": 2,
                        "status": "unavailable",
                        "reason": (
                            "The complete dataset has no "
                            "SELECTION_CANDIDATES.json; common evaluation was "
                            "not silently skipped."
                        ),
                        "dataset_root": str(dataset_root),
                    },
                )
                self.console.print(
                    "[yellow]Common evaluation unavailable: the complete "
                    "selection-candidate evidence is missing.[/yellow]"
                )
                continue
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            eligible = set(
                int(value)
                for value in payload["evaluation_anchor"]["observation_candidates"]
            )
            configured_anchors = [
                config.evaluation.anchor_marker_id for _, _, config in group
            ]
            explicit_anchors = {
                int(value) for value in configured_anchors if isinstance(value, int)
            }
            if len(explicit_anchors) != 1 or any(
                not isinstance(value, int) for value in configured_anchors
            ):
                final = (
                    experiment / "evaluations" / "COMMON_EVALUATION_UNAVAILABLE.json"
                )
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_text(
                    json.dumps(
                        {
                            "status": "unavailable",
                            "reason": (
                                "The queue did not contain exactly one "
                                "evaluation anchor frozen for every method "
                                "during preflight."
                            ),
                            "configured_anchor_marker_ids": sorted(explicit_anchors),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue
            requested_anchor = next(iter(explicit_anchors))
            ranked = [
                item
                for item in payload["ap03_single_scale_marker"]["candidates"]
                if int(item["id"]) == requested_anchor and int(item["id"]) in eligible
            ]
            methods: list[tuple[str, Path]] = []
            for result_path, manifest, _ in group:
                method_id = str(manifest.get("method_id", ""))
                if method_id not in directory_by_method:
                    continue
                if method_id == "ap02":
                    status_path = (
                        result_path / "diagnostics" / "method" / "METHOD_STATUS.json"
                    )
                    if status_path.is_file():
                        try:
                            method_status = json.loads(
                                status_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            method_status = {}
                        if not method_status.get("comparison_eligible", True):
                            self.console.print(
                                "[yellow]AP02 diagnostic partial result is "
                                "excluded from common primary-method "
                                "evaluation.[/yellow]"
                            )
                            continue
                variant = result_path.name
                methods.append(
                    (
                        f"{label_by_method[method_id]}__{variant}",
                        result_path / directory_by_method[method_id],
                    )
                )
            if not methods or not ranked:
                continue

            selection: dict[str, Any] | None = None
            for item in ranked:
                anchor = int(item["id"])
                eval_sha = evaluation_fingerprint(first_config, anchor)[:8]
                method_identity = [
                    {
                        "method_id": manifest.get("method_id"),
                        "variant": manifest.get("variant"),
                        "method_fingerprint": manifest.get("method_fingerprint"),
                    }
                    for _, manifest, _ in group
                ]
                job_fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "evaluation": first_config.evaluation.model_dump(
                                mode="json"
                            ),
                            "anchor": anchor,
                            "methods": method_identity,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                transaction_evaluations = (
                    queue_temporary_root(first_config, queue.id)
                    / "results"
                    / "evaluations"
                )
                output = transaction_evaluations / f"anchor_marker_{anchor}_{eval_sha}"
                previous_status = output / "COMMON_ANCHOR_STATUS.json"
                if previous_status.is_file() and not any(
                    config.project.duplicate_policy == "force" for _, _, config in group
                ):
                    previous = json.loads(previous_status.read_text(encoding="utf-8"))
                    if (
                        previous.get("success_for_every_method")
                        and previous.get("evaluation_job_fingerprint")
                        == job_fingerprint
                    ):
                        selection = previous
                        self.console.print(
                            f"[dim]Exact common evaluation already exists; "
                            f"skipped: {output}[/dim]"
                        )
                        break
                argv = [
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.evaluation.marker_consistency",
                    "--dataset",
                    str(dataset_root),
                    "--results-root",
                    str(experiment),
                    "--observations-root",
                    str(observations),
                    "--output-root",
                    str(output),
                    "--anchor-marker-id",
                    str(anchor),
                    "--marker-length-m",
                    str(first_config.markers.length_m),
                    "--reprojection-threshold-px",
                    str(first_config.evaluation.reprojection_threshold_px),
                    "--min-inliers",
                    str(first_config.evaluation.minimum_inliers),
                    "--ransac-iters",
                    str(first_config.evaluation.ransac_iterations),
                    "--min-triangulation-angle-deg",
                    str(first_config.evaluation.minimum_triangulation_angle_deg),
                    "--max-moving-observations-per-marker",
                    str(first_config.evaluation.maximum_moving_observations_per_marker),
                    "--cameras",
                    ",".join(camera.id for camera in first_config.static_cameras),
                ]
                for label, directory in methods:
                    argv += ["--method", f"{label}={directory.resolve()}"]
                self.console.print(
                    f"[cyan]Common evaluation with frozen marker {anchor} "
                    f"for {len(methods)} methods[/cyan]"
                )
                started = time.monotonic()
                completed = subprocess.run(
                    argv,
                    cwd=self.repository_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                runtime_seconds = time.monotonic() - started
                self.console.print(
                    f"[cyan]Common evaluation finished in "
                    f"{runtime_seconds:.1f} s[/cyan]"
                )
                (output / "evaluation.log").parent.mkdir(parents=True, exist_ok=True)
                (output / "evaluation.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                summary_path = (
                    output
                    / "marker_consistency"
                    / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json"
                )
                summaries = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.is_file()
                    else []
                )
                success = (
                    completed.returncode == 0
                    and len(summaries) == len(methods)
                    and all(
                        not str(row.get("status", "")).startswith("NOT_AVAILABLE")
                        for row in summaries
                    )
                )
                _status = {
                    "anchor_marker_id": anchor,
                    "success_for_every_method": success,
                    "evaluation_job_fingerprint": job_fingerprint,
                    "runtime_seconds": runtime_seconds,
                    "method_statuses": {
                        str(row.get("method")): str(row.get("status"))
                        for row in summaries
                    },
                    "output": str(output),
                }
                (output / "COMMON_ANCHOR_STATUS.json").write_text(
                    json.dumps(_status, indent=2) + "\n",
                    encoding="utf-8",
                )
                if success:
                    selection = _status
                    comparison = output / "comparison"
                    comparison.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        summary_path,
                        comparison / "COMMON_METHOD_EVALUATION_SUMMARY.json",
                    )
                    support_path = (
                        output / "marker_consistency" / "COMMON_SUPPORT_REPORT.json"
                    )
                    if support_path.is_file():
                        shutil.copy2(
                            support_path,
                            comparison / "COMMON_SUPPORT_REPORT.json",
                        )
                    # AP03 Single is a diagnostic scale on the same COLMAP
                    # reconstruction. Evaluate it separately with the selected
                    # common anchor, but never let this diagnostic replace or
                    # block AP03 Multi in the common method comparison.
                    for result_path, manifest, config in group:
                        if manifest.get("method_id") != "ap03":
                            continue
                        single_output = (
                            output / "diagnostics" / f"ap03_single_{result_path.name}"
                        )
                        single_argv = [
                            sys.executable,
                            "-m",
                            "camera_rig_calibration.evaluation.marker_consistency",
                            "--dataset",
                            str(dataset_root),
                            "--results-root",
                            str(result_path),
                            "--observations-root",
                            str(observations),
                            "--output-root",
                            str(single_output),
                            "--anchor-marker-id",
                            str(anchor),
                            "--marker-length-m",
                            str(config.markers.length_m),
                            "--reprojection-threshold-px",
                            str(config.evaluation.reprojection_threshold_px),
                            "--min-inliers",
                            str(config.evaluation.minimum_inliers),
                            "--ransac-iters",
                            str(config.evaluation.ransac_iterations),
                            "--min-triangulation-angle-deg",
                            str(config.evaluation.minimum_triangulation_angle_deg),
                            "--max-moving-observations-per-marker",
                            str(
                                config.evaluation.maximum_moving_observations_per_marker
                            ),
                            "--cameras",
                            ",".join(camera.id for camera in config.static_cameras),
                            "--method",
                            (
                                "AP03_SINGLE="
                                + str(
                                    (
                                        result_path
                                        / "diagnostics"
                                        / "method"
                                        / "scale_single"
                                    ).resolve()
                                )
                            ),
                        ]
                        diagnostic = subprocess.run(
                            single_argv,
                            cwd=self.repository_root,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        single_output.mkdir(parents=True, exist_ok=True)
                        (single_output / "evaluation.log").write_text(
                            diagnostic.stdout, encoding="utf-8"
                        )
                        _write_json(
                            single_output / "DIAGNOSTIC_STATUS.json",
                            {
                                "role": "diagnostic",
                                "method": "AP03_SINGLE",
                                "anchor_marker_id": anchor,
                                "returncode": diagnostic.returncode,
                                "success": diagnostic.returncode == 0,
                            },
                        )
                    break
            final = (
                queue_temporary_root(first_config, queue.id)
                / "results"
                / "evaluations"
                / (
                    "SELECTED_COMMON_EVALUATION.json"
                    if selection is not None
                    else "COMMON_EVALUATION_UNAVAILABLE.json"
                )
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_text(
                json.dumps(
                    selection
                    or {
                        "status": "unavailable",
                        "reason": (
                            "The single preflight-frozen evaluation anchor was "
                            "not reconstructable for every completed method. "
                            "Calibration results remain available and no "
                            "replacement anchor was attempted."
                        ),
                        "candidate_marker_ids": [int(item["id"]) for item in ranked],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


__all__ = ["QueueEvaluationMixin"]
