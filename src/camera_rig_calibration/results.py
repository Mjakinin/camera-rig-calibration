from __future__ import annotations

import csv
import json
import math
import os
import shutil
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
    category: str = ""
    experiment_id: str = ""
    input_id: str = ""
    variant: str = ""
    legacy: bool = False


def _display_method_id(method: object) -> str:
    """Collapse legacy AP03 branches into the combined public AP03 method."""

    value = str(method)
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"ap03_single", "ap03_multi"}:
        return "ap03"
    return value


def _display_method_sort_key(method: str) -> tuple[int, str]:
    order = {"ap01": 1, "ap02": 2, "ap03": 3}
    return order.get(method.lower(), 99), method.lower()


def index_results(output_root: Path) -> list[ResultEntry]:
    root = output_root.resolve()
    entries: list[ResultEntry] = []
    migrated_legacy_roots: set[Path] = set()
    if root.is_dir():
        legacy_manifests = list(root.rglob("legacy_manifest.json"))
        for manifest_path in sorted(legacy_manifests):
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            experiment_root = manifest_path.parent
            migrated_legacy_roots.add(experiment_root)
            methods: list[str] = []
            legacy_results = experiment_root / "legacy_results"
            if legacy_results.is_dir():
                for name in ("AP01", "AP02", "AP03"):
                    if any(
                        path.is_dir()
                        for path in legacy_results.glob(f"*/{name}")
                    ):
                        methods.append(name.lower())
            entries.append(
                ResultEntry(
                    dataset_id=str(
                        payload.get(
                            "experiment_id", experiment_root.name
                        )
                    ),
                    run_id="migrated_legacy",
                    status=str(payload.get("status", "legacy")),
                    path=experiment_root,
                    methods=tuple(methods),
                    category=str(payload.get("category", "")),
                    experiment_id=str(
                        payload.get(
                            "experiment_id", experiment_root.name
                        )
                    ),
                    input_id=str(payload.get("input_id", "")),
                    legacy=True,
                )
            )
    manifests = (
        [
            path
            for path in root.rglob("run_manifest.json")
            if "run_history" not in path.parts
        ]
        if root.is_dir()
        else []
    )
    new_method_statuses: set[Path] = set()
    for path in manifests:
        if "_views" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_directory = path.parent
        schema_version = int(payload.get("schema_version", 1))
        experiment_root_value = payload.get("experiment_root")
        experiment_root = (
            Path(str(experiment_root_value)).resolve()
            if experiment_root_value
            else None
        )
        if schema_version >= 5:
            if (
                experiment_root is None
                or not (experiment_root / "PUBLISHED.json").is_file()
            ):
                continue
        new_method_statuses.update(run_directory.rglob("METHOD_STATUS.json"))
        entries.append(
            ResultEntry(
                dataset_id=str(payload.get("dataset_id", path.parents[2].name)),
                run_id=str(payload.get("run_id", run_directory.name)),
                status=str(payload.get("status", "unknown")),
                path=run_directory,
                methods=tuple(payload.get("enabled_methods", [])),
                category=str(
                    payload.get("result_category")
                    or (
                        "simulation"
                        if payload.get("scene_type") == "simulation"
                        else "real_vehicle"
                        if payload.get("scene_type")
                        else ""
                    )
                ),
                experiment_id=str(payload.get("experiment_id", "")),
                input_id=str(payload.get("input_id", "")),
                variant=str(payload.get("variant", "")),
            )
        )
    seen_legacy: set[Path] = set()
    new_run_directories = {path.parent for path in manifests}
    if root.is_dir():
        legacy_dataset_roots: set[Path] = set()
        for status_path in root.rglob("METHOD_STATUS.json"):
            if status_path in new_method_statuses:
                continue
            legacy_dataset_roots.add(status_path.parent.parent)
        for final_directory in root.rglob("99_FINAL_RESULTS"):
            if final_directory.parent in new_run_directories:
                continue
            legacy_dataset_roots.add(final_directory.parent)

        for dataset_root in sorted(legacy_dataset_roots):
            if any(
                dataset_root == migrated
                or migrated in dataset_root.parents
                for migrated in migrated_legacy_roots
            ):
                continue
            if dataset_root in seen_legacy:
                continue
            seen_legacy.add(dataset_root)
            methods = []
            statuses = []
            for candidate in dataset_root.rglob("METHOD_STATUS.json"):
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                methods.append(str(payload.get("method", candidate.parent.name)))
                statuses.append(str(payload.get("status", "unknown")))
            summary_candidates = list(
                (dataset_root / "99_FINAL_RESULTS").glob(
                    "**/*MARKER_CONSISTENCY_SUMMARY.json"
                )
            )
            for summary_path in summary_candidates[:1]:
                try:
                    summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary_rows = []
                if isinstance(summary_rows, list):
                    for row in summary_rows:
                        if not isinstance(row, dict):
                            continue
                        methods.append(str(row.get("method", "unknown")))
                        statuses.append(str(row.get("status", "unknown")))
            entries.append(
                ResultEntry(
                    dataset_id=dataset_root.name,
                    run_id="legacy",
                    status="; ".join(statuses) or "legacy",
                    path=dataset_root,
                    methods=tuple(dict.fromkeys(methods)),
                    legacy=True,
                )
            )
    canonical_groups: dict[tuple[str, str], list[ResultEntry]] = {}
    ungrouped: list[ResultEntry] = []
    for entry in entries:
        experiment = entry.experiment_id or (
            entry.dataset_id if entry.category else ""
        )
        if entry.category in {"simulation", "real_vehicle"} and experiment:
            canonical_groups.setdefault(
                (entry.category, experiment), []
            ).append(entry)
        else:
            ungrouped.append(entry)
    aggregated: list[ResultEntry] = []
    for (category, experiment), group in canonical_groups.items():
        status_counts: dict[str, int] = {}
        for item in group:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        status = (
            group[0].status
            if len(group) == 1
            else "; ".join(
                f"{name}: {count}"
                for name, count in sorted(status_counts.items())
            )
        )
        manifest_roots = []
        for item in group:
            manifest_path = item.path / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest_payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            value = manifest_payload.get("experiment_root")
            if value:
                manifest_roots.append(Path(str(value)).resolve())
        experiment_root = (
            manifest_roots[0]
            if manifest_roots
            else root / category / experiment
        )
        if (experiment_root / "INPUT_REMOVED.json").is_file():
            status = "results available; input cleaned; not rerunnable"
        aggregated.append(
            ResultEntry(
                dataset_id=experiment,
                run_id=f"{len(group)} execution{'s' if len(group) != 1 else ''}",
                status=status,
                path=experiment_root,
                methods=tuple(
                    sorted(
                        {
                            _display_method_id(method)
                            for item in group
                            for method in item.methods
                        },
                        key=_display_method_sort_key,
                    )
                ),
                category=category,
                experiment_id=experiment,
                input_id=";".join(
                    sorted(
                        {
                            item.input_id
                            for item in group
                            if item.input_id
                        }
                    )
                ),
                legacy=all(item.legacy for item in group),
            )
        )
    return sorted(
        [*aggregated, *ungrouped],
        key=lambda item: (
            item.category,
            item.experiment_id or item.dataset_id,
            item.run_id,
        ),
        reverse=True,
    )


def create_simulation_factor_views(
    output_root: Path,
    experiment_root: Path,
    *,
    experiment_id: str,
    parameters: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    world_id: str = "bus",
) -> list[Path]:
    """Expose a world's baseline under its factor-default paths.

    Non-baseline experiments already live at their one canonical factor or
    ``mixed`` path and therefore never receive aliases.  This keeps discovery
    manifest-driven and prevents one experiment from being counted twice.
    """
    simulation_root = output_root.resolve() / "simulation"
    experiment = experiment_root.resolve()
    try:
        experiment.relative_to(simulation_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Simulation view target is outside results/simulation: {experiment}"
        ) from exc
    defaults = {
        "route": "route2",
        "moving_width": 1280,
        "moving_height": 720,
        "moving_hfov_deg": 69.1,
        "lighting": "baseline",
        "lighting_scale": 1.0,
        "motion_blur_kernel": 0,
        "motion_blur_angle_deg": 0.0,
        "target_route_frames": 189,
        "route_sampling_strategy": "original_route_poses",
        "settle_seconds": 0.35,
        "post_pose_skip": 5,
        "frame_timeout_seconds": 3.0,
        "startup_timeout_seconds": 60.0,
    }
    supplied_baseline = dict(baseline or {})
    if "route_name" in supplied_baseline:
        supplied_baseline["route"] = supplied_baseline.pop("route_name")
    defaults.update(supplied_baseline)
    comparable = {
        key: parameters.get(key, value) for key, value in defaults.items()
    }
    if any(
        (
            not math.isclose(
                float(comparable[key]),
                float(value),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            if isinstance(value, (int, float))
            and isinstance(comparable[key], (int, float))
            else comparable[key] != value
        )
        for key, value in defaults.items()
    ):
        return []

    route = str(comparable["route"])
    width = int(parameters.get("moving_width", 1280))
    height = int(parameters.get("moving_height", 720))
    fov = float(parameters.get("moving_hfov_deg", 69.1))
    lighting = str(parameters.get("lighting", "baseline"))
    lighting_scale = float(parameters.get("lighting_scale", 1.0))
    blur = int(parameters.get("motion_blur_kernel", 0))
    blur_angle = float(parameters.get("motion_blur_angle_deg", 0.0))
    route_frames = int(parameters.get("target_route_frames", 189))
    sampling = str(
        parameters.get(
            "route_sampling_strategy", "original_route_poses"
        )
    )
    settle = float(parameters.get("settle_seconds", 0.35))
    post_pose_skip = int(parameters.get("post_pose_skip", 5))
    frame_timeout = float(parameters.get("frame_timeout_seconds", 3.0))
    startup_timeout = float(
        parameters.get("startup_timeout_seconds", 60.0)
    )
    scope = (
        simulation_root
        if world_id == "bus"
        else simulation_root / "worlds" / world_id
    )
    lighting_label = lighting
    if abs(lighting_scale - 1.0) >= 1e-9:
        lighting_label = f"{lighting}__scale_{lighting_scale:g}"
    blur_label = f"kernel_{blur}"
    if abs(blur_angle) >= 1e-9:
        blur_label += f"__angle_{blur_angle:g}deg"
    factor_targets = {
        "fov": scope / "fov" / f"{fov:g}deg",
        "resolution": scope / "resolution" / f"{width}x{height}",
        "lighting": scope / "lighting" / lighting_label,
        "motion_blur": scope / "motion_blur" / blur_label,
        "route": scope / "route" / route,
        "density": (
            scope
            / "density"
            / (
                "original_route_poses"
                if sampling == "original_route_poses"
                else f"frames_{route_frames}_{sampling}"
            )
        ),
    }
    targets = list(factor_targets.values())
    created: list[Path] = []
    for link in targets:
        if link.resolve() == experiment:
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        relative_target = Path(os.path.relpath(experiment, link.parent.resolve()))
        if link.is_symlink():
            if link.resolve() == experiment:
                created.append(link)
                continue
            raise RuntimeError(f"Simulation factor view conflicts: {link}")
        if link.exists():
            raise RuntimeError(
                f"Simulation factor view path is not a symlink: {link}"
            )
        link.symlink_to(relative_target, target_is_directory=True)
        created.append(link)
    return created


def write_comparison(run_directory: Path, method_results: dict[str, dict[str, Any]]) -> None:
    comparison = run_directory / "07_COMPARISON"
    final = run_directory / "99_FINAL_RESULTS"
    comparison.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    evaluation_path = (
        run_directory
        / "06_EVALUATION/marker_consistency/REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json"
    )
    evaluation_rows = []
    if evaluation_path.is_file():
        try:
            evaluation_rows = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except Exception:
            evaluation_rows = []
    label_by_id = {
        "ap01": "AP01",
        "ap02": "AP02",
        "ap03": "AP03_MULTI",
    }
    evaluation_by_label = {
        str(row.get("method")): row
        for row in evaluation_rows
        if isinstance(row, dict)
    }
    rows = []
    for method_id, payload in method_results.items():
        evaluation = evaluation_by_label.get(label_by_id.get(method_id, method_id), {})
        rows.append(
            {
                "method": method_id,
                "status": payload.get("status", "MISSING"),
                "success": payload.get("success", False),
                "static_camera_count": len(payload.get("available_static_cameras", [])),
                "runtime_seconds": payload.get("runtime_seconds", ""),
                "registered_moving_frames": evaluation.get(
                    "registered_moving_frames", ""
                ),
                "evaluated_markers": evaluation.get(
                    "evaluated_non_anchor_markers", ""
                ),
                "cross_camera_reprojection_rmse_px": evaluation.get(
                    "moving_to_static_reprojection_rmse_px", ""
                ),
                "median_marker_size_error_cm": evaluation.get(
                    "median_absolute_size_error_cm", ""
                ),
                "warning": evaluation.get("error", payload.get("error", "")),
                "output_directory": payload.get("directory", ""),
            }
        )
    (comparison / "method_status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (comparison / "method_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    evaluation_report = run_directory / "06_EVALUATION" / "REAL_DATA_MARKER_CONSISTENCY.txt"
    if evaluation_report.is_file():
        shutil.copy2(evaluation_report, final / "MARKER_CONSISTENCY.txt")
    summary = [
        "CAMERA RIG CALIBRATION RESULT",
        "=" * 72,
        "",
        f"Run: {run_directory.name}",
        "",
        "Methods:",
    ]
    summary.extend(
        (
            f"- {row['method']}: {row['status']} (success={row['success']}, "
            f"static cameras={row['static_camera_count']}, "
            f"moving frames={row['registered_moving_frames'] or 'NA'}, "
            f"cross RMSE={row['cross_camera_reprojection_rmse_px'] or 'NA'} px)"
        )
        for row in rows
    )
    summary.extend(
        [
            "",
            f"Comparison: {comparison}",
            f"Evaluation: {run_directory / '06_EVALUATION'}",
            "",
        ]
    )
    (final / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")
    (final / "SUMMARY.json").write_text(
        json.dumps({"run": run_directory.name, "methods": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
