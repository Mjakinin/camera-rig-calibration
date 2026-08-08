from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


_INSTALLED = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _authoritative_anchor(experiment_root: Path) -> int | None:
    selected = _read_json(
        experiment_root / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    )
    value = selected.get("anchor_marker_id")
    if value is None:
        selection = _read_json(
            experiment_root / "observations" / "SELECTION_CANDIDATES.json"
        )
        value = selection.get("evaluation_anchor", {}).get("selected")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _anchor_export_rows(experiment_root: Path) -> tuple[int | None, list[dict[str, Any]]]:
    anchor = _authoritative_anchor(experiment_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (experiment_root / "methods").glob("*/*/camera_extrinsics_anchor.json")
    ):
        payload = _read_json(path)
        method = str(payload.get("method") or path.parents[1].name)
        # AP03's shared container is provenance. The scale-specific derived
        # result, especially ap03_multi, is the public AP03 estimate.
        if method == "ap03":
            continue
        try:
            payload_anchor = int(payload.get("anchor_marker_id"))
        except (TypeError, ValueError):
            continue
        if anchor is not None and payload_anchor != anchor:
            continue
        for camera in payload.get("cameras", []):
            if not isinstance(camera, dict) or not camera.get("camera_id"):
                continue
            rows.append(
                {
                    "method": method,
                    "label": str(payload.get("label") or path.parent.name),
                    "anchor_marker_id": payload_anchor,
                    "camera_id": str(camera["camera_id"]),
                    "parent_frame": str(payload.get("parent_frame", "")),
                    "x": camera.get("x_m"),
                    "y": camera.get("y_m"),
                    "z": camera.get("z_m"),
                    "roll": camera.get("roll_rad"),
                    "pitch": camera.get("pitch_rad"),
                    "yaw": camera.get("yaw_rad"),
                    "qx": camera.get("qx"),
                    "qy": camera.get("qy"),
                    "qz": camera.get("qz"),
                    "qw": camera.get("qw"),
                    "deployment_eligible": camera.get("deployment_eligible", True),
                    "quality_status": camera.get("quality_status", "accepted"),
                }
            )
    return anchor, rows


def _public_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one primary public variant per AP01/AP02/AP03 family for the compact export."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row["method"])
        family = "ap03" if method in {"ap03_single", "ap03_multi"} else method
        grouped.setdefault(family, []).append(row)
    selected: list[dict[str, Any]] = []
    for family, family_rows in sorted(grouped.items()):
        labels = sorted({str(row["label"]) for row in family_rows})
        if family == "ap03":
            preferred_methods = ["ap03_multi", "ap03_single"]
            chosen_method = next(
                (
                    method
                    for method in preferred_methods
                    if any(row["method"] == method for row in family_rows)
                ),
                str(family_rows[0]["method"]),
            )
            candidates = [row for row in family_rows if row["method"] == chosen_method]
        else:
            candidates = family_rows
        candidate_labels = sorted({str(row["label"]) for row in candidates})
        chosen_label = "baseline" if "baseline" in candidate_labels else candidate_labels[0]
        selected.extend(row for row in candidates if str(row["label"]) == chosen_label)
    return selected


def _write_common_anchor_exports(experiment_root: Path) -> dict[str, Any]:
    anchor, rows = _anchor_export_rows(experiment_root)
    public_rows = _public_method_rows(rows)
    payload = {
        "schema_version": 1,
        "anchor_marker_id": anchor,
        "parent_frame": (
            f"evaluation_anchor_marker_{anchor}" if anchor is not None else None
        ),
        "translation_unit": "m",
        "rotation_unit": "rad",
        "rpy_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "transform_convention": "T_anchor_camera; p_anchor = T_anchor_camera @ p_camera",
        "rows": public_rows,
        "all_published_variant_rows": rows,
    }
    _write_json(experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.json", payload)
    _write_text(
        experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    csv_path = experiment_root / "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv"
    fields = [
        "method",
        "label",
        "anchor_marker_id",
        "camera_id",
        "parent_frame",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "qx",
        "qy",
        "qz",
        "qw",
        "deployment_eligible",
        "quality_status",
    ]
    temporary = csv_path.with_suffix(".csv.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_rows)
    temporary.replace(csv_path)
    return payload


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "null"


def _sixdof_text(experiment_root: Path) -> tuple[str, dict[str, Any]]:
    payload = _write_common_anchor_exports(experiment_root)
    anchor = payload.get("anchor_marker_id")
    rows = payload.get("rows", [])
    lines = [
        "COMMON-ANCHOR STATIC-CAMERA 6DOF EXPORTS",
        "-" * 138,
        f"Reference frame: evaluation_anchor_marker_{anchor}",
        "Translation: metres; roll/pitch/yaw: radians; optical-camera convention x right, y down, z forward.",
        "These are method outputs expressed in the common evaluation/export anchor frame; no GT alignment is applied.",
        "",
    ]
    if not rows:
        lines.append("No common-anchor 6DoF camera export is available.")
    else:
        by_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            by_variant.setdefault((str(row["method"]), str(row["label"])), []).append(row)
        for (method, label), cameras in sorted(by_variant.items()):
            lines.append(f"{method}/{label}:")
            for camera in sorted(cameras, key=lambda item: str(item["camera_id"])):
                lines.extend(
                    [
                        f"  {camera['camera_id']}:",
                        f"    x: {_fmt(camera.get('x'))}",
                        f"    y: {_fmt(camera.get('y'))}",
                        f"    z: {_fmt(camera.get('z'))}",
                        f"    roll: {_fmt(camera.get('roll'))}",
                        f"    pitch: {_fmt(camera.get('pitch'))}",
                        f"    yaw: {_fmt(camera.get('yaw'))}",
                    ]
                )
            lines.append("")
    lines.extend(
        [
            "Machine-readable exports:",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.json",
            "- CAMERA_EXTRINSICS_COMMON_ANCHOR.csv",
            "",
        ]
    )
    return "\n".join(lines), payload


def _real_summary_text(method_payloads: list[dict[str, Any]]) -> str:
    lines = [
        "REAL-DATA METHOD RESULT SUMMARY (NO INDEPENDENT GROUND TRUTH)",
        "-" * 138,
        "The values below report completion, internal fit and reconstruction support; they are not absolute real-world accuracy.",
    ]
    for payload in method_payloads:
        method = str(payload.get("method", "-"))
        if method == "ap03":
            continue
        label = str(payload.get("label", "-"))
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        fragments = [
            f"cameras={payload.get('static_camera_count', 0)}",
            f"quality={payload.get('quality_status', '-')}",
        ]
        if payload.get("runtime_seconds") is not None:
            fragments.append(f"runtime={float(payload['runtime_seconds']):.1f}s")
        if method == "ap01":
            scale = metrics.get("ap01_scale", {})
            if scale.get("scale_m_per_colmap_unit") is not None:
                fragments.append(
                    "scale=" + _fmt(scale.get("scale_m_per_colmap_unit")) + "m/COLMAP-unit"
                )
            consensus = metrics.get("ap01_consensus", {})
            fragments.append(
                f"unstable_targets={len(consensus.get('unstable_targets', []))}"
            )
        elif method == "ap02":
            reproj = metrics.get("combined_reprojection_rmse_px")
            if reproj is not None:
                fragments.append(f"combined_RMSE={_fmt(reproj)}px")
            solver = metrics.get("solver", {})
            if solver:
                fragments.append(
                    f"solver_success={str(bool(solver.get('success'))).lower()}"
                )
                fragments.append(
                    f"nfev={solver.get('nfev', '-')}/{solver.get('maximum_function_evaluations', '-')}"
                )
        elif method in {"ap03_single", "ap03_multi"}:
            registration = metrics.get("ap03_registration", {})
            if registration:
                fragments.append(
                    f"registered_static={registration.get('registered_static_cameras', '-') }"
                )
                fragments.append(
                    f"registered_moving={registration.get('registered_moving_frames', '-') }"
                )
                fragments.append(f"sparse_points={registration.get('sparse_points', '-')}")
            rel = metrics.get("ap03_scale_relative_std")
            if rel is not None:
                fragments.append(f"scale_rel_std={_fmt(rel)}")
        lines.append(f"- {method}/{label}: " + ", ".join(fragments))
    lines.append("")
    return "\n".join(lines)


def _install_reporting_outputs() -> None:
    from .evaluation import reporting

    original_real = reporting._real_results_text
    if not getattr(original_real, "_rigcal_common_anchor_6dof", False):
        def real_results_text(experiment_root, method_payloads, dataset_root=None):
            text, payload = original_real(experiment_root, method_payloads, dataset_root)
            summary = _real_summary_text(method_payloads)
            sixdof, aggregate = _sixdof_text(Path(experiment_root))
            payload["common_anchor_camera_6dof"] = aggregate
            return text.rstrip() + "\n\n" + summary + "\n" + sixdof, payload

        real_results_text._rigcal_common_anchor_6dof = True  # type: ignore[attr-defined]
        reporting._real_results_text = real_results_text

    original_sim = reporting._simulation_results
    if not getattr(original_sim, "_rigcal_common_anchor_6dof", False):
        def simulation_results(experiment_root, dataset_root, method_payloads):
            text, payload = original_sim(experiment_root, dataset_root, method_payloads)
            sixdof, aggregate = _sixdof_text(Path(experiment_root))
            payload["common_anchor_camera_6dof"] = aggregate
            return text.rstrip() + "\n\n" + sixdof, payload

        simulation_results._rigcal_common_anchor_6dof = True  # type: ignore[attr-defined]
        reporting._simulation_results = simulation_results


def _default_visible_variants(variants: list[dict[str, Any]]) -> set[tuple[str, str]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for item in variants:
        method = str(item["method"])
        family = "ap03" if method in {"ap03_single", "ap03_multi"} else method
        families.setdefault(family, []).append(item)
    selected: set[tuple[str, str]] = set()
    for family, items in families.items():
        if family == "ap03":
            multi = [item for item in items if item["method"] == "ap03_multi"]
            pool = multi or items
        else:
            pool = items
        baseline = [item for item in pool if item["label"] == "baseline"]
        chosen = sorted(
            baseline or pool,
            key=lambda item: (str(item["method"]), str(item["label"])),
        )[0]
        selected.add((str(chosen["method"]), str(chosen["label"])))
    return selected


def _install_rviz_defaults() -> None:
    from .visualization import scene

    original = scene._rviz_config
    if getattr(original, "_rigcal_all_primary_methods_visible", False):
        return

    def rviz_config(fixed_frame, variants, *, ground_truth_available):
        visible = _default_visible_variants(variants)
        displays = [
            scene._rviz_display(
                name="AP03 Multi COLMAP context",
                topic="/rigcal/scene/points",
                enabled=True,
                class_name="rviz_default_plugins/PointCloud2",
            ),
            scene._rviz_display(
                name="Common evaluation/export anchor",
                topic="/rigcal/scene/anchor",
                enabled=True,
            ),
        ]
        if ground_truth_available:
            displays.append(
                scene._rviz_display(
                    name="Ground truth cameras",
                    topic="/rigcal/ground_truth/cameras",
                    enabled=True,
                )
            )
        for variant in variants:
            name = f"{variant['method']}/{variant['label']}"
            enabled = (str(variant["method"]), str(variant["label"])) in visible
            displays.extend(
                [
                    scene._rviz_display(
                        name=name,
                        topic=scene._topic(variant["method"], variant["label"], "cameras"),
                        enabled=enabled,
                    ),
                    scene._rviz_display(
                        name=f"{name} anchor edges",
                        topic=scene._topic(variant["method"], variant["label"], "anchor_edges"),
                        enabled=enabled,
                    ),
                ]
            )
            if ground_truth_available:
                displays.append(
                    scene._rviz_display(
                        name=f"{name} estimate-to-GT errors",
                        topic=scene._topic(variant["method"], variant["label"], "error_lines"),
                        enabled=False,
                    )
                )
        return (
            "Panels:\n"
            "  - Class: rviz_common/Displays\n"
            "Visualization Manager:\n"
            f"  Global Options:\n    Fixed Frame: {fixed_frame}\n"
            "    Background Color: 35; 35; 35\n"
            "  Displays:\n"
            + "\n".join(displays)
            + "\n"
            "  Tools:\n"
            "    - Class: rviz_default_plugins/Interact\n"
            "    - Class: rviz_default_plugins/MoveCamera\n"
            "Window Geometry:\n  Width: 1400\n  Height: 900\n"
        )

    rviz_config._rigcal_all_primary_methods_visible = True  # type: ignore[attr-defined]
    scene._rviz_config = rviz_config


def install_result_output_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_reporting_outputs()
    _install_rviz_defaults()
    _INSTALLED = True
