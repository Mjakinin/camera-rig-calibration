from __future__ import annotations

import json
import re
import subprocess
import sys
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


def _authoritative_anchor(experiment_root: Path, dataset_root: Path) -> int | None:
    selected = _read_json(
        experiment_root / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    )
    value = selected.get("anchor_marker_id")
    if value is None:
        selection = _read_json(
            dataset_root / "observations" / "SELECTION_CANDIDATES.json"
        )
        value = selection.get("evaluation_anchor", {}).get("selected")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_authoritative_marker_consistency(
    experiment_root: Path,
    dataset_root: Path,
) -> Path | None:
    """Regenerate real marker-length/Cross-RMSE diagnostics from native metric results.

    Evaluation-only: completed AP01/AP02/AP03 outputs are read, never rerun or
    re-optimized. The common marker is used only for frame alignment where needed;
    it is not used to re-scale reconstructed marker lengths. Ground Truth is never
    consulted.
    """

    from ..evaluation import reporting

    anchor = _authoritative_anchor(experiment_root, dataset_root)
    if anchor is None:
        return None
    output = experiment_root / "evaluations" / "method_anchors_reconciled"
    report = output / "REAL_DATA_MARKER_CONSISTENCY.txt"

    methods: list[tuple[str, Path]] = []
    for result_path in sorted((experiment_root / "methods").glob("*/*/RESULT.json")):
        method = result_path.parents[1].name
        label = result_path.parent.name
        method_root = result_path.parent / "diagnostics" / "method"
        if not method_root.is_dir():
            continue
        payload = _read_json(result_path)
        config_summary = payload.get("config_summary", {})
        if method == "ap01":
            display_name = (
                "AP01 "
                f"(root {config_summary.get('root_camera', 'auto')}, "
                f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
            )
        elif method == "ap02":
            display_name = (
                "AP02 "
                f"(ref {config_summary.get('reference_marker_id', 'auto')}, "
                f"nfev {config_summary.get('combined_max_nfev', '-')}, "
                f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
            )
        elif method in {"ap03", "ap03_multi", "ap03_single"}:
            mode = (
                "Multi"
                if method == "ap03_multi"
                else "Single"
                if method == "ap03_single"
                else ""
            )
            display_name = (
                f"AP03{(' ' + mode) if mode else ''} "
                f"(multi {config_summary.get('multi_marker_count', '-')} markers, "
                f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
            )
        else:
            display_name = f"{method.upper()} ({label})"
        methods.append((display_name, method_root))

    if not methods:
        return None

    dataset = _read_json(dataset_root / "dataset.json")
    cameras = [
        str(item["id"])
        for item in dataset.get("static_cameras", [])
        if isinstance(item, dict) and item.get("id")
    ]
    first_config = next(
        (
            result_path.parent / "provenance" / "resolved_config.yaml"
            for result_path in sorted((experiment_root / "methods").glob("*/*/RESULT.json"))
            if (result_path.parent / "provenance" / "resolved_config.yaml").is_file()
        ),
        None,
    )
    marker_length = 0.17
    if first_config is not None:
        try:
            resolved = yaml.safe_load(first_config.read_text(encoding="utf-8")) or {}
            marker_length = float(resolved.get("markers", {}).get("length_m", marker_length))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass

    command = [
        sys.executable,
        "-m",
        "camera_rig_calibration.evaluation.real_marker_consistency_native",
        "--dataset",
        str(dataset_root),
        "--results-root",
        str(experiment_root),
        "--observations-root",
        str(dataset_root / "observations"),
        "--output-root",
        str(output),
        "--anchor-marker-id",
        str(anchor),
        "--marker-length-m",
        str(marker_length),
        "--cameras",
        ",".join(cameras),
    ]
    for name, method_root in methods:
        command.extend(["--method", f"{name}={method_root.resolve()}"])

    completed = subprocess.run(
        command,
        cwd=reporting._repository_root(experiment_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.log").write_text(completed.stdout, encoding="utf-8")
    status = {
        "schema_version": 5,
        "layout_version": 2,
        "anchor_marker_id": anchor,
        "evaluation_scope": "native_metric_marker_length_and_cross_reprojection",
        "metric_scale_source": "native_method_outputs_no_evaluation_rescale",
        "method_rerun": False,
        "colmap_rerun": False,
        "ground_truth_used": False,
        "return_code": completed.returncode,
    }
    if completed.returncode != 0 or not report.is_file():
        status.update(
            {
                "status": "unavailable",
                "reason": "Real marker consistency evaluation failed; see evaluation.log.",
            }
        )
        reporting._write_json(output / "COMMON_ANCHOR_STATUS.json", status)
        return None
    status.update(
        {
            "status": "available",
            "report": str(report.relative_to(experiment_root)),
            "methods": [name for name, _ in methods],
        }
    )
    reporting._write_json(output / "COMMON_ANCHOR_STATUS.json", status)
    return report


def _install_real_results_authority() -> None:
    from ..evaluation import reporting

    original = reporting._real_results_text
    if getattr(original, "_rigcal_real_marker_reporting_authority", False):
        return

    def real_results_text(experiment_root, method_payloads, dataset_root=None):
        root = Path(experiment_root)
        dataset = Path(dataset_root) if dataset_root is not None else root
        _run_authoritative_marker_consistency(root, dataset)
        text, payload = original(experiment_root, method_payloads, dataset_root)

        selected = _read_json(root / "evaluations" / "SELECTED_COMMON_EVALUATION.json")
        authoritative = selected.get("anchor_marker_id")
        if authoritative is None:
            return text, payload
        try:
            authoritative_id = int(authoritative)
        except (TypeError, ValueError):
            return text, payload
        selection = _read_json(dataset / "observations" / "SELECTION_CANDIDATES.json")
        preflight_value = selection.get("evaluation_anchor", {}).get("selected")
        try:
            preflight_id = int(preflight_value)
        except (TypeError, ValueError):
            preflight_id = None

        replacement = f"Common evaluation/export anchor: marker {authoritative_id}"
        text = re.sub(
            r"^Common evaluation anchor: marker .*?$",
            replacement,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        payload["evaluation_anchor"] = {
            **selection.get("evaluation_anchor", {}),
            **selected,
            "selected": authoritative_id,
            "preflight_selected": preflight_id,
            "ground_truth_used": False,
        }
        payload["marker_consistency_anchor_marker_id"] = authoritative_id
        return text, payload

    real_results_text._rigcal_real_marker_reporting_authority = True  # type: ignore[attr-defined]
    reporting._real_results_text = real_results_text


def install_real_marker_reporting_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_real_results_authority()
    _INSTALLED = True
