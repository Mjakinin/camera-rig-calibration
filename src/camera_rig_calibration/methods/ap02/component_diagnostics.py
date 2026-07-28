from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from .common import read_csv, write_csv
from .initialize_stage import run as run_initialization
from .optimize_stage import run as run_optimization


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
        dtype=np.float64,
    )


def _write_component_result(
    component_root: Path,
    *,
    component: dict[str, Any],
    reference_marker_id: int,
) -> dict[str, Any]:
    poses_path = (
        component_root
        / "07_graph_ba"
        / "with_moving"
        / "optimized_static_camera_poses_ref_marker.csv"
    )
    poses = read_csv(poses_path)
    camera_ids = set(map(str, component["static_cameras"]))
    camera_rows = [
        row for row in poses if str(row.get("entity_id")) in camera_ids
    ]
    exported = component_root / "camera_extrinsics.csv"
    fields = list(camera_rows[0]) if camera_rows else []
    write_csv(exported, camera_rows, fields)
    pair_rows: list[dict[str, Any]] = []
    by_camera = {str(row["entity_id"]): row for row in camera_rows}
    for first, second in combinations(sorted(by_camera), 2):
        pair_rows.append(
            {
                "camera_a": first,
                "camera_b": second,
                "baseline_m": float(
                    np.linalg.norm(
                        _position(by_camera[first])
                        - _position(by_camera[second])
                    )
                ),
                "observability": "within_component",
            }
        )
    write_csv(
        component_root / "pairwise_camera_extrinsics.csv",
        pair_rows,
        ["camera_a", "camera_b", "baseline_m", "observability"],
    )
    optimizer = component_root / "07_graph_ba/with_moving/optimizer_report.json"
    optimizer_payload = (
        json.loads(optimizer.read_text(encoding="utf-8"))
        if optimizer.is_file()
        else {}
    )
    result = {
        "schema_version": 5,
        "status": "available",
        "quality_status": (
            "converged"
            if optimizer_payload.get("success")
            else "optimizer_warning"
        ),
        "component_id": component["component_id"],
        "reference_marker_id": reference_marker_id,
        "reference_frame": f"marker_{reference_marker_id}",
        "static_cameras": sorted(by_camera),
        "static_camera_count": len(by_camera),
        "marker_ids": list(component["marker_ids"]),
        "moving_frame_count": int(component["moving_frame_count"]),
        "camera_extrinsics": "camera_extrinsics.csv",
        "pairwise_camera_extrinsics": "pairwise_camera_extrinsics.csv",
        "cross_component_extrinsics": "not_observable",
        "optimizer": optimizer_payload,
    }
    _write_json(component_root / "COMPONENT_RESULT.json", result)
    (component_root / "COMPONENT_RESULT.txt").write_text(
        "\n".join(
            [
                "AP02 DISCONNECTED-COMPONENT DIAGNOSTIC",
                "=" * 72,
                "",
                f"Component: {component['component_id']}",
                f"Status: {result['status']}",
                f"Quality: {result['quality_status']}",
                f"Local reference frame: marker_{reference_marker_id}",
                "Static cameras: " + ", ".join(sorted(by_camera)),
                "Marker IDs: "
                + ", ".join(map(str, component["marker_ids"])),
                (
                    "Cross-component camera relationships: not observable "
                    "from these observations"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def run(
    *,
    output_root: Path,
    maximum_function_evaluations: int,
    robust_loss: str,
    robust_loss_scale_px: float,
) -> StageResult:
    stage_root = output_root / "09_component_diagnostics"
    manifest_path = (
        output_root / "02_aruco_observations/component_manifest.json"
    )

    def action() -> dict[str, Path | int]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        primary_id = manifest.get("primary_component_id")
        results: list[dict[str, Any]] = []
        for component in manifest.get("components", []):
            component_id = str(component["component_id"])
            if component_id == primary_id:
                results.append(
                    {
                        **component,
                        "execution_status": "primary_component",
                        "result_path": "../07_graph_ba/with_moving",
                    }
                )
                continue
            if not component.get("calibratable"):
                results.append(
                    {
                        **component,
                        "execution_status": "not_calibratable",
                        "reason": (
                            "AP02 rig diagnostics require at least two static "
                            "cameras and one moving-frame observer"
                        ),
                        "cross_component_extrinsics": "not_observable",
                    }
                )
                continue
            component_root = stage_root / component_id
            observations = component_root / "02_aruco_observations"
            source = (
                output_root
                / "02_aruco_observations"
                / "components"
                / component_id
            )
            observations.mkdir(parents=True, exist_ok=True)
            for item in source.glob("*.csv"):
                shutil.copy2(item, observations / item.name)
            reference = int(component["anchor_marker_id"])
            try:
                run_initialization(
                    output_root=component_root,
                    reference_marker_id=reference,
                    mode="with_moving",
                    log_path=(
                        component_root / "logs" / "initialization.log"
                    ),
                )
                optimization = run_optimization(
                    output_root=component_root,
                    reference_marker_id=reference,
                    mode="with_moving",
                    maximum_function_evaluations=(
                        maximum_function_evaluations
                    ),
                    robust_loss=robust_loss,
                    robust_loss_scale_px=robust_loss_scale_px,
                    log_path=(
                        component_root / "logs" / "optimizer.log"
                    ),
                )
                if optimization.status != "COMPLETED":
                    raise RuntimeError(
                        f"optimizer status is {optimization.status}"
                    )
                result = _write_component_result(
                    component_root,
                    component=component,
                    reference_marker_id=reference,
                )
                results.append(
                    {
                        **component,
                        "execution_status": "available",
                        "local_reference_marker_id": reference,
                        "result_path": component_id,
                        "quality_status": result["quality_status"],
                    }
                )
            except Exception as exc:
                failure = {
                    **component,
                    "execution_status": "failed_diagnostic",
                    "local_reference_marker_id": reference,
                    "error": f"{type(exc).__name__}: {exc}",
                    "cross_component_extrinsics": "not_observable",
                }
                _write_json(
                    component_root / "COMPONENT_RESULT.json", failure
                )
                results.append(failure)
        camera_component: dict[str, str] = {
            str(camera): str(component["component_id"])
            for component in manifest.get("components", [])
            for camera in component.get("static_cameras", [])
        }
        expected_cameras = sorted(
            map(str, manifest.get("expected_static_cameras", []))
        )
        pair_observability = [
            {
                "camera_a": first,
                "camera_b": second,
                "status": (
                    "within_component"
                    if camera_component.get(first)
                    == camera_component.get(second)
                    else "not_observable"
                ),
                "component_id": (
                    camera_component.get(first)
                    if camera_component.get(first)
                    == camera_component.get(second)
                    else None
                ),
            }
            for first, second in combinations(expected_cameras, 2)
        ]
        primary_component = next(
            (
                component
                for component in manifest.get("components", [])
                if component.get("component_id") == primary_id
            ),
            {},
        )
        disconnected = set(expected_cameras) != set(
            map(str, primary_component.get("static_cameras", []))
        )
        summary = {
            "schema_version": 5,
            "status": (
                "partial_coverage" if disconnected else "complete_graph"
            ),
            "primary_component_id": primary_id,
            "component_count": len(results),
            "calibrated_diagnostic_component_count": sum(
                item.get("execution_status") == "available"
                for item in results
            ),
            "cross_component_extrinsics": (
                "not_observable" if disconnected else "not_applicable"
            ),
            "camera_pair_observability": pair_observability,
            "components": results,
        }
        summary_path = stage_root / "AP02_COMPONENT_RESULTS.json"
        _write_json(summary_path, summary)
        (stage_root / "AP02_COMPONENT_RESULTS.txt").write_text(
            "\n".join(
                [
                    "AP02 DISCONNECTED COMPONENT RESULTS",
                    "=" * 72,
                    "",
                    f"Primary component: {primary_id}",
                    f"Detected components: {len(results)}",
                    (
                        "Cross-component camera relationships: not observable; "
                        "components are not artificially aligned"
                    ),
                    "",
                    *[
                        (
                            f"{item['component_id']}: "
                            f"{item['execution_status']} | cameras="
                            f"{','.join(item['static_cameras']) or '-'} | "
                            f"markers={','.join(map(str, item['marker_ids']))}"
                        )
                        for item in results
                    ],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "component_results": summary_path,
            "component_count": len(results),
            "diagnostic_components_calibrated": (
                summary["calibrated_diagnostic_component_count"]
            ),
        }

    return run_stage(
        "ap02.component_diagnostics",
        stage_root,
        action,
        inputs={"component_manifest": manifest_path},
        parameters={
            "maximum_function_evaluations": maximum_function_evaluations,
            "robust_loss": robust_loss,
            "robust_loss_scale_px": robust_loss_scale_px,
            "cross_component_alignment": False,
        },
        failure_is_diagnostic=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, required=True)
    parser.add_argument(
        "--robust-loss",
        choices=["soft_l1", "huber", "linear"],
        required=True,
    )
    parser.add_argument("--robust-loss-scale-px", type=float, required=True)
    args = parser.parse_args()
    run(
        output_root=args.out.resolve(),
        maximum_function_evaluations=args.max_nfev,
        robust_loss=args.robust_loss,
        robust_loss_scale_px=args.robust_loss_scale_px,
    )


if __name__ == "__main__":
    main()
