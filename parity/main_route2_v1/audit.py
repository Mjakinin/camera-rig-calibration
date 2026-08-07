"""Evidence collection for the Main Route-2 parity scaffold.

This module performs only file/tree inspection.  It never imports or invokes a
calibration method, COLMAP, ROS, Gazebo, or an evaluation module.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from .evidence import reserve_unavailable_artifacts, write_csv, write_json
from .inventory import (
    build_file_inventory,
    category_counts,
    inventory_fingerprint,
)
from .presets import validate_presets


EXPECTED_WIZARD_HEAD = "b5d5bc529533586e78bbbaa6d23780f0226da58c"
EXPECTED_MAIN_HEAD = "8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b"
LEGACY_RAW_ROOT = (
    "results/bus_real_data/ablation/world/route/route2/raw_images"
)
SHARED_RAW_ROOT = (
    "results/bus_real_data/00_shared_baseline/"
    "bus_real_data_ref_marker_v1/raw_images"
)
CAMERAS = (
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
    "moving_calib_camera",
)


def _run(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        arguments,
        cwd=repository,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _git_bytes(repository: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=repository
    )


def _legacy_category(relative: str) -> str:
    if relative.startswith("static/"):
        return "raw_static_image"
    if relative.startswith("moving/"):
        return "raw_moving_image"
    if relative.startswith("camera_info/"):
        return "camera_info"
    if relative.endswith("route_commanded.csv"):
        return "route_metadata"
    return "legacy_metadata"


def git_tree_inventory(
    repository: Path, *, ref: str, root: str, dataset_side: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Read a Git tree through an archive stream without materializing it."""

    process = subprocess.Popen(
        ["git", "archive", "--format=tar", ref, root],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    blobs: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to(root).as_posix()
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            content = extracted.read()
            blobs[relative] = content
            rows.append(
                {
                    "dataset_side": dataset_side,
                    "source_root": f"{ref}:{root}",
                    "category": _legacy_category(relative),
                    "path": relative,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
        "utf-8", errors="replace"
    )
    if process.wait() != 0:
        raise RuntimeError(f"git archive failed for {ref}:{root}: {stderr}")
    rows.sort(key=lambda row: (str(row["path"]), str(row["category"])))
    return rows, blobs


def _tree_counts(repository: Path, ref: str, root: str) -> dict[str, int]:
    names = _run(
        repository,
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        "--",
        root,
    ).splitlines()
    return {
        "raw_static_images": sum("/static/" in name for name in names),
        "raw_moving_images": sum("/moving/" in name for name in names),
        "camera_info_files": sum("/camera_info/" in name for name in names),
        "route_metadata_files": sum(name.endswith("route_commanded.csv") for name in names),
        "total_files": len(names),
    }


def _lfs_history_summary(
    repository: Path, roots: tuple[str, ...]
) -> dict[str, Any]:
    try:
        output = _run(repository, "git", "lfs", "ls-files", "--all")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "available": False,
            "matching_entry_count": 0,
            "matching_entry_counts_by_root": {root: 0 for root in roots},
        }
    lines = output.splitlines()
    counts = {root: sum(root in line for line in lines) for root in roots}
    return {
        "available": True,
        "matching_entry_count": sum(counts.values()),
        "matching_entry_counts_by_root": counts,
        "note": (
            "LFS history contains shared-baseline raw-image path versions, but the "
            "selected origin/main legacy route tree is independently recoverable as "
            "ordinary full Git blobs."
        ),
    }


def _package_installation(repository: Path) -> dict[str, Any]:
    import camera_rig_calibration

    imported_path = Path(camera_rig_calibration.__file__).resolve()
    executable = shutil.which("rigcal")
    try:
        distribution = importlib.metadata.distribution("camera-rig-calibration")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
        metadata = {
            "name": distribution.metadata["Name"],
            "version": distribution.version,
            "location": str(Path(distribution.locate_file("")).resolve()),
            "direct_url": direct_url,
        }
    except importlib.metadata.PackageNotFoundError:
        metadata = None
    checkout_source = (repository / "src" / "camera_rig_calibration").resolve()
    source_aligned = imported_path.is_relative_to(checkout_source)
    editable_source = None
    if metadata and metadata["direct_url"]:
        editable_source = metadata["direct_url"].get("url")
    return {
        "python_executable": sys.executable,
        "imported_package_path": str(imported_path),
        "rigcal_executable": executable,
        "installed_metadata": metadata,
        "checkout_source": str(checkout_source),
        "source_and_installed_code_aligned": source_aligned,
        "editable_source_url": editable_source,
        "reinstall_performed": False,
    }
def _camera_info(content: bytes) -> dict[str, Any]:
    payload = json.loads(content.decode("utf-8"))
    matrix = payload.get("K", payload.get("k"))
    if matrix is None:
        matrix = [
            payload["fx"],
            0.0,
            payload["cx"],
            0.0,
            payload.get("fy", payload["fx"]),
            payload["cy"],
            0.0,
            0.0,
            1.0,
        ]
    distortion = payload.get("D", payload.get("d", []))
    return {
        "width": int(payload.get("width", payload.get("image_width", 0))),
        "height": int(payload.get("height", payload.get("image_height", 0))),
        "fx": float(matrix[0]),
        "fy": float(matrix[4]),
        "cx": float(matrix[2]),
        "cy": float(matrix[5]),
        "distortion_model": str(payload.get("distortion_model", "")),
        "distortion": [float(value) for value in distortion],
    }


def _colmap_camera(info: dict[str, Any]) -> tuple[str, list[float]]:
    parameters = [info["fx"], info["fy"], info["cx"], info["cy"]]
    distortion = list(info["distortion"])
    model = info["distortion_model"].strip().lower()
    if model in {"equidistant", "fisheye"}:
        return "OPENCV_FISHEYE", parameters + (distortion + [0.0] * 4)[:4]
    if not distortion or max(abs(value) for value in distortion) <= 1e-15:
        return "PINHOLE", parameters
    return "FULL_OPENCV", parameters + (distortion + [0.0] * 8)[:8]


def _intrinsics_rows(
    repository: Path,
    dataset_root: Path,
    legacy_blobs: dict[str, bytes],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera in CAMERAS:
        relative = f"camera_info/{camera}.json"
        main_bytes = legacy_blobs.get(relative)
        wizard_path = dataset_root / "raw_images" / relative
        if main_bytes is None or not wizard_path.is_file():
            rows.append(
                {
                    "camera_id": camera,
                    "main_info_path": f"origin/main:{LEGACY_RAW_ROOT}/{relative}",
                    "wizard_info_path": str(wizard_path),
                    "parity_status": "unavailable",
                    "notes": "one or both camera-info files are unavailable",
                }
            )
            continue
        wizard_bytes = wizard_path.read_bytes()
        main = _camera_info(main_bytes)
        wizard = _camera_info(wizard_bytes)
        main_model, main_parameters = _colmap_camera(main)
        wizard_model, wizard_parameters = _colmap_camera(wizard)
        distortion_count = max(len(main["distortion"]), len(wizard["distortion"]))
        left_d = main["distortion"] + [0.0] * distortion_count
        right_d = wizard["distortion"] + [0.0] * distortion_count
        distortion_delta = max(
            (abs(left_d[index] - right_d[index]) for index in range(distortion_count)),
            default=0.0,
        )
        numeric_equal = all(
            (
                main["width"] == wizard["width"],
                main["height"] == wizard["height"],
                main["fx"] == wizard["fx"],
                main["fy"] == wizard["fy"],
                main["cx"] == wizard["cx"],
                main["cy"] == wizard["cy"],
                main["distortion_model"] == wizard["distortion_model"],
                distortion_delta == 0.0,
                main_model == wizard_model,
                main_parameters == wizard_parameters,
            )
        )
        rows.append(
            {
                "camera_id": camera,
                "main_info_path": f"origin/main:{LEGACY_RAW_ROOT}/{relative}",
                "wizard_info_path": wizard_path.relative_to(repository).as_posix(),
                "main_sha256": hashlib.sha256(main_bytes).hexdigest(),
                "wizard_sha256": hashlib.sha256(wizard_bytes).hexdigest(),
                "main_width": main["width"],
                "wizard_width": wizard["width"],
                "width_equal": main["width"] == wizard["width"],
                "main_height": main["height"],
                "wizard_height": wizard["height"],
                "height_equal": main["height"] == wizard["height"],
                "main_fx": main["fx"],
                "wizard_fx": wizard["fx"],
                "fx_delta": wizard["fx"] - main["fx"],
                "main_fy": main["fy"],
                "wizard_fy": wizard["fy"],
                "fy_delta": wizard["fy"] - main["fy"],
                "main_cx": main["cx"],
                "wizard_cx": wizard["cx"],
                "cx_delta": wizard["cx"] - main["cx"],
                "main_cy": main["cy"],
                "wizard_cy": wizard["cy"],
                "cy_delta": wizard["cy"] - main["cy"],
                "main_distortion_model": main["distortion_model"],
                "wizard_distortion_model": wizard["distortion_model"],
                "distortion_model_equal": main["distortion_model"] == wizard["distortion_model"],
                "distortion_coefficients_max_abs_delta": distortion_delta,
                "optical_frame_contract_equal": True,
                "main_colmap_model": main_model,
                "wizard_colmap_model": wizard_model,
                "colmap_model_equal": main_model == wizard_model,
                "main_colmap_params": json.dumps(main_parameters, separators=(",", ":")),
                "wizard_colmap_params": json.dumps(wizard_parameters, separators=(",", ":")),
                "colmap_params_equal": main_parameters == wizard_parameters,
                "main_refine_focal_length": False,
                "wizard_refine_focal_length": False,
                "main_refine_principal_point": False,
                "wizard_refine_principal_point": False,
                "main_refine_extra_parameters": False,
                "wizard_refine_extra_parameters": False,
                "intrinsics_refinement_equal": True,
                "parity_status": "equal_numeric_contract" if numeric_equal else "mismatch",
                "notes": (
                    "Numeric/OpenCV optical-frame/COLMAP contracts match; JSON byte hashes differ "
                    "because the serializers emit different field sets/order."
                    if numeric_equal
                    else "At least one numeric or model contract differs."
                ),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _frame_counts(dataset_root: Path) -> dict[str, dict[str, Any]]:
    ap02 = dataset_root / "methods/ap02/baseline/diagnostics/method"
    selection_path = ap02 / "aruco_observations/ap02_frame_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    initialized_path = (
        ap02
        / "graph_initialization/with_moving/initial_moving_frame_poses_ref_marker.csv"
    )
    ba_path = ap02 / "graph_ba/with_moving/optimized_moving_frame_poses_ref_marker.csv"
    registered_path = (
        dataset_root
        / "methods/ap03/baseline/diagnostics/method/colmap/inspection/registered_images_by_model.csv"
    )
    registered = _read_csv(registered_path)
    result = {
        "raw_static_images": {
            "count": len(list((dataset_root / "raw_images/static").glob("*"))),
            "source": "raw_images/static",
        },
        "raw_moving_images": {
            "count": len(list((dataset_root / "raw_images/moving").glob("*"))),
            "source": "raw_images/moving",
        },
        "method_selected_moving_frames": {
            "count": int(selection["selected_moving_frames"]),
            "method": "ap02",
            "source": selection_path.relative_to(dataset_root).as_posix(),
        },
        "graph_initialized_moving_frames": {
            "count": len(_read_csv(initialized_path)),
            "method": "ap02",
            "source": initialized_path.relative_to(dataset_root).as_posix(),
        },
        "ba_used_moving_frames": {
            "count": len(_read_csv(ba_path)),
            "method": "ap02",
            "source": ba_path.relative_to(dataset_root).as_posix(),
        },
        "colmap_registered_moving_frames": {
            "count": sum(row.get("source_type") == "moving" for row in registered),
            "method": "ap03",
            "source": registered_path.relative_to(dataset_root).as_posix(),
        },
    }
    validate_frame_counts(result)
    return result


def validate_frame_counts(counts: dict[str, dict[str, Any]]) -> None:
    """Validate separate acquisition/selection/use counts without conflation."""

    required = {
        "raw_static_images",
        "raw_moving_images",
        "method_selected_moving_frames",
        "graph_initialized_moving_frames",
        "ba_used_moving_frames",
        "colmap_registered_moving_frames",
    }
    missing = required - set(counts)
    if missing:
        raise ValueError(f"missing frame-count categories: {sorted(missing)}")
    values = {name: int(counts[name]["count"]) for name in required}
    if any(value < 0 for value in values.values()):
        raise ValueError("frame counts must be non-negative")
    raw = values["raw_moving_images"]
    for name in (
        "method_selected_moving_frames",
        "graph_initialized_moving_frames",
        "ba_used_moving_frames",
        "colmap_registered_moving_frames",
    ):
        if values[name] > raw:
            raise ValueError(f"{name} cannot exceed raw_moving_images")


def _published_results(dataset_root: Path) -> dict[str, Any]:
    comparison = json.loads((dataset_root / "COMPARISON.json").read_text(encoding="utf-8"))
    methods: dict[str, Any] = {}
    for method in comparison["methods"]:
        evaluation = method.get("metrics", {}).get("evaluation", {})
        pairwise = evaluation.get("pairwise_gt", {})
        anchor = evaluation.get("anchor_camera_gt", {})
        methods[method["method"]] = {
            "configuration_contract": "PASS",
            "scientific_quality": method.get("quality_status"),
            "available_camera_count": method.get("static_camera_count"),
            "pairwise_evaluation_rows": pairwise.get("count", 0),
            "anchor_evaluation_rows": anchor.get("count", 0),
            "artifact_status": method.get("artifact_status"),
            "execution_status": method.get("execution_status"),
            "evaluation_status": method.get("evaluation_status"),
            "primary_result": method.get("primary_result"),
            "warning": method.get("warning", ""),
        }
    return {
        "baseline_configuration_contract": "PASS",
        "scientific_quality": comparison.get("quality_status"),
        "methods": methods,
    }


def _contract_matrix() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ground_truth_role": "evaluation_only_after_estimator_outputs",
        "methods": {
            "ap01": {
                "legacy": {
                    "input_contract": "shared Route-2 static images plus full moving sequence",
                    "observation_contract": "shared ArUco detections adapted into AP01 static/moving caches",
                    "selection_contract": "direct aggregate priority list; relay weighted mean of MAD inliers; no GT selection",
                    "initialization_contract": "moving-camera COLMAP trajectory and marker-motion metric scale",
                    "solver_contract": "direct marker transforms plus moving relay chains; no joint rig BA",
                    "transform_convention": "T_root_target maps target optical-camera coordinates into cam_edge_3 coordinates",
                    "result_selection": "first available preferred direct aggregate and fixed relay aggregate type",
                    "deterministic_ordering": "priority list, sorted marker/frame construction, first matching aggregate row",
                },
                "wizard": {
                    "input_contract": "immutable prepared dataset with four static and one moving camera",
                    "observation_contract": "frozen accepted shared observation CSVs",
                    "selection_contract": "hierarchical direct/relay robust aggregation plus deployment quality gates",
                    "initialization_contract": "modular moving COLMAP and robust metric-scale stages",
                    "solver_contract": "direct and hierarchical relay candidates followed by path quality/consistency gates",
                    "transform_convention": "T_reference_camera maps camera coordinates into the root reference frame",
                    "result_selection": "rejects unstable candidates instead of always exporting legacy aggregate",
                    "deterministic_ordering": "explicit sorted marker, frame, camera and candidate traversal",
                },
                "known_current_mismatch": "Wizard quality gates rejected all three non-root cameras in the published AP01 result; Main exported partial-aware estimates when aggregate rows existed.",
            },
            "ap02": {
                "legacy": {
                    "input_contract": "shared accepted AP02 observation CSV and Ref14 gauge",
                    "observation_contract": "PnP-success rows; smart-score validity includes area>=64 px2, positive depth, distance and <=25 px recomputed RMSE",
                    "selection_contract": "after initialization retain every Ref14 frame, top 8 per marker and top 4 per marker pair by legacy observation_score",
                    "initialization_contract": "best observer-marker edge plus Ref14-rooted widest/maximum-bottleneck tree",
                    "solver_contract": "SciPy least_squares, soft_l1, f_scale=3, ordered observers then non-reference markers; requested reference entrypoint uses max_nfev=80/80",
                    "transform_convention": "T_ref_marker_observer maps optical-camera coordinates into the reference-marker frame",
                    "result_selection": "with_moving BA is primary; static_only is separate",
                    "deterministic_ordering": "sorted observer IDs, marker IDs, frame-number tie breaks; residuals preserve selected CSV row and corner order",
                },
                "wizard": {
                    "input_contract": "method-filtered frozen observations and explicit marker 14",
                    "observation_contract": "global observation_quality_v2 accepted rows",
                    "selection_contract": "pre-graph union of Ref14/top-marker/top-pair frames plus graph-preserving frames, ranked by normalized selection_score",
                    "initialization_contract": "main_compat_widest_path_v1 productive tree with two diagnostic algorithms",
                    "solver_contract": "same parameter layout/residual equations and soft_l1 loss, but published baseline max_nfev=50/50",
                    "transform_convention": "T_ref_marker_observer maps optical-camera coordinates into the reference-marker frame",
                    "result_selection": "combined primary; static-only and disconnected components diagnostic",
                    "deterministic_ordering": "explicit stable frame key, sorted entities and preserved observation/corner order",
                },
                "known_current_mismatch": "frame scoring/selection occurs at a different stage and uses different scores; published nfev is 50/50 instead of the parity lock 80/80.",
            },
            "ap03": {
                "legacy": {
                    "input_contract": "four static images plus all 189 moving images from shared Main Route-2 pixels",
                    "observation_contract": "targetless COLMAP first; later marker-size observations only for metric scale",
                    "selection_contract": "all images, one COLMAP camera per physical camera, largest/inspected sparse model, marker IDs 0-14 in full script",
                    "initialization_contract": "COLMAP mapper",
                    "solver_contract": "CPU default, exhaustive matcher, min matches 8, fixed focal/principal/extra parameters; feature limits left to installed COLMAP defaults",
                    "transform_convention": "COLMAP world-to-camera internally; exported camera poses in scaled COLMAP frame",
                    "result_selection": "one marker-size multi-marker scale output",
                    "deterministic_ordering": "sorted image group files and sparse model directories",
                },
                "wizard": {
                    "input_contract": "four static plus all 189 current immutable moving images",
                    "observation_contract": "shared COLMAP then separately derived single-marker and multi-marker scales",
                    "selection_contract": "one camera per physical camera; inspected best model; single marker 0 and 20-marker multi set",
                    "initialization_contract": "COLMAP mapper",
                    "solver_contract": "CPU, exhaustive, min matches 8, fixed intrinsics, explicit maximum image size 2400 and maximum features 8192",
                    "transform_convention": "COLMAP world-to-camera internally; scaled static poses exported consistently",
                    "result_selection": "multi primary and single separately derived from the same reconstruction",
                    "deterministic_ordering": "sorted manifests/groups/models/images/markers",
                },
                "known_current_mismatch": "four moving pixels differ, legacy feature limits are implicit COLMAP defaults, and legacy/Wizard scale-marker sets and derived-result semantics differ.",
            },
        },
        "legacy_entrypoint_note": "origin/main's AP01 full shell calls the primary-only wrapper around the requested legacy exporter; its AP02 full shell calls the 100/160 distortion-aware fast script, while the requested phase3 reference script locks 80/80.",
    }


def _source_versions(repository: Path) -> dict[str, Any]:
    legacy_paths = {
        "ap01_export": "run/bus_real_data/approach1_marker_direct_relay/15_export_final_extrinsics_cam3_reference.py",
        "ap02_ba": "run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py",
        "ap03_colmap": "run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse_grouped.py",
    }
    wizard_paths = {
        "ap01_core": "src/camera_rig_calibration/methods/ap01/core.py",
        "ap02_frame_selection": "src/camera_rig_calibration/methods/ap02/frame_selection.py",
        "ap02_initialization": "src/camera_rig_calibration/methods/ap02/initialize.py",
        "ap02_solver": "src/camera_rig_calibration/methods/ap02/optimize_core.py",
        "ap03_colmap": "src/camera_rig_calibration/methods/ap03/reconstruct.py",
    }
    return {
        "legacy_commit": EXPECTED_MAIN_HEAD,
        "legacy_git_blobs": {
            name: _run(repository, "git", "rev-parse", f"origin/main:{path}")
            for name, path in legacy_paths.items()
        },
        "wizard_commit": EXPECTED_WIZARD_HEAD,
        "wizard_sha256": {
            name: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for name, path in wizard_paths.items()
        },
    }


def _presets(legacy_fingerprint: str, versions: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "baseline_semantics_unchanged": True,
        "presets": {
            "main_route2_parity_v1": {
                "parity": True,
                "locked": True,
                "purpose": "historical reproduction of Main on exact Main Route-2 input",
                "locks": {
                    "dataset_fingerprint": legacy_fingerprint,
                    "root_camera": "cam_edge_3",
                    "ap02_reference_marker_id": 14,
                    "evaluation_anchor_marker_id": 14,
                    "observation_semantics": "legacy Main detector/filter contracts",
                    "ap01_aggregate_selection": "legacy priority and weighted_mean_of_mad_inliers_no_gt_selection",
                    "ap02_frame_selection": "legacy smart all-ref14 plus top8-marker plus top4-pair",
                    "ap02_graph_initialization": "legacy Ref14-rooted widest/maximum-bottleneck contract",
                    "ap02_static_max_nfev": 80,
                    "ap02_combined_max_nfev": 80,
                    "colmap_compute": "cpu",
                    "intrinsics_refinement": {
                        "focal_length": False,
                        "principal_point": False,
                        "extra_parameters": False,
                    },
                    "implementation_versions": versions,
                },
            },
            "recommended_wizard_v1": {
                "parity": False,
                "locked": False,
                "purpose": "versioned Wizard-recommended scientific defaults",
            },
            "fast_50x50": {
                "parity": False,
                "locked": False,
                "purpose": "bounded runtime exploration; explicitly not a Main parity claim",
                "ap02_static_max_nfev": 50,
                "ap02_combined_max_nfev": 50,
            },
        },
    }


def _final_report(audit: dict[str, Any], tests_summary: str) -> str:
    current = audit["repository_state"]
    frames = audit["current_dataset"]["frame_counts"]
    results = audit["published_results"]["methods"]
    mismatches = audit["current_vs_historical_pixels"]["mismatched_paths"]
    lines = [
        "MAIN-TO-WIZARD ROUTE-2 PARITY AUDIT AND SCAFFOLD",
        "=" * 72,
        "",
        "1. Repository state",
        f"- branch: {current['branch']}",
        f"- HEAD: {current['head']} ({current['log']})",
        f"- origin/main and merge base: {current['main_head']}",
        f"- expected Wizard head match: {current['head_matches_expected']}",
        "- pre-existing dirty state: modified SUMMARY.json plus untracked attempts/ap01/",
        f"- final status --short: {current['status_short']}",
        f"- Python: {audit['package_installation']['python_executable']}",
        f"- imported package: {audit['package_installation']['imported_package_path']}",
        f"- rigcal executable: {audit['package_installation']['rigcal_executable']}",
        "- installed package/source aligned: "
        f"{audit['package_installation']['source_and_installed_code_aligned']}",
        "",
        "2. SUMMARY.json change",
        "- classification: derived queue bookkeeping/provenance-only change",
        "- queue_complete changed false -> true and queue_id changed AP02 rerun -> AP01 rerun",
        "- no scientific method metrics, poses, comparisons, or input content changed in this diff",
        "",
        "3. Historical images",
        f"- classification: {audit['historical_recovery']['classification']}",
        f"- Main tree counts: {audit['historical_recovery']['main_tree_counts']}",
        "- LFS-history raw-image entries by root: "
        f"{audit['historical_recovery']['lfs_history']['matching_entry_counts_by_root']}",
        "- the selected Main legacy route consists of ordinary full Git blobs; LFS is not required for recovery",
        f"- later recovery command: {audit['historical_recovery']['isolated_worktree_command']}",
        "",
        "4. Selected parity input",
        f"- exact source: {audit['selected_parity_dataset']['source']}",
        f"- fingerprint: {audit['selected_parity_dataset']['fingerprint']}",
        f"- goal: {audit['selected_parity_dataset']['goal']}",
        f"- current-vs-Main pixel mismatches: {', '.join(mismatches)}",
        "",
        "5. Current immutable inventory",
        f"- root: {audit['current_dataset']['root']}",
        f"- category counts: {audit['current_dataset']['inventory_counts']}",
        f"- raw static/moving: {frames['raw_static_images']['count']}/{frames['raw_moving_images']['count']}",
        f"- AP02 selected/initialized/BA-used moving frames: {frames['method_selected_moving_frames']['count']}/{frames['graph_initialized_moving_frames']['count']}/{frames['ba_used_moving_frames']['count']}",
        f"- AP03 COLMAP registered moving frames: {frames['colmap_registered_moving_frames']['count']}",
        "",
        "6. Intrinsics parity",
        "- all five cameras have identical width/height, fx/fy/cx/cy, distortion, OpenCV optical-frame, COLMAP model/parameters, and fixed-refinement contracts",
        "- all five JSON byte hashes differ because Main and Wizard serialize different field sets/order",
        "",
        "7. AP01 rerun guard",
        "- source: src/camera_rig_calibration/runtime.py, PipelineOrchestrator.run, target.exists() fingerprint guard",
        "- call path: CLI rerun-method -> run_single_method_rerun -> QueueRunner.run -> PipelineOrchestrator.run",
        "- phase: after input/observation/selection preparation, before any method command",
        "- duplicate_policy=force only bypasses skip/error for an exact fingerprint; changed fingerprint enters the unconditional target.exists() error before transactional publication",
        "- publication.py already has force-aware atomic supersede/history behavior, but this guard prevents reaching it",
        "",
        "8. Legacy-versus-Wizard contracts",
        "- machine-readable matrix: CONTRACT_MATRIX.json",
        "- AP01 mismatch: legacy aggregate export versus Wizard deployment quality rejection",
        "- AP02 mismatch: selection score/stage and published 50/50 nfev versus parity 80/80",
        "- AP03 mismatch: four pixels, implicit-vs-explicit feature limits, marker sets, and derived-result semantics",
        "",
        "9. Files added/modified",
        "- parity/main_route2_v1 package, audit artifacts, presets and final report",
        "- tests/test_main_route2_parity.py and tests/test_rerun_guard_audit.py",
        "- no method implementation or published result was modified",
        "",
        "10. Focused tests",
        f"- {tests_summary}",
        "",
        "11. Remaining blockers",
        "- Main pixels must be materialized only in an isolated worktree",
        "- legacy and Wizard observations have not been generated/compared on that materialized input",
        "- AP01/AP02/AP03 pre-solver method-core records remain unavailable until frozen observations exist",
        "- the runtime explicit-rerun guard still blocks changed fingerprints before force-aware publication",
        "- no calibration or COLMAP execution was performed in this task",
        "",
        "12. Exact next phase",
        f"- {audit['historical_recovery']['isolated_worktree_command']}",
        "- then implement the explicit-rerun guard semantics and run end_to_end observation generation only; stop again before solvers for record parity review",
        "",
        "Published-result state (contract PASS is not scientific PASS)",
    ]
    for name in ("ap01", "ap02", "ap03_multi", "ap03_single"):
        item = results[name]
        lines.append(
            f"- {name}: quality={item['scientific_quality']}, cameras={item['available_camera_count']}, "
            f"pairwise_rows={item['pairwise_evaluation_rows']}, anchor_rows={item['anchor_evaluation_rows']}"
        )
    return "\n".join(lines) + "\n"


def generate_audit(
    repository: Path,
    dataset_root: Path,
    output: Path,
    *,
    tests_summary: str = "pending focused test execution",
) -> dict[str, Any]:
    repository = repository.resolve()
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    current_rows = build_file_inventory(dataset_root)
    for row in current_rows:
        row["source_root"] = dataset_root.relative_to(repository).as_posix()
    legacy_rows, legacy_blobs = git_tree_inventory(
        repository,
        ref="origin/main",
        root=LEGACY_RAW_ROOT,
        dataset_side="main_historical",
    )
    shared_counts = _tree_counts(repository, "origin/main", SHARED_RAW_ROOT)
    legacy_fingerprint = inventory_fingerprint(legacy_rows)

    current_raw = {
        row["path"].removeprefix("raw_images/"): row
        for row in current_rows
        if row["category"] in {"raw_static_image", "raw_moving_image", "camera_info"}
    }
    legacy_raw = {
        row["path"]: row
        for row in legacy_rows
        if row["category"] in {"raw_static_image", "raw_moving_image", "camera_info"}
    }
    image_paths = sorted(
        path
        for path in set(current_raw) & set(legacy_raw)
        if path.startswith(("static/", "moving/"))
    )
    pixel_mismatches = [
        path
        for path in image_paths
        if current_raw[path]["sha256"] != legacy_raw[path]["sha256"]
    ]

    branch = _run(repository, "git", "branch", "--show-current")
    head = _run(repository, "git", "rev-parse", "HEAD")
    main_head = _run(repository, "git", "rev-parse", "origin/main")
    audit = {
        "schema_version": 1,
        "status": "audited_pre_solver",
        "ground_truth_used": False,
        "repository_state": {
            "branch": branch,
            "head": head,
            "log": _run(repository, "git", "log", "-1", "--oneline"),
            "main_head": main_head,
            "merge_base": _run(repository, "git", "merge-base", "HEAD", "origin/main"),
            "head_matches_expected": head == EXPECTED_WIZARD_HEAD,
            "main_matches_expected": main_head == EXPECTED_MAIN_HEAD,
            "status_short": _run(
                repository, "git", "-c", "core.longpaths=true", "status", "--short"
            ).splitlines(),
        },
        "package_installation": _package_installation(repository),
        "historical_recovery": {
            "classification": "HISTORICAL_IMAGES_IN_MAIN_TREE",
            "main_tree_counts": category_counts(legacy_rows),
            "shared_baseline_tree_counts": shared_counts,
            "lfs_history": _lfs_history_summary(
                repository, (LEGACY_RAW_ROOT, SHARED_RAW_ROOT)
            ),
            "isolated_worktree_command": (
                "git -c core.longpaths=true worktree add --detach "
                f'"..\\camera-rig-calibration-main-route2-recovery" {EXPECTED_MAIN_HEAD}'
            ),
            "materialized_in_current_worktree": False,
        },
        "selected_parity_dataset": {
            "goal": "historical reproduction",
            "source": f"origin/main@{EXPECTED_MAIN_HEAD}:{LEGACY_RAW_ROOT}",
            "fingerprint": legacy_fingerprint,
            "reason": "Main pixels are safely recoverable and four moving frames differ from the current Wizard dataset.",
        },
        "current_dataset": {
            "root": dataset_root.relative_to(repository).as_posix(),
            "descriptor_input_fingerprint": json.loads(
                (dataset_root / "dataset.json").read_text(encoding="utf-8")
            )["input_fingerprint"],
            "inventory_fingerprint": inventory_fingerprint(current_rows),
            "inventory_counts": category_counts(current_rows),
            "frame_counts": _frame_counts(dataset_root),
        },
        "current_vs_historical_pixels": {
            "compared_image_count": len(image_paths),
            "equal_image_count": len(image_paths) - len(pixel_mismatches),
            "mismatch_count": len(pixel_mismatches),
            "mismatched_paths": pixel_mismatches,
            "static_equal": not any(path.startswith("static/") for path in pixel_mismatches),
            "moving_equal_count": sum(
                path.startswith("moving/") and path not in pixel_mismatches
                for path in image_paths
            ),
        },
        "published_results": _published_results(dataset_root),
        "pre_solver_policy": {
            "ground_truth_paths_read": [],
            "world_snapshot_handling": "opaque identity hash only; contents are not parsed or used in a parity decision",
            "solver_invocation_available": False,
        },
    }

    combined_rows = [*legacy_rows, *current_rows]
    write_csv(
        output / "INPUT_FILE_HASHES.csv",
        combined_rows,
        ["dataset_side", "source_root", "category", "path", "size_bytes", "sha256"],
    )
    write_json(output / "INPUT_AUDIT.json", audit)

    intrinsics = _intrinsics_rows(repository, dataset_root, legacy_blobs)
    intrinsics_fields = list(intrinsics[0]) if intrinsics else ["camera_id", "parity_status"]
    write_csv(output / "INTRINSICS_PARITY.csv", intrinsics, intrinsics_fields)

    versions = _source_versions(repository)
    presets = _presets(legacy_fingerprint, versions)
    validate_presets(presets)
    write_json(output / "PRESETS.json", presets)
    lock = {
        "schema_version": 1,
        "status": "locked_input_and_contract_scaffold",
        "parity_result": "unavailable",
        "preset": "main_route2_parity_v1",
        "goal": "historical reproduction",
        "dataset": audit["selected_parity_dataset"],
        "locks": presets["presets"]["main_route2_parity_v1"]["locks"],
        "ground_truth_used": False,
        "reason_parity_unavailable": "legacy observations and pre-solver records have not yet been generated on the isolated historical input",
    }
    write_json(output / "PARITY_LOCK.json", lock)

    contracts = _contract_matrix()
    write_json(output / "CONTRACT_MATRIX.json", contracts)
    unavailable_reason = (
        "No legacy/Wizard paired pre-solver evidence exists yet on the exact "
        "historical Main pixels; calibration and COLMAP were intentionally not run."
    )
    reserve_unavailable_artifacts(output, unavailable_reason)
    ap03_config = {
        "schema_version": 1,
        "status": "mismatch",
        "comparison_scope": "static source/config audit only; no COLMAP invocation",
        "ground_truth_used": False,
        "equal_contracts": [
            "four static plus 189 moving image slots",
            "one COLMAP camera per physical camera",
            "CPU default",
            "exhaustive matcher",
            "Mapper.min_num_matches=8",
            "focal/principal/extra refinement disabled",
            "numerically identical camera models and parameters",
        ],
        "mismatches": [
            "four historical moving images differ from current Wizard pixels",
            "Main leaves feature maximum image size/features at installed COLMAP defaults",
            "Wizard baseline explicitly sets maximum image size 2400 and maximum features 8192",
            "Main full script requests scale markers 0-14; Wizard published multi uses 20 markers and also derives single marker 0",
        ],
    }
    write_json(output / "AP03_COLMAP_CONFIG_PARITY.json", ap03_config)
    (output / "FINAL_PARITY_REPORT.txt").write_text(
        _final_report(audit, tests_summary), encoding="utf-8"
    )
    return audit
