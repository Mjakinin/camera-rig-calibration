#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run existing grouped calibrated AP03 COLMAP and marker-size metric scaling in an isolated real-data root."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--marker-ids", default="0-20")
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--min-area-px2", type=float, default=100.0)
    parser.add_argument("--reproj-thresh-px", type=float, default=5.0)
    parser.add_argument("--ransac-iters", type=int, default=1000)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--reuse-colmap", action="store_true")
    parser.add_argument("--reuse-all", action="store_true")
    parser.add_argument("--cameras", default=",".join(CAMERAS))
    parser.add_argument("--moving-camera-id", default="moving_calib_camera")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--colmap-executable", default="colmap")
    parser.add_argument("--max-image-size", type=int)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1])
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "METHOD_STATUS.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_main(module, argv: list[str]) -> None:
    previous = sys.argv[:]
    try:
        sys.argv = [str(getattr(module, "__file__", "module")), *argv]
        module.main()
    finally:
        sys.argv = previous


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def prepare_dataset(dataset: Path, out: Path) -> None:
    raw = dataset / "raw_images"
    static_dir = raw / "static"
    moving_dir = raw / "moving"
    dataset_root = out / "01_colmap_dataset"
    image_dir = dataset_root / "images"

    shutil.rmtree(dataset_root, ignore_errors=True)
    image_dir.mkdir(parents=True)

    rows = []
    for camera in CAMERAS:
        source = static_dir / f"{camera}.png"
        if not source.is_file():
            raise RuntimeError(f"Missing AP03 static image: {source}")
        name = f"static_{camera}.png"
        link_or_copy(source, image_dir / name)
        rows.append({
            "image_name": name,
            "source_type": "static",
            "source_id": camera,
            "source_path": str(source),
        })

    moving_files = sorted(moving_dir.glob("frame_*.png"))
    if not moving_files:
        raise RuntimeError(f"No AP03 moving images in {moving_dir}")

    for source in moving_files:
        name = f"moving_{source.name}"
        link_or_copy(source, image_dir / name)
        rows.append({
            "image_name": name,
            "source_type": "moving",
            "source_id": source.stem,
            "source_path": str(source),
        })

    write_csv(
        dataset_root / "image_manifest.csv",
        rows,
        ["image_name", "source_type", "source_id", "source_path"],
    )
    (dataset_root / "README_REAL_AP03_DATASET.txt").write_text(
        "\n".join([
            "AP03 REAL-DATA COLMAP DATASET",
            "=" * 72,
            "",
            f"Source dataset: {dataset}",
            "Static input: one selected image per physical static camera.",
            "Moving input: all extracted 3 Hz moving-camera frames.",
            "Intrinsics: fixed and grouped per physical camera.",
            f"Static images: {len(CAMERAS)}",
            f"Moving images: {len(moving_files)}",
            f"Total images: {len(rows)}",
            "",
        ]) + "\n",
        encoding="utf-8",
    )


def configure_grouped_module(
    module, out: Path, dataset: Path, moving_camera_id: str
) -> None:
    dataset_root = out / "01_colmap_dataset"
    run_root = out / "02_colmap_sparse"
    module.AP3_ROOT = out
    module.DATASET_ROOT = dataset_root
    module.IMAGE_DIR = dataset_root / "images"
    module.MANIFEST = dataset_root / "image_manifest.csv"
    module.SHARED_RAW = dataset / "raw_images"
    module.RUN_ROOT = run_root
    module.DB = run_root / "database.db"
    module.SPARSE_ROOT = run_root / "sparse"
    module.TXT_ROOT = run_root / "sparse_txt"
    module.GROUP_ROOT = run_root / "camera_groups"
    module.STATIC_CAMERAS = list(CAMERAS)
    module.MOVING_CAMERA = moving_camera_id


def configure_inspect_module(module, out: Path) -> None:
    module.AP3_ROOT = out
    module.DATASET_ROOT = out / "01_colmap_dataset"
    module.TXT_ROOT = out / "02_colmap_sparse" / "sparse_txt"
    module.OUT_ROOT = out / "03_reconstruction_inspection"
    module.STATIC_EXPECTED = [f"static_{camera}.png" for camera in CAMERAS]


def configure_scale_common(module, out: Path) -> None:
    module.AP3_ROOT = out
    module.TXT_ROOT = out / "02_colmap_sparse" / "sparse_txt"
    module.IMAGE_DIR = out / "01_colmap_dataset" / "images"
    module.INSPECT_SUMMARY = (
        out / "03_reconstruction_inspection" / "colmap_model_summary.csv"
    )
    module.OUT_ROOT = out / "06_triangulated_ref_aruco_registration"
    module.AP3_CMP = out / "07_final_results"
    module.COMBINED = out / "07_final_results"
    module.STATIC_CAMERAS = list(CAMERAS)


def pose_positions(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    return {
        row["entity_id"]: np.asarray([
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"]),
        ], dtype=np.float64)
        for row in rows
        if row.get("entity_id") in CAMERAS
    }


def pairwise_rows(positions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for first, second in combinations(CAMERAS, 2):
        if first not in positions or second not in positions:
            continue
        rows.append({
            "camera_a": first,
            "camera_b": second,
            "distance_m": float(np.linalg.norm(positions[first] - positions[second])),
        })
    return rows


def best_model_diagnostics(summary_path: Path) -> dict:
    rows = read_csv(summary_path)
    if not rows:
        return {}
    best = sorted(
        rows,
        key=lambda row: (
            int(row["registered_static_cameras"]),
            int(row["registered_images"]),
            int(row["num_3d_points"]),
        ),
        reverse=True,
    )[0]
    return {
        "best_model": best["model"],
        "registered_images": int(best["registered_images"]),
        "registered_static_cameras": int(best["registered_static_cameras"]),
        "registered_moving_frames": int(best["registered_moving_frames"]),
        "num_3d_points": int(best["num_3d_points"]),
        "static_registered_list": best["static_registered_list"],
        "static_missing_list": best["static_missing_list"],
    }


def main() -> None:
    global CAMERAS
    args = parse_args()
    CAMERAS = [value.strip() for value in args.cameras.split(",") if value.strip()]
    if not CAMERAS:
        raise RuntimeError("--cameras must contain at least one camera ID")
    started = time.time()

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[2]
    bus_run = repo / "run" / "bus_real_data"
    ap03_dir = bus_run / "approach3_targetless_colmap_aruco_scale"

    try:
        if not ap03_dir.is_dir():
            raise RuntimeError(f"Missing existing AP03 implementation: {ap03_dir}")
        if str(bus_run) not in sys.path:
            sys.path.insert(0, str(bus_run))
        if str(ap03_dir) not in sys.path:
            sys.path.insert(0, str(ap03_dir))

        final_dir = out / "07_final_results"
        pose_file = final_dir / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
        metadata_file = final_dir / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"

        if args.reuse_all and pose_file.is_file() and metadata_file.is_file():
            print("[REUSE] complete AP03 result:", pose_file)
        else:
            prepare_dataset(dataset, out)

            sparse_summary = (
                out / "02_colmap_sparse" / "AP03_GROUPED_COLMAP_REPORT.txt"
            )
            sparse_models = out / "02_colmap_sparse" / "sparse_txt"

            if not (
                args.reuse_colmap
                and sparse_models.is_dir()
                and any(sparse_models.iterdir())
            ):
                grouped = load_module(
                    "ap03_real_grouped_colmap",
                    ap03_dir / "02_run_colmap_sparse_grouped.py",
                )
                configure_grouped_module(
                    grouped, out, dataset, args.moving_camera_id
                )
                grouped_args = [
                    "--colmap", args.colmap_executable,
                    "--use-gpu", str(args.use_gpu),
                    "--matcher", args.matcher,
                    "--sequential-overlap", str(args.sequential_overlap),
                    "--mapper-min-matches", str(args.mapper_min_matches),
                ]
                if args.max_image_size is not None:
                    grouped_args += ["--max-image-size", str(args.max_image_size)]
                if args.max_features is not None:
                    grouped_args += ["--max-features", str(args.max_features)]
                if args.loop_detection is not None:
                    grouped_args += ["--loop-detection", str(args.loop_detection)]
                run_main(grouped, grouped_args)
            else:
                print("[REUSE] AP03 COLMAP sparse models:", sparse_models)

            inspector = load_module(
                "ap03_real_inspector",
                ap03_dir / "03_inspect_colmap_reconstruction.py",
            )
            configure_inspect_module(inspector, out)
            run_main(inspector, [])

            scale_common = load_module(
                "ap03_scale_common",
                ap03_dir / "ap03_scale_common.py",
            )
            configure_scale_common(scale_common, out)
            sys.modules["ap03_scale_common"] = scale_common

            scale_module = load_module(
                "ap03_real_marker_scale",
                ap03_dir / "10_estimate_scale_from_marker_size_only.py",
            )
            scale_module.STATIC_CAMERAS = list(CAMERAS)
            run_main(scale_module, [
                "--out-dir", str(final_dir),
                "--marker-ids", args.marker_ids,
                "--marker-length-m", str(args.marker_length_m),
                "--min-area-px2", str(args.min_area_px2),
                "--reproj-thresh-px", str(args.reproj_thresh_px),
                "--ransac-iters", str(args.ransac_iters),
                "--min-inliers", str(args.min_inliers),
                "--dictionary", args.dictionary,
            ])

        if not metadata_file.is_file():
            raise RuntimeError(f"Missing AP03 metadata: {metadata_file}")

        metadata = json.loads(metadata_file.read_text())
        positions = pose_positions(pose_file) if pose_file.is_file() else {}

        diagnostics = {
            "approach": "AP03_targetless_grouped_COLMAP_marker_size_scale",
            "marker_scale": metadata,
            "reconstruction": best_model_diagnostics(
                out / "03_reconstruction_inspection" / "colmap_model_summary.csv"
            ),
            "available_static_cameras": sorted(positions),
            "missing_static_cameras": sorted(set(CAMERAS) - set(positions)),
            "runtime_seconds": time.time() - started,
            "ground_truth_used": False,
        }
        diagnostics_path = final_dir / "AP03_DIAGNOSTICS.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        write_csv(
            final_dir / "AP03_PAIRWISE_DISTANCES.csv",
            pairwise_rows(positions),
            ["camera_a", "camera_b", "distance_m"],
        )

        original_status = str(metadata.get("status", "UNKNOWN"))
        expected_count = len(CAMERAS)
        if len(positions) == expected_count:
            status = "OK_FULL" if original_status.startswith("OK") else original_status
        elif positions:
            status = f"PARTIAL_{len(positions)}_OF_{expected_count}"
        else:
            status = original_status if original_status.startswith("FAILED") else "FAILED"

        success = (
            len(positions) == expected_count
            and metadata.get("scale_m_per_colmap_unit") is not None
        )
        write_status(out, {
            "method": "AP03",
            "status": status,
            "success": success,
            "available_static_cameras": sorted(positions),
            "runtime_seconds": time.time() - started,
            "pose_file": str(pose_file),
            "diagnostics_file": str(diagnostics_path),
        })

        print("\nAP03 REAL-DATA RESULT")
        print("=" * 72)
        print("status:", status)
        print("static cameras:", sorted(positions))
        print("registered images:", metadata.get("registered_images"))
        print("scale:", metadata.get("scale_m_per_colmap_unit"))
        print("pose file:", pose_file)

        if len(positions) < expected_count:
            raise RuntimeError(
                f"AP03 registered only {len(positions)}/{expected_count} static cameras; "
                f"method status remains recorded as {status}"
            )

    except Exception as exc:
        existing = {}
        path = out / "METHOD_STATUS.json"
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        failure = {
            **existing,
            "method": "AP03",
            "status": existing.get("status", "FAILED"),
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.time() - started,
        }
        write_status(out, failure)
        print(failure["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
