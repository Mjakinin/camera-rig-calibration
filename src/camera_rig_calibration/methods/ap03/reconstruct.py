"""Run AP03 COLMAP with one calibrated camera model per physical camera.

The four static cameras each receive their own COLMAP camera id. All moving
frames share one camera id. Intrinsics are read from the shared camera_info
JSON files and are fixed during mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
import subprocess
from pathlib import Path


AP3_ROOT = Path("workspace/standalone_methods/ap03")
DATASET_ROOT = AP3_ROOT / "01_colmap_dataset"
IMAGE_DIR = DATASET_ROOT / "images"
MANIFEST = DATASET_ROOT / "image_manifest.csv"
SHARED_RAW = Path("results/simulation/baseline/route2/raw_images")
RUN_ROOT = AP3_ROOT / "02_colmap_sparse"
DB = RUN_ROOT / "database.db"
SPARSE_ROOT = RUN_ROOT / "sparse"
TXT_ROOT = RUN_ROOT / "sparse_txt"
GROUP_ROOT = RUN_ROOT / "camera_groups"

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
MOVING_CAMERA = "moving_calib_camera"


def run(command: list[str]) -> None:
    print("\n[CMD]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_manifest(manifest: Path = MANIFEST) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise RuntimeError(f"Missing AP03 image manifest: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty AP03 image manifest: {manifest}")
    return rows


def read_camera_info(
    camera_name: str, shared_raw: Path = SHARED_RAW
) -> dict:
    path = shared_raw / "camera_info" / f"{camera_name}.json"
    if not path.is_file():
        raise RuntimeError(f"Missing camera_info for {camera_name}: {path}")
    data = json.loads(path.read_text())

    flat_k = data.get("K", data.get("k"))
    if flat_k is None and "camera_matrix" in data:
        value = data["camera_matrix"]
        flat_k = value.get("data") if isinstance(value, dict) else value
    if flat_k is None:
        fx = float(data["fx"])
        fy = float(data.get("fy", fx))
        cx = float(data["cx"])
        cy = float(data["cy"])
    else:
        fx = float(flat_k[0])
        fy = float(flat_k[4])
        cx = float(flat_k[2])
        cy = float(flat_k[5])

    distortion = data.get("D", data.get("d"))
    if distortion is None and "distortion_coefficients" in data:
        value = data["distortion_coefficients"]
        distortion = value.get("data") if isinstance(value, dict) else value
    if distortion is None:
        distortion = data.get("distortion", [])
    distortion = [float(value) for value in distortion]

    return {
        "path": str(path),
        "width": int(data.get("width", data.get("image_width", 0)) or 0),
        "height": int(data.get("height", data.get("image_height", 0)) or 0),
        "distortion_model": str(data.get("distortion_model", "plumb_bob")),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "D": distortion,
    }


def colmap_camera(info: dict) -> tuple[str, str]:
    """Map ROS camera_info models to fixed COLMAP camera models."""
    fx, fy = info["fx"], info["fy"]
    cx, cy = info["cx"], info["cy"]
    model = info["distortion_model"].strip().lower()
    d = list(info["D"])

    if model in {"equidistant", "fisheye"}:
        d = (d + [0.0] * 4)[:4]
        params = [fx, fy, cx, cy, d[0], d[1], d[2], d[3]]
        return "OPENCV_FISHEYE", ",".join(f"{x:.17g}" for x in params)

    if model not in {"", "none", "plumb_bob", "rational_polynomial"}:
        raise RuntimeError(
            f"Unsupported distortion_model={info['distortion_model']!r} "
            f"in {info['path']}"
        )

    if not d or max(abs(value) for value in d) <= 1e-15:
        params = [fx, fy, cx, cy]
        return "PINHOLE", ",".join(f"{x:.17g}" for x in params)

    # ROS plumb_bob ordering is k1,k2,p1,p2,k3. FULL_OPENCV extends this
    # with k4,k5,k6 and is therefore lossless for the standard ROS model.
    d = (d + [0.0] * 8)[:8]
    params = [fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]
    return "FULL_OPENCV", ",".join(f"{x:.17g}" for x in params)


def build_groups(
    rows: list[dict[str, str]],
    static_cameras: tuple[str, ...] = tuple(STATIC_CAMERAS),
    moving_camera: str = MOVING_CAMERA,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {camera: [] for camera in static_cameras}
    groups[moving_camera] = []

    for row in rows:
        image_name = row["image_name"]
        source_type = row.get("source_type", "")
        source_id = row.get("source_id", "")
        if source_type == "static":
            if source_id not in groups or source_id == moving_camera:
                raise RuntimeError(f"Unknown static camera in manifest: {source_id}")
            groups[source_id].append(image_name)
        elif source_type == "moving":
            groups[moving_camera].append(image_name)
        else:
            raise RuntimeError(f"Unknown source_type in manifest row: {row}")

    missing = [name for name, images in groups.items() if not images]
    if missing:
        raise RuntimeError(f"AP03 camera groups without images: {missing}")

    return groups


def write_group_files(
    groups: dict[str, list[str]],
    group_root: Path = GROUP_ROOT,
    shared_raw: Path = SHARED_RAW,
) -> list[dict]:
    group_root.mkdir(parents=True, exist_ok=True)
    records = []
    for group_name, image_names in groups.items():
        list_path = group_root / f"{group_name}.txt"
        list_path.write_text("\n".join(sorted(image_names)) + "\n")
        info = read_camera_info(group_name, shared_raw)
        model, params = colmap_camera(info)
        records.append(
            {
                "group": group_name,
                "image_count": len(image_names),
                "image_list_path": str(list_path),
                "camera_info_path": info["path"],
                "camera_model": model,
                "camera_params": params,
                "width": info["width"],
                "height": info["height"],
            }
        )
    return records


def validate_database_groups(
    records: list[dict],
    groups: dict[str, list[str]],
    database: Path = DB,
    run_root: Path = RUN_ROOT,
) -> None:
    connection = sqlite3.connect(database)
    try:
        image_to_camera = {
            name: int(camera_id)
            for name, camera_id in connection.execute(
                "SELECT name, camera_id FROM images"
            )
        }
        camera_rows = {
            int(camera_id): (int(model), int(width), int(height))
            for camera_id, model, width, height in connection.execute(
                "SELECT camera_id, model, width, height FROM cameras"
            )
        }
    finally:
        connection.close()

    assignment_rows = []
    group_camera_ids: dict[str, int] = {}
    for record in records:
        group = record["group"]
        camera_ids = {
            image_to_camera[name]
            for name in groups[group]
            if name in image_to_camera
        }
        if len(camera_ids) != 1:
            raise RuntimeError(
                f"COLMAP group {group} must map to one camera id, got {camera_ids}"
            )
        camera_id = next(iter(camera_ids))
        group_camera_ids[group] = camera_id
        db_model, db_width, db_height = camera_rows[camera_id]
        assignment_rows.append(
            {
                **record,
                "camera_id": camera_id,
                "database_model_id": db_model,
                "database_width": db_width,
                "database_height": db_height,
            }
        )

    if len(set(group_camera_ids.values())) != len(group_camera_ids):
        raise RuntimeError(
            "Different physical cameras unexpectedly share a COLMAP camera id: "
            f"{group_camera_ids}"
        )

    fields = list(assignment_rows[0].keys())
    with (run_root / "camera_group_assignments.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignment_rows)


def count_registered_images(images_txt: Path) -> int:
    if not images_txt.is_file():
        return 0
    count = 0
    for line in images_txt.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                int(parts[0])
                int(parts[8])
                count += 1
            except ValueError:
                pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--shared-raw", type=Path, default=SHARED_RAW)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--cameras", default=",".join(STATIC_CAMERAS)
    )
    parser.add_argument("--moving-camera", default=MOVING_CAMERA)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--max-image-size", type=int)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1])
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    image_dir = dataset_root / "images"
    manifest = dataset_root / "image_manifest.csv"
    shared_raw = args.shared_raw.resolve()
    run_root = args.run_root.resolve()
    database = run_root / "database.db"
    sparse_root = run_root / "sparse"
    txt_root = run_root / "sparse_txt"
    group_root = run_root / "camera_groups"
    static_cameras = tuple(
        value.strip() for value in args.cameras.split(",") if value.strip()
    )
    moving_camera = args.moving_camera

    executable = shutil.which(args.colmap)
    if executable is None:
        raise RuntimeError(f"COLMAP executable not found: {args.colmap}")
    if not image_dir.is_dir():
        raise RuntimeError(f"Missing AP03 image directory: {image_dir}")

    if (
        args.reuse
        and txt_root.is_dir()
        and any(txt_root.rglob("images.txt"))
    ):
        print(f"[REUSE] AP03 grouped COLMAP: {txt_root}", flush=True)
        return
    shutil.rmtree(run_root, ignore_errors=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    txt_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(manifest)
    groups = build_groups(manifest_rows, static_cameras, moving_camera)
    records = write_group_files(groups, group_root, shared_raw)

    for record in records:
        command = [
            executable,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--image_list_path",
            record["image_list_path"],
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            record["camera_model"],
            "--ImageReader.camera_params",
            record["camera_params"],
            "--SiftExtraction.use_gpu",
            str(args.use_gpu),
        ]
        if args.max_image_size is not None:
            command += ["--SiftExtraction.max_image_size", str(args.max_image_size)]
        if args.max_features is not None:
            command += ["--SiftExtraction.max_num_features", str(args.max_features)]
        run(command)

    validate_database_groups(records, groups, database, run_root)

    if args.matcher == "exhaustive":
        command = [
            executable,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            str(args.use_gpu),
        ]
    else:
        command = [
            executable,
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            str(args.use_gpu),
            "--SequentialMatching.overlap",
            str(args.sequential_overlap),
        ]
        if args.loop_detection is not None:
            command += [
                "--SequentialMatching.loop_detection",
                str(args.loop_detection),
            ]
    run(command)

    run(
        [
            executable,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_root),
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.min_num_matches",
            str(args.mapper_min_matches),
        ]
    )

    model_dirs = sorted(
        path for path in sparse_root.iterdir() if path.is_dir()
    )
    if not model_dirs:
        raise RuntimeError("COLMAP mapper produced no sparse model")

    model_counts = []
    for model_dir in model_dirs:
        out_dir = txt_root / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                executable,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(out_dir),
                "--output_type",
                "TXT",
            ]
        )
        model_counts.append((model_dir.name, count_registered_images(out_dir / "images.txt")))

    report_lines = [
        "AP03 grouped calibrated COLMAP run",
        "===================================",
        "",
        "Physical camera contract:",
        "- one COLMAP camera id per static physical camera",
        "- one shared COLMAP camera id for all moving-camera frames",
        "- camera intrinsics and distortion loaded from camera_info JSON",
        "- intrinsic refinement disabled during mapping",
        "",
        "Groups:",
    ]
    for record in records:
        report_lines.append(
            f"- {record['group']}: {record['image_count']} images, "
            f"{record['camera_model']}, {record['camera_info_path']}"
        )
    report_lines.extend(["", "Sparse models:"])
    for model, count in model_counts:
        report_lines.append(f"- {model}: {count} registered images")
    (run_root / "colmap_grouped_report.txt").write_text(
        "\n".join(report_lines) + "\n"
    )

    print("\n[OK] AP03 grouped calibrated COLMAP run complete")
    print(
        f"[OK] camera assignment audit: "
        f"{run_root / 'camera_group_assignments.csv'}"
    )
    print(f"[OK] report: {run_root / 'colmap_grouped_report.txt'}")


if __name__ == "__main__":
    main()
