"""Inspect and select the deterministic AP03 sparse reconstruction."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


AP3_ROOT = Path("workspace/standalone_methods/ap03")
DATASET_ROOT = AP3_ROOT / "01_colmap_dataset"
TXT_ROOT = AP3_ROOT / "02_colmap_sparse" / "sparse_txt"
OUT_ROOT = AP3_ROOT / "03_reconstruction_inspection"

STATIC_EXPECTED = [
    "static_cam_edge_0.png",
    "static_cam_edge_1.png",
    "static_cam_edge_3.png",
    "static_cam_edge_5.png",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_manifest(dataset_root: Path = DATASET_ROOT):
    path = dataset_root / "image_manifest.csv"
    if not path.exists():
        return []

    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows, fields):
    ensure_dir(path.parent)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_images_txt(path: Path):
    images = []

    if not path.exists():
        return images

    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = [float(x) for x in parts[1:5]]
            tx, ty, tz = [float(x) for x in parts[5:8]]
            camera_id = int(parts[8])
            name = parts[9]

            images.append({
                "image_id": image_id,
                "camera_id": camera_id,
                "image_name": name,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "point3d_ids": [],
            })

            if i + 1 < len(lines):
                points = lines[i + 1].strip().split()
                images[-1]["point3d_ids"] = [
                    int(points[index])
                    for index in range(2, len(points), 3)
                    if int(points[index]) >= 0
                ]
            i += 2
        else:
            i += 1

    return images


def parse_points3D_txt(path: Path):
    points = {}
    if not path.exists():
        return points

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        point_id = int(parts[0])
        track = [
            (int(parts[index]), int(parts[index + 1]))
            for index in range(8, len(parts) - 1, 2)
        ]
        points[point_id] = {
            "error_px": float(parts[7]),
            "track": track,
        }
    return points


def parse_cameras_txt(path: Path):
    cameras = []
    if not path.exists():
        return cameras

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 5:
            cameras.append({
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": " ".join(parts[4:]),
            })

    return cameras


def inspect_models(
    txt_root: Path = TXT_ROOT,
    static_expected: tuple[str, ...] = tuple(STATIC_EXPECTED),
):
    if not txt_root.exists():
        raise RuntimeError(f"Missing COLMAP txt model root: {txt_root}")

    rows = []
    image_rows = []

    for model_dir in sorted(txt_root.iterdir()):
        if not model_dir.is_dir():
            continue

        images = parse_images_txt(model_dir / "images.txt")
        cameras = parse_cameras_txt(model_dir / "cameras.txt")
        points = parse_points3D_txt(model_dir / "points3D.txt")
        num_points = len(points)

        image_names = {r["image_name"] for r in images}
        static_registered = [
            name for name in static_expected if name in image_names
        ]
        moving_registered = [name for name in image_names if name.startswith("moving_")]

        rows.append({
            "model": model_dir.name,
            "registered_images": len(images),
            "registered_static_cameras": len(static_registered),
            "registered_moving_frames": len(moving_registered),
            "num_3d_points": num_points,
            "static_registered_list": ";".join(static_registered),
            "static_missing_list": ";".join(
                x for x in static_expected if x not in image_names
            ),
            "model_path": str(model_dir),
        })

        for r in images:
            image_rows.append({
                "model": model_dir.name,
                "image_name": r["image_name"],
                "source_type": "static" if r["image_name"].startswith("static_") else "moving",
                "camera_id": r["camera_id"],
                "qw": r["qw"],
                "qx": r["qx"],
                "qy": r["qy"],
                "qz": r["qz"],
                "tx": r["tx"],
                "ty": r["ty"],
                "tz": r["tz"],
            })

    return rows, image_rows


def reconstruction_diagnostics(
    model_dir: Path,
    manifest: list[dict[str, str]],
    static_expected: tuple[str, ...],
) -> dict:
    """Create GT-free per-camera COLMAP support diagnostics."""

    images = parse_images_txt(model_dir / "images.txt")
    points = parse_points3D_txt(model_dir / "points3D.txt")
    by_name = {str(item["image_name"]): item for item in images}
    source_by_name = {
        str(row["image_name"]): {
            "source_type": str(row.get("source_type", "")),
            "source_id": str(row.get("source_id", "")),
        }
        for row in manifest
    }
    moving_point_ids: set[int] = set()
    for image in images:
        source = source_by_name.get(str(image["image_name"]), {})
        if source.get("source_type") == "moving":
            moving_point_ids.update(int(value) for value in image["point3d_ids"])

    camera_rows: list[dict] = []
    for expected_name in static_expected:
        camera_id = expected_name.removeprefix("static_").removesuffix(
            ".png"
        )
        image = by_name.get(expected_name)
        if image is None:
            camera_rows.append(
                {
                    "camera_id": camera_id,
                    "image_name": expected_name,
                    "registered": False,
                    "colmap_camera_id": None,
                    "track_support": 0,
                    "shared_tracks_with_moving": 0,
                    "median_reprojection_error_px": None,
                    "reprojection_rmse_px": None,
                    "warnings": ["unregistered_static_camera"],
                }
            )
            continue
        point_ids = [
            int(value)
            for value in image["point3d_ids"]
            if int(value) in points
        ]
        errors = [float(points[value]["error_px"]) for value in point_ids]
        camera_rows.append(
            {
                "camera_id": camera_id,
                "image_name": expected_name,
                "registered": True,
                "colmap_camera_id": int(image["camera_id"]),
                "track_support": len(set(point_ids)),
                "shared_tracks_with_moving": len(
                    set(point_ids) & moving_point_ids
                ),
                "median_reprojection_error_px": (
                    float(statistics.median(errors)) if errors else None
                ),
                "reprojection_rmse_px": (
                    math.sqrt(
                        sum(error * error for error in errors) / len(errors)
                    )
                    if errors
                    else None
                ),
                "warnings": [],
            }
        )

    registered = [row for row in camera_rows if row["registered"]]
    track_median = (
        float(statistics.median(row["track_support"] for row in registered))
        if registered
        else 0.0
    )
    reprojection_values = [
        float(row["median_reprojection_error_px"])
        for row in registered
        if row["median_reprojection_error_px"] is not None
    ]
    reprojection_median = (
        float(statistics.median(reprojection_values))
        if reprojection_values
        else 0.0
    )
    track_threshold = max(20.0, 0.25 * track_median)
    reprojection_threshold = max(3.0, 2.5 * reprojection_median)
    warnings: list[dict[str, object]] = []
    for row in camera_rows:
        if not row["registered"]:
            warnings.append(
                {
                    "camera_id": row["camera_id"],
                    "code": "unregistered_static_camera",
                }
            )
            continue
        moving_threshold = max(10.0, 0.10 * float(row["track_support"]))
        if float(row["track_support"]) < track_threshold:
            row["warnings"].append("weak_track_support")
        if float(row["shared_tracks_with_moving"]) < moving_threshold:
            row["warnings"].append("weak_shared_moving_tracks")
        median_error = row["median_reprojection_error_px"]
        if (
            median_error is None
            or float(median_error) > reprojection_threshold
        ):
            row["warnings"].append("high_median_reprojection")
        for code in row["warnings"]:
            warnings.append({"camera_id": row["camera_id"], "code": code})
        row["track_support_threshold"] = track_threshold
        row["shared_moving_track_threshold"] = moving_threshold
        row["median_reprojection_threshold_px"] = reprojection_threshold

    assignments: dict[str, set[int]] = {}
    for image in images:
        source = source_by_name.get(str(image["image_name"]), {})
        source_id = str(source.get("source_id", ""))
        if source_id:
            assignments.setdefault(source_id, set()).add(
                int(image["camera_id"])
            )
    stable_groups = bool(assignments) and all(
        len(values) == 1 for values in assignments.values()
    )
    distinct_groups = (
        stable_groups
        and len({next(iter(values)) for values in assignments.values()})
        == len(assignments)
    )
    if not stable_groups or not distinct_groups:
        warnings.append(
            {
                "camera_id": None,
                "code": "unstable_physical_camera_group_assignment",
            }
        )
    return {
        "schema_version": 5,
        "algorithm": "ap03_colmap_support_diagnostics_v1",
        "ground_truth_used": False,
        "best_model": model_dir.name,
        "sparse_point_count": len(points),
        "registered_image_count": len(images),
        "registered_static_camera_count": len(registered),
        "registered_moving_frame_count": sum(
            source_by_name.get(str(image["image_name"]), {}).get(
                "source_type"
            )
            == "moving"
            for image in images
        ),
        "static_track_median": track_median,
        "static_median_reprojection_error_px": reprojection_median,
        "thresholds": {
            "minimum_track_support": track_threshold,
            "minimum_shared_moving_tracks": (
                "max(10, 10% of that camera track support)"
            ),
            "maximum_median_reprojection_error_px": (
                reprojection_threshold
            ),
        },
        "camera_groups": {
            "assignment": {
                key: sorted(values)
                for key, values in sorted(assignments.items())
            },
            "one_camera_id_per_physical_camera": stable_groups,
            "physical_camera_ids_are_distinct": distinct_groups,
            "intrinsics_refinement": False,
        },
        "static_cameras": camera_rows,
        "warnings": warnings,
        "quality_status": (
            "good" if not warnings else "warning_weak_reconstruction_support"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--txt-root", type=Path, default=TXT_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--cameras", required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    txt_root = args.txt_root.resolve()
    output_root = args.out.resolve()
    static_expected = tuple(
        f"static_{value.strip()}.png"
        for value in args.cameras.split(",")
        if value.strip()
    )
    ensure_dir(output_root)

    manifest = read_manifest(dataset_root)
    total_images = len(manifest)
    total_static = len([r for r in manifest if r.get("source_type") == "static"])
    total_moving = len([r for r in manifest if r.get("source_type") == "moving"])

    model_rows, image_rows = inspect_models(txt_root, static_expected)

    write_csv(
        output_root / "colmap_model_summary.csv",
        model_rows,
        [
            "model",
            "registered_images",
            "registered_static_cameras",
            "registered_moving_frames",
            "num_3d_points",
            "static_registered_list",
            "static_missing_list",
            "model_path",
        ],
    )

    write_csv(
        output_root / "registered_images_by_model.csv",
        image_rows,
        [
            "model",
            "image_name",
            "source_type",
            "camera_id",
            "qw",
            "qx",
            "qy",
            "qz",
            "tx",
            "ty",
            "tz",
        ],
    )

    if model_rows:
        best = sorted(
            model_rows,
            key=lambda r: (
                int(r["registered_static_cameras"]),
                int(r["registered_images"]),
                int(r["num_3d_points"]),
            ),
            reverse=True,
        )[0]
    else:
        best = None
    diagnostics = None
    if best is not None:
        diagnostics = reconstruction_diagnostics(
            txt_root / str(best["model"]),
            manifest,
            static_expected,
        )
        (output_root / "AP03_RECONSTRUCTION_DIAGNOSTICS.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(
            output_root / "AP03_RECONSTRUCTION_DIAGNOSTICS.csv",
            diagnostics["static_cameras"],
            [
                "camera_id",
                "image_name",
                "registered",
                "colmap_camera_id",
                "track_support",
                "track_support_threshold",
                "shared_tracks_with_moving",
                "shared_moving_track_threshold",
                "median_reprojection_error_px",
                "median_reprojection_threshold_px",
                "reprojection_rmse_px",
                "warnings",
            ],
        )

    lines = [
        "AP03 COLMAP Reconstruction Inspection",
        "=====================================",
        "",
        "AP03 Phase:",
        "- targetless COLMAP/SfM feasibility",
        "- no ArUco scale/registration yet",
        "- reconstruction scale is arbitrary",
        "",
        "Input dataset:",
        f"- total images: {total_images}",
        f"- static images: {total_static}",
        f"- moving images: {total_moving}",
        "",
        f"Number of COLMAP models: {len(model_rows)}",
        "",
    ]

    if not model_rows:
        lines += [
            "No sparse COLMAP model was created.",
            "",
            "Interpretation:",
            "- targetless feature matching / SfM failed on this dataset configuration",
            "- next options: reduce frame stride, improve matching, add more overlap, or use ArUco-assisted registration only after a valid model exists",
            "",
        ]
    else:
        lines.append("Models:")
        for r in model_rows:
            lines.append(
                f"- model {r['model']}: "
                f"{r['registered_images']} images, "
                f"{r['registered_static_cameras']}/"
                f"{len(static_expected)} static cameras, "
                f"{r['registered_moving_frames']} moving frames, "
                f"{r['num_3d_points']} 3D points"
            )
            lines.append(f"  static registered: {r['static_registered_list'] or 'none'}")
            lines.append(f"  static missing: {r['static_missing_list'] or 'none'}")

        lines += [
            "",
            "Best model by static-camera coverage:",
            f"- model: {best['model']}",
            f"- registered images: {best['registered_images']}",
            f"- registered static cameras: "
            f"{best['registered_static_cameras']}/{len(static_expected)}",
            f"- registered moving frames: {best['registered_moving_frames']}",
            f"- 3D points: {best['num_3d_points']}",
            "",
            "Interpretation:",
        ]

        if int(best["registered_static_cameras"]) == len(static_expected):
            lines += [
                "- COLMAP registered all static cameras.",
                "- AP03 can proceed to scale/registration with a known metric reference.",
            ]
        elif int(best["registered_static_cameras"]) > 0:
            lines += [
                "- COLMAP registered some but not all static cameras.",
                "- This is a partial targetless baseline.",
                "- We can still inspect whether the missing static cameras lack visual overlap/features.",
            ]
        else:
            lines += [
                "- COLMAP did not register any static bus cameras.",
                "- Moving trajectory may reconstruct, but rig extrinsics are not available yet.",
            ]

        lines += [
            "",
            "Important:",
            "- COLMAP poses are currently in arbitrary scale and arbitrary coordinate frame.",
            "- Metric scale and Ref-ArUco alignment are planned for AP03 Phase 2.",
            "",
        ]

    if diagnostics is not None:
        lines += [
            "GT-free static-camera support diagnostics:",
        ]
        for camera in diagnostics["static_cameras"]:
            lines.append(
                f"- {camera['camera_id']}: "
                f"tracks={camera['track_support']}, "
                f"shared-moving={camera['shared_tracks_with_moving']}, "
                f"reprojection RMSE="
                f"{camera['reprojection_rmse_px'] if camera['reprojection_rmse_px'] is not None else 'unavailable'} px, "
                f"warnings={','.join(camera['warnings']) or 'none'}"
            )
        lines += [
            f"- quality status: {diagnostics['quality_status']}",
            "- Ground truth was not read by this inspection.",
            "",
        ]

    lines += [
        "Output files:",
        f"- {output_root / 'colmap_model_summary.csv'}",
        f"- {output_root / 'registered_images_by_model.csv'}",
        f"- {output_root / 'AP03_RECONSTRUCTION_DIAGNOSTICS.json'}",
        f"- {output_root / 'AP03_RECONSTRUCTION_DIAGNOSTICS.csv'}",
        f"- {output_root / 'ap03_colmap_inspection_report.txt'}",
        "",
    ]

    report = "\n".join(lines) + "\n"
    (output_root / "ap03_colmap_inspection_report.txt").write_text(report)

    print(report)


if __name__ == "__main__":
    main()
