#!/usr/bin/env python3

import csv
from pathlib import Path


AP3_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
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


def read_manifest():
    path = DATASET_ROOT / "image_manifest.csv"
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
            })

            # Skip points2D line.
            i += 2
        else:
            i += 1

    return images


def parse_points3D_txt(path: Path):
    count = 0
    if not path.exists():
        return 0

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        count += 1
    return count


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


def inspect_models():
    if not TXT_ROOT.exists():
        raise RuntimeError(f"Missing COLMAP txt model root: {TXT_ROOT}")

    rows = []
    image_rows = []

    for model_dir in sorted(TXT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        images = parse_images_txt(model_dir / "images.txt")
        cameras = parse_cameras_txt(model_dir / "cameras.txt")
        num_points = parse_points3D_txt(model_dir / "points3D.txt")

        image_names = {r["image_name"] for r in images}
        static_registered = [name for name in STATIC_EXPECTED if name in image_names]
        moving_registered = [name for name in image_names if name.startswith("moving_")]

        rows.append({
            "model": model_dir.name,
            "registered_images": len(images),
            "registered_static_cameras": len(static_registered),
            "registered_moving_frames": len(moving_registered),
            "num_3d_points": num_points,
            "static_registered_list": ";".join(static_registered),
            "static_missing_list": ";".join([x for x in STATIC_EXPECTED if x not in image_names]),
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


def main():
    ensure_dir(OUT_ROOT)

    manifest = read_manifest()
    total_images = len(manifest)
    total_static = len([r for r in manifest if r.get("source_type") == "static"])
    total_moving = len([r for r in manifest if r.get("source_type") == "moving"])

    model_rows, image_rows = inspect_models()

    write_csv(
        OUT_ROOT / "colmap_model_summary.csv",
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
        OUT_ROOT / "registered_images_by_model.csv",
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
                f"{r['registered_static_cameras']}/4 static cameras, "
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
            f"- registered static cameras: {best['registered_static_cameras']}/4",
            f"- registered moving frames: {best['registered_moving_frames']}",
            f"- 3D points: {best['num_3d_points']}",
            "",
            "Interpretation:",
        ]

        if int(best["registered_static_cameras"]) == 4:
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

    lines += [
        "Output files:",
        f"- {OUT_ROOT / 'colmap_model_summary.csv'}",
        f"- {OUT_ROOT / 'registered_images_by_model.csv'}",
        f"- {OUT_ROOT / 'ap03_colmap_inspection_report.txt'}",
        "",
    ]

    report = "\n".join(lines) + "\n"
    (OUT_ROOT / "ap03_colmap_inspection_report.txt").write_text(report)

    print(report)


if __name__ == "__main__":
    main()
