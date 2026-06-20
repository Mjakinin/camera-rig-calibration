#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from pathlib import Path


SHARED_RAW = Path("results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1")
OUT_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
DATASET_ROOT = OUT_ROOT / "01_colmap_dataset"
IMAGE_DIR = DATASET_ROOT / "images"


STATIC_CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.link(src, dst)
    except Exception:
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


def write_csv(path: Path, rows, fields):
    ensure_dir(path.parent)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moving-stride", type=int, default=1)
    ap.add_argument("--max-moving", type=int, default=0)
    ap.add_argument("--copy-mode", choices=["link", "copy"], default="link")
    args = ap.parse_args()

    if not SHARED_RAW.exists():
        raise RuntimeError(f"Missing shared raw dataset: {SHARED_RAW}")

    static_dir = SHARED_RAW / "static"
    moving_dir = SHARED_RAW / "moving"

    if not static_dir.exists():
        raise RuntimeError(f"Missing static raw folder: {static_dir}")
    if not moving_dir.exists():
        raise RuntimeError(f"Missing moving raw folder: {moving_dir}")

    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    ensure_dir(IMAGE_DIR)

    rows = []

    # Static images first.
    for cam in STATIC_CAMS:
        src = static_dir / f"{cam}.png"
        if not src.exists():
            raise RuntimeError(f"Missing static image: {src}")

        dst_name = f"static_{cam}.png"
        dst = IMAGE_DIR / dst_name

        if args.copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            link_or_copy(src, dst)

        rows.append({
            "image_name": dst_name,
            "source_type": "static",
            "source_id": cam,
            "source_path": str(src),
        })

    moving_files = sorted(moving_dir.glob("*.png"))
    if args.moving_stride > 1:
        moving_files = moving_files[::args.moving_stride]
    if args.max_moving and args.max_moving > 0:
        moving_files = moving_files[:args.max_moving]

    for src in moving_files:
        frame_id = src.stem
        dst_name = f"moving_{frame_id}.png"
        dst = IMAGE_DIR / dst_name

        if args.copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            link_or_copy(src, dst)

        rows.append({
            "image_name": dst_name,
            "source_type": "moving",
            "source_id": frame_id,
            "source_path": str(src),
        })

    write_csv(
        DATASET_ROOT / "image_manifest.csv",
        rows,
        ["image_name", "source_type", "source_id", "source_path"],
    )

    (DATASET_ROOT / "README_AP03_DATASET.txt").write_text(
        "\n".join([
            "AP03 COLMAP Dataset",
            "====================",
            "",
            "This dataset is prepared from the shared raw image dataset.",
            "",
            f"Shared raw source: {SHARED_RAW}",
            "",
            "Image naming:",
            "- static_cam_edge_X.png: static bus camera snapshots",
            "- moving_frame_XXXX.png: moving camera sequence frames",
            "",
            "AP03 Phase 1 uses this as a targetless COLMAP/SfM input.",
            "No ArUco detections are used in this phase.",
            "COLMAP reconstruction scale is arbitrary until later scale/registration.",
            "",
            f"Static images: {len([r for r in rows if r['source_type'] == 'static'])}",
            f"Moving images: {len([r for r in rows if r['source_type'] == 'moving'])}",
            f"Total images: {len(rows)}",
            "",
        ]) + "\n"
    )

    print("[OK] prepared AP03 COLMAP dataset")
    print("[OK] image dir:", IMAGE_DIR)
    print("[OK] manifest:", DATASET_ROOT / "image_manifest.csv")
    print("[OK] static images:", len([r for r in rows if r["source_type"] == "static"]))
    print("[OK] moving images:", len([r for r in rows if r["source_type"] == "moving"]))
    print("[OK] total images:", len(rows))


if __name__ == "__main__":
    main()
