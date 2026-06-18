#!/usr/bin/env python3

import argparse
import math
import shutil
import subprocess
from pathlib import Path


WIDTH = 1280
HEIGHT = 720
HFOV_DEG = 69.1


def run(cmd):
    print()
    print("[CMD]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def fx_from_hfov(width, hfov_deg):
    hfov = math.radians(hfov_deg)
    return width / (2.0 * math.tan(hfov / 2.0))


def count_registered_images(images_txt):
    if not images_txt.exists():
        return 0

    count = 0
    lines = images_txt.read_text(errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 10:
            try:
                int(parts[0])
                float(parts[1])
                float(parts[2])
                float(parts[3])
                float(parts[4])
                float(parts[5])
                float(parts[6])
                float(parts[7])
                int(parts[8])
                count += 1
            except Exception:
                pass

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="results/bus_real_data/05_moving_camera_sequence_run3")
    ap.add_argument("--out", default="results/bus_real_data/06_colmap_moving_sequence_run3")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--use-gpu", type=int, default=0)
    args = ap.parse_args()

    seq_dir = Path(args.sequence)
    img_dir = seq_dir / "images"

    if not img_dir.exists():
        raise RuntimeError(f"Image directory not found: {img_dir}")

    out_dir = Path(args.out)
    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_txt_dir = out_dir / "sparse_txt"
    best_txt_dir = out_dir / "sparse_txt_best"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    sparse_txt_dir.mkdir(parents=True, exist_ok=True)

    fx = fx_from_hfov(WIDTH, HFOV_DEG)
    fy = fx
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0
    camera_params = f"{fx:.8f},{fy:.8f},{cx:.8f},{cy:.8f}"

    print("[INFO] sequence:", seq_dir)
    print("[INFO] images:", img_dir)
    print("[INFO] output:", out_dir)
    print("[INFO] camera model: PINHOLE")
    print("[INFO] camera params:", camera_params)

    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", camera_params,
        "--SiftExtraction.use_gpu", str(args.use_gpu),
        "--SiftExtraction.max_num_features", "8192",
    ])

    if args.matcher == "exhaustive":
        run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", str(args.use_gpu),
        ])
    else:
        run([
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", str(args.use_gpu),
            "--SequentialMatching.overlap", "20",
        ])

    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ])

    model_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError("COLMAP mapper produced no sparse model.")

    best_model = None
    best_count = -1

    for model_dir in model_dirs:
        model_name = model_dir.name
        txt_out = sparse_txt_dir / model_name
        txt_out.mkdir(parents=True, exist_ok=True)

        run([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(txt_out),
            "--output_type", "TXT",
        ])

        n = count_registered_images(txt_out / "images.txt")
        print(f"[INFO] model {model_name}: registered images = {n}")

        if n > best_count:
            best_count = n
            best_model = txt_out

    if best_model is None:
        raise RuntimeError("Could not select best sparse model.")

    if best_txt_dir.exists():
        shutil.rmtree(best_txt_dir)
    shutil.copytree(best_model, best_txt_dir)

    report = out_dir / "colmap_report.txt"
    report.write_text(
        "COLMAP moving sequence report\n"
        "=============================\n\n"
        f"Sequence: {seq_dir}\n"
        f"Images: {img_dir}\n"
        f"Matcher: {args.matcher}\n"
        f"Camera model: PINHOLE\n"
        f"Camera params: {camera_params}\n"
        f"Best model txt: {best_txt_dir}\n"
        f"Registered images in best model: {best_count}\n"
    )

    print()
    print("[OK] COLMAP done")
    print("[OK] best model:", best_txt_dir)
    print("[OK] registered images:", best_count)
    print("[OK] report:", report)


if __name__ == "__main__":
    main()
