#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


W, H = 1280.0, 720.0
CX, CY = W / 2.0, H / 2.0
HALF_DIAG = math.sqrt(CX * CX + CY * CY)


def pnp_ok(row):
    val = str(row.get("pnp_success", "True")).strip().lower()
    return val not in ("false", "0", "no", "none", "nan")


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def marker_area_px(row):
    pts = np.array([
        [f(row, "corner0_u"), f(row, "corner0_v")],
        [f(row, "corner1_u"), f(row, "corner1_v")],
        [f(row, "corner2_u"), f(row, "corner2_v")],
        [f(row, "corner3_u"), f(row, "corner3_v")],
    ], dtype=np.float32)

    if not np.isfinite(pts).all():
        return 0.0

    return float(cv2.contourArea(pts))


def center_error_norm(row):
    du = f(row, "center_u") - CX
    dv = f(row, "center_v") - CY

    if not np.isfinite(du) or not np.isfinite(dv):
        return float("inf")

    return math.sqrt(du * du + dv * dv) / HALF_DIAG


def load_registered_frames(images_txt):
    frames = set()

    if not images_txt.exists():
        raise FileNotFoundError(images_txt)

    for line in images_txt.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = re.search(r"frame_(\d+)\.png", line)
        if m:
            frames.add(int(m.group(1)))

    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--moving-det",
        required=True,
        help="moving_detections.csv from the sequence",
    )
    ap.add_argument(
        "--colmap-images",
        required=True,
        help="COLMAP sparse_txt_best/images.txt",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV. Default: <sequence>/best_marker_frames/best_registered_moving_frame_by_marker.csv",
    )
    args = ap.parse_args()

    moving_det = Path(args.moving_det)
    colmap_images = Path(args.colmap_images)

    if args.out is None:
        out_csv = moving_det.parent / "best_marker_frames" / "best_registered_moving_frame_by_marker.csv"
    else:
        out_csv = Path(args.out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    registered_frames = load_registered_frames(colmap_images)

    if not registered_frames:
        raise RuntimeError(f"No registered frames found in {colmap_images}")

    best_by_marker = {}

    with moving_det.open() as fp:
        for row in csv.DictReader(fp):
            if not pnp_ok(row):
                continue

            frame = int(row["frame"])
            if frame not in registered_frames:
                continue

            marker_id = int(row["marker_id"])
            area = marker_area_px(row)
            center_norm = center_error_norm(row)
            dist = f(row, "distance_m")

            if area <= 0.0:
                continue
            if not np.isfinite(center_norm):
                continue
            if not np.isfinite(dist) or dist <= 0.0:
                dist = 1.0

            # Higher is better:
            # - large marker area
            # - near image center
            # - closer marker
            score = area / (1.0 + center_norm) / dist

            out = dict(row)
            out["score"] = f"{score:.12g}"
            out["area_px"] = f"{area:.12g}"
            out["center_norm"] = f"{center_norm:.12g}"
            out["registered_in_colmap"] = "True"

            old = best_by_marker.get(marker_id)
            if old is None or score > float(old["score"]):
                best_by_marker[marker_id] = out

    if not best_by_marker:
        raise RuntimeError("No marker detections overlap with registered COLMAP frames.")

    rows = [best_by_marker[k] for k in sorted(best_by_marker)]

    preferred_fields = [
        "marker_id",
        "frame",
        "score",
        "area_px",
        "center_norm",
        "registered_in_colmap",
        "image",
        "distance_m",
        "route_x",
        "route_y",
        "route_z",
        "route_roll",
        "route_pitch",
        "route_yaw",
    ]

    extra_fields = []
    for row in rows:
        for key in row.keys():
            if key not in preferred_fields and key not in extra_fields:
                extra_fields.append(key)

    fields = preferred_fields + extra_fields

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("[OK] wrote:", out_csv)
    print("[OK] registered frames:", len(registered_frames))
    print("[OK] selected markers:", sorted(best_by_marker.keys()))


if __name__ == "__main__":
    main()
