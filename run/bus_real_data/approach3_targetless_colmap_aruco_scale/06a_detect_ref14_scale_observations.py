#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.io_utils import ensure_dir, write_csv
from ap03_scale_common import (
    load_best_colmap_model,
    detect_ref14_observations,
    REF_MARKER_ID,
    MARKER_LENGTH_M,
)

DEFAULT_OUT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-area-px2", type=float, default=1000.0)
    args = ap.parse_args()

    ensure_dir(args.out_root)

    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()
    obs = detect_ref14_observations(images, args.min_area_px2)

    if len(obs) < 8:
        raise RuntimeError(
            f"Too few Ref{REF_MARKER_ID} corner observations: {len(obs)}. "
            f"Try lower --min-area-px2."
        )

    write_csv(
        args.out_root / "ap03_ref14_corner_observations.csv",
        obs,
        ["image_name", "corner_idx", "u", "v", "area_px2"],
    )

    meta = {
        "stage": "06a_detect_ref14_scale_observations",
        "best_colmap_model": best_model,
        "model_dir": str(model_dir),
        "registered_images": len(images),
        "registered_cameras": len(cameras),
        "num_sparse_points3d": len(points3d),
        "reference_marker_id": REF_MARKER_ID,
        "marker_length_m": MARKER_LENGTH_M,
        "min_area_px2": args.min_area_px2,
        "ref14_corner_observation_count": len(obs),
        "ref14_unique_anchor_images": len(set(o["image_name"] for o in obs)),
    }
    (args.out_root / "06a_detection_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("AP03 06a complete")
    print(f"best_colmap_model: {best_model}")
    print(f"registered_images: {len(images)}")
    print(f"ref14_corner_observations: {len(obs)}")
    print(f"ref14_anchor_images: {len(set(o['image_name'] for o in obs))}")
    print(f"out: {args.out_root}")


if __name__ == "__main__":
    main()
