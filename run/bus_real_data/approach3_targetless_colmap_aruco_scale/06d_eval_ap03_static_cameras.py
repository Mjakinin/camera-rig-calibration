#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.io_utils import ensure_dir, read_csv, write_csv
from ap03_scale_common import (
    eval_static_cameras_vs_gt,
    eval_fields,
    AP3_CMP,
)

DEFAULT_OUT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ensure_dir(args.out_root)
    ensure_dir(AP3_CMP)

    static_rows = read_csv(args.out_root / "ap03_static_camera_poses_ref_aruco.csv")
    eval_rows = eval_static_cameras_vs_gt(static_rows)

    write_csv(args.out_root / "ap03_static_cameras_ref_aruco_vs_gt.csv", eval_rows, eval_fields())
    write_csv(AP3_CMP / "ap03_static_cameras_ref_aruco_vs_gt_repo_like.csv", eval_rows, eval_fields())

    t = [float(r["translation_error_cm"]) for r in eval_rows]
    r = [float(r["rotation_error_deg"]) for r in eval_rows]

    print("AP03 06d complete")
    print(f"camera_count: {len(eval_rows)}")
    print(f"mean_translation_error_cm: {sum(t) / len(t):.12f}")
    print(f"mean_rotation_error_deg: {sum(r) / len(r):.12f}")
    print(f"out: {args.out_root}")


if __name__ == "__main__":
    main()
