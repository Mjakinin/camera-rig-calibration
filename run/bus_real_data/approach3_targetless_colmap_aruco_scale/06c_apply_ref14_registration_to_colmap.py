#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.io_utils import ensure_dir, write_csv
from ap03_scale_common import (
    load_best_colmap_model,
    load_sim3_metadata,
    apply_registration_to_colmap,
    pose_fields,
)

DEFAULT_OUT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ensure_dir(args.out_root)

    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()
    meta, scale, R_ref_col, t_ref_col = load_sim3_metadata(args.out_root / "ap03_ref14_sim3_metadata.json")

    static_rows, moving_rows, point_rows = apply_registration_to_colmap(
        images,
        points3d,
        scale,
        R_ref_col,
        t_ref_col,
    )

    write_csv(args.out_root / "ap03_static_camera_poses_ref_aruco.csv", static_rows, pose_fields())
    write_csv(args.out_root / "ap03_moving_frame_poses_ref_aruco.csv", moving_rows, pose_fields())
    write_csv(
        args.out_root / "ap03_sparse_points3d_ref_aruco.csv",
        point_rows,
        ["point3d_id", "x_ref_aruco_m", "y_ref_aruco_m", "z_ref_aruco_m", "r", "g", "b", "colmap_reprojection_error"],
    )

    print("AP03 06c complete")
    print(f"static_camera_count: {len(static_rows)}")
    print(f"moving_frame_count: {len(moving_rows)}")
    print(f"sparse_point_count: {len(point_rows)}")
    print(f"out: {args.out_root}")


if __name__ == "__main__":
    main()
