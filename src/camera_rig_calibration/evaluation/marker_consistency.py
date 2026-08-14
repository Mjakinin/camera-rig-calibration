"""Stable facade and command for real-data marker-consistency evaluation."""

from __future__ import annotations

from .marker_consistency_core import (
    CAMERAS,
    METHOD_DIRS,
    T,
    Rq,
    angle,
    area,
    best_static,
    camera,
    colmap_images,
    errors,
    evaluate,
    frame_id,
    inv,
    load_ap01,
    load_ap02,
    load_ap03,
    load_poses,
    marker_lengths,
    mean_transform,
    med,
    moving_rows,
    num,
    obs,
    pnp_pose,
    pose_rows,
    project,
    qR,
    quality,
    ray,
    read_csv,
    rmse,
    robust_triangulate,
    rotation_error,
    scaled_colmap_poses,
    selected_marker_rows,
    static_poses,
    status,
    success,
    triangulate,
    und,
    vals,
    write_csv,
)
from .marker_consistency_reporting import (
    format_value as fmt,
    report,
    text_table,
)
from .marker_consistency_runner import main, run_evaluation

__all__ = [
    "CAMERAS",
    "METHOD_DIRS",
    "evaluate",
    "load_poses",
    "main",
    "report",
    "run_evaluation",
]


if __name__ == "__main__":
    main()
