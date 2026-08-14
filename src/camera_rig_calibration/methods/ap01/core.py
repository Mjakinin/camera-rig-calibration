"""Compatibility facade for the AP01 scientific core."""
from __future__ import annotations

from .core_candidates import (
    R_to_quat,
    aggregate_candidates,
    aggregate_direct_marker_estimates,
    aggregate_relay_marker_chains,
    best_static_by_camera_marker,
    direct_candidates,
    moving_by_contract,
    moving_by_marker,
    quat_to_R,
    relay_candidates,
    rotation_difference_deg,
    weighted_rotation_mean,
)
from .core_colmap import (
    colmap_camera_model,
    colmap_feature_extractor_command,
    colmap_mapper_command,
    count_colmap_images,
    run_colmap,
    run_command,
)
from .core_geometry import (
    T_from_observation,
    invT,
    legacy_detection_quality,
    make_T,
    marker_area_from_corners,
    observation_quality,
    parse_colmap_poses,
    qvec_to_R,
)
from .core_io import (
    CAMERAS,
    ROOT_CAMERA,
    frame_number,
    is_success,
    load_camera_info,
    parse_args,
    read_csv,
    safe_float,
    status_path,
    write_csv,
    write_status,
)
from .core_legacy import (
    aggregate_legacy_direct_candidates,
    aggregate_legacy_relay_candidates,
    legacy_medoid_inliers,
    legacy_se3_medoid,
    weighted_transform_mean,
)
from .core_runner import (
    R_to_rpy_deg,
    pairwise_rows,
    pose_row,
    rpy_deg_to_R,
    serializable_candidate,
    serialize_final_pose,
)
from .core_scale import prepare_observations, robust_scale


def main() -> None:
    """Run AP01 while honoring hooks patched on this compatibility module."""
    from . import core_runner

    core_runner.main()


if __name__ == "__main__":
    main()
