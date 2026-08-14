"""Deterministic AP02 pose-graph initialization."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .common import (
    AP02_ROOT,
    DEFAULT_REF_MARKER_ID,
    ensure_dir,
    read_csv,
    write_csv,
    make_T,
    T_from_detection_row,
    make_observer_known_from_marker,
    make_marker_known_from_observer,
    pose_row,
    pose_fields,
)

from camera_rig_calibration.observation_quality import (
    observation_succeeded as is_success,
    pnp_reprojection_rmse,
)


OBS_CSV = (
    AP02_ROOT
    / "02_aruco_observations"
    / "ap02_all_aruco_observations.csv"
)


Node = tuple[str, str | int]



from .initialize_graph import (
    best_observations,
    build_graph,
    deterministic_breadth_first_tree,
    edge_quality,
    filter_mode,
    main_compat_widest_path_tree,
    main_observation_score,
    marker_node,
    maximum_bottleneck_tree,
    observation_pnp_rmse,
)
from .initialize_poses import (
    _camera_path_diagnostics,
    _rotation_difference_deg,
    _tree_path_metrics,
    initialize_from_tree,
)
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "static_only",
            "with_moving",
        ],
        required=True,
    )

    parser.add_argument(
        "--ref-marker-id",
        type=int,
        default=DEFAULT_REF_MARKER_ID,
    )

    parser.add_argument(
        "--out-root",
        default=str(
            AP02_ROOT
            / "05_graph_initialization"
        ),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=OBS_CSV,
    )
    parser.add_argument(
        "--initialization-algorithm",
        choices=(
            "legacy_maximum_bottleneck_v1",
            "wizard_maximum_bottleneck_v2",
            "unweighted_bfs_diagnostic",
        ),
        default="legacy_maximum_bottleneck_v1",
    )
    parser.add_argument(
        "--edge-weight-policy",
        choices=(
            "legacy_observation_quality_v1",
            "wizard_selection_score_v2",
        ),
        default="legacy_observation_quality_v1",
    )

    args = parser.parse_args()

    out = ensure_dir(
        Path(args.out_root) / args.mode
    )

    all_rows = read_csv(args.observations)
    mode_rows = filter_mode(all_rows, args.mode)
    selected_rows = best_observations(
        mode_rows, edge_weight_policy=args.edge_weight_policy
    )

    adjacency = build_graph(
        selected_rows,
        preserve_input_order=(
            args.initialization_algorithm
            == "legacy_maximum_bottleneck_v1"
        ),
    )

    start = marker_node(args.ref_marker_id)

    bfs_parent = deterministic_breadth_first_tree(adjacency, start)
    legacy_parent, legacy_metrics = main_compat_widest_path_tree(
        adjacency, start, edge_weight_policy=args.edge_weight_policy
    )
    v2_parent, v2_metrics = maximum_bottleneck_tree(
        adjacency, start, edge_weight_policy=args.edge_weight_policy
    )
    if args.initialization_algorithm == "legacy_maximum_bottleneck_v1":
        parent, path_metrics = legacy_parent, legacy_metrics
        productive_algorithm = "legacy_maximum_bottleneck_v1"
    elif args.initialization_algorithm == "wizard_maximum_bottleneck_v2":
        parent, path_metrics = v2_parent, v2_metrics
        productive_algorithm = "wizard_maximum_bottleneck_v2"
    else:
        parent = bfs_parent
        path_metrics = _tree_path_metrics(bfs_parent, start)
        productive_algorithm = "unweighted_bfs_diagnostic"

    (
        marker_poses,
        observer_poses,
        init_log,
        used_edges,
    ) = initialize_from_tree(
        parent,
        args.ref_marker_id,
        path_metrics=path_metrics,
        algorithm=productive_algorithm,
        edge_weight_policy=args.edge_weight_policy,
    )
    (
        _v2_marker_poses,
        v2_observer_poses,
        v2_init_log,
        _v2_used_edges,
    ) = initialize_from_tree(
        v2_parent,
        args.ref_marker_id,
        path_metrics=v2_metrics,
        algorithm="maximum_bottleneck_v2",
        edge_weight_policy=args.edge_weight_policy,
    )
    (
        _bfs_marker_poses,
        bfs_observer_poses,
        bfs_init_log,
        _bfs_used_edges,
    ) = initialize_from_tree(
        bfs_parent,
        args.ref_marker_id,
        path_metrics=_tree_path_metrics(bfs_parent, start),
        algorithm="unweighted_first_hit_bfs_diagnostic",
        edge_weight_policy=args.edge_weight_policy,
    )

    static_pose_rows = []
    moving_pose_rows = []
    static_observer_ids = {
        row["observer_id"]
        for row in selected_rows
        if row.get("observer_type") == "static"
    }

    for observer_id, transform in sorted(
        observer_poses.items()
    ):
        if observer_id in static_observer_ids:
            static_pose_rows.append(
                pose_row(
                    "static_camera",
                    observer_id,
                    transform,
                    source=f"{args.mode}_{productive_algorithm}",
                )
            )

        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(
                pose_row(
                    "moving_frame",
                    observer_id,
                    transform,
                    source=f"{args.mode}_{productive_algorithm}",
                )
            )

    marker_pose_rows = [
        pose_row(
            "marker",
            str(marker_id),
            transform,
            source=f"{args.mode}_{productive_algorithm}",
        )
        for marker_id, transform
        in sorted(marker_poses.items())
    ]

    write_csv(
        out / "initial_static_camera_poses_ref_marker.csv",
        static_pose_rows,
        pose_fields(),
    )

    write_csv(
        out / "initial_moving_frame_poses_ref_marker.csv",
        moving_pose_rows,
        pose_fields(),
    )

    write_csv(
        out / "initial_marker_poses_ref_marker.csv",
        marker_pose_rows,
        pose_fields(),
    )

    init_fields = [
        "initialization_algorithm",
        "initialized_type",
        "initialized_id",
        "from_type",
        "from_id",
        "observed_marker_id",
        "observer_id",
        "frame_id",
        "edge_quality",
        "path_length",
        "path_bottleneck_score",
        "pnp_reprojection_rmse_px",
        "area_px2",
        "distance_m",
    ]

    write_csv(
        out / "initialization_log.csv",
        init_log,
        init_fields,
    )

    if used_edges:
        write_csv(
            out / "used_initialization_edges.csv",
            used_edges,
            list(used_edges[0].keys()),
        )
    else:
        write_csv(
            out / "used_initialization_edges.csv",
            [],
            [
                "observer_type",
                "observer_id",
                "frame_id",
                "marker_id",
            ],
        )

    productive_static_poses = {
        camera_id: transform
        for camera_id, transform in observer_poses.items()
        if camera_id in static_observer_ids
    }
    bfs_static_poses = {
        camera_id: transform
        for camera_id, transform in bfs_observer_poses.items()
        if camera_id in static_observer_ids
    }
    v2_static_poses = {
        camera_id: transform
        for camera_id, transform in v2_observer_poses.items()
        if camera_id in static_observer_ids
    }
    productive_paths = _camera_path_diagnostics(
        parent,
        path_metrics,
        productive_static_poses,
        algorithm=productive_algorithm,
    )
    v2_paths = _camera_path_diagnostics(
        v2_parent,
        v2_metrics,
        v2_static_poses,
        algorithm="maximum_bottleneck_v2",
    )
    bfs_metrics = _tree_path_metrics(bfs_parent, start)
    bfs_paths = _camera_path_diagnostics(
        bfs_parent,
        bfs_metrics,
        bfs_static_poses,
        algorithm="unweighted_first_hit_bfs_diagnostic",
    )
    productive_by_camera = {
        str(row["camera_id"]): row for row in productive_paths
    }
    v2_by_camera = {str(row["camera_id"]): row for row in v2_paths}
    bfs_by_camera = {str(row["camera_id"]): row for row in bfs_paths}
    comparison_rows: list[dict[str, object]] = []
    for camera_id in sorted(
        set(productive_static_poses)
        | set(v2_static_poses)
        | set(bfs_static_poses)
    ):
        productive_pose = productive_static_poses.get(camera_id)
        bfs_pose = bfs_static_poses.get(camera_id)
        v2_pose = v2_static_poses.get(camera_id)
        productive_path = productive_by_camera.get(camera_id, {})
        v2_path = v2_by_camera.get(camera_id, {})
        bfs_path = bfs_by_camera.get(camera_id, {})
        comparison_rows.append(
            {
                "camera_id": camera_id,
                "productive_algorithm": productive_algorithm,
                "productive_path": " -> ".join(
                    str(edge["to"])
                    for edge in productive_path.get(
                        "selected_graph_path", []
                    )
                ),
                "productive_bottleneck_score": productive_path.get(
                    "path_bottleneck_score"
                ),
                "v2_algorithm": "maximum_bottleneck_v2",
                "v2_path": " -> ".join(
                    str(edge["to"])
                    for edge in v2_path.get("selected_graph_path", [])
                ),
                "v2_bottleneck_score": v2_path.get(
                    "path_bottleneck_score"
                ),
                "diagnostic_algorithm": (
                    "unweighted_first_hit_bfs_diagnostic"
                ),
                "diagnostic_path": " -> ".join(
                    str(edge["to"])
                    for edge in bfs_path.get("selected_graph_path", [])
                ),
                "diagnostic_bottleneck_score": bfs_path.get(
                    "path_bottleneck_score"
                ),
                "translation_difference_m": (
                    float(
                        np.linalg.norm(
                            productive_pose[:3, 3] - bfs_pose[:3, 3]
                        )
                    )
                    if productive_pose is not None and bfs_pose is not None
                    else ""
                ),
                "rotation_difference_deg": (
                    _rotation_difference_deg(productive_pose, bfs_pose)
                    if productive_pose is not None and bfs_pose is not None
                    else ""
                ),
                "productive_v2_translation_difference_m": (
                    float(
                        np.linalg.norm(
                            productive_pose[:3, 3] - v2_pose[:3, 3]
                        )
                    )
                    if productive_pose is not None and v2_pose is not None
                    else ""
                ),
                "productive_v2_rotation_difference_deg": (
                    _rotation_difference_deg(productive_pose, v2_pose)
                    if productive_pose is not None and v2_pose is not None
                    else ""
                ),
            }
        )
    write_csv(
        out / "initialization_algorithm_comparison.csv",
        comparison_rows,
        list(comparison_rows[0])
        if comparison_rows
        else [
            "camera_id",
            "productive_algorithm",
            "diagnostic_algorithm",
        ],
    )
    diagnostics_payload = {
        "schema_version": 5,
        "mode": args.mode,
        "reference_marker_id": args.ref_marker_id,
        "productive_algorithm": productive_algorithm,
        "diagnostic_algorithms": [
            "maximum_bottleneck_v2",
            "unweighted_first_hit_bfs_diagnostic",
        ],
        "path_tie_breakers": [
            "maximum bottleneck selection score",
            "minimum path length",
            "maximum mean selection score",
            "minimum worst PnP reprojection RMSE",
            "maximum minimum marker area ratio",
            "lexicographic node/frame/image path",
        ],
        "productive_camera_paths": productive_paths,
        "maximum_bottleneck_v2_camera_paths": v2_paths,
        "diagnostic_camera_paths": bfs_paths,
        "comparison": comparison_rows,
    }
    (out / "initialization_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    bfs_root = ensure_dir(
        out / "diagnostics" / "unweighted_first_hit_bfs"
    )
    write_csv(
        bfs_root / "initialization_log.csv",
        bfs_init_log,
        init_fields,
    )
    (bfs_root / "camera_paths.json").write_text(
        json.dumps(bfs_paths, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    v2_root = ensure_dir(out / "diagnostics" / "maximum_bottleneck_v2")
    write_csv(v2_root / "initialization_log.csv", v2_init_log, init_fields)
    (v2_root / "camera_paths.json").write_text(
        json.dumps(v2_paths, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    parity_rows: list[dict[str, object]] = []
    for key, rows in itertools.groupby(
        sorted(
            mode_rows,
            key=lambda row: (
                str(row.get("observer_id", "")),
                int(float(row.get("marker_id", -1))),
            ),
        ),
        key=lambda row: (
            str(row.get("observer_id", "")),
            int(float(row.get("marker_id", -1))),
        ),
    ):
        group = list(rows)
        current_order = sorted(
            group,
            key=lambda row: (
                -edge_quality(row),
                observation_pnp_rmse(row),
                str(row.get("frame_id", "")),
                str(row.get("image_path", "")),
            ),
        )
        main_order = sorted(
            group,
            key=lambda row: (
                -main_observation_score(row),
                observation_pnp_rmse(row),
                str(row.get("frame_id", "")),
                str(row.get("image_path", "")),
            ),
        )
        parity_rows.append(
            {
                "observer_id": key[0],
                "marker_id": key[1],
                "same_top_observation": bool(
                    current_order
                    and main_order
                    and current_order[0] is main_order[0]
                ),
                "selection_score_top_frame": (
                    current_order[0].get("frame_id") if current_order else None
                ),
                "main_score_top_frame": (
                    main_order[0].get("frame_id") if main_order else None
                ),
                "selection_score": (
                    edge_quality(
                        current_order[0], "wizard_selection_score_v2"
                    )
                    if current_order
                    else None
                ),
                "main_observation_score": (
                    main_observation_score(main_order[0])
                    if main_order
                    else None
                ),
            }
        )
    parity = {
        "schema_version": 5,
        "algorithm_version": productive_algorithm,
        "reference_marker_id": args.ref_marker_id,
        "mode": args.mode,
        "productive_uses_ground_truth": False,
        "observation_score_parity": {
            "pair_count": len(parity_rows),
            "same_top_count": sum(
                bool(row["same_top_observation"]) for row in parity_rows
            ),
            "pairs": parity_rows,
        },
        "camera_paths": {
            camera_id: {
                "main_compatible": productive_by_camera.get(camera_id),
                "maximum_bottleneck_v2": v2_by_camera.get(camera_id),
                "unweighted_first_hit_bfs_diagnostic": bfs_by_camera.get(
                    camera_id
                ),
                "explicit_cam_edge_5_evidence": camera_id == "cam_edge_5",
            }
            for camera_id in sorted(
                set(productive_by_camera) | set(v2_by_camera) | set(bfs_by_camera)
            )
        },
    }
    (out / "AP02_MAIN_COMPAT_INITIALIZATION_PARITY.json").write_text(
        json.dumps(parity, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    successful_rows = [
        row
        for row in mode_rows
        if is_success(row)
    ]

    all_static = sorted({
        row["observer_id"]
        for row in successful_rows
        if row.get("observer_type") == "static"
    })

    all_moving = sorted({
        row["observer_id"]
        for row in successful_rows
        if row.get("observer_type") == "moving"
    })

    all_markers = sorted({
        int(float(row["marker_id"]))
        for row in successful_rows
    })

    initialized_static = sorted(
        row["entity_id"]
        for row in static_pose_rows
    )

    initialized_moving = sorted(
        row["entity_id"]
        for row in moving_pose_rows
    )

    initialized_markers = sorted(
        int(row["entity_id"])
        for row in marker_pose_rows
    )

    missing_static = sorted(
        set(all_static) - set(initialized_static)
    )

    missing_moving = sorted(
        set(all_moving) - set(initialized_moving)
    )

    missing_markers = sorted(
        set(all_markers) - set(initialized_markers)
    )

    report = [
        "AP02 main-compatible widest-path pose graph initialization",
        "==========================================================",
        "",
        f"Mode: {args.mode}",
        f"Reference marker id: {args.ref_marker_id}",
        f"Productive algorithm: {productive_algorithm}",
        "Diagnostics: maximum_bottleneck_v2, "
        "unweighted_first_hit_bfs_diagnostic",
        "",
        f"Raw observations in mode: {len(mode_rows)}",
        f"Observer-marker initialization edges: {len(selected_rows)}",
        f"Used initialization edges: {len(used_edges)}",
        "",
        (
            "Initialized static cameras: "
            f"{len(initialized_static)} / {len(all_static)}"
        ),
        (
            "Initialized moving frames: "
            f"{len(initialized_moving)} / {len(all_moving)}"
        ),
        (
            "Initialized markers: "
            f"{len(initialized_markers)} / {len(all_markers)}"
        ),
        "",
        f"Initialized static cameras: {initialized_static}",
        f"Missing static cameras: {missing_static}",
        "",
        f"Initialized markers: {initialized_markers}",
        f"Missing markers: {missing_markers}",
        "",
        f"Missing moving frames count: {len(missing_moving)}",
        "",
        "Interpretation:",
        (
            "- Productive BA poses use the validated main-compatible maximum "
            "frontier tree from the configured reference marker."
        ),
        (
            "- First-hit BFS is retained only in diagnostics and never "
            "initializes bundle adjustment."
        ),
        "- Ground truth is not read by either initialization.",
        "",
    ]

    (
        out / "graph_connectivity_report.txt"
    ).write_text(
        "\n".join(report) + "\n"
    )

    print(f"[OK] wrote {out}")
    print(
        "[OK] initialized static cameras:",
        initialized_static,
    )
    print(
        "[OK] initialized markers:",
        len(initialized_markers),
        "/",
        len(all_markers),
    )


if __name__ == "__main__":
    main()
