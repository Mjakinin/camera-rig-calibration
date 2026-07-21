#!/usr/bin/env python3

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from ap02_common import (
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


OBS_CSV = AP02_ROOT / "02_aruco_observations" / "ap02_all_aruco_observations.csv"


def is_success(row):
    return str(row.get("pnp_success", "")).strip().lower() in ["true", "1", "yes"]


def obs_quality(row):
    try:
        area = float(row.get("area_px2", 0.0))
    except Exception:
        area = 0.0

    try:
        dist = float(row.get("distance_m", 99.0))
    except Exception:
        dist = 99.0

    if dist <= 0:
        dist = 99.0

    return area / (dist * dist + 1e-9)


def filter_mode(rows, mode):
    if mode == "static_only":
        return [r for r in rows if r.get("observer_type") == "static"]
    if mode == "with_moving":
        return rows
    raise RuntimeError(f"Unknown mode: {mode}")


def best_observations(rows):
    best = {}
    for r in rows:
        if not is_success(r):
            continue
        try:
            marker_id = int(float(r["marker_id"]))
        except Exception:
            continue

        key = (r["observer_id"], marker_id)
        q = obs_quality(r)

        if key not in best or q > obs_quality(best[key]):
            best[key] = r

    return list(best.values())


def initialize_graph(rows, ref_marker_id):
    marker_poses = {
        int(ref_marker_id): make_T(np.eye(3), np.zeros(3))
    }
    observer_poses = {}

    observations_by_marker = defaultdict(list)
    observations_by_observer = defaultdict(list)

    for r in rows:
        try:
            marker_id = int(float(r["marker_id"]))
        except Exception:
            continue

        observations_by_marker[marker_id].append(r)
        observations_by_observer[r["observer_id"]].append(r)

    queue = deque()
    queue.append(("marker", int(ref_marker_id)))

    init_log = []
    used_edges = []

    while queue:
        kind, entity_id = queue.popleft()

        if kind == "marker":
            marker_id = int(entity_id)
            T_ref_marker = marker_poses[marker_id]

            for obs in observations_by_marker.get(marker_id, []):
                observer_id = obs["observer_id"]

                if observer_id in observer_poses:
                    continue

                T_obs_marker = T_from_detection_row(obs)
                if T_obs_marker is None:
                    continue

                T_ref_observer = make_observer_known_from_marker(T_ref_marker, T_obs_marker)
                observer_poses[observer_id] = T_ref_observer
                queue.append(("observer", observer_id))

                init_log.append({
                    "initialized_type": obs["observer_type"],
                    "initialized_id": observer_id,
                    "from_type": "marker",
                    "from_id": marker_id,
                    "observed_marker_id": marker_id,
                    "quality": obs_quality(obs),
                })
                used_edges.append(obs)

        elif kind == "observer":
            observer_id = str(entity_id)
            T_ref_observer = observer_poses[observer_id]

            for obs in observations_by_observer.get(observer_id, []):
                marker_id = int(float(obs["marker_id"]))

                if marker_id in marker_poses:
                    continue

                T_obs_marker = T_from_detection_row(obs)
                if T_obs_marker is None:
                    continue

                T_ref_marker = make_marker_known_from_observer(T_ref_observer, T_obs_marker)
                marker_poses[marker_id] = T_ref_marker
                queue.append(("marker", marker_id))

                init_log.append({
                    "initialized_type": "marker",
                    "initialized_id": marker_id,
                    "from_type": obs["observer_type"],
                    "from_id": observer_id,
                    "observed_marker_id": marker_id,
                    "quality": obs_quality(obs),
                })
                used_edges.append(obs)

    return marker_poses, observer_poses, init_log, used_edges


def write_used_edges(path, rows):
    if not rows:
        fields = ["observer_type", "observer_id", "frame_id", "marker_id"]
        write_csv(path, [], fields)
        return

    fields = list(rows[0].keys())
    write_csv(path, rows, fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static_only", "with_moving"], required=True)
    ap.add_argument("--ref-marker-id", type=int, default=DEFAULT_REF_MARKER_ID)
    ap.add_argument("--out-root", default=str(AP02_ROOT / "05_graph_initialization"))
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_root) / args.mode)

    all_rows = read_csv(OBS_CSV)
    mode_rows = filter_mode(all_rows, args.mode)
    best_rows = best_observations(mode_rows)

    marker_poses, observer_poses, init_log, used_edges = initialize_graph(best_rows, args.ref_marker_id)

    static_pose_rows = []
    moving_pose_rows = []
    for observer_id, T in sorted(observer_poses.items()):
        if observer_id.startswith("cam_edge_"):
            static_pose_rows.append(pose_row("static_camera", observer_id, T, source=args.mode))
        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(pose_row("moving_frame", observer_id, T, source=args.mode))

    marker_pose_rows = []
    for marker_id, T in sorted(marker_poses.items()):
        marker_pose_rows.append(pose_row("marker", str(marker_id), T, source=args.mode))

    write_csv(out / "initial_static_camera_poses_ref_marker.csv", static_pose_rows, pose_fields())
    write_csv(out / "initial_moving_frame_poses_ref_marker.csv", moving_pose_rows, pose_fields())
    write_csv(out / "initial_marker_poses_ref_marker.csv", marker_pose_rows, pose_fields())

    init_log_fields = [
        "initialized_type",
        "initialized_id",
        "from_type",
        "from_id",
        "observed_marker_id",
        "quality",
    ]
    write_csv(out / "initialization_log.csv", init_log, init_log_fields)
    write_used_edges(out / "used_initialization_edges.csv", used_edges)

    all_static = sorted({r["observer_id"] for r in mode_rows if r.get("observer_type") == "static"})
    all_moving = sorted({r["observer_id"] for r in mode_rows if r.get("observer_type") == "moving"})
    all_markers = sorted({int(float(r["marker_id"])) for r in mode_rows})

    initialized_static = sorted([r["entity_id"] for r in static_pose_rows])
    initialized_moving = sorted([r["entity_id"] for r in moving_pose_rows])
    initialized_markers = sorted([int(r["entity_id"]) for r in marker_pose_rows])

    missing_static = sorted(set(all_static) - set(initialized_static))
    missing_moving = sorted(set(all_moving) - set(initialized_moving))
    missing_markers = sorted(set(all_markers) - set(initialized_markers))

    report = [
        "AP02 reference-marker pose graph initialization",
        "================================================",
        "",
        f"Mode: {args.mode}",
        f"Reference marker id: {args.ref_marker_id}",
        "",
        f"Raw observations in mode: {len(mode_rows)}",
        f"Best observer-marker observations: {len(best_rows)}",
        f"Used initialization edges: {len(used_edges)}",
        "",
        f"Initialized static cameras: {len(initialized_static)} / {len(all_static)}",
        f"Initialized moving frames: {len(initialized_moving)} / {len(all_moving)}",
        f"Initialized markers: {len(initialized_markers)} / {len(all_markers)}",
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
        "- static_only shows what can be connected using only static camera images.",
        "- with_moving shows whether moving-camera observations connect additional markers/cameras to the reference marker.",
        "- No AP01 result files are used in this pipeline.",
        "",
    ]

    (out / "graph_connectivity_report.txt").write_text("\n".join(report) + "\n")

    print("[OK] wrote", out)
    print("[OK] mode:", args.mode)
    print("[OK] initialized static cameras:", initialized_static)
    print("[OK] initialized markers:", len(initialized_markers), "/", len(all_markers))
    print("[OK] initialized moving frames:", len(initialized_moving), "/", len(all_moving))


if __name__ == "__main__":
    main()
