#!/usr/bin/env bash
set -eo pipefail

# Add AP02 Phase 2 graph-initialization scripts.
#
# This patch assumes AP02 Phase 1 already created:
#   results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv
#
# It does not read AP01 internals.
# It does not modify AP01.
#
# Creates/overwrites:
#   run/bus_real_data/approach2_ref_marker_graph_ba/04_single_ref_marker_pnp_baseline.py
#   run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph.py
#   run/bus_real_data/approach2_ref_marker_graph_ba/06_compare_graph_initialization.py
#   run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh

PROJECT_ROOT="/workspaces/project"
if [ -d "$PROJECT_ROOT" ]; then
  cd "$PROJECT_ROOT"
fi

mkdir -p run/bus_real_data/approach2_ref_marker_graph_ba

cat > run/bus_real_data/approach2_ref_marker_graph_ba/04_single_ref_marker_pnp_baseline.py <<'PY'
#!/usr/bin/env python3

import argparse
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
    pose_row,
    pose_fields,
)


OBS_CSV = AP02_ROOT / "02_aruco_observations" / "ap02_static_aruco_observations.csv"


def is_success(row):
    return str(row.get("pnp_success", "")).strip().lower() in ["true", "1", "yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-marker-id", type=int, default=DEFAULT_REF_MARKER_ID)
    ap.add_argument("--out", default=str(AP02_ROOT / "04_single_ref_marker_pnp"))
    args = ap.parse_args()

    out = ensure_dir(Path(args.out))

    rows = read_csv(OBS_CSV)
    T_ref_marker = make_T(np.eye(3), np.zeros(3))

    pose_rows = []
    seen_ref = []
    all_static_cams = sorted({r["observer_id"] for r in rows if r.get("observer_type") == "static"})

    for r in rows:
        if not is_success(r):
            continue
        if int(float(r["marker_id"])) != args.ref_marker_id:
            continue

        T_cam_marker = T_from_detection_row(r)
        if T_cam_marker is None:
            continue

        T_ref_cam = make_observer_known_from_marker(T_ref_marker, T_cam_marker)

        pose_rows.append(
            pose_row(
                entity_type="static_camera",
                entity_id=r["observer_id"],
                T=T_ref_cam,
                source=f"single_ref_marker_{args.ref_marker_id}_pnp",
            )
        )
        seen_ref.append(r["observer_id"])

    seen_ref = sorted(set(seen_ref))
    not_seen = sorted(set(all_static_cams) - set(seen_ref))

    write_csv(out / "single_ref_marker_static_camera_poses_ref_marker.csv", pose_rows, pose_fields())

    report = [
        "AP02 single reference-marker PnP baseline",
        "=========================================",
        "",
        f"Reference marker id: {args.ref_marker_id}",
        "",
        "Goal:",
        "Estimate T_ref_marker_cam directly for every static camera that sees the reference marker.",
        "",
        f"Static cameras in AP02 observations: {all_static_cams}",
        f"Static cameras seeing reference marker: {seen_ref}",
        f"Static cameras NOT seeing reference marker: {not_seen}",
        "",
        f"Estimated static camera poses: {len(pose_rows)}",
        "",
        "Interpretation:",
        "- If a static camera is not listed, it cannot be calibrated by the naive single-reference-marker method.",
        "- Missing cameras motivate the marker-map graph method.",
        "- This AP02 script reads only AP02 observations, not AP01 result internals.",
        "",
    ]

    (out / "single_ref_marker_report.txt").write_text("\n".join(report) + "\n")

    print("[OK] wrote", out)
    print("[OK] reference marker:", args.ref_marker_id)
    print("[OK] direct static cameras:", seen_ref)
    print("[OK] missing static cameras:", not_seen)


if __name__ == "__main__":
    main()
PY
chmod +x run/bus_real_data/approach2_ref_marker_graph_ba/04_single_ref_marker_pnp_baseline.py

cat > run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph.py <<'PY'
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
PY
chmod +x run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph.py

cat > run/bus_real_data/approach2_ref_marker_graph_ba/06_compare_graph_initialization.py <<'PY'
#!/usr/bin/env python3

from ap02_common import AP02_ROOT, ensure_dir, read_csv, write_csv


def count_rows(path):
    if not path.exists():
        return 0
    return len(read_csv(path))


def read_report_value(report_path, prefix):
    if not report_path.exists():
        return ""
    for line in report_path.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def main():
    out = ensure_dir(AP02_ROOT / "06_graph_initialization_comparison")

    rows = []
    for mode in ["static_only", "with_moving"]:
        root = AP02_ROOT / "05_graph_initialization" / mode
        report = root / "graph_connectivity_report.txt"

        rows.append({
            "mode": mode,
            "static_camera_pose_count": count_rows(root / "initial_static_camera_poses_ref_marker.csv"),
            "moving_frame_pose_count": count_rows(root / "initial_moving_frame_poses_ref_marker.csv"),
            "marker_pose_count": count_rows(root / "initial_marker_poses_ref_marker.csv"),
            "initialized_static_cameras": read_report_value(report, "Initialized static cameras"),
            "missing_static_cameras": read_report_value(report, "Missing static cameras"),
            "initialized_markers": read_report_value(report, "Initialized markers"),
            "missing_markers": read_report_value(report, "Missing markers"),
        })

    fields = [
        "mode",
        "static_camera_pose_count",
        "moving_frame_pose_count",
        "marker_pose_count",
        "initialized_static_cameras",
        "missing_static_cameras",
        "initialized_markers",
        "missing_markers",
    ]

    write_csv(out / "graph_initialization_static_vs_moving.csv", rows, fields)

    report_lines = [
        "AP02 graph initialization comparison",
        "====================================",
        "",
        "This compares the AP02 reference-marker pose graph before bundle adjustment.",
        "",
    ]

    for r in rows:
        report_lines += [
            f"Mode: {r['mode']}",
            f"- static camera poses: {r['static_camera_pose_count']}",
            f"- moving frame poses: {r['moving_frame_pose_count']}",
            f"- marker poses: {r['marker_pose_count']}",
            f"- initialized static cameras: {r['initialized_static_cameras']}",
            f"- missing static cameras: {r['missing_static_cameras']}",
            "",
        ]

    report_lines += [
        "Main question:",
        "Does adding moving-camera observations connect more static cameras and markers to the reference ArUco?",
        "",
    ]

    (out / "graph_initialization_comparison_report.txt").write_text("\n".join(report_lines) + "\n")

    print("[OK] wrote", out)
    print((out / "graph_initialization_comparison_report.txt").read_text())


if __name__ == "__main__":
    main()
PY
chmod +x run/bus_real_data/approach2_ref_marker_graph_ba/06_compare_graph_initialization.py

cat > run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh <<'SH'
#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

echo "=== AP02 Phase 2: sanity check AP02 independence ==="
if grep -R "01_marker_direct_relay\|AP1_ROOT\|AP01_ROOT\|moving_detections.csv\|01_static_a4_marker_detection" -n run/bus_real_data/approach2_ref_marker_graph_ba; then
  echo "[ERROR] AP02 scripts reference AP01 result internals. Stop."
  exit 1
fi
echo "[OK] AP02 scripts do not reference AP01 internals."

echo
echo "=== AP02 Phase 2: single reference marker PnP baseline ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/04_single_ref_marker_pnp_baseline.py

echo
echo "=== AP02 Phase 2: graph initialization static_only ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph.py --mode static_only

echo
echo "=== AP02 Phase 2: graph initialization with_moving ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph.py --mode with_moving

echo
echo "=== AP02 Phase 2: compare graph initialization variants ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/06_compare_graph_initialization.py

echo
echo "=== Single Reference Marker Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/04_single_ref_marker_pnp/single_ref_marker_report.txt

echo
echo "=== Static-only Graph Connectivity ==="
cat results/bus_real_data/02_ref_marker_graph_ba/05_graph_initialization/static_only/graph_connectivity_report.txt

echo
echo "=== With-moving Graph Connectivity ==="
cat results/bus_real_data/02_ref_marker_graph_ba/05_graph_initialization/with_moving/graph_connectivity_report.txt

echo
echo "=== Comparison Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/06_graph_initialization_comparison/graph_initialization_comparison_report.txt

echo
echo "[OK] AP02 Phase 2 complete."
SH
chmod +x run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh

echo "[OK] AP02 Phase 2 scripts added."
echo
echo "Run next:"
echo "  bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh"
