#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = ROOT / "run/bus_real_data/approach_comparison_ref_aruco/04_make_ap02_gazebo_graph_overlay.py"

spec = importlib.util.spec_from_file_location("ap02_overlay_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = ROOT / "results/bus_real_data/90_approach_comparison_ref_aruco/91_gazebo_ap02_graph_debug"

PRESENTATION_SDF = OUT / "ap02_graph_presentation_observation_overlay_model.sdf"
SPAWN_SCRIPT = OUT / "spawn_presentation_observation_overlay.sh"
INFO_TXT = OUT / "AP02_PRESENTATION_OBSERVATION_OVERLAY_INFO.txt"

COLORS = {
    "ref":         (1.00, 0.85, 0.05, 1.00),
    "camera":      (0.05, 0.35, 1.00, 1.00),
    "marker":      (0.05, 0.85, 0.20, 0.92),
    "moving":      (0.70, 0.20, 1.00, 0.82),
    "static_edge": (1.00, 0.45, 0.05, 0.23),
    "moving_edge": (0.70, 0.20, 1.00, 0.22),
    "frustum":     (0.05, 0.60, 1.00, 0.55),
}

# Make camera frustums more subtle than in the debug overlay.
base.COLORS["frustum"] = COLORS["frustum"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def safe_marker_id(row):
    try:
        return int(float(row["marker_id"]))
    except Exception:
        return None


def collect_static_edges(obs_rows, cams, markers):
    """
    Unique AP02 observation edges:
      static camera node -> marker node
    """
    edges = set()

    for r in obs_rows:
        obs_id = r.get("observer_id", "")
        mid = safe_marker_id(r)

        if obs_id not in cams:
            continue
        if mid is None or mid not in markers:
            continue

        edges.add((obs_id, mid))

    return sorted(edges)


def collect_moving_observations(obs_rows, moving, markers):
    """
    Map moving frame id -> observed marker ids.
    """
    by_frame = {}

    for r in obs_rows:
        obs_id = r.get("observer_id", "")
        mid = safe_marker_id(r)

        if obs_id not in moving:
            continue
        if mid is None or mid not in markers:
            continue

        by_frame.setdefault(obs_id, set()).add(mid)

    return by_frame


def choose_representative_moving_frame(moving, moving_obs_by_frame):
    """
    Choose one moving-camera frame near x=0, but prefer frames with more marker observations.
    This keeps the slide readable while still showing the moving camera's graph role.
    """
    candidates = [k for k in moving.keys() if k in moving_obs_by_frame and len(moving_obs_by_frame[k]) > 0]
    if not candidates:
        return None

    def score(frame_id):
        p = moving[frame_id][:3, 3]
        n_obs = len(moving_obs_by_frame.get(frame_id, []))
        return (abs(float(p[0])), -n_obs)

    return sorted(candidates, key=score)[0]


def nearest_markers_to_point(p, marker_ids, markers, max_edges=8):
    scored = []
    for mid in marker_ids:
        q = markers[mid][:3, 3]
        scored.append((float(np.linalg.norm(q - p)), mid))
    scored.sort()
    return [mid for _, mid in scored[:max_edges]]


def make_overlay_sdf():
    ensure_dir(OUT)

    T_W_ref, cams, markers, moving = base.load_estimated_world_poses()
    obs_rows = read_csv(base.OBS_EDGE_CSV)

    static_edges = collect_static_edges(obs_rows, cams, markers)
    moving_obs_by_frame = collect_moving_observations(obs_rows, moving, markers)
    moving_frame = choose_representative_moving_frame(moving, moving_obs_by_frame)

    visuals = []

    # Ref14 node and plane.
    ref_p = T_W_ref[:3, 3]
    visuals.append(base.visual_sphere("NODE_REF_ARUCO_14", ref_p, 0.12, COLORS["ref"]))
    visuals.append(base.visual_box_T("PLANE_REF_ARUCO_14", T_W_ref, (0.20, 0.20, 0.015), COLORS["ref"]))

    # Static cameras.
    for cam, T in sorted(cams.items()):
        p = T[:3, 3]
        visuals.append(base.visual_sphere(f"NODE_STATIC_CAMERA_{cam}", p, 0.095, COLORS["camera"]))
        base.add_camera_frustum(visuals, f"FRUSTUM_{cam}", T, scale=0.30)

    # Marker map.
    for mid, T in sorted(markers.items()):
        p = T[:3, 3]
        r = 0.070 if mid == base.REF_MARKER_ID else 0.052
        visuals.append(base.visual_sphere(f"NODE_ARUCO_MARKER_{mid:03d}", p, r, COLORS["marker"]))
        visuals.append(base.visual_box_T(f"PLANE_ARUCO_MARKER_{mid:03d}", T, (0.17, 0.17, 0.010), COLORS["marker"]))

    # Static camera -> marker observation edges.
    for i, (cam, mid) in enumerate(static_edges):
        p_cam = cams[cam][:3, 3]
        p_marker = markers[mid][:3, 3]
        visuals.append(
            base.visual_cylinder_line(
                f"OBS_STATIC_EDGE_{i:03d}_{cam}_to_marker_{mid:03d}",
                p_cam,
                p_marker,
                0.006,
                COLORS["static_edge"],
            )
        )

    # One representative moving frame -> marker observation edges.
    moving_edges = []
    if moving_frame is not None:
        T_mov = moving[moving_frame]
        p_mov = T_mov[:3, 3]
        visuals.append(base.visual_sphere(f"NODE_REPRESENTATIVE_MOVING_FRAME_{moving_frame}", p_mov, 0.075, COLORS["moving"]))
        base.add_camera_frustum(visuals, f"FRUSTUM_{moving_frame}", T_mov, scale=0.24)

        observed_markers = sorted(moving_obs_by_frame.get(moving_frame, []))
        shown_markers = nearest_markers_to_point(p_mov, observed_markers, markers, max_edges=8)

        for j, mid in enumerate(shown_markers):
            p_marker = markers[mid][:3, 3]
            moving_edges.append((moving_frame, mid))
            visuals.append(
                base.visual_cylinder_line(
                    f"OBS_MOVING_EDGE_{j:03d}_{moving_frame}_to_marker_{mid:03d}",
                    p_mov,
                    p_marker,
                    0.005,
                    COLORS["moving_edge"],
                )
            )

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="ap02_graph_presentation_observation_overlay">
    <static>true</static>
    <link name="ap02_presentation_overlay_link">
{''.join(visuals)}
    </link>
  </model>
</sdf>
"""
    PRESENTATION_SDF.write_text(sdf)

    info = []
    info.append("AP02 Presentation Observation Overlay")
    info.append("====================================")
    info.append("")
    info.append("Meaning:")
    info.append("- blue nodes/frustums: estimated static cameras")
    info.append("- green nodes/planes: estimated ArUco marker map")
    info.append("- yellow node/plane: reference marker 14")
    info.append("- orange transparent lines: static camera -> marker observation edges")
    info.append("- purple node/frustum: one representative moving-camera frame")
    info.append("- purple transparent lines: moving frame -> marker observation edges")
    info.append("")
    info.append("Important:")
    info.append("- No direct camera-camera edges are drawn.")
    info.append("- No direct marker-marker edges are drawn.")
    info.append("- AP02 connects entities through observation edges: camera/frame observes marker.")
    info.append("")
    info.append(f"Static observation edges shown: {len(static_edges)}")
    info.append(f"Representative moving frame: {moving_frame if moving_frame else 'none'}")
    info.append(f"Moving observation edges shown: {len(moving_edges)}")
    info.append("")
    if moving_edges:
        info.append("Moving edges:")
        for frame_id, mid in moving_edges:
            info.append(f"- {frame_id} -> marker {mid}")
    INFO_TXT.write_text("\n".join(info) + "\n")

    return static_edges, moving_frame, moving_edges


def make_spawn_script():
    txt = f"""#!/usr/bin/env bash
set -eo pipefail

cd "$REPO_ROOT"

SDF="{PRESENTATION_SDF}"

echo "[INFO] SDF overlay:"
echo "$SDF"

if ! command -v ign >/dev/null 2>&1; then
  echo "[ERROR] ign command not found. Run inside the ROS/Gazebo container."
  exit 1
fi

CREATE_SERVICE=$(ign service -l | grep -E '/world/.*/create$' | head -n 1 || true)

if [ -z "$CREATE_SERVICE" ]; then
  echo "[ERROR] Could not find Gazebo create service."
  echo "Start the bus world first, then run this script again."
  echo
  echo "Available services:"
  ign service -l | head -50
  exit 1
fi

echo "[INFO] Using create service: $CREATE_SERVICE"

ign service -s "$CREATE_SERVICE" \\
  --reqtype ignition.msgs.EntityFactory \\
  --reptype ignition.msgs.Boolean \\
  --timeout 5000 \\
  --req "sdf_filename: \\"$SDF\\" name: \\"ap02_graph_presentation_observation_overlay\\" allow_renaming: true"

echo "[OK] Spawn request sent."
echo
echo "Presentation overlay:"
echo "- blue = static cameras"
echo "- green = ArUco markers"
echo "- yellow = Ref14"
echo "- orange = static camera -> marker observation edges"
echo "- purple = one representative moving-camera frame and its observation edges"
"""
    SPAWN_SCRIPT.write_text(txt)
    SPAWN_SCRIPT.chmod(0o755)


def main():
    static_edges, moving_frame, moving_edges = make_overlay_sdf()
    make_spawn_script()

    print("[OK] wrote presentation overlay")
    print("-", PRESENTATION_SDF)
    print("-", SPAWN_SCRIPT)
    print("-", INFO_TXT)
    print(f"[INFO] static observation edges shown: {len(static_edges)}")
    print(f"[INFO] representative moving frame: {moving_frame}")
    print(f"[INFO] moving observation edges shown: {len(moving_edges)}")


if __name__ == "__main__":
    main()
