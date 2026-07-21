#!/usr/bin/env python3

import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np


WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

AP02_ROOT = Path("results/bus_real_data/02_ref_marker_graph_ba")
BA_ROOT = AP02_ROOT / "07_graph_ba" / "with_moving"
FINAL_ROOT = AP02_ROOT / "08_final_results"

CAM_EVAL = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/plots/02_ref_marker_graph_ba/ap02_static_cameras_ref_aruco_vs_gt.csv")
MARKER_EVAL = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/plots/02_ref_marker_graph_ba/ap02_markers_ref_aruco_vs_gt.csv")

EST_CAM_CSV = FINAL_ROOT / "ap02_with_moving_static_camera_poses_ref_marker.csv"
EST_MARKER_CSV = FINAL_ROOT / "ap02_with_moving_marker_poses_ref_marker.csv"
EST_MOVING_CSV = FINAL_ROOT / "ap02_with_moving_moving_frame_poses_ref_marker.csv"

OBS_EDGE_CSV = BA_ROOT / "reprojection_errors_by_observation.csv"

OUT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/plots/91_gazebo_ap02_graph_debug")

REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"
STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

# SDF marker model frame vs OpenCV marker frame.
# OpenCV marker frame:
#   x right, y down, z marker normal
# SDF model marker visual plane:
#   x horizontal, z vertical, normal roughly -y
R_MODEL_FROM_OPENCV_MARKER = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=np.float64)


COLORS = {
    "ref":       (1.00, 0.85, 0.05, 1.0),
    "camera":    (0.05, 0.35, 1.00, 1.0),
    "marker":    (0.05, 0.85, 0.20, 1.0),
    "moving":    (0.70, 0.20, 1.00, 1.0),
    "ref_cam":   (0.05, 0.45, 1.00, 0.85),
    "ref_marker":(0.05, 0.80, 0.20, 0.55),
    "obs":       (1.00, 0.45, 0.05, 0.25),
    "traj":      (0.75, 0.20, 1.00, 0.80),
    "gt":        (0.05, 0.05, 0.05, 0.75),
    "error":     (1.00, 0.05, 0.05, 0.90),
    "frustum":   (0.05, 0.60, 1.00, 0.90),
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows, fields):
    ensure_dir(path.parent)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rvec_to_R(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)

    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def parse_pose_text(text):
    vals = [float(x) for x in text.split()]
    if len(vals) < 6:
        raise RuntimeError(f"Invalid pose text: {text}")
    x, y, z, roll, pitch, yaw = vals[:6]
    return make_T(rpy_to_R(roll, pitch, yaw), [x, y, z]), vals[:6]


def parse_world_poses():
    if not WORLD_SDF.exists():
        raise RuntimeError(f"Missing world SDF: {WORLD_SDF}")

    tree = ET.parse(WORLD_SDF)
    root = tree.getroot()
    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "").strip()
        pose_el = model.find("pose")
        if not name or pose_el is None or not pose_el.text:
            continue
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None or not name_el.text or not pose_el.text:
            continue
        name = name_el.text.strip()
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    return poses


def sdf_marker_model_to_opencv_frame(T_W_model):
    T_model_cv = make_T(R_MODEL_FROM_OPENCV_MARKER, [0, 0, 0])
    return T_W_model @ T_model_cv


def get_ref_world_pose():
    poses = parse_world_poses()
    if REF_MARKER_ENTITY not in poses:
        raise RuntimeError(f"Could not find {REF_MARKER_ENTITY} in {WORLD_SDF}")
    return sdf_marker_model_to_opencv_frame(poses[REF_MARKER_ENTITY]["T_W_model"])


def T_from_pose_csv_row(row):
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=np.float64)
    t = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=np.float64)
    return make_T(rvec_to_R(rvec), t)


def T_from_eval_row(row, prefix):
    # prefix: est_ref_aruco or gt_ref_aruco
    rvec = np.array([
        float(row[f"{prefix}_rvec_x"]),
        float(row[f"{prefix}_rvec_y"]),
        float(row[f"{prefix}_rvec_z"]),
    ], dtype=np.float64)
    t = np.array([
        float(row[f"{prefix}_x_m"]),
        float(row[f"{prefix}_y_m"]),
        float(row[f"{prefix}_z_m"]),
    ], dtype=np.float64)
    return make_T(rvec_to_R(rvec), t)


def pose_str_from_T(T):
    t = T[:3, 3]
    rpy = R_to_rpy(T[:3, :3])
    return f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"


def material_xml(color):
    r, g, b, a = color
    return f"""
          <material>
            <ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>
            <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>
          </material>"""


def visual_sphere(name, p, radius, color):
    return f"""
      <visual name="{name}">
        <pose>{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0</pose>
        <geometry><sphere><radius>{radius:.4f}</radius></sphere></geometry>
        {material_xml(color)}
      </visual>"""


def visual_box_T(name, T, size_xyz, color):
    sx, sy, sz = size_xyz
    return f"""
      <visual name="{name}">
        <pose>{pose_str_from_T(T)}</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        {material_xml(color)}
      </visual>"""


def cylinder_T_between(p1, p2):
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    length = float(np.linalg.norm(d))

    if length < 1e-9:
        return None, 0.0

    z_axis = d / length

    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tmp, z_axis))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = np.cross(tmp, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return make_T(R, mid), length


def visual_cylinder_line(name, p1, p2, radius, color):
    T, length = cylinder_T_between(p1, p2)
    if T is None:
        return ""

    return f"""
      <visual name="{name}">
        <pose>{pose_str_from_T(T)}</pose>
        <geometry>
          <cylinder>
            <radius>{radius:.5f}</radius>
            <length>{length:.6f}</length>
          </cylinder>
        </geometry>
        {material_xml(color)}
      </visual>"""


def add_camera_frustum(visuals, base_name, T_W_cam, scale=0.35):
    c = T_W_cam[:3, 3]
    z = scale
    w = scale * 0.45
    h = scale * 0.28

    corners_local = [
        np.array([-w, -h, z, 1.0]),
        np.array([ w, -h, z, 1.0]),
        np.array([ w,  h, z, 1.0]),
        np.array([-w,  h, z, 1.0]),
    ]

    corners = [(T_W_cam @ p)[:3] for p in corners_local]

    for i, p in enumerate(corners):
        visuals.append(visual_cylinder_line(f"{base_name}_frustum_ray_{i}", c, p, 0.006, COLORS["frustum"]))

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        visuals.append(visual_cylinder_line(f"{base_name}_frustum_rect_{i}", p1, p2, 0.005, COLORS["frustum"]))


def moving_index(name):
    m = re.search(r"(\d+)$", name)
    if not m:
        return 0
    return int(m.group(1))


def load_estimated_world_poses():
    T_W_ref = get_ref_world_pose()

    cams = {}
    for r in read_csv(EST_CAM_CSV):
        cams[r["entity_id"]] = T_W_ref @ T_from_pose_csv_row(r)

    markers = {}
    for r in read_csv(EST_MARKER_CSV):
        mid = int(float(r["entity_id"]))
        markers[mid] = T_W_ref @ T_from_pose_csv_row(r)

    moving = {}
    for r in read_csv(EST_MOVING_CSV):
        moving[r["entity_id"]] = T_W_ref @ T_from_pose_csv_row(r)

    return T_W_ref, cams, markers, moving


def load_eval_world_poses():
    T_W_ref = get_ref_world_pose()

    cam_eval = {}
    for r in read_csv(CAM_EVAL):
        cam = r["entity_id"]
        cam_eval[cam] = {
            "est": T_W_ref @ T_from_eval_row(r, "est_ref_aruco"),
            "gt": T_W_ref @ T_from_eval_row(r, "gt_ref_aruco"),
            "t_err_cm": float(r["translation_error_cm"]),
            "r_err_deg": float(r["rotation_error_deg"]),
        }

    marker_eval = {}
    for r in read_csv(MARKER_EVAL):
        mid = int(float(r["marker_id"]))
        marker_eval[mid] = {
            "entity_id": r["entity_id"],
            "est": T_W_ref @ T_from_eval_row(r, "est_ref_aruco"),
            "gt": T_W_ref @ T_from_eval_row(r, "gt_ref_aruco"),
            "t_err_cm": float(r["translation_error_cm"]),
            "r_err_deg": float(r["rotation_error_deg"]),
        }

    return cam_eval, marker_eval


def make_overlay_sdf(name, full_debug=False):
    T_W_ref, cams, markers, moving = load_estimated_world_poses()
    cam_eval, marker_eval = load_eval_world_poses()

    visuals = []
    distance_rows = []

    ref_p = T_W_ref[:3, 3]
    visuals.append(visual_sphere("NODE_REF_ARUCO_14_ORIGIN", ref_p, 0.16, COLORS["ref"]))

    # Reference marker plane.
    visuals.append(visual_box_T("REF_ARUCO_14_PLANE", T_W_ref, (0.20, 0.20, 0.018), COLORS["ref"]))

    # Static cameras + frustums + lines from ref.
    for cam, T in sorted(cams.items()):
        p = T[:3, 3]
        dist = float(np.linalg.norm(p - ref_p))
        visuals.append(visual_sphere(f"NODE_EST_CAMERA_{cam}_dist_{dist:.2f}m", p, 0.12, COLORS["camera"]))
        add_camera_frustum(visuals, f"CAMERA_{cam}", T, scale=0.45)
        visuals.append(visual_cylinder_line(f"EDGE_REF14_TO_CAMERA_{cam}_{dist:.2f}m", ref_p, p, 0.018, COLORS["ref_cam"]))

        distance_rows.append({
            "entity_type": "static_camera",
            "entity_id": cam,
            "distance_from_ref14_m": f"{dist:.6f}",
            "x_world_m": f"{p[0]:.6f}",
            "y_world_m": f"{p[1]:.6f}",
            "z_world_m": f"{p[2]:.6f}",
        })

    # Markers + lines from ref.
    for mid, T in sorted(markers.items()):
        p = T[:3, 3]
        dist = float(np.linalg.norm(p - ref_p))
        node_name = "NODE_EST_REF_MARKER_14" if mid == REF_MARKER_ID else f"NODE_EST_MARKER_{mid:03d}_dist_{dist:.2f}m"

        visuals.append(visual_sphere(node_name, p, 0.075 if mid != REF_MARKER_ID else 0.12, COLORS["marker"]))
        visuals.append(visual_box_T(f"PLANE_EST_MARKER_{mid:03d}", T, (0.17, 0.17, 0.012), COLORS["marker"]))

        if mid != REF_MARKER_ID:
            visuals.append(visual_cylinder_line(f"EDGE_REF14_TO_MARKER_{mid:03d}_{dist:.2f}m", ref_p, p, 0.010, COLORS["ref_marker"]))

        distance_rows.append({
            "entity_type": "aruco_marker",
            "entity_id": f"marker_{mid:03d}" if mid != REF_MARKER_ID else REF_MARKER_ENTITY,
            "distance_from_ref14_m": f"{dist:.6f}",
            "x_world_m": f"{p[0]:.6f}",
            "y_world_m": f"{p[1]:.6f}",
            "z_world_m": f"{p[2]:.6f}",
        })

    if full_debug:
        # Moving frames + trajectory.
        moving_items = sorted(moving.items(), key=lambda kv: moving_index(kv[0]))
        last_p = None
        for obs_id, T in moving_items:
            p = T[:3, 3]
            visuals.append(visual_sphere(f"NODE_MOVING_{obs_id}", p, 0.045, COLORS["moving"]))

            if last_p is not None:
                visuals.append(visual_cylinder_line(f"TRAJ_{obs_id}", last_p, p, 0.008, COLORS["traj"]))
            last_p = p

        # Observation edges used by BA.
        obs_rows = read_csv(OBS_EDGE_CSV)
        for i, r in enumerate(obs_rows):
            obs_id = r["observer_id"]
            mid = int(float(r["marker_id"]))

            if obs_id in cams:
                p_obs = cams[obs_id][:3, 3]
            elif obs_id in moving:
                p_obs = moving[obs_id][:3, 3]
            else:
                continue

            if mid not in markers:
                continue

            p_marker = markers[mid][:3, 3]
            visuals.append(visual_cylinder_line(f"OBS_EDGE_{i:04d}_{obs_id}_to_marker_{mid:03d}", p_obs, p_marker, 0.0045, COLORS["obs"]))

        # GT nodes and error vectors.
        for cam, payload in sorted(cam_eval.items()):
            p_est = payload["est"][:3, 3]
            p_gt = payload["gt"][:3, 3]
            visuals.append(visual_sphere(f"NODE_GT_CAMERA_{cam}", p_gt, 0.075, COLORS["gt"]))
            visuals.append(visual_cylinder_line(f"ERROR_VECTOR_CAMERA_{cam}_{payload['t_err_cm']:.1f}cm", p_gt, p_est, 0.014, COLORS["error"]))

        for mid, payload in sorted(marker_eval.items()):
            p_est = payload["est"][:3, 3]
            p_gt = payload["gt"][:3, 3]
            visuals.append(visual_sphere(f"NODE_GT_MARKER_{mid:03d}", p_gt, 0.045, COLORS["gt"]))
            if mid != REF_MARKER_ID:
                visuals.append(visual_cylinder_line(f"ERROR_VECTOR_MARKER_{mid:03d}_{payload['t_err_cm']:.1f}cm", p_gt, p_est, 0.008, COLORS["error"]))

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="ap02_graph_overlay_link">
{''.join(visuals)}
    </link>
  </model>
</sdf>
"""

    return sdf, distance_rows


def make_spawn_script(script_path, sdf_path):
    abs_sdf = (REPO_ROOT / sdf_path).resolve()

    txt = f"""#!/usr/bin/env bash
set -eo pipefail

cd "$REPO_ROOT"

SDF="{abs_sdf}"

echo "[INFO] SDF overlay:"
echo "$SDF"

if ! command -v ign >/dev/null 2>&1; then
  echo "[ERROR] ign command not found. Run inside the ROS/Gazebo container."
  exit 1
fi

CREATE_SERVICE=$(ign service -l | grep -E '/world/.*/create$' | head -n 1 || true)

if [ -z "$CREATE_SERVICE" ]; then
  echo "[ERROR] Could not find Gazebo create service."
  echo "Start your bus world first, then run this script again."
  echo
  echo "Expected something like:"
  echo "  /world/<world_name>/create"
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
  --req "sdf_filename: \\"$SDF\\" name: \\"$(basename "$SDF" .sdf)\\" allow_renaming: true"

echo "[OK] Spawn request sent."
echo
echo "In Gazebo, look for:"
echo "- yellow sphere/plane = Ref-ArUco 14"
echo "- blue nodes/frustums = estimated static cameras"
echo "- green nodes/planes = estimated ArUco markers"
echo "- purple nodes/line = selected moving trajectory, full overlay only"
echo "- orange lines = observation graph edges, full overlay only"
echo "- red lines = GT error vectors, full overlay only"
"""
    script_path.write_text(txt)
    script_path.chmod(0o755)


def write_readme():
    readme = f"""AP02 Gazebo Graph Visualization
===============================

Purpose
-------
This folder contains Gazebo SDF overlay models for visual debugging of AP02
inside the real bus Gazebo world.

Main idea
---------
AP02 estimates everything in the reference ArUco coordinate frame:

  aruco_marker_14 / {REF_MARKER_ENTITY}

For visualization, the script converts the AP02 result back into the Gazebo
world frame and spawns a visual-only overlay model.

Files
-----
- ap02_graph_clean_overlay_model.sdf
  Clean presentation overlay:
  Ref-ArUco, static cameras, marker map, ref-to-node edges, camera frustums.

- ap02_graph_full_debug_overlay_model.sdf
  Full debug overlay:
  everything from clean overlay, plus selected moving frames, moving trajectory,
  observation edges, GT nodes, and red estimated-vs-GT error vectors.

- spawn_clean_overlay.sh
  Spawns the clean overlay into the currently running Gazebo world.

- spawn_full_debug_overlay.sh
  Spawns the full debug overlay into the currently running Gazebo world.

Legend
------
- yellow sphere/plane:
  reference ArUco marker 14

- blue spheres/frustums:
  estimated static camera poses

- green spheres/planes:
  estimated marker-map poses

- thin blue/green lines:
  distance edges from Ref14 to cameras/markers

- purple nodes/line:
  selected moving-camera frames and trajectory

- orange transparent lines:
  AP02 observation edges used in BA

- black ghost nodes:
  GT camera/marker positions

- red lines:
  estimated-to-GT error vectors

How to use
----------
Terminal 1: start the bus world normally.

Terminal 2:
  cd "$REPO_ROOT"
  bash results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/plots/91_gazebo_ap02_graph_debug/spawn_clean_overlay.sh

or:

  bash results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/plots/91_gazebo_ap02_graph_debug/spawn_full_debug_overlay.sh

Recommended for presentation
----------------------------
Use clean overlay first. It is easier to explain.

Use full debug overlay when explaining why the moving camera helps:
the orange observation edges and purple trajectory show how the graph connects
markers and cameras that do not directly see Ref14.

"""
    (OUT / "AP02_GAZEBO_GRAPH_VISUALIZATION_README.txt").write_text(readme)


def main():
    ensure_dir(OUT)

    clean_sdf, distance_rows = make_overlay_sdf("ap02_graph_clean_overlay", full_debug=False)
    full_sdf, _ = make_overlay_sdf("ap02_graph_full_debug_overlay", full_debug=True)

    clean_path = OUT / "ap02_graph_clean_overlay_model.sdf"
    full_path = OUT / "ap02_graph_full_debug_overlay_model.sdf"

    clean_path.write_text(clean_sdf)
    full_path.write_text(full_sdf)

    fields = ["entity_type", "entity_id", "distance_from_ref14_m", "x_world_m", "y_world_m", "z_world_m"]
    write_csv(OUT / "ap02_ref_distances.csv", distance_rows, fields)

    make_spawn_script(OUT / "spawn_clean_overlay.sh", clean_path)
    make_spawn_script(OUT / "spawn_full_debug_overlay.sh", full_path)

    write_readme()

    print("[OK] wrote AP02 Gazebo graph debug overlay:")
    print("-", clean_path)
    print("-", full_path)
    print("-", OUT / "spawn_clean_overlay.sh")
    print("-", OUT / "spawn_full_debug_overlay.sh")
    print("-", OUT / "ap02_ref_distances.csv")
    print("-", OUT / "AP02_GAZEBO_GRAPH_VISUALIZATION_README.txt")


if __name__ == "__main__":
    main()
