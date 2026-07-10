#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

if [[ -f install/setup.bash ]]; then
  set +u
  source install/setup.bash
  set -u
fi

export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

ROOT="results/bus_real_data/ablation/world/route"
WORLD="$PWD/src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"

STATIC_CAPTURE="$PWD/run/bus_real_data/approach2_ref_marker_graph_ba/01_capture_shared_raw_dataset.py"
ROUTE_CAPTURE="$PWD/run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"
DETECTOR="$PWD/run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"
COMMON_RUN="$PWD/run/bus_real_data/ablation/_shared/12_run_one_clean_variant_common.sh"

ROUTE1="$PWD/src/calib_lab/bus_real_data/config/moving_camera_route1_interpolated_final.json"
ROUTE2="$PWD/src/calib_lab/bus_real_data/config/moving_camera_route2_interpolated_final.json"

mkdir -p "$ROOT"

for required in "$WORLD" "$STATIC_CAPTURE" "$ROUTE_CAPTURE" "$DETECTOR" "$COMMON_RUN" "$ROUTE1" "$ROUTE2"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] missing required file: $required"
    exit 1
  fi
done

if ! command -v ign >/dev/null 2>&1; then
  echo "[ERROR] ign not found"
  exit 127
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found"
  echo "Install with: sudo apt update && sudo apt install colmap"
  exit 127
fi

SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
SHARED_BACKUP="results/bus_real_data/.backup_shared_before_route_ablation_$(date +%Y%m%d_%H%M%S)"

if [[ -d "$SHARED" ]]; then
  echo "[INFO] Backup shared baseline:"
  echo "       $SHARED_BACKUP"
  cp -a "$SHARED" "$SHARED_BACKUP"
fi

restore_shared() {
  set +e
  pkill -f "parameter_bridge" 2>/dev/null || true
  pkill -f "ros_gz_bridge" 2>/dev/null || true
  pkill -f "ign gazebo" 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true

  if [[ -d "$SHARED_BACKUP" ]]; then
    echo "[INFO] Restoring shared baseline backup..."
    rm -rf "$SHARED"
    mkdir -p "$(dirname "$SHARED")"
    cp -a "$SHARED_BACKUP" "$SHARED"
    echo "[OK] shared baseline restored"
  fi
}

trap restore_shared EXIT INT TERM

stop_sim() {
  set +e
  if [[ -n "${BRIDGE_PID:-}" ]]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [[ -n "${GZ_PID:-}" ]]; then
    kill "$GZ_PID" 2>/dev/null || true
    wait "$GZ_PID" 2>/dev/null || true
  fi
  pkill -f "parameter_bridge" 2>/dev/null || true
  pkill -f "ros_gz_bridge" 2>/dev/null || true
  pkill -f "ign gazebo" 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true
  set -e
  sleep 2
}

capture_one_route() {
  local VARIANT="$1"
  local ROUTE="$2"

  local VAR_ROOT="$ROOT/$VARIANT"
  local RAW="$VAR_ROOT/raw_images"
  local META="$VAR_ROOT/metadata"
  local LOGS="$VAR_ROOT/logs"
  local TMP_CAPTURE="$VAR_ROOT/_moving_route_capture"
  local OBS="$VAR_ROOT/aruco_observations"

  echo
  echo "================================================================================"
  echo "CAPTURE ROUTE VARIANT: $VARIANT"
  echo "ROUTE: $ROUTE"
  echo "WORLD: $WORLD"
  echo "OUT:   $VAR_ROOT"
  echo "================================================================================"

  rm -rf "$RAW" "$META" "$LOGS" "$TMP_CAPTURE" "$OBS" "$VAR_ROOT/FINAL_RESULTS"
  mkdir -p "$RAW" "$META" "$LOGS"

  python3 - "$VAR_ROOT" "$VARIANT" "$ROUTE" "$WORLD" <<'PY'
import json
import sys
from pathlib import Path

var_root = Path(sys.argv[1])
variant = sys.argv[2]
route = Path(sys.argv[3])
world = Path(sys.argv[4])

data = json.loads(route.read_text())
frames = data.get("frames", [])

metadata = {
    "group": "world/route",
    "variant": variant,
    "parameter": "moving camera route",
    "description": "Route ablation comparing two manually designed moving-camera trajectories.",
    "world": str(world),
    "route_file": str(route),
    "num_route_frames": len(frames),
    "note": "Static camera images are identical world/camera setup. Moving-camera trajectory differs between route1 and route2."
}

(var_root / "VARIANT_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("[OK] wrote", var_root / "VARIANT_METADATA.json")
PY

  stop_sim

  echo "[INFO] Starting Gazebo server-only..."
  ign gazebo -r -s "$WORLD" > "$LOGS/gazebo.log" 2>&1 &
  GZ_PID=$!

  WORLD_NAME=""
  for _ in $(seq 1 60); do
    SERVICES="$(ign service -l 2>/dev/null || true)"
    WORLD_NAME="$(
      printf '%s\n' "$SERVICES" \
      | sed -n 's#^/world/\([^/]*\)/set_pose$#\1#p' \
      | head -n 1
    )"

    if [[ -n "$WORLD_NAME" ]]; then
      break
    fi

    sleep 1
  done

  if [[ -z "$WORLD_NAME" ]]; then
    echo "[ERROR] Gazebo set_pose service was not detected."
    tail -100 "$LOGS/gazebo.log" || true
    exit 1
  fi

  echo "[OK] Gazebo world name: $WORLD_NAME"

  BRIDGE_TOPICS=(
    "/bus_real_data/cam_edge_0/image@sensor_msgs/msg/Image@gz.msgs.Image"
    "/bus_real_data/cam_edge_0/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"

    "/bus_real_data/cam_edge_1/image@sensor_msgs/msg/Image@gz.msgs.Image"
    "/bus_real_data/cam_edge_1/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"

    "/bus_real_data/cam_edge_3/image@sensor_msgs/msg/Image@gz.msgs.Image"
    "/bus_real_data/cam_edge_3/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"

    "/bus_real_data/cam_edge_5/image@sensor_msgs/msg/Image@gz.msgs.Image"
    "/bus_real_data/cam_edge_5/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"

    "/bus_real_data/moving_calib_camera/image@sensor_msgs/msg/Image@gz.msgs.Image"
    "/bus_real_data/moving_calib_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"
  )

  echo "[INFO] Starting ROS-GZ bridge..."
  ros2 run ros_gz_bridge parameter_bridge "${BRIDGE_TOPICS[@]}" \
    > "$LOGS/bridge.log" 2>&1 &
  BRIDGE_PID=$!

  REQUIRED_ROS_TOPICS=(
    "/bus_real_data/cam_edge_0/image"
    "/bus_real_data/cam_edge_0/camera_info"
    "/bus_real_data/cam_edge_1/image"
    "/bus_real_data/cam_edge_1/camera_info"
    "/bus_real_data/cam_edge_3/image"
    "/bus_real_data/cam_edge_3/camera_info"
    "/bus_real_data/cam_edge_5/image"
    "/bus_real_data/cam_edge_5/camera_info"
    "/bus_real_data/moving_calib_camera/image"
    "/bus_real_data/moving_calib_camera/camera_info"
  )

  for topic in "${REQUIRED_ROS_TOPICS[@]}"; do
    FOUND=0
    for _ in $(seq 1 60); do
      if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
        FOUND=1
        break
      fi
      sleep 1
    done

    if [[ "$FOUND" != "1" ]]; then
      echo "[ERROR] ROS topic did not appear: $topic"
      tail -100 "$LOGS/bridge.log" || true
      exit 1
    fi

    echo "[OK] topic: $topic"
  done

  echo
  echo "=== Park moving camera outside static camera views ==="
  ign service \
    -s "/world/$WORLD_NAME/set_pose" \
    --reqtype ignition.msgs.Pose \
    --reptype ignition.msgs.Boolean \
    --timeout 1000 \
    --req 'name: "moving_calib_camera" position {x: 0 y: 30 z: 5} orientation {x: 0 y: 0 z: 0 w: 1}' || true

  sleep 2.0

  echo
  echo "=== Capture four static cameras ==="
  python3 "$STATIC_CAPTURE" \
    --out "$RAW" \
    --static-only \
    --overwrite \
    2>&1 | tee "$LOGS/static_capture.log"

  echo
  echo "=== Capture moving-camera CameraInfo ==="
  RAW_CAMERA_INFO="$RAW/camera_info/moving_calib_camera.json"

  python3 - "$RAW_CAMERA_INFO" <<'PY'
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

topic = "/bus_real_data/moving_calib_camera/camera_info"
output = Path(sys.argv[1])

class Grabber(Node):
    def __init__(self):
        super().__init__("route_ablation_moving_camera_info_capture")
        self.message = None
        self.create_subscription(
            CameraInfo,
            topic,
            self.callback,
            qos_profile_sensor_data,
        )

    def callback(self, message):
        self.message = message

rclpy.init()
node = Grabber()

deadline = time.time() + 30.0
while rclpy.ok() and node.message is None and time.time() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

message = node.message
node.destroy_node()
rclpy.shutdown()

if message is None:
    raise RuntimeError(f"No CameraInfo received from {topic}")

data = {
    "camera_name": "moving_calib_camera",
    "width": int(message.width),
    "height": int(message.height),
    "image_width": int(message.width),
    "image_height": int(message.height),
    "distortion_model": str(message.distortion_model),
    "d": [float(value) for value in message.d],
    "D": [float(value) for value in message.d],
    "k": [float(value) for value in message.k],
    "K": [float(value) for value in message.k],
    "r": [float(value) for value in message.r],
    "R": [float(value) for value in message.r],
    "p": [float(value) for value in message.p],
    "P": [float(value) for value in message.p],
    "fx": float(message.k[0]),
    "fy": float(message.k[4]),
    "cx": float(message.k[2]),
    "cy": float(message.k[5]),
}

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(data, indent=2) + "\n")
print(f"[OK] wrote {output}")
PY

  echo
  echo "=== Capture commanded moving-camera route ==="
  python3 "$ROUTE_CAPTURE" \
    --route "$ROUTE" \
    --world "$WORLD_NAME" \
    --name moving_calib_camera \
    --out "$TMP_CAPTURE" \
    --clean \
    2>&1 | tee "$LOGS/moving_route_capture.log"

  mkdir -p "$RAW/moving" "$RAW/ap1_metadata" "$META"

  cp -a "$TMP_CAPTURE/images/." "$RAW/moving/"

  python3 - \
    "$TMP_CAPTURE/route_commanded.csv" \
    "$META/route_commanded.csv" \
    "$RAW" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
raw = Path(sys.argv[3])

rows = list(csv.DictReader(source.open()))
if not rows:
    raise RuntimeError("route_commanded.csv contains no rows")

fields = list(rows[0])

for row in rows:
    frame = int(row["frame"])
    row["image"] = str(raw / "moving" / f"frame_{frame:04d}.png")

destination.parent.mkdir(parents=True, exist_ok=True)

with destination.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote normalized route: {destination}")
PY

  cp "$META/route_commanded.csv" "$RAW/ap1_metadata/route_commanded.csv"
  cp "$ROUTE" "$META/$(basename "$ROUTE")"
  cp "$WORLD" "$META/$(basename "$WORLD")"

  EXPECTED_MOVING="$(
    python3 - "$ROUTE" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print(len(data["frames"]))
PY
  )"

  STATIC_COUNT="$(find "$RAW/static" -maxdepth 1 -name 'cam_edge_*.png' | wc -l)"
  MOVING_COUNT="$(find "$RAW/moving" -maxdepth 1 -name 'frame_*.png' | wc -l)"
  INFO_COUNT="$(find "$RAW/camera_info" -maxdepth 1 -name '*.json' | wc -l)"

  echo
  echo "=== Capture audit ==="
  echo "static images:       $STATIC_COUNT"
  echo "moving images:       $MOVING_COUNT"
  echo "expected moving:     $EXPECTED_MOVING"
  echo "camera-info files:   $INFO_COUNT"

  if [[ "$STATIC_COUNT" -ne 4 ]]; then
    echo "[ERROR] Expected exactly four static images."
    exit 1
  fi

  if [[ "$MOVING_COUNT" -ne "$EXPECTED_MOVING" ]]; then
    echo "[ERROR] Moving-frame count does not match route."
    exit 1
  fi

  if [[ "$INFO_COUNT" -ne 5 ]]; then
    echo "[ERROR] Expected five CameraInfo files."
    exit 1
  fi

  rm -rf "$TMP_CAPTURE"

  echo
  echo "=== Stop sim before detection/pipelines ==="
  stop_sim

  echo
  echo "=== ArUco detection for $VARIANT ==="
  rm -rf "$OBS"
  python3 "$DETECTOR" \
    --dataset "$RAW" \
    --out "$OBS" \
    --dictionary DICT_4X4_50 \
    2>&1 | tee "$LOGS/aruco_detection.log"

  if [[ -f "$OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" ]]; then
    cat "$OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt"
  else
    echo "[ERROR] Detector did not create summary."
    exit 1
  fi

  echo
  echo "=== Run AP01/AP02/AP03 for $VARIANT ==="
  REFRESH_CANONICAL_FINAL=0 bash "$COMMON_RUN" "$ROOT" "WORLD ROUTE ABLATION" "$VARIANT"

  echo
  echo "[OK] completed variant: $VARIANT"
}

capture_one_route "route1" "$ROUTE1"
capture_one_route "route2" "$ROUTE2"

echo
echo "================================================================================"
echo "BUILD CUSTOM 99_FINAL_RESULTS ROUTE COMPARISON"
echo "================================================================================"

python3 - <<'PY'
import csv
import json
from pathlib import Path
from typing import Any

BUS = Path("results/bus_real_data")
ROOT = BUS / "ablation/world/route"
FINAL99 = BUS / "99_FINAL_RESULTS_FOR_REPORT"

VARIANTS = [
    ("route1", "Route 1"),
    ("route2", "Route 2"),
]

METHODS = ["AP01", "AP02", "AP03"]

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def write_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def safe(value):
    text = "" if value is None else str(value).strip()
    return text if text else "-"

def num(value):
    try:
        return float(value)
    except Exception:
        return None

def fmt(value, digits=2):
    parsed = num(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}"

all_primary_summary = []
all_primary_detail = []
all_secondary_summary = []
all_secondary_detail = []
all_run_status = []
all_ap02_full_map = []

report_lines = [
    "ROUTE ABLATION — AP01 / AP02 / AP03",
    "=" * 100,
    "",
    "This custom report compares the two manually designed moving-camera routes.",
    "Group: world/route",
    "Variants: route1, route2",
    "",
]

primary_detail_report = [
    "ROUTE ABLATION — DETAILED CAMERA-TO-CAMERA RESULTS",
    "=" * 100,
    "",
]

secondary_detail_report = [
    "ROUTE ABLATION — DETAILED REF14/WORLD MAP-TO-GT RESULTS",
    "=" * 100,
    "",
]

for variant, label in VARIANTS:
    var_final = ROOT / variant / "FINAL_RESULTS"
    if not var_final.is_dir():
        raise SystemExit(f"[ERROR] missing FINAL_RESULTS: {var_final}")

    metadata_path = ROOT / variant / "VARIANT_METADATA.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())

    primary_summary = read_csv(var_final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv")
    primary_detail = read_csv(var_final / "BASELINE_FINAL_PAIRWISE_DETAIL.csv")
    secondary_summary = read_csv(var_final / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv")
    secondary_detail = read_csv(var_final / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv")
    full_map = read_csv(var_final / "DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.csv")

    status = {}
    status_path = var_final / "RUN_STATUS.txt"
    if status_path.is_file():
        for line in status_path.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                status[k.strip()] = v.strip()

    status_row = {
        "group": "world/route",
        "variant": variant,
        "parameter": label,
        **status,
    }
    all_run_status.append(status_row)

    for row in primary_summary:
        all_primary_summary.append({
            "group": "world/route",
            "variant": variant,
            "parameter": label,
            **row,
        })

    for row in primary_detail:
        all_primary_detail.append({
            "group": "world/route",
            "variant": variant,
            "parameter": label,
            **row,
        })

    for row in secondary_summary:
        all_secondary_summary.append({
            "group": "world/route",
            "variant": variant,
            "parameter": label,
            **row,
        })

    for row in secondary_detail:
        all_secondary_detail.append({
            "group": "world/route",
            "variant": variant,
            "parameter": label,
            **row,
        })

    for row in full_map:
        all_ap02_full_map.append({
            "group": "world/route",
            "variant": variant,
            "parameter": label,
            **row,
        })

    report_lines += [
        "#" * 100,
        f"{label} — {variant}",
        "#" * 100,
        "",
        f"route_file: {metadata.get('route_file', '-')}",
        f"num_route_frames: {metadata.get('num_route_frames', '-')}",
        "",
        "RUN STATUS",
        "-" * 60,
        f"AP01_STATUS={status.get('AP01_STATUS', '-')}",
        f"AP02_STATUS={status.get('AP02_STATUS', '-')}",
        f"AP02_GT_FULL_MAP_STATUS={status.get('AP02_GT_FULL_MAP_STATUS', '-')}",
        f"AP03_STATUS={status.get('AP03_STATUS', '-')}",
        "",
        "PRIMARY CAMERA-TO-CAMERA",
        "-" * 90,
        f"{'Method':8s}{'Status':24s}{'Cameras':>10s}{'Pairs':>9s}{'Mean t':>13s}{'Mean r':>13s}",
        "-" * 90,
    ]

    by_method = {row.get("method", ""): row for row in primary_summary}
    for method in METHODS:
        row = by_method.get(method, {})
        report_lines.append(
            f"{method:8s}"
            f"{safe(row.get('status'))[:24]:24s}"
            f"{safe(row.get('camera_count')) + '/4':>10s}"
            f"{safe(row.get('pair_count_ok')) + '/6':>9s}"
            f"{fmt(row.get('mean_pair_t_cm')) + ' cm':>13s}"
            f"{fmt(row.get('mean_pair_r_deg')) + ' deg':>13s}"
        )

    report_lines += [
        "",
        "OPTIONAL SECONDARY REF14/WORLD MAP",
        "-" * 90,
        f"{'Method':8s}{'Status':24s}{'Cameras':>10s}{'Mean t':>13s}{'Mean r':>13s}",
        "-" * 90,
    ]

    by_method = {row.get("method", ""): row for row in secondary_summary}
    for method in METHODS:
        row = by_method.get(method, {})
        report_lines.append(
            f"{method:8s}"
            f"{safe(row.get('status'))[:24]:24s}"
            f"{safe(row.get('camera_count')) + '/4':>10s}"
            f"{fmt(row.get('mean_translation_error_cm')) + ' cm':>13s}"
            f"{fmt(row.get('mean_rotation_error_deg')) + ' deg':>13s}"
        )

    report_lines.append("")

    primary_detail_report += [
        "#" * 100,
        f"{label} — {variant}",
        "#" * 100,
        "",
    ]
    for row in primary_detail:
        primary_detail_report.append(str(row))
    primary_detail_report.append("")

    secondary_detail_report += [
        "#" * 100,
        f"{label} — {variant}",
        "#" * 100,
        "",
    ]
    for row in secondary_detail:
        secondary_detail_report.append(str(row))
    secondary_detail_report.append("")

# Write custom 99 files.
(FINAL99 / "ablations").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/primary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/secondary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "data/primary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "data/secondary").mkdir(parents=True, exist_ok=True)

(FINAL99 / "ablations/05_ROUTE_PATH_ALL_METHODS.txt").write_text("\n".join(report_lines) + "\n")
(FINAL99 / "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt").write_text("\n".join(primary_detail_report) + "\n")
(FINAL99 / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt").write_text("\n".join(secondary_detail_report) + "\n")

write_csv_union(FINAL99 / "data/primary/ROUTE_PATH_ABLATION_SUMMARY.csv", all_primary_summary)
write_csv_union(FINAL99 / "data/primary/ROUTE_PATH_ABLATION_DETAIL.csv", all_primary_detail)
write_csv_union(FINAL99 / "data/secondary/ROUTE_PATH_ABLATION_SUMMARY.csv", all_secondary_summary)
write_csv_union(FINAL99 / "data/secondary/ROUTE_PATH_ABLATION_DETAIL.csv", all_secondary_detail)
write_csv_union(FINAL99 / "data/secondary/ROUTE_PATH_AP02_GT_ALIGNED_FULL_MAP.csv", all_ap02_full_map)
write_csv_union(FINAL99 / "data/ROUTE_PATH_RUN_STATUS.csv", all_run_status)

print("[OK] wrote custom route ablation report:")
print(FINAL99 / "ablations/05_ROUTE_PATH_ALL_METHODS.txt")
print(FINAL99 / "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt")
print(FINAL99 / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt")
print(FINAL99 / "data/primary/ROUTE_PATH_ABLATION_SUMMARY.csv")
print(FINAL99 / "data/secondary/ROUTE_PATH_ABLATION_SUMMARY.csv")
PY

echo
echo "================================================================================"
echo "[OK] ROUTE ABLATION COMPLETE"
echo "================================================================================"
echo "Variant results:"
echo "  results/bus_real_data/ablation/world/route/route1/FINAL_RESULTS"
echo "  results/bus_real_data/ablation/world/route/route2/FINAL_RESULTS"
echo
echo "Custom 99_FINAL_RESULTS comparison:"
echo "  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/ablations/05_ROUTE_PATH_ALL_METHODS.txt"
echo "  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt"
echo "  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt"
echo "  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/data/primary/ROUTE_PATH_ABLATION_SUMMARY.csv"
echo "  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/data/secondary/ROUTE_PATH_ABLATION_SUMMARY.csv"
