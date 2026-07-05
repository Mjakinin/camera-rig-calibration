#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

VARIANT="${1:-}"

case "$VARIANT" in
    ceiling_dark_extreme|ceiling_low|ceiling_normal|ceiling_bright)
        ;;
    *)
        echo "Usage:"
        echo "  $0 ceiling_dark_extreme"
        echo "  $0 ceiling_low"
        echo "  $0 ceiling_normal"
        echo "  $0 ceiling_bright"
        exit 2
        ;;
esac

if ! command -v ign >/dev/null 2>&1; then
    echo "[ERROR] ign is required because the existing route-capture"
    echo "        uses 'ign service' for camera pose updates."
    exit 127
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u 2>/dev/null || true
fi

if [[ -f install/setup.bash ]]; then
    set +u
    source install/setup.bash
    set -u 2>/dev/null || true
fi

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

ROOT="results/bus_real_data/ablation/world/lighting"
VAR_ROOT="$ROOT/$VARIANT"
RAW="$VAR_ROOT/raw_images"
META="$VAR_ROOT/metadata"
LOGS="$VAR_ROOT/logs"
TMP_CAPTURE="$VAR_ROOT/_moving_route_capture"

WORLD="$PWD/src/calib_lab/bus_real_data/worlds/lighting/bus_real_data_moving_camera_light_${VARIANT}.sdf"

ROUTE="$PWD/src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json"

STATIC_CAPTURE="$PWD/run/bus_real_data/approach2_ref_marker_graph_ba/01_capture_shared_raw_dataset.py"

ROUTE_CAPTURE="$PWD/run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"

if [[ ! -f "$WORLD" ]]; then
    echo "[ERROR] Missing world: $WORLD"
    exit 1
fi

python3   run/bus_real_data/ablation/world/lighting/17_validate_lighting_world.py   "$WORLD"

if [[ ! -f "$ROUTE" ]]; then
    echo "[ERROR] Missing route: $ROUTE"
    exit 1
fi

mkdir -p "$VAR_ROOT" "$META" "$LOGS"

rm -rf \
  "$RAW" \
  "$TMP_CAPTURE" \
  "$VAR_ROOT/aruco_observations" \
  "$VAR_ROOT/FINAL_RESULTS"

mkdir -p "$RAW" "$META" "$LOGS"

pkill -f "ign gazebo" 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ros_gz_bridge" 2>/dev/null || true
pkill -f "parameter_bridge" 2>/dev/null || true
sleep 2

GZ_PID=""
BRIDGE_PID=""

cleanup() {
    set +e

    if [[ -n "$BRIDGE_PID" ]]; then
        kill "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi

    if [[ -n "$GZ_PID" ]]; then
        kill "$GZ_PID" 2>/dev/null || true
        wait "$GZ_PID" 2>/dev/null || true
    fi

    pkill -f "parameter_bridge" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "================================================================================"
echo "LIGHTING CAPTURE: $VARIANT"
echo "world: $WORLD"
echo "raw:   $RAW"
echo "================================================================================"

echo "[INFO] Starting Gazebo server-only."

ign gazebo -r -s "$WORLD" \
  > "$LOGS/gazebo.log" 2>&1 &

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

echo "[INFO] Starting image and CameraInfo bridges."

ros2 run ros_gz_bridge parameter_bridge \
  "${BRIDGE_TOPICS[@]}" \
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
        super().__init__("lighting_moving_camera_info_capture")
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
    raise RuntimeError(
        f"No CameraInfo received from {topic}"
    )

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

mkdir -p \
  "$RAW/moving" \
  "$RAW/ap1_metadata" \
  "$META"

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
    row["image"] = str(
        raw / "moving" / f"frame_{frame:04d}.png"
    )

destination.parent.mkdir(parents=True, exist_ok=True)

with destination.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote normalized route: {destination}")
PY

cp \
  "$META/route_commanded.csv" \
  "$RAW/ap1_metadata/route_commanded.csv"

cp \
  src/calib_lab/bus_real_data/worlds/lighting/LIGHTING_VARIANTS.json \
  "$META/LIGHTING_VARIANTS.json"

cp \
  "$WORLD" \
  "$META/$(basename "$WORLD")"

EXPECTED_MOVING="$(
  python3 - "$ROUTE" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(len(data["frames"]))
PY
)"

STATIC_COUNT="$(
  find "$RAW/static" -maxdepth 1 -name 'cam_edge_*.png' \
  | wc -l
)"

MOVING_COUNT="$(
  find "$RAW/moving" -maxdepth 1 -name 'frame_*.png' \
  | wc -l
)"

INFO_COUNT="$(
  find "$RAW/camera_info" -maxdepth 1 -name '*.json' \
  | wc -l
)"

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
echo "[OK] Lighting capture complete: $VARIANT"
