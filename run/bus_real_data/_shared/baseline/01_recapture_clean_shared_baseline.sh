#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! command -v ign >/dev/null 2>&1; then
    echo "[ERROR] ign is required by the route capture."
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

STAMP="$(date +%Y%m%d_%H%M%S)"

SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"

STAGE_ROOT="results/bus_real_data/_capture_staging/shared_baseline_${STAMP}"
STAGE_RAW="$STAGE_ROOT/raw_images"
STAGE_META="$STAGE_ROOT/metadata"
STAGE_LOG="$STAGE_ROOT/logs"
ROUTE_CAPTURE="$STAGE_ROOT/_moving_route_capture"

BACKUP_DIR="results/bus_real_data/_baseline_backups"
BACKUP_TAR="$BACKUP_DIR/shared_baseline_before_recapture_${STAMP}.tar"

WORLD="$STAGE_ROOT/bus_real_data_moving_camera_clean_HEAD.sdf"
ROUTE="src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json"

STATIC_CAPTURE="run/bus_real_data/approach2_ref_marker_graph_ba/01_capture_shared_raw_dataset.py"
MOVING_CAPTURE="run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"

mkdir -p \
  "$STAGE_RAW" \
  "$STAGE_META" \
  "$STAGE_LOG" \
  "$BACKUP_DIR"

git show \
  HEAD:src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf \
  > "$WORLD"

if grep -qiE \
  'motion[_ -]?blur|GaussianBlur|filter2D|kernel_size' \
  "$WORLD"
then
    echo "[ERROR] Unexpected blur reference in clean world."
    exit 1
fi

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
echo "CLEAN SHARED BASELINE RECAPTURE"
echo "stage:  $STAGE_ROOT"
echo "shared: $SHARED"
echo "world:  committed HEAD standard world"
echo "================================================================================"

ign gazebo -r -s "$WORLD" \
  > "$STAGE_LOG/gazebo.log" 2>&1 &

GZ_PID=$!

WORLD_NAME=""

for _ in $(seq 1 60); do
    WORLD_NAME="$(
      ign service -l 2>/dev/null \
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
    tail -100 "$STAGE_LOG/gazebo.log" || true
    exit 1
fi

echo "[OK] world name: $WORLD_NAME"

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

ros2 run ros_gz_bridge parameter_bridge \
  "${BRIDGE_TOPICS[@]}" \
  > "$STAGE_LOG/bridge.log" 2>&1 &

BRIDGE_PID=$!

REQUIRED_TOPICS=(
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

for topic in "${REQUIRED_TOPICS[@]}"; do
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
        tail -100 "$STAGE_LOG/bridge.log" || true
        exit 1
    fi

    echo "[OK] topic: $topic"
done

echo
echo "=== Capture four static cameras ==="

python3 "$STATIC_CAPTURE" \
  --out "$STAGE_RAW" \
  --static-only \
  --overwrite \
  2>&1 | tee "$STAGE_LOG/static_capture.log"

echo
echo "=== Capture moving-camera CameraInfo ==="

python3 - "$STAGE_RAW/camera_info/moving_calib_camera.json" <<'PY'
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo


TOPIC = "/bus_real_data/moving_calib_camera/camera_info"
OUTPUT = Path(sys.argv[1])


class Grabber(Node):
    def __init__(self):
        super().__init__("shared_baseline_moving_camera_info")
        self.message = None

        self.create_subscription(
            CameraInfo,
            TOPIC,
            self.callback,
            qos_profile_sensor_data,
        )

    def callback(self, message):
        self.message = message


rclpy.init()
node = Grabber()

deadline = time.time() + 30.0

while (
    rclpy.ok()
    and node.message is None
    and time.time() < deadline
):
    rclpy.spin_once(node, timeout_sec=0.1)

message = node.message

node.destroy_node()
rclpy.shutdown()

if message is None:
    raise RuntimeError(
        f"No CameraInfo received from {TOPIC}"
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

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(data, indent=2) + "\n"
)

print(f"[OK] wrote {OUTPUT}")
PY

echo
echo "=== Capture complete moving-camera route ==="

python3 "$MOVING_CAPTURE" \
  --route "$ROUTE" \
  --world "$WORLD_NAME" \
  --name moving_calib_camera \
  --out "$ROUTE_CAPTURE" \
  --settle 0.35 \
  --timeout 3.0 \
  --clean \
  2>&1 | tee "$STAGE_LOG/moving_capture.log"

mkdir -p \
  "$STAGE_RAW/moving" \
  "$STAGE_RAW/ap1_metadata" \
  "$STAGE_META"

cp -a \
  "$ROUTE_CAPTURE/images/." \
  "$STAGE_RAW/moving/"

python3 - \
  "$ROUTE_CAPTURE/route_commanded.csv" \
  "$STAGE_META/route_commanded.csv" \
  "$STAGE_RAW" <<'PY'
import csv
import sys
from pathlib import Path


source = Path(sys.argv[1])
destination = Path(sys.argv[2])
raw_root = Path(sys.argv[3])

rows = list(csv.DictReader(source.open()))

if not rows:
    raise RuntimeError(
        "route_commanded.csv has no rows"
    )

fields = list(rows[0])

for row in rows:
    frame = int(row["frame"])

    row["image"] = str(
        raw_root
        / "moving"
        / f"frame_{frame:04d}.png"
    )

destination.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with destination.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote {destination}")
PY

cp \
  "$STAGE_META/route_commanded.csv" \
  "$STAGE_RAW/ap1_metadata/route_commanded.csv"

COMMIT_SHA="$(git rev-parse HEAD)"

python3 - \
  "$STAGE_META/capture_metadata.json" \
  "$COMMIT_SHA" \
  "$WORLD" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


output = Path(sys.argv[1])

data = {
    "dataset": "bus_real_data_ref_marker_v1",
    "capture_type": "clean_shared_baseline_recapture",
    "source_commit": sys.argv[2],
    "world_source": (
        "HEAD:src/calib_lab/bus_real_data/worlds/"
        "bus_real_data_moving_camera.sdf"
    ),
    "staged_world": sys.argv[3],
    "moving_route": (
        "src/calib_lab/bus_real_data/config/"
        "moving_camera_route_interpolated.json"
    ),
    "moving_settle_seconds": 0.35,
    "software_motion_blur_applied": False,
    "captured_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

output.write_text(
    json.dumps(data, indent=2) + "\n"
)
PY

EXPECTED_MOVING="$(
  python3 - "$ROUTE" <<'PY'
import json
import sys
from pathlib import Path

print(
    len(
        json.loads(
            Path(sys.argv[1]).read_text()
        )["frames"]
    )
)
PY
)"

STATIC_COUNT="$(
  find "$STAGE_RAW/static" \
    -maxdepth 1 \
    -name 'cam_edge_*.png' \
    | wc -l
)"

MOVING_COUNT="$(
  find "$STAGE_RAW/moving" \
    -maxdepth 1 \
    -name 'frame_*.png' \
    | wc -l
)"

INFO_COUNT="$(
  find "$STAGE_RAW/camera_info" \
    -maxdepth 1 \
    -name '*.json' \
    | wc -l
)"

echo
echo "=== Capture audit ==="
echo "static images:       $STATIC_COUNT / 4"
echo "moving images:       $MOVING_COUNT / $EXPECTED_MOVING"
echo "camera-info files:   $INFO_COUNT / 5"

if [[ "$STATIC_COUNT" -ne 4 ]]; then
    echo "[ERROR] Expected four static images."
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

echo
echo "=== Old versus new sharpness diagnostic ==="

OLD_MOVING="$SHARED/raw_images/moving" \
NEW_MOVING="$STAGE_RAW/moving" \
python3 - <<'PY'
import os
from pathlib import Path
from statistics import median

import cv2


old_root = Path(os.environ["OLD_MOVING"])
new_root = Path(os.environ["NEW_MOVING"])


def index_images(root):
    result = {}

    if not root.is_dir():
        return result

    for path in root.glob("frame_*.png"):
        try:
            index = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue

        result[index] = path

    return result


def sharpness(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    return float(
        cv2.Laplacian(
            image,
            cv2.CV_64F,
        ).var()
    )


old = index_images(old_root)
new = index_images(new_root)
common = sorted(set(old) & set(new))

if not common:
    print("[INFO] No old corresponding images available.")
    raise SystemExit(0)

old_values = []
new_values = []

for index in common:
    old_value = sharpness(old[index])
    new_value = sharpness(new[index])

    if old_value is None or new_value is None:
        continue

    old_values.append(old_value)
    new_values.append(new_value)

old_median = median(old_values)
new_median = median(new_values)

print(f"common frames:             {len(old_values)}")
print(f"old median sharpness:      {old_median:.6f}")
print(f"new median sharpness:      {new_median:.6f}")
print(
    "new / old sharpness:      "
    f"{new_median / max(old_median, 1e-9):.6f}"
)
PY

echo
echo "=== Back up existing shared baseline ==="

if [[ -d "$SHARED" ]]; then
    tar \
      -C "$(dirname "$SHARED")" \
      -cf "$BACKUP_TAR" \
      "$(basename "$SHARED")"

    echo "[OK] backup: $BACKUP_TAR"
fi

echo
echo "=== Promote clean capture to shared baseline ==="

mkdir -p "$SHARED"

for component in \
  raw_images \
  aruco_observations \
  metadata
do
    if [[ -e "$SHARED/$component" ]]; then
        chmod -R u+rwX "$SHARED/$component" 2>/dev/null || true
        rm -rf "$SHARED/$component"
    fi
done

cp -a \
  "$STAGE_RAW" \
  "$SHARED/raw_images"

cp -a \
  "$STAGE_META" \
  "$SHARED/metadata"

cat > "$SHARED/README_RECAPTURE.txt" <<TXT
Clean shared baseline recapture
===============================

Captured from committed standard world:
HEAD:src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf

Moving frames: $MOVING_COUNT
Static images: $STATIC_COUNT
CameraInfo files: $INFO_COUNT
Moving settle seconds: 0.35
Software motion blur applied: false
Source commit: $COMMIT_SHA
Backup of previous baseline: $BACKUP_TAR
TXT

rm -rf "$ROUTE_CAPTURE"

echo
echo "[OK] Clean shared baseline installed:"
echo "     $SHARED"
echo
echo "[INFO] Staging data retained at:"
echo "       $STAGE_ROOT"
