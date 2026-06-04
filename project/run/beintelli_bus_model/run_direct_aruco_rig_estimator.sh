#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-gui}"
OUT_DIR="${2:-results/beintelli_bus_model/aruco_direct_rig/pitch40_fov90_baseline}"
MARKER_SIZE_M="${3:-0.60}"

WORLD="src/calib_lab/beintelli_bus_model/worlds/bus_individual_marker_visibility_test.sdf"
ESTIMATOR="src/calib_lab/beintelli_bus_model/scripts/bus_aruco_direct_rig_estimator.py"

export IGN_GAZEBO_RESOURCE_PATH="$PROJECT_DIR/src/calib_lab/beintelli_bus_model/models:$PROJECT_DIR/src/calib_lab/beintelli_bus_model/models/aruco_individual:$PROJECT_DIR/src/calib_lab/minimal_world/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

cleanup() {
  for PID in ${BRIDGE_PID:-}; do kill "$PID" 2>/dev/null || true; done
  kill "$GAZEBO_PID" 2>/dev/null || true
}
trap cleanup EXIT

pkill -9 -f "ign gazebo" || true
pkill -9 -f "ign-gazebo" || true
pkill -9 -f "ruby.*ign" || true
pkill -9 -f "ros2 run ros_gz_image" || true
pkill -9 -f "ros2 run ros_gz_bridge" || true

source /opt/ros/humble/setup.bash

if [ "$MODE" = "headless" ]; then
  ign gazebo -s "$WORLD" -r -v 2 &
else
  export DISPLAY="${DISPLAY:-:0}"
  export QT_X11_NO_MITSHM=1
  ign gazebo "$WORLD" -r -v 3 &
fi

GAZEBO_PID=$!
sleep 6

ros2 run ros_gz_image image_bridge \
  /front_static_camera/image \
  /rear_static_camera/image &
IMAGE_BRIDGE_PID=$!

ros2 run ros_gz_bridge parameter_bridge \
  /front_static_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /rear_static_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo &
INFO_BRIDGE_PID=$!

BRIDGE_PID="$IMAGE_BRIDGE_PID $INFO_BRIDGE_PID"

sleep 3

python3 "$ESTIMATOR" \
  --dictionary DICT_4X4_50 \
  --marker_size_m "$MARKER_SIZE_M" \
  --horizontal_fov_deg 90.0 \
  --output_dir "$OUT_DIR" \
  --wait_sec 5

echo ""
echo "[OK] Direct ArUco rig result:"
echo "     $OUT_DIR/direct_aruco_rig_summary.txt"
cat "$OUT_DIR/direct_aruco_rig_summary.txt" || true
