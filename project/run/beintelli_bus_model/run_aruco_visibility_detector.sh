#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-gui}"
OUT_DIR="${2:-results/beintelli_bus_model/aruco_visibility/current}"

WORLD="src/calib_lab/beintelli_bus_model/worlds/bus_individual_marker_visibility_test.sdf"
DETECTOR="src/calib_lab/beintelli_bus_model/scripts/bus_aruco_visibility_detector.py"

export IGN_GAZEBO_RESOURCE_PATH="$PROJECT_DIR/src/calib_lab/beintelli_bus_model/models:$PROJECT_DIR/src/calib_lab/beintelli_bus_model/models/aruco_individual:$PROJECT_DIR/src/calib_lab/minimal_world/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

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
  ign gazebo "$WORLD" -r -v 4 &
fi

GAZEBO_PID=$!
sleep 6

ros2 run ros_gz_image image_bridge \
  /front_static_camera/image \
  /rear_static_camera/image &
BRIDGE_PID=$!

sleep 3

python3 "$DETECTOR" \
  --dictionary DICT_4X4_50 \
  --output_dir "$OUT_DIR" \
  --wait_sec 5

echo "[OK] Results:"
echo "     $OUT_DIR/bus_aruco_visibility_summary.txt"
cat "$OUT_DIR/bus_aruco_visibility_summary.txt" || true

kill "$BRIDGE_PID" 2>/dev/null || true
kill "$GAZEBO_PID" 2>/dev/null || true
