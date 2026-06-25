#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

VARIANT="${1:-}"
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"

if [ -z "$VARIANT" ]; then
  echo "Usage:"
  echo "  ./run/bus_real_data/ablation/moving_cam_res/02_capture_one_res_variant.sh res_640x360"
  echo
  echo "Variants:"
  echo "  res_640x360"
  echo "  res_960x540"
  echo "  res_1280x720_baseline"
  echo "  res_1920x1080"
  exit 1
fi

WORLD_ROOT="src/calib_lab/bus_real_data/worlds/ablation/moving_cam_res"
OUT_ROOT="results/bus_real_data/ablation/moving_cam_res/00_captures"
LOG_ROOT="$OUT_ROOT/_logs"

CAPTURE_SCRIPT="run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"
CAMERA_NAME="moving_calib_camera"
IMAGE_TOPIC="/bus_real_data/moving_calib_camera/image"

WORLD_FILE="$WORLD_ROOT/bus_real_data_moving_camera_${VARIANT}.sdf"
OUT_DIR="$OUT_ROOT/$VARIANT"

GZ_LOG="$LOG_ROOT/${VARIANT}_gazebo.log"
BRIDGE_LOG="$LOG_ROOT/${VARIANT}_bridge.log"
CAPTURE_LOG="$LOG_ROOT/${VARIANT}_capture.log"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$WORLD_FILE" ]; then
  echo "[ERROR] missing world file:"
  echo "  $WORLD_FILE"
  exit 1
fi

echo "[INFO] cleaning old Gazebo / bridge processes..."
pkill -f "ign gazebo" || true
pkill -f "gz sim" || true
pkill -f "ros_gz_bridge" || true
pkill -f "parameter_bridge" || true
sleep 2

if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u 2>/dev/null || true
fi

if [ -f "install/setup.bash" ]; then
  set +u
  source "install/setup.bash"
  set -u 2>/dev/null || true
fi

GZ_PID=""
BRIDGE_PID=""

cleanup() {
  set +e
  echo "[INFO] cleanup..."
  if [ -n "$BRIDGE_PID" ]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [ -n "$GZ_PID" ]; then
    kill "$GZ_PID" 2>/dev/null || true
    wait "$GZ_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_world_name() {
  local timeout="${1:-60}"
  local start
  start="$(date +%s)"

  while true; do
    local services
    services="$(ign service -l 2>/dev/null || true)"

    local world
    world="$(echo "$services" | sed -n 's#^/world/\([^/]*\)/set_pose$#\1#p' | head -n 1)"

    if [ -n "$world" ]; then
      echo "$world"
      return 0
    fi

    if [ $(( $(date +%s) - start )) -gt "$timeout" ]; then
      echo "[ERROR] timed out waiting for Gazebo world set_pose service." >&2
      echo "[DEBUG] Gazebo log:" >&2
      tail -80 "$GZ_LOG" >&2 || true
      return 1
    fi

    sleep 1
  done
}

wait_for_ros_topic() {
  local topic="$1"
  local timeout="${2:-60}"
  local start
  start="$(date +%s)"

  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi

    if [ $(( $(date +%s) - start )) -gt "$timeout" ]; then
      echo "[ERROR] timed out waiting for ROS topic: $topic" >&2
      echo "[DEBUG] bridge log:" >&2
      tail -80 "$BRIDGE_LOG" >&2 || true
      return 1
    fi

    sleep 1
  done
}

wait_for_one_image() {
  local topic="$1"
  echo "[INFO] waiting for one actual ROS image message..."

  if timeout 45 ros2 topic echo --once "$topic" --field header >/dev/null 2>&1; then
    echo "[OK] received image message."
    return 0
  fi

  echo "[WARN] --field header failed; trying full echo once..."
  if timeout 45 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    echo "[OK] received image message."
    return 0
  fi

  echo "[ERROR] no actual image message received on $topic"
  echo "[DEBUG] ROS topics:"
  ros2 topic list || true
  echo "[DEBUG] Gazebo log:"
  tail -80 "$GZ_LOG" || true
  echo "[DEBUG] bridge log:"
  tail -80 "$BRIDGE_LOG" || true
  return 1
}

echo
echo "============================================================"
echo "[RUN] moving_cam_res single variant: $VARIANT"
echo "============================================================"
echo "[INFO] world:"
echo "  $WORLD_FILE"
echo "[INFO] output:"
echo "  $OUT_DIR"

echo
echo "[INFO] starting Gazebo..."
ign gazebo -r "$WORLD_FILE" > "$GZ_LOG" 2>&1 &
GZ_PID=$!

echo "[INFO] waiting a few seconds for Gazebo GUI/server startup..."
sleep 8

WORLD_NAME="$(wait_for_world_name 60)"
echo "[OK] detected Gazebo world: $WORLD_NAME"

echo
echo "[INFO] starting bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  "${IMAGE_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image" \
  > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

wait_for_ros_topic "$IMAGE_TOPIC" 60
echo "[OK] ROS topic exists: $IMAGE_TOPIC"

wait_for_one_image "$IMAGE_TOPIC"

echo
echo "[READY] Gazebo and image bridge look ready."

if [ "$AUTO_CONFIRM" = "1" ]; then
  echo "[INFO] AUTO_CONFIRM=1, starting capture automatically."
else
  echo "Press ENTER to start capture for $VARIANT."
  read -r _
fi

echo "[INFO] starting route capture..."
python3 "$CAPTURE_SCRIPT" \
  --world "$WORLD_NAME" \
  --name "$CAMERA_NAME" \
  --out "$OUT_DIR" \
  --clean \
  2>&1 | tee "$CAPTURE_LOG"

FRAME_COUNT="$(find "$OUT_DIR/images" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l || true)"

echo
echo "[OK] capture finished."
echo "[INFO] frames captured: $FRAME_COUNT"
echo "[INFO] output: $OUT_DIR"

if [ "$FRAME_COUNT" -lt 10 ]; then
  echo "[WARN] very few frames captured. Check logs:"
  echo "  $GZ_LOG"
  echo "  $BRIDGE_LOG"
  echo "  $CAPTURE_LOG"
  exit 2
fi

echo
echo "[DONE] $VARIANT"
