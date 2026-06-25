#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

VARIANTS=(
  "res_640x360"
  "res_960x540"
  "res_1280x720_baseline"
  "res_1920x1080"
)

WORLD_ROOT="src/calib_lab/bus_real_data/worlds/ablation/moving_cam_res"
OUT_ROOT="results/bus_real_data/ablation/moving_cam_res/00_captures"
LOG_ROOT="results/bus_real_data/ablation/moving_cam_res/00_captures/_logs"

CAPTURE_SCRIPT="run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"
CAMERA_NAME="moving_calib_camera"
IMAGE_TOPIC="/bus_real_data/moving_calib_camera/image"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [ -f "install/setup.bash" ]; then
  source "install/setup.bash"
fi

cleanup() {
  set +e
  if [ -n "${BRIDGE_PID:-}" ]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [ -n "${GZ_PID:-}" ]; then
    kill "$GZ_PID" 2>/dev/null || true
    wait "$GZ_PID" 2>/dev/null || true
  fi
  BRIDGE_PID=""
  GZ_PID=""
}
trap cleanup EXIT

wait_for_gazebo_world() {
  local timeout="${1:-30}"
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
      echo "[ERROR] timed out waiting for Gazebo /world/<name>/set_pose service" >&2
      echo "[DEBUG] current ign services:" >&2
      ign service -l 2>/dev/null | head -100 >&2 || true
      return 1
    fi

    sleep 1
  done
}

wait_for_ros_topic() {
  local topic="$1"
  local timeout="${2:-30}"
  local start
  start="$(date +%s)"

  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi

    if [ $(( $(date +%s) - start )) -gt "$timeout" ]; then
      echo "[ERROR] timed out waiting for ROS topic: $topic" >&2
      echo "[DEBUG] current ROS topics:" >&2
      ros2 topic list 2>/dev/null >&2 || true
      return 1
    fi

    sleep 1
  done
}

for variant in "${VARIANTS[@]}"; do
  echo
  echo "============================================================"
  echo "[RUN] moving_cam_res variant: $variant"
  echo "============================================================"

  WORLD_FILE="$WORLD_ROOT/bus_real_data_moving_camera_${variant}.sdf"
  OUT_DIR="$OUT_ROOT/$variant"
  GZ_LOG="$LOG_ROOT/${variant}_gazebo.log"
  BRIDGE_LOG="$LOG_ROOT/${variant}_bridge.log"
  CAPTURE_LOG="$LOG_ROOT/${variant}_capture.log"

  if [ ! -f "$WORLD_FILE" ]; then
    echo "[ERROR] missing world file: $WORLD_FILE"
    exit 1
  fi

  cleanup

  echo "[INFO] starting Gazebo:"
  echo "       $WORLD_FILE"

  # Normal mode with rendering. This is usually safer for camera sensors than pure server-only mode.
  ign gazebo -r "$WORLD_FILE" > "$GZ_LOG" 2>&1 &
  GZ_PID=$!

  WORLD_NAME="$(wait_for_gazebo_world 45)"
  echo "[INFO] detected Gazebo world: $WORLD_NAME"

  echo "[INFO] starting ROS-GZ bridge for $IMAGE_TOPIC"
  ros2 run ros_gz_bridge parameter_bridge \
    "${IMAGE_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image" \
    > "$BRIDGE_LOG" 2>&1 &
  BRIDGE_PID=$!

  wait_for_ros_topic "$IMAGE_TOPIC" 45
  echo "[INFO] ROS image topic is available."

  echo "[INFO] capturing route to: $OUT_DIR"
  python3 "$CAPTURE_SCRIPT" \
    --world "$WORLD_NAME" \
    --name "$CAMERA_NAME" \
    --out "$OUT_DIR" \
    --clean \
    > "$CAPTURE_LOG" 2>&1

  echo "[OK] capture finished: $OUT_DIR"

  cleanup

  echo "[INFO] quick frame count:"
  find "$OUT_DIR/images" -maxdepth 1 -name 'frame_*.png' | wc -l
done

trap - EXIT
cleanup

echo
echo "[OK] all moving_cam_res captures finished."
echo "[INFO] captures:"
find "$OUT_ROOT" -maxdepth 2 -type d | sort
