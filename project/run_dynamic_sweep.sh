#!/usr/bin/env bash
set -u

# Platform-independent project root detection.
# This script is expected to live in the project/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$PROJECT_DIR" || {
  echo "[ERROR] Could not cd into PROJECT_DIR=$PROJECT_DIR"
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
IGN_BIN="${IGN_BIN:-ign}"
ROS2_BIN="${ROS2_BIN:-ros2}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

if [ $# -lt 2 ]; then
  echo "Usage:"
  echo "  ./run_dynamic_sweep.sh checkerboard|aruco|charuco res320x240|res640x480 [distance|yaw|shift|height|mixed|all]"
  exit 1
fi

METHOD="$1"
RES_NAME="$2"
REQUESTED_GROUP="${3:-all}"

if [ "$REQUESTED_GROUP" = "combination" ]; then
  REQUESTED_GROUP="mixed"
fi

for cmd in "$PYTHON_BIN" "$IGN_BIN" "$ROS2_BIN"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd"
    echo "[INFO] Check ROS 2 / Gazebo environment setup."
    exit 1
  fi
done

if ! command -v "$TIMEOUT_BIN" >/dev/null 2>&1; then
  echo "[WARN] Command '$TIMEOUT_BIN' not found. Evaluator timeout will be disabled."
  TIMEOUT_BIN=""
fi

export IGN_GAZEBO_RESOURCE_PATH="$PROJECT_DIR/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

WORLD_FILE="src/calib_lab/worlds/dynamic/${METHOD}_${RES_NAME}.sdf"
WORLD_NAME="dynamic_${METHOD}_${RES_NAME}"
POSE_CSV="src/calib_lab/worlds/dynamic/scenario_poses.csv"

case "$METHOD" in
  checkerboard)
    TARGET_NAME="target_9x6_square0_12"
    EVALUATOR="src/calib_lab/scripts/checkerboard/checkerboard_rig_evaluator.py"
    ;;
  aruco)
    TARGET_NAME="target_aruco_6x4_marker0_15_sep0_06"
    EVALUATOR="src/calib_lab/scripts/aruco/aruco_rig_evaluator.py"
    ;;
  charuco)
    TARGET_NAME="target_charuco_current"
    EVALUATOR="src/calib_lab/scripts/charuco/charuco_rig_evaluator.py"
    ;;
  *)
    echo "[ERROR] Unknown method: $METHOD"
    exit 1
    ;;
esac

RESULT_BASE="results/${METHOD}/${TARGET_NAME}/${RES_NAME}"
ANALYZER="src/calib_lab/scripts/tools/analyze_checkerboard_results.py"
SET_POSE="src/calib_lab/scripts/tools/set_gazebo_model_pose.py"
GAZEBO_LOG="${TMPDIR:-/tmp}/${METHOD}_${RES_NAME}_${REQUESTED_GROUP}_gazebo_dynamic.log"

is_valid_group() {
  case "$1" in
    distance|yaw|shift|height|mixed|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_run_group() {
  local scenario_group="$1"

  if [ "$REQUESTED_GROUP" = "all" ]; then
    return 0
  fi

  if [ "$REQUESTED_GROUP" = "$scenario_group" ]; then
    return 0
  fi

  return 1
}

prepare_group_folder() {
  local group="$1"
  local folder="${RESULT_BASE}/${group}"

  rm -rf "$folder"
  mkdir -p "${folder}/debug_images"
  mkdir -p "${folder}/evaluator_logs"
  echo "[INFO] Prepared clean folder: $folder"
}

cleanup() {
  pkill -9 -f "ros2 run ros_gz_image image_bridge" 2>/dev/null || true
  pkill -9 -f "ros2 run ros_gz_bridge parameter_bridge" 2>/dev/null || true
  pkill -9 -f "ign gazebo" 2>/dev/null || true
  pkill -9 -f "ign-gazebo" 2>/dev/null || true
  pkill -9 -f "ruby.*ign" 2>/dev/null || true
  sleep 2
}

if ! is_valid_group "$REQUESTED_GROUP"; then
  echo "[ERROR] Invalid group: $REQUESTED_GROUP"
  echo "[INFO] Valid groups: distance yaw shift height mixed all"
  exit 1
fi

if [ ! -f "$EVALUATOR" ]; then
  echo "[ERROR] Evaluator not found: $EVALUATOR"
  echo "[INFO] Method '$METHOD' is not implemented yet or the script is in another location."
  exit 1
fi

if [ ! -f "$WORLD_FILE" ]; then
  echo "[ERROR] Dynamic world not found: $WORLD_FILE"
  echo "[INFO] Generate it first. Example for checkerboard:"
  echo "  $PYTHON_BIN src/calib_lab/scripts/tools/generate_dynamic_worlds.py --method checkerboard --resolution $RES_NAME --target_uri model://checkerboard_target"
  echo "[INFO] Example for aruco:"
  echo "  $PYTHON_BIN src/calib_lab/scripts/tools/generate_dynamic_worlds.py --method aruco --resolution $RES_NAME --target_uri model://aruco_target"
  exit 1
fi

if [ ! -f "$POSE_CSV" ]; then
  echo "[ERROR] Pose CSV not found: $POSE_CSV"
  echo "[INFO] Regenerate dynamic worlds with generate_dynamic_worlds.py."
  exit 1
fi

HEADER="$(head -n 1 "$POSE_CSV" | tr -d '\r')"
if [ "$HEADER" != "scenario,group,x,y,z,roll,pitch,yaw" ]; then
  echo "[ERROR] scenario_poses.csv has wrong header:"
  echo "$HEADER"
  echo "[INFO] Regenerate it with generate_dynamic_worlds.py."
  exit 1
fi

trap cleanup EXIT

echo "[INFO] Project dir:  $PROJECT_DIR"
echo "[INFO] Method:       $METHOD"
echo "[INFO] Target:       $TARGET_NAME"
echo "[INFO] Resolution:   $RES_NAME"
echo "[INFO] Group:        $REQUESTED_GROUP"
echo "[INFO] Result base:  $RESULT_BASE"
echo "[INFO] Gazebo log:   $GAZEBO_LOG"

echo "[INFO] Cleaning old processes..."
cleanup

mkdir -p "$RESULT_BASE"

echo "[INFO] Clearing requested result folder(s)..."
if [ "$REQUESTED_GROUP" = "all" ]; then
  prepare_group_folder "distance"
  prepare_group_folder "yaw"
  prepare_group_folder "shift"
  prepare_group_folder "height"
  prepare_group_folder "mixed"
else
  prepare_group_folder "$REQUESTED_GROUP"
fi

echo "[INFO] Starting one Gazebo instance:"
echo "[INFO] $WORLD_FILE"
"$IGN_BIN" gazebo -s "$WORLD_FILE" -r -v 1 > "$GAZEBO_LOG" 2>&1 &

echo "[INFO] Waiting for set_pose service..."
SET_POSE_SERVICE=""

for _ in $(seq 1 30); do
  SET_POSE_SERVICE="$($IGN_BIN service -l 2>/dev/null | grep -E "/world/${WORLD_NAME}/set_pose$" | head -n 1 || true)"

  if [ -n "$SET_POSE_SERVICE" ]; then
    echo "[INFO] Found set_pose service: $SET_POSE_SERVICE"
    break
  fi

  sleep 1
done

if [ -z "$SET_POSE_SERVICE" ]; then
  echo "[ERROR] No set_pose service found for world: $WORLD_NAME"
  echo "[DEBUG] Available set_pose services:"
  "$IGN_BIN" service -l | grep set_pose || true
  echo "[DEBUG] Gazebo log:"
  tail -n 120 "$GAZEBO_LOG" || true
  exit 1
fi

echo "[INFO] Starting bridges once..."
"$ROS2_BIN" run ros_gz_image image_bridge /camera_1/image /camera_2/image > /dev/null 2>&1 &

"$ROS2_BIN" run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock \
  > /dev/null 2>&1 &

sleep 5

echo "[INFO] Running dynamic sweep..."

tail -n +2 "$POSE_CSV" | while IFS=, read -r SCENARIO SCENARIO_GROUP X Y Z ROLL PITCH YAW; do
  SCENARIO_GROUP="$(echo "$SCENARIO_GROUP" | tr -d '\r')"

  if [ -z "$SCENARIO" ]; then
    continue
  fi

  if ! is_valid_group "$SCENARIO_GROUP"; then
    echo "[ERROR] Invalid scenario group from CSV: '$SCENARIO_GROUP' for scenario '$SCENARIO'"
    echo "[ERROR] This would create wrong folders. Aborting."
    exit 1
  fi

  if ! should_run_group "$SCENARIO_GROUP"; then
    continue
  fi

  RESULT_ROOT="${RESULT_BASE}/${SCENARIO_GROUP}"
  OUT_CSV="${RESULT_ROOT}/raw_results.csv"
  DEBUG_DIR="${RESULT_ROOT}/debug_images"
  LOG_DIR="${RESULT_ROOT}/evaluator_logs"

  echo ""
  echo "============================================================"
  echo "[INFO] Scenario: $SCENARIO"
  echo "[INFO] Group:    $SCENARIO_GROUP"
  echo "[INFO] Pose:     x=$X y=$Y z=$Z yaw=$YAW"
  echo "[INFO] Output:   $RESULT_ROOT"
  echo "============================================================"

  "$PYTHON_BIN" "$SET_POSE" \
    --service "$SET_POSE_SERVICE" \
    --model calibration_target \
    --x "$X" --y "$Y" --z "$Z" \
    --roll "$ROLL" --pitch "$PITCH" --yaw "$YAW" \
    > /dev/null

  sleep 1

  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" 120s "$PYTHON_BIN" "$EVALUATOR" \
      --ros-args \
      -p scenario_name:="$SCENARIO" \
      -p max_valid_samples:=1 \
      -p max_attempts:=1 \
      -p ready_timeout_sec:=20.0 \
      -p output_csv:="$OUT_CSV" \
      -p debug_dir:="$DEBUG_DIR" \
      > "${LOG_DIR}/${SCENARIO}_evaluator.log" 2>&1
  else
    "$PYTHON_BIN" "$EVALUATOR" \
      --ros-args \
      -p scenario_name:="$SCENARIO" \
      -p max_valid_samples:=1 \
      -p max_attempts:=1 \
      -p ready_timeout_sec:=20.0 \
      -p output_csv:="$OUT_CSV" \
      -p debug_dir:="$DEBUG_DIR" \
      > "${LOG_DIR}/${SCENARIO}_evaluator.log" 2>&1
  fi

  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    echo "[OK] $SCENARIO completed"
  elif [ "$STATUS" -eq 124 ]; then
    echo "[WARN] $SCENARIO timed out"
  else
    echo "[WARN] $SCENARIO returned status $STATUS"
  fi
done

echo ""
echo "[INFO] Running analysis for requested groups..."

for G in distance yaw shift height mixed; do
  if ! should_run_group "$G"; then
    continue
  fi

  RESULT_ROOT="${RESULT_BASE}/${G}"
  OUT_CSV="${RESULT_ROOT}/raw_results.csv"
  SUMMARY_CSV="${RESULT_ROOT}/summary.csv"

  if [ ! -f "$OUT_CSV" ]; then
    echo "[WARN] No raw_results.csv found for group: $G"
    continue
  fi

  if [ -f "$ANALYZER" ]; then
    "$PYTHON_BIN" "$ANALYZER" \
      --input_csv "$OUT_CSV" \
      --output_csv "$SUMMARY_CSV" \
      --max_error_cm 10 \
      --max_rot_deg 10 \
      > "${RESULT_ROOT}/analysis_printout.txt"

    echo ""
    echo "================ ANALYSIS: $G ================"
    cat "${RESULT_ROOT}/analysis_printout.txt"
    echo "=============================================="
  fi

  echo "[INFO] Raw CSV:     $OUT_CSV"
  echo "[INFO] Summary CSV: $SUMMARY_CSV"
  echo "[INFO] Debug dir:   ${RESULT_ROOT}/debug_images"
  echo "[INFO] Logs dir:    ${RESULT_ROOT}/evaluator_logs"
done

echo ""
echo "[DONE] Dynamic sweep finished."
