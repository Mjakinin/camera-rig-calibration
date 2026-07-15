#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SOURCE_DIR="/mnt/c/Users/maxim/Desktop/Application of Robotics and Autonomous Systems/appras_sose26/1x"
CHECKERBOARD_VIDEO=""
DRIVE_VIDEOS=()
GPU=0
MATCHER="exhaustive"
MAX_IMAGE_SIZE=2400
PREPARE_ONLY=0
OVERWRITE=1

usage() {
  cat <<'EOF'
Usage:
  bash run/real_vehicle_data/run_1x_two_video_pipeline.sh [options]

Defaults match the current Windows folder and filenames:
  source directory: .../appras_sose26/1x
  checkerboard:      IMG_4364.mov
  drive videos:      IMG_4317.mov and IMG_4318.mov

Options:
  --source-dir PATH
  --checkerboard PATH
  --drive-video PATH       Repeat for additional independent videos.
  --gpu 0|1
  --matcher exhaustive|sequential
  --max-image-size PIXELS
  --prepare-only           Calibrate, extract 3 Hz frames and detect ArUco only.
  --no-overwrite           Refuse to replace an existing 00_shared_input.
  -h, --help

Each drive video is processed as an independent dataset. This avoids silently
joining two routes into one COLMAP sequence and lets the final report compare
the two recordings separately.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --checkerboard)
      CHECKERBOARD_VIDEO="$2"
      shift 2
      ;;
    --drive-video)
      DRIVE_VIDEOS+=("$2")
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --matcher)
      MATCHER="$2"
      shift 2
      ;;
    --max-image-size)
      MAX_IMAGE_SIZE="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$GPU" in
  0|1) ;;
  *)
    echo "[ERROR] --gpu must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$MATCHER" in
  exhaustive|sequential) ;;
  *)
    echo "[ERROR] --matcher must be exhaustive or sequential" >&2
    exit 2
    ;;
esac

if [[ -z "$CHECKERBOARD_VIDEO" ]]; then
  CHECKERBOARD_VIDEO="$SOURCE_DIR/IMG_4364.mov"
fi
if [[ ${#DRIVE_VIDEOS[@]} -eq 0 ]]; then
  DRIVE_VIDEOS=(
    "$SOURCE_DIR/IMG_4317.mov"
    "$SOURCE_DIR/IMG_4318.mov"
  )
fi

for path in "$CHECKERBOARD_VIDEO" "${DRIVE_VIDEOS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Video not found: $path" >&2
    exit 1
  fi
done

for command in python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $command" >&2
    exit 127
  fi
done

python3 - <<'PY'
import cv2
import numpy
print("[OK] Python dependencies: OpenCV and NumPy")
PY

INTRINSIC_WORK="results/real_vehicle_data/real_1x_4k_intrinsics_v1"
INTRINSIC_JSON="$INTRINSIC_WORK/moving_calib_camera.json"

printf '\n%s\n' "================================================================================"
echo "1X INTRINSIC CALIBRATION"
printf '%s\n' "================================================================================"
echo "Checkerboard video: $CHECKERBOARD_VIDEO"
echo

python3 run/real_vehicle_data/02_calibrate_intrinsics_and_archive.py \
  --video "$CHECKERBOARD_VIDEO" \
  --out "$INTRINSIC_WORK" \
  --result-name "iphone_1x_4k"

if [[ ! -f "$INTRINSIC_JSON" ]]; then
  echo "[ERROR] Intrinsic calibration did not produce: $INTRINSIC_JSON" >&2
  exit 1
fi

RUN_ROOTS=()

for video in "${DRIVE_VIDEOS[@]}"; do
  stem="$(basename "$video")"
  stem="${stem%.*}"
  safe_stem="$(printf '%s' "$stem" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  dataset_name="real_1x_4k_3hz_${safe_stem}_v1"
  result_root="results/real_vehicle_data/$dataset_name"
  shared_input="$result_root/00_shared_input"
  observations="$shared_input/aruco_observations"

  RUN_ROOTS+=("$result_root")

  printf '\n%s\n' "================================================================================"
  echo "PREPARE $dataset_name"
  printf '%s\n' "================================================================================"
  echo "Drive video: $video"

  prepare_args=(
    --video "$video"
    --intrinsics-json "$INTRINSIC_JSON"
    --dataset-name "$dataset_name"
    --sampling-hz 3
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi

  python3 run/real_vehicle_data/03_prepare_real_moving_video_dataset.py \
    "${prepare_args[@]}"

  printf '\n%s\n' "--------------------------------------------------------------------------------"
  echo "ARUCO OBSERVATIONS AND DEBUG IMAGES"
  printf '%s\n' "--------------------------------------------------------------------------------"

  python3 run/real_vehicle_data/01_detect_moving_aruco_from_raw.py \
    --dataset "$shared_input" \
    --observations-root "$observations"

  python3 run/real_vehicle_data/00_validate_and_prepare_shared_input.py \
    --dataset "$shared_input" \
    --deep-images

  echo
  echo "Debug contact sheet:"
  echo "$observations/debug_images/MOVING_ARUCO_DEBUG_CONTACT_SHEET.jpg"

  if [[ "$PREPARE_ONLY" == "0" ]]; then
    printf '\n%s\n' "--------------------------------------------------------------------------------"
    echo "AP01 / AP02 / AP03 / FINAL REPORT"
    printf '%s\n' "--------------------------------------------------------------------------------"

    bash run/real_vehicle_data/run_full_real_pipeline.sh \
      --dataset "$shared_input" \
      --results-root "$result_root" \
      --observations-root "$observations" \
      --gpu "$GPU" \
      --matcher "$MATCHER" \
      --max-image-size "$MAX_IMAGE_SIZE"
  fi
done

printf '\n%s\n' "================================================================================"
echo "1X TWO-VIDEO PROCESSING COMPLETE"
printf '%s\n' "================================================================================"
echo "Intrinsic result:"
echo "  $INTRINSIC_WORK/INTRINSICS_REPORT.txt"
echo "  results/real_vehicle_data/INTRINSIC_RESULTS/"
echo
for result_root in "${RUN_ROOTS[@]}"; do
  echo "Dataset: $result_root"
  echo "  Debug:  $result_root/00_shared_input/aruco_observations/debug_images/MOVING_ARUCO_DEBUG_CONTACT_SHEET.jpg"
  if [[ "$PREPARE_ONLY" == "0" ]]; then
    echo "  Report: $result_root/99_FINAL_RESULTS/REAL_DATA_ALL_METHODS.txt"
  fi
done
