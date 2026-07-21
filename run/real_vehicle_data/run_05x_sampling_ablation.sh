#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"

SOURCE_DIR="/mnt/c/Users/maxim/Desktop/Application of Robotics and Autonomous Systems/appras_sose26/0.5"
DRIVE_VIDEO=""
INTRINSICS_JSON="results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/raw_images/camera_info/moving_calib_camera.json"
TEMPLATE_DATASET="results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input"
RESULTS_ROOT="results/real_vehicle_data"
RATES=(1 5)
CUSTOM_RATES=0
GPU=0
MATCHER="exhaustive"
MAX_IMAGE_SIZE=2400
AP02_MAX_NFEV_STATIC=80
AP02_MAX_NFEV_MOVING=120
OVERWRITE=1

usage() {
  cat <<'EOF'
Usage:
  bash run/real_vehicle_data/run_05x_sampling_ablation.sh [options]

Purpose:
  Build independent 0.5x datasets from the original moving-camera video and run
  AP01/AP02/AP03 plus the common marker-consistency evaluation. The default
  temporal sampling rates are 1 Hz and 5 Hz. The existing 3 Hz baseline is not
  overwritten.

Options:
  --source-dir PATH
  --drive-video PATH
  --intrinsics-json PATH
  --template-dataset PATH
  --results-root PATH
  --rate HZ                    Repeat to replace the default 1 Hz and 5 Hz list.
  --gpu 0|1
  --matcher exhaustive|sequential
  --max-image-size PIXELS
  --ap02-max-nfev-static N     Default: 80
  --ap02-max-nfev-moving N     Default: 120
  --no-overwrite
  -h, --help

Video discovery:
  When --drive-video is omitted, the script searches --source-dir for one video
  that is not named like an intrinsic/checkerboard/calibration recording. A
  unique filename containing "0.5_60FPS" is preferred.

Result names:
  real_05x_4k_<rate>hz_ap02nfev<moving-limit>_v1

Important comparison rule:
  AP02 must use the same max_nfev values at every sampling rate being compared.
  The canonical 3 Hz result used 100/160, while this ablation defaults to 80/120.
  Therefore an exact AP02 1-vs-3-vs-5 Hz comparison requires adding --rate 3 so
  that all three rates are rerun with the same AP02 budget.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --drive-video)
      DRIVE_VIDEO="$2"
      shift 2
      ;;
    --intrinsics-json)
      INTRINSICS_JSON="$2"
      shift 2
      ;;
    --template-dataset)
      TEMPLATE_DATASET="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --rate)
      if [[ "$CUSTOM_RATES" == "0" ]]; then
        RATES=()
        CUSTOM_RATES=1
      fi
      RATES+=("$2")
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
    --ap02-max-nfev-static)
      AP02_MAX_NFEV_STATIC="$2"
      shift 2
      ;;
    --ap02-max-nfev-moving)
      AP02_MAX_NFEV_MOVING="$2"
      shift 2
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

for value in "$AP02_MAX_NFEV_STATIC" "$AP02_MAX_NFEV_MOVING"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] AP02 max_nfev values must be positive integers: $value" >&2
    exit 2
  fi
done

if [[ ${#RATES[@]} -eq 0 ]]; then
  echo "[ERROR] At least one --rate is required" >&2
  exit 2
fi

for rate in "${RATES[@]}"; do
  if ! python3 - "$rate" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
  then
    echo "[ERROR] Invalid positive sampling rate: $rate" >&2
    exit 2
  fi
done

if [[ -z "$DRIVE_VIDEO" ]]; then
  if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "[ERROR] Source directory not found: $SOURCE_DIR" >&2
    exit 1
  fi

  candidates=()
  preferred=()
  while IFS= read -r -d '' path; do
    name="$(basename "$path")"
    lower="${name,,}"
    if [[ "$lower" == *intrinsic* || "$lower" == *checker* || "$lower" == *calib* ]]; then
      continue
    fi
    candidates+=("$path")
    if [[ "$lower" == *0.5_60fps* || "$lower" == *05_60fps* ]]; then
      preferred+=("$path")
    fi
  done < <(
    find "$SOURCE_DIR" -maxdepth 1 -type f \
      \( -iname '*.mov' -o -iname '*.mp4' -o -iname '*.m4v' -o -iname '*.avi' \) \
      -print0
  )

  if [[ ${#preferred[@]} -eq 1 ]]; then
    DRIVE_VIDEO="${preferred[0]}"
  elif [[ ${#candidates[@]} -eq 1 ]]; then
    DRIVE_VIDEO="${candidates[0]}"
  else
    echo "[ERROR] Could not identify one unique 0.5x moving video in: $SOURCE_DIR" >&2
    echo "Candidates:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Pass --drive-video with the exact path." >&2
    exit 1
  fi
fi

for path in "$DRIVE_VIDEO" "$INTRINSICS_JSON"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file not found: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$TEMPLATE_DATASET" ]]; then
  echo "[ERROR] Template dataset not found: $TEMPLATE_DATASET" >&2
  exit 1
fi

# Maintain one stable, clean intrinsic result location. The same 0.5x intrinsic
# calibration is reused for every temporal sampling rate; sampling frequency does
# not change the lens, resolution, sensor crop or intrinsic camera model.
INTRINSIC_CATALOG="$RESULTS_ROOT/INTRINSIC_RESULTS/iphone_05x_4k"
mkdir -p "$INTRINSIC_CATALOG"
cp -f "$INTRINSICS_JSON" "$INTRINSIC_CATALOG/moving_calib_camera.json"
flat_report="$(find "$RESULTS_ROOT/INTRINSIC_RESULTS" -maxdepth 1 -type f -iname 'iphone_05x_4k*INTRINSICS_REPORT.txt' | sort | head -n 1 || true)"
if [[ -n "$flat_report" && -f "$flat_report" ]]; then
  cp -f "$flat_report" "$INTRINSIC_CATALOG/INTRINSICS_REPORT.txt"
fi
cat > "$INTRINSIC_CATALOG/SOURCE.txt" <<EOF
Canonical camera mode: iPhone 0.5x, native 4K
Canonical CameraInfo source: $INTRINSICS_JSON
Used unchanged for sampling rates: ${RATES[*]} Hz
Sampling frequency changes frame density only; it does not change intrinsics.
EOF

printf '\n%s\n' "================================================================================"
echo "0.5X TEMPORAL SAMPLING ABLATION"
printf '%s\n' "================================================================================"
echo "Moving video:             $DRIVE_VIDEO"
echo "Intrinsic CameraInfo:     $INTRINSICS_JSON"
echo "Stable intrinsic catalog: $INTRINSIC_CATALOG"
echo "Sampling rates:           ${RATES[*]} Hz"
echo "AP02 max_nfev static:     $AP02_MAX_NFEV_STATIC"
echo "AP02 max_nfev moving:     $AP02_MAX_NFEV_MOVING"
echo

FAILED=0
RUN_ROOTS=()

run_step() {
  local label="$1"
  local log="$2"
  shift 2

  mkdir -p "$(dirname "$log")"
  printf '\n%s\n' "--------------------------------------------------------------------------------"
  echo "$label"
  echo "LOG: $log"
  printf '%s\n' "--------------------------------------------------------------------------------"

  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if [[ "$code" -ne 0 ]]; then
    echo "[WARN] $label exited with code $code"
    FAILED=1
  else
    echo "[OK] $label"
  fi
  return 0
}

for rate in "${RATES[@]}"; do
  rate_label="$(python3 - "$rate" <<'PY'
import sys
x = float(sys.argv[1])
if x.is_integer():
    print(str(int(x)))
else:
    print((f"{x:g}").replace('.', 'p'))
PY
)"
  dataset_name="real_05x_4k_${rate_label}hz_ap02nfev${AP02_MAX_NFEV_MOVING}_v1"
  result_root="$RESULTS_ROOT/$dataset_name"
  shared_input="$result_root/00_shared_input"
  observations="$shared_input/aruco_observations"
  logs="$result_root/_pipeline_logs"
  RUN_ROOTS+=("$result_root")

  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf \
      "$result_root/02_ap01_real" \
      "$result_root/03_ap02_real" \
      "$result_root/04_ap03_real" \
      "$result_root/99_FINAL_RESULTS" \
      "$result_root/_pipeline_logs"
  fi

  prepare_args=(
    --video "$DRIVE_VIDEO"
    --intrinsics-json "$INTRINSIC_CATALOG/moving_calib_camera.json"
    --dataset-name "$dataset_name"
    --template-dataset "$TEMPLATE_DATASET"
    --results-root "$RESULTS_ROOT"
    --sampling-hz "$rate"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi

  run_step "PREPARE ${rate} HZ DATASET" "$logs/00_prepare_dataset.log" \
    python3 run/real_vehicle_data/03_prepare_real_moving_video_dataset.py \
      "${prepare_args[@]}"

  run_step "DETECT MOVING ARUCO ${rate} HZ" "$logs/01_detect_moving_aruco.log" \
    python3 run/real_vehicle_data/01_detect_moving_aruco_from_raw.py \
      --dataset "$shared_input" \
      --observations-root "$observations"

  run_step "VALIDATE ${rate} HZ INPUT" "$logs/02_validate_input.log" \
    python3 run/real_vehicle_data/00_validate_and_prepare_shared_input.py \
      --dataset "$shared_input" \
      --deep-images

  cat > "$result_root/EXPERIMENT_CONFIG.txt" <<EOF
Experiment: 0.5x temporal sampling ablation
Source moving video: $DRIVE_VIDEO
Sampling rate [Hz]: $rate
Intrinsic CameraInfo: $INTRINSIC_CATALOG/moving_calib_camera.json
Template static dataset: $TEMPLATE_DATASET
AP02 max_nfev static: $AP02_MAX_NFEV_STATIC
AP02 max_nfev moving: $AP02_MAX_NFEV_MOVING
Matcher: $MATCHER
GPU: $GPU
Maximum image size: $MAX_IMAGE_SIZE
EOF

  run_step "AP01 ${rate} HZ" "$logs/03_ap01_outer.log" \
    bash run/real_vehicle_data/run_full_real_pipeline.sh \
      --dataset "$shared_input" \
      --results-root "$result_root" \
      --observations-root "$observations" \
      --gpu "$GPU" \
      --matcher "$MATCHER" \
      --max-image-size "$MAX_IMAGE_SIZE" \
      --only ap01

  ref_marker="$(tr -d '[:space:]' < "$observations/REFERENCE_MARKER_ID.txt")"
  run_step "AP02 ${rate} HZ (NFEV ${AP02_MAX_NFEV_STATIC}/${AP02_MAX_NFEV_MOVING})" "$logs/AP02_REAL.log" \
    python3 run/real_vehicle_data/08_run_ap02_real.py \
      --observations-root "$observations" \
      --out "$result_root/03_ap02_real" \
      --ref-marker-id "$ref_marker" \
      --max-nfev-static "$AP02_MAX_NFEV_STATIC" \
      --max-nfev-moving "$AP02_MAX_NFEV_MOVING"

  run_step "AP03 ${rate} HZ" "$logs/04_ap03_outer.log" \
    bash run/real_vehicle_data/run_full_real_pipeline.sh \
      --dataset "$shared_input" \
      --results-root "$result_root" \
      --observations-root "$observations" \
      --gpu "$GPU" \
      --matcher "$MATCHER" \
      --max-image-size "$MAX_IMAGE_SIZE" \
      --only ap03

  run_step "FINAL REPORT ${rate} HZ" "$logs/05_final_report_outer.log" \
    bash run/real_vehicle_data/run_full_real_pipeline.sh \
      --dataset "$shared_input" \
      --results-root "$result_root" \
      --observations-root "$observations" \
      --only report

  run_step "MARKER CONSISTENCY ${rate} HZ" "$logs/06_marker_consistency.log" \
    bash run/real_vehicle_data/run_real_marker_consistency.sh \
      --dataset "$shared_input" \
      --results-root "$result_root" \
      --observations-root "$observations"
done

printf '\n%s\n' "================================================================================"
echo "0.5X TEMPORAL SAMPLING ABLATION COMPLETE"
printf '%s\n' "================================================================================"
for root in "${RUN_ROOTS[@]}"; do
  echo "Dataset: $root"
  echo "  Method report: $root/99_FINAL_RESULTS/REAL_DATA_ALL_METHODS.txt"
  echo "  Marker report: $root/99_FINAL_RESULTS/REAL_DATA_MARKER_CONSISTENCY.txt"
done

echo
echo "Stable intrinsic result:"
echo "  $INTRINSIC_CATALOG"

exit "$FAILED"
