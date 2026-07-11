#!/usr/bin/env bash
set -o pipefail

cd "$(git rev-parse --show-toplevel)"

# Validate canonical tracked shared input and create local legacy link.
python3 run/real_vehicle_data/00_validate_and_prepare_shared_input.py

DATASET="$PWD/results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input"
RESULTS_ROOT="$PWD/results/real_vehicle_data/real_05x_4k_3hz_v1"
OBS_ROOT=""
GPU=0
MATCHER="exhaustive"
MAX_IMAGE_SIZE=2400
ONLY="all"
REUSE_COLMAP=0

usage() {
  cat <<'EOF'
Usage:
  bash run/real_vehicle_data/run_full_real_pipeline.sh [options]

Options:
  --dataset PATH
  --results-root PATH
  --observations-root PATH
  --gpu 0|1
  --matcher exhaustive|sequential
  --max-image-size PIXELS
  --only all|ap01|ap02|ap03|report
  --reuse-colmap
  -h, --help

Default result layout:
  02_ap01_real/
  03_ap02_real/
  04_ap03_real/
  99_FINAL_RESULTS/REAL_DATA_ALL_METHODS.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --observations-root)
      OBS_ROOT="$2"
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
    --only)
      ONLY="$2"
      shift 2
      ;;
    --reuse-colmap)
      REUSE_COLMAP=1
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

if [[ -z "$OBS_ROOT" ]]; then
  OBS_ROOT="$RESULTS_ROOT/00_shared_input/aruco_observations"
fi

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

case "$ONLY" in
  all|ap01|ap02|ap03|report) ;;
  *)
    echo "[ERROR] --only must be all, ap01, ap02, ap03, or report" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULTS_ROOT/_pipeline_logs"

REF_FILE="$OBS_ROOT/REFERENCE_MARKER_ID.txt"
if [[ ! -f "$REF_FILE" ]]; then
  echo "[ERROR] Missing reference marker file: $REF_FILE" >&2
  exit 1
fi
REF_MARKER="$(tr -d '[:space:]' < "$REF_FILE")"

echo
echo "================================================================================"
echo "REAL-DATA FULL PIPELINE"
echo "================================================================================"
echo "DATASET=$DATASET"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "OBS_ROOT=$OBS_ROOT"
echo "REF_MARKER=$REF_MARKER"
echo "GPU=$GPU"
echo "MATCHER=$MATCHER"
echo "MAX_IMAGE_SIZE=$MAX_IMAGE_SIZE"
echo "ONLY=$ONLY"
echo "REUSE_COLMAP=$REUSE_COLMAP"
echo

python3 - "$DATASET" "$OBS_ROOT" "$REF_MARKER" <<'PY'
from pathlib import Path
import csv
import json
import sys

dataset = Path(sys.argv[1])
obs = Path(sys.argv[2])
ref_marker = int(sys.argv[3])

raw = dataset / "raw_images"
cameras = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

required = [
    raw / "camera_info" / "moving_calib_camera.json",
    obs / "shared_static_aruco_observations.csv",
    obs / "shared_moving_aruco_observations.csv",
    obs / "shared_all_aruco_observations.csv",
]

for camera in cameras:
    required.append(raw / "static" / f"{camera}.png")
    required.append(raw / "camera_info" / f"{camera}.json")

missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("[ERROR] missing required files:\n" + "\n".join(missing))

moving = sorted((raw / "moving").glob("frame_*.png"))
if not moving:
    raise SystemExit("[ERROR] no moving frames")

with (obs / "shared_all_aruco_observations.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

markers = {
    int(float(row["marker_id"]))
    for row in rows
    if str(row.get("pnp_success", "")).lower() in {"true", "1", "yes"}
}
if ref_marker not in markers:
    raise SystemExit(f"[ERROR] reference marker {ref_marker} not in observations")

moving_info = json.loads(
    (raw / "camera_info" / "moving_calib_camera.json").read_text()
)

print("[OK] preflight")
print(" moving frames:", len(moving))
print(" observations:", len(rows))
print(" marker IDs:", sorted(markers))
print(" reference marker:", ref_marker)
print(
    " moving intrinsics:",
    f"{moving_info['width']}x{moving_info['height']}",
    moving_info["distortion_model"],
)
PY

if [[ "$ONLY" != "report" ]] && ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] COLMAP not found in PATH; AP01/AP03 cannot run." >&2
  exit 127
fi

python3 - <<'PY'
import cv2
import numpy
import scipy
print("[OK] Python dependencies: OpenCV, NumPy, SciPy")
PY

FAILED=0

run_logged() {
  local name="$1"
  shift

  local log="$RESULTS_ROOT/_pipeline_logs/${name}.log"
  echo
  echo "================================================================================"
  echo "RUNNING $name"
  echo "LOG: $log"
  echo "================================================================================"

  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}

  if [[ "$code" -ne 0 ]]; then
    echo "[WARN] $name exited with code $code"
    FAILED=1
  else
    echo "[OK] $name completed"
  fi
}

run_ap01() {
  local extra=()
  if [[ "$REUSE_COLMAP" == "1" ]]; then
    extra+=(--reuse-colmap)
  fi

  run_logged AP01_REAL \
    python3 run/real_vehicle_data/07_run_ap01_real.py \
      --dataset "$DATASET" \
      --observations-root "$OBS_ROOT" \
      --out "$RESULTS_ROOT/02_ap01_real" \
      --matcher "$MATCHER" \
      --use-gpu "$GPU" \
      --max-image-size "$MAX_IMAGE_SIZE" \
      "${extra[@]}"
}

run_ap02() {
  run_logged AP02_REAL \
    python3 run/real_vehicle_data/08_run_ap02_real.py \
      --observations-root "$OBS_ROOT" \
      --out "$RESULTS_ROOT/03_ap02_real" \
      --ref-marker-id "$REF_MARKER"
}

run_ap03() {
  local extra=()
  if [[ "$REUSE_COLMAP" == "1" ]]; then
    extra+=(--reuse-colmap)
  fi

  run_logged AP03_REAL \
    python3 run/real_vehicle_data/09_run_ap03_real.py \
      --dataset "$DATASET" \
      --out "$RESULTS_ROOT/04_ap03_real" \
      --matcher "$MATCHER" \
      --use-gpu "$GPU" \
      --marker-ids "0-20" \
      --marker-length-m "0.17" \
      "${extra[@]}"
}

case "$ONLY" in
  all)
    run_ap01
    run_ap02
    run_ap03
    ;;
  ap01)
    run_ap01
    ;;
  ap02)
    run_ap02
    ;;
  ap03)
    run_ap03
    ;;
  report)
    ;;
esac

run_logged FINAL_REPORT \
  python3 run/real_vehicle_data/10_write_real_final_report.py \
    --dataset "$DATASET" \
    --results-root "$RESULTS_ROOT"

REPORT="$RESULTS_ROOT/99_FINAL_RESULTS/REAL_DATA_ALL_METHODS.txt"

echo
echo "================================================================================"
echo "FINAL REPORT"
echo "================================================================================"
if [[ -f "$REPORT" ]]; then
  cat "$REPORT"
else
  echo "[ERROR] final report missing: $REPORT" >&2
  FAILED=1
fi

echo
echo "Report path:"
echo "$REPORT"
echo
echo "Windows path:"
wslpath -w "$REPORT" 2>/dev/null || true

exit "$FAILED"
