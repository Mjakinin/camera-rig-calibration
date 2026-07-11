#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

GPU=0
SECTION="all"
RECAPTURE_FOV=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:?missing value for --gpu}"
      shift 2
      ;;
    --section)
      SECTION="${2:?missing value for --section}"
      shift 2
      ;;
    --recapture-fov)
      RECAPTURE_FOV=1
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1"
      echo "usage: $0 [--gpu 0|1] [--section all|preflight|real|simulation] [--recapture-fov]"
      exit 2
      ;;
  esac
done

if [[ "$GPU" != "0" && "$GPU" != "1" ]]; then
  echo "[ERROR] --gpu must be 0 or 1"
  exit 2
fi
if [[ "$SECTION" != "all" && "$SECTION" != "preflight" && "$SECTION" != "real" && "$SECTION" != "simulation" ]]; then
  echo "[ERROR] invalid --section: $SECTION"
  exit 2
fi

REAL_FINAL_DIR="results/real_vehicle_data/real_05x_4k_3hz_v1/99_FINAL_RESULTS"
SIM_FINAL_DIR="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
mkdir -p "$REAL_FINAL_DIR" "$SIM_FINAL_DIR"

# This legacy global log directory is no longer part of the output contract.
rm -rf results/_full_final_rerun_logs

if [[ -n "$(git status --porcelain --untracked-files=no)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "[ERROR] tracked working tree changes exist."
  echo "Commit/stash them first, or run with ALLOW_DIRTY=1 after reviewing the risk."
  git status --short
  exit 1
fi

print_header() {
  local label="$1"
  echo "================================================================================"
  echo "$label"
  echo "section=$SECTION gpu=$GPU recapture_fov=$RECAPTURE_FOV"
  echo "commit=$(git rev-parse HEAD)"
  echo "================================================================================"
}

run_preflight() {
  echo
  echo "=== Python syntax ==="
  find run -name '*.py' -print0 | xargs -0 python3 -m py_compile

  echo
  echo "=== Shell syntax ==="
  while IFS= read -r -d '' script_path; do
    bash -n "$script_path"
  done < <(find run -name '*.sh' -print0)

  echo
  echo "=== Required executables ==="
  command -v python3
  command -v colmap
  if [[ "$SECTION" == "all" || "$SECTION" == "simulation" ]]; then
    command -v ign
    command -v ros2
  fi

  echo
  echo "=== Canonical real shared-input audit ==="
  python3 run/real_vehicle_data/00_validate_and_prepare_shared_input.py

  echo
  echo "=== Simulation marker-map audit ==="
  python3 - <<'PY'
import re
from pathlib import Path

path = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
text = path.read_text(encoding="utf-8")
ids = sorted({int(value) for value in re.findall(r"a4_aruco_marker_(\d{3})", text)})
expected = list(range(0, 15)) + list(range(16, 21))
if ids != expected:
    raise SystemExit(f"[ERROR] marker IDs {ids} != expected {expected}")
print("[OK] simulation marker map contains exactly 20 markers:", ids)
PY

  echo
  echo "=== Route audit ==="
  python3 - <<'PY'
import json
from pathlib import Path

for name, expected in [
    ("moving_camera_route1_interpolated_final.json", 352),
    ("moving_camera_route2_interpolated_final.json", 189),
]:
    path = Path("src/calib_lab/bus_real_data/config") / name
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if len(frames) != expected or int(data.get("num_frames", -1)) != expected:
        raise SystemExit(
            f"[ERROR] {name}: len={len(frames)} num_frames={data.get('num_frames')} "
            f"expected={expected}"
        )
    numbers = [int(row["frame"]) for row in frames]
    if numbers != list(range(expected)):
        raise SystemExit(f"[ERROR] {name}: frame indices are not contiguous")
    print(f"[OK] {name}: {expected} bounded route frames")
PY
}

run_real() {
  echo
  echo "================================================================================"
  echo "REAL 0.5x FULL END-TO-END CALIBRATION"
  echo "================================================================================"
  local pipeline_code=0
  bash run/real_vehicle_data/run_full_real_pipeline.sh --gpu "$GPU" || pipeline_code=$?

  local legacy_logs="results/real_vehicle_data/real_05x_4k_3hz_v1/_pipeline_logs"
  if [[ -d "$legacy_logs" ]]; then
    mkdir -p "$REAL_FINAL_DIR/logs"
    find "$legacy_logs" -maxdepth 1 -type f -exec mv -f {} "$REAL_FINAL_DIR/logs/" \;
    rm -rf "$legacy_logs"
  fi

  if [[ "$pipeline_code" -ne 0 ]]; then
    echo "[ERROR] real pipeline exited with code $pipeline_code"
    return "$pipeline_code"
  fi

  local final="$REAL_FINAL_DIR/REAL_DATA_ALL_METHODS.txt"
  if [[ ! -s "$final" ]]; then
    echo "[ERROR] missing final real report: $final"
    exit 1
  fi
  echo "[OK] real final report: $final"
}

run_group() {
  local root="$1"
  local label="$2"
  shift 2
  bash run/bus_real_data/ablation/_shared/30_run_existing_variant_group.sh \
    "$root" "$label" "$@"
}

run_simulation() {
  echo
  echo "================================================================================"
  echo "FOV ABLATION"
  echo "================================================================================"
  if [[ "$RECAPTURE_FOV" == "1" ]]; then
    FORCE_CAPTURE=1 bash run/bus_real_data/ablation/moving_cam/fov/01_make_all_fov_ready.sh
  fi
  run_group \
    results/bus_real_data/ablation/moving_cam/fov \
    moving_cam_fov \
    fov_40deg \
    fov_69deg_baseline \
    fov_100deg \
    fov_140deg_extreme

  echo
  echo "================================================================================"
  echo "RESOLUTION ABLATION"
  echo "================================================================================"
  run_group \
    results/bus_real_data/ablation/moving_cam/res \
    moving_cam_resolution \
    moving_res_160x90_extreme_pixel \
    moving_res_320x180_low \
    moving_res_1280x720_baseline \
    moving_res_2560x1440_upscaled

  echo
  echo "================================================================================"
  echo "MOTION-BLUR ABLATION"
  echo "================================================================================"
  run_group \
    results/bus_real_data/ablation/moving_cam/motion_blur \
    moving_cam_motion_blur \
    moving_blur_k00_baseline \
    moving_blur_k09_mild \
    moving_blur_k21_strong \
    moving_blur_k41_extreme

  echo
  echo "================================================================================"
  echo "FRAME-DENSITY ABLATION"
  echo "================================================================================"
  bash run/bus_real_data/ablation/moving_cam/density/21_run_all_density_methods.sh

  echo
  echo "================================================================================"
  echo "LIGHTING ABLATION"
  echo "================================================================================"
  run_group \
    results/bus_real_data/ablation/world/lighting \
    world_lighting \
    ceiling_dark_extreme \
    ceiling_low \
    ceiling_normal \
    ceiling_bright

  echo
  echo "================================================================================"
  echo "ROUTE ABLATION — FULL RECAPTURE, ROUTE 2 LAST"
  echo "================================================================================"
  bash run/bus_real_data/ablation/world/route/01_run_route_ablation_all.sh

  echo
  echo "=== Promote current Route-2 method outputs into canonical final reports ==="
  bash run/bus_real_data/reporting/run_refresh_final_results.sh \
    --reuse-baseline --promote

  local route_report="$SIM_FINAL_DIR/ablations/05_ROUTE_PATH_ALL_METHODS.txt"
  if [[ ! -s "$route_report" ]]; then
    echo "[ERROR] route report was not generated: $route_report"
    exit 1
  fi
  echo "[OK] route report: $route_report"
}

run_preflight_logged() {
  local logs=()
  case "$SECTION" in
    real)
      logs+=("$REAL_FINAL_DIR/PREFLIGHT.log")
      ;;
    simulation)
      logs+=("$SIM_FINAL_DIR/PREFLIGHT.log")
      ;;
    all|preflight)
      logs+=(
        "$REAL_FINAL_DIR/PREFLIGHT.log"
        "$SIM_FINAL_DIR/PREFLIGHT.log"
      )
      ;;
  esac

  {
    print_header "FINAL FULL RERUN — PREFLIGHT"
    run_preflight
    echo
    echo "[OK] PREFLIGHT COMPLETE"
  } 2>&1 | tee "${logs[@]}"
}

run_real_logged() {
  {
    print_header "FINAL FULL RERUN — REAL VEHICLE"
    run_real
    echo
    echo "[OK] REAL VEHICLE RERUN COMPLETE"
  } 2>&1 | tee "$REAL_FINAL_DIR/FULL_REAL_RERUN.log"
}

run_simulation_logged() {
  {
    print_header "FINAL FULL RERUN — SIMULATION"
    run_simulation
    echo
    echo "[OK] SIMULATION RERUN COMPLETE"
  } 2>&1 | tee "$SIM_FINAL_DIR/FULL_SIMULATION_RERUN.log"
}

run_preflight_logged

case "$SECTION" in
  preflight)
    ;;
  real)
    run_real_logged
    ;;
  simulation)
    run_simulation_logged
    ;;
  all)
    run_real_logged
    run_simulation_logged
    ;;
esac

echo
echo "================================================================================"
echo "[OK] FINAL RERUN SECTION COMPLETE"
echo "section=$SECTION"
echo "real results: $REAL_FINAL_DIR"
echo "simulation results: $SIM_FINAL_DIR"
echo "================================================================================"
