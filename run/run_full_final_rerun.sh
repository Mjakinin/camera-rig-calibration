#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

GPU=0
SECTION="all"

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
      # Kept for compatibility. Simulation reruns now always recapture FOV variants.
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1"
      echo "usage: $0 [--gpu 0|1] [--section all|preflight|simulation|real]"
      exit 2
      ;;
  esac
done

if [[ "$GPU" != "0" && "$GPU" != "1" ]]; then
  echo "[ERROR] --gpu must be 0 or 1"
  exit 2
fi
if [[ "$SECTION" != "all" && "$SECTION" != "preflight" && "$SECTION" != "simulation" && "$SECTION" != "real" ]]; then
  echo "[ERROR] invalid --section: $SECTION"
  exit 2
fi

REAL_ROOT="results/real_vehicle_data/real_05x_4k_3hz_v1"
REAL_FINAL_DIR="$REAL_ROOT/99_FINAL_RESULTS"
SIM_FINAL_DIR="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
ROUTE_ROOT="results/bus_real_data/ablation/world/route"
ROUTE2_ROOT="$ROUTE_ROOT/route2"
SHARED_ROOT="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
BASELINE_HELPER="run/bus_real_data/ablation/_shared/31_reuse_route2_baseline.py"
DETECTOR="run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"
TMP_RUN_DIR="$(mktemp -d -t camera-rig-overnight-XXXXXX)"

cleanup() {
  rm -rf "$TMP_RUN_DIR"
}
trap cleanup EXIT INT TERM

mkdir -p "$REAL_FINAL_DIR" "$SIM_FINAL_DIR"
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
  echo "section=$SECTION gpu=$GPU"
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

config = Path("src/calib_lab/bus_real_data/config")
checks = [
    ("moving_camera_route1_interpolated_final.json", 352),
    ("moving_camera_route2_interpolated_final.json", 189),
]
for name, expected in checks:
    path = config / name
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if len(frames) != expected or int(data.get("num_frames", -1)) != expected:
        raise SystemExit(
            f"[ERROR] {name}: len={len(frames)} num_frames={data.get('num_frames')} expected={expected}"
        )
    if [int(row["frame"]) for row in frames] != list(range(expected)):
        raise SystemExit(f"[ERROR] {name}: frame indices are not contiguous")
    print(f"[OK] {name}: {expected} bounded route frames")

canonical = config / "moving_camera_route_interpolated.json"
route2 = config / "moving_camera_route2_interpolated_final.json"
if canonical.read_bytes() != route2.read_bytes():
    raise SystemExit(
        "[ERROR] moving_camera_route_interpolated.json is not byte-identical to Route 2"
    )
print("[OK] canonical moving-camera route is exactly Route 2")
PY
}

copy_log_after_run() {
  local source="$1"
  local destination="$2"
  mkdir -p "$(dirname "$destination")"
  cp -f "$source" "$destination"
}

run_preflight_logged() {
  local log="$TMP_RUN_DIR/PREFLIGHT.log"
  local code=0
  set +e
  (
    set -euo pipefail
    print_header "FINAL OVERNIGHT RERUN — PREFLIGHT"
    run_preflight
    echo
    echo "[OK] PREFLIGHT COMPLETE"
  ) 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  set -e

  case "$SECTION" in
    real)
      copy_log_after_run "$log" "$REAL_FINAL_DIR/PREFLIGHT.log"
      ;;
    simulation)
      copy_log_after_run "$log" "$SIM_FINAL_DIR/PREFLIGHT.log"
      ;;
    all|preflight)
      copy_log_after_run "$log" "$REAL_FINAL_DIR/PREFLIGHT.log"
      copy_log_after_run "$log" "$SIM_FINAL_DIR/PREFLIGHT.log"
      ;;
  esac
  return "$code"
}

install_route2_dataset_as_canonical() {
  python3 "$BASELINE_HELPER" \
    --source "$ROUTE2_ROOT" \
    --target "$SHARED_ROOT" \
    --variant route2_nominal \
    --group canonical/route2 \
    --dataset-only
}

reuse_route2_baseline() {
  local target="$1"
  local variant="$2"
  local group="$3"
  python3 "$BASELINE_HELPER" \
    --source "$ROUTE2_ROOT" \
    --target "$target" \
    --variant "$variant" \
    --group "$group"
}

run_group() {
  local root="$1"
  local label="$2"
  shift 2
  bash run/bus_real_data/ablation/_shared/30_run_existing_variant_group.sh \
    "$root" "$label" "$@"
}

run_route_ablation() {
  echo
  echo "================================================================================"
  echo "ROUTE ABLATION — ROUTE 1 AND ROUTE 2"
  echo "================================================================================"
  bash run/bus_real_data/ablation/world/route/01_run_route_ablation_all.sh

  for variant in route1 route2; do
    test -s "$ROUTE_ROOT/$variant/FINAL_RESULTS/RUN_STATUS.txt"
  done
  test -s "$SIM_FINAL_DIR/ablations/05_ROUTE_PATH_ALL_METHODS.txt"

  install_route2_dataset_as_canonical
  echo "[OK] Route 2 is now the canonical baseline dataset for all following ablations."
}

prepare_fov_route2_variants() {
  local root="results/bus_real_data/ablation/moving_cam/fov"
  local prep="$root/00_prepared_datasets"
  local obs="$root/01_shared_observations"
  local variants=(fov_40deg fov_100deg fov_140deg_extreme)

  echo
  echo "================================================================================"
  echo "FOV ABLATION — ROUTE 2; BASELINE REUSED"
  echo "================================================================================"

  python3 run/bus_real_data/ablation/moving_cam/fov/00_generate_fov_world_variants.py
  for variant in "${variants[@]}"; do
    AUTO_CONFIRM=1 bash \
      run/bus_real_data/ablation/moving_cam/fov/02_capture_one_fov_variant.sh \
      "$variant"
  done
  python3 run/bus_real_data/ablation/moving_cam/fov/03_prepare_fov_raw_datasets.py

  rm -rf "$prep/fov_69deg_baseline/raw_images" "$obs/fov_69deg_baseline"
  mkdir -p "$prep/fov_69deg_baseline" "$obs"
  cp -a "$ROUTE2_ROOT/raw_images" "$prep/fov_69deg_baseline/raw_images"
  cp -a "$ROUTE2_ROOT/aruco_observations" "$obs/fov_69deg_baseline"

  for variant in "${variants[@]}"; do
    rm -rf "$obs/$variant"
    python3 "$DETECTOR" \
      --dataset "$prep/$variant/raw_images" \
      --out "$obs/$variant" \
      --dictionary DICT_4X4_50
  done

  python3 run/bus_real_data/ablation/moving_cam/fov/05_summarize_shared_observations_fov.py
  python3 run/bus_real_data/ablation/moving_cam/fov/06_materialize_clean_fov_structure.py

  reuse_route2_baseline \
    "$root/fov_69deg_baseline" \
    fov_69deg_baseline \
    moving_cam/fov

  REUSE_EXISTING_OBSERVATIONS=1 run_group \
    "$root" moving_cam_fov "${variants[@]}"

  rm -rf "$root/01_shared_observations" "$root/99_summary"
}

prepare_resolution_route2_variants() {
  local root="results/bus_real_data/ablation/moving_cam/res"
  echo
  echo "================================================================================"
  echo "RESOLUTION ABLATION — ROUTE 2; 1280x720 BASELINE REUSED"
  echo "================================================================================"

  python3 run/bus_real_data/ablation/moving_cam/res/10_make_clean_res_raw_variants.py
  reuse_route2_baseline \
    "$root/moving_res_1280x720_baseline" \
    moving_res_1280x720_baseline \
    moving_cam/res
  run_group \
    "$root" moving_cam_resolution \
    moving_res_160x90_extreme_pixel \
    moving_res_320x180_low \
    moving_res_2560x1440_upscaled
}

prepare_motion_blur_route2_variants() {
  local root="results/bus_real_data/ablation/moving_cam/motion_blur"
  echo
  echo "================================================================================"
  echo "MOTION-BLUR ABLATION — ROUTE 2; K00 BASELINE REUSED"
  echo "================================================================================"

  python3 run/bus_real_data/ablation/moving_cam/motion_blur/10_make_clean_motion_blur_raw_variants.py
  reuse_route2_baseline \
    "$root/moving_blur_k00_baseline" \
    moving_blur_k00_baseline \
    moving_cam/motion_blur
  run_group \
    "$root" moving_cam_motion_blur \
    moving_blur_k09_mild \
    moving_blur_k21_strong \
    moving_blur_k41_extreme
}

prepare_density_route2_variants() {
  local root="results/bus_real_data/ablation/moving_cam/density"
  echo
  echo "================================================================================"
  echo "FRAME-DENSITY ABLATION — ROUTE 2; STRIDE-1 BASELINE REUSED"
  echo "================================================================================"

  python3 run/bus_real_data/ablation/moving_cam/density/00_generate_density_variants.py \
    --source "$SHARED_ROOT" \
    --out "$root"
  reuse_route2_baseline \
    "$root/density_stride_1_100pct" \
    density_stride_1_100pct \
    moving_cam/density
  run_group \
    "$root" moving_cam_density \
    density_stride_2_50pct \
    density_stride_4_25pct \
    density_stride_8_12p5pct
}

prepare_lighting_route2_variants() {
  local root="results/bus_real_data/ablation/world/lighting"
  local variants=(ceiling_dark_extreme ceiling_low ceiling_normal ceiling_bright)

  echo
  echo "================================================================================"
  echo "LIGHTING ABLATION — ALL FOUR PHYSICAL LIGHTING WORLDS ON ROUTE 2"
  echo "================================================================================"
  echo "[INFO] ceiling_normal is rendered separately because its ceiling-light world"
  echo "       is not pixel-identical to the nominal Route-2 world."

  bash run/bus_real_data/ablation/world/lighting/19_prepare_lighting_datasets.sh
  for variant in "${variants[@]}"; do
    bash run/bus_real_data/ablation/world/lighting/18_capture_one_lighting_variant.sh \
      "$variant"
  done
  run_group "$root" world_lighting "${variants[@]}"
}

preserve_route_report_artifacts() {
  local save="$TMP_RUN_DIR/route_artifacts"
  local paths=(
    "ablations/05_ROUTE_PATH_ALL_METHODS.txt"
    "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt"
    "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt"
    "data/primary/ROUTE_PATH_ABLATION_SUMMARY.csv"
    "data/primary/ROUTE_PATH_ABLATION_DETAIL.csv"
    "data/secondary/ROUTE_PATH_ABLATION_SUMMARY.csv"
    "data/secondary/ROUTE_PATH_ABLATION_DETAIL.csv"
    "data/secondary/ROUTE_PATH_AP02_GT_ALIGNED_FULL_MAP.csv"
    "data/ROUTE_PATH_RUN_STATUS.csv"
  )

  rm -rf "$save"
  for relative in "${paths[@]}"; do
    if [[ -f "$SIM_FINAL_DIR/$relative" ]]; then
      mkdir -p "$save/$(dirname "$relative")"
      cp -f "$SIM_FINAL_DIR/$relative" "$save/$relative"
    fi
  done
}

restore_route_report_artifacts() {
  local save="$TMP_RUN_DIR/route_artifacts"
  [[ -d "$save" ]] || return 0
  while IFS= read -r -d '' file; do
    local relative="${file#"$save/"}"
    mkdir -p "$SIM_FINAL_DIR/$(dirname "$relative")"
    cp -f "$file" "$SIM_FINAL_DIR/$relative"
  done < <(find "$save" -type f -print0)
}

install_density_report_artifacts() {
  local source="results/bus_real_data/ablation/moving_cam/density/ABLATION_SUMMARY"
  test -s "$source/FULL_ABLATION_REPORT.txt"

  mkdir -p \
    "$SIM_FINAL_DIR/ablations" \
    "$SIM_FINAL_DIR/data/primary" \
    "$SIM_FINAL_DIR/data/secondary"

  cp -f "$source/FULL_ABLATION_REPORT.txt" \
    "$SIM_FINAL_DIR/ablations/06_FRAME_DENSITY_ALL_METHODS.txt"
  cp -f "$source/RUN_STATUS_ALL_VARIANTS.csv" \
    "$SIM_FINAL_DIR/data/FRAME_DENSITY_RUN_STATUS.csv"
  cp -f "$source/PRIMARY_PAIRWISE_SUMMARY_ALL_VARIANTS.csv" \
    "$SIM_FINAL_DIR/data/primary/FRAME_DENSITY_ABLATION_SUMMARY.csv"
  cp -f "$source/PRIMARY_PAIRWISE_DETAIL_ALL_VARIANTS.csv" \
    "$SIM_FINAL_DIR/data/primary/FRAME_DENSITY_ABLATION_DETAIL.csv"
  cp -f "$source/SECONDARY_REF14_WORLD_SUMMARY_ALL_VARIANTS.csv" \
    "$SIM_FINAL_DIR/data/secondary/FRAME_DENSITY_ABLATION_SUMMARY.csv"
  cp -f "$source/SECONDARY_REF14_WORLD_DETAIL_ALL_VARIANTS.csv" \
    "$SIM_FINAL_DIR/data/secondary/FRAME_DENSITY_ABLATION_DETAIL.csv"

  cat >> "$SIM_FINAL_DIR/02_ALL_ABLATIONS_ALL_METHODS.txt" <<'TXT'


FRAME-DENSITY ABLATION
====================================================================================================
The complete Route-2 frame-density report is stored in:
  ablations/06_FRAME_DENSITY_ALL_METHODS.txt
Baseline density_stride_1_100pct reuses the exact Route-2 nominal method run.
TXT

  python3 - "$SIM_FINAL_DIR/EXTRA_ABLATIONS_MANIFEST.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "baseline_policy": (
        "Route 2 is the single nominal simulation run. Exactly equivalent FOV, "
        "resolution, no-blur and 100%-density baselines reuse its method outputs."
    ),
    "route_ablation": {
        "variants": ["route1", "route2"],
        "report": "ablations/05_ROUTE_PATH_ALL_METHODS.txt",
    },
    "frame_density_ablation": {
        "variants": [
            "density_stride_1_100pct",
            "density_stride_2_50pct",
            "density_stride_4_25pct",
            "density_stride_8_12p5pct",
        ],
        "report": "ablations/06_FRAME_DENSITY_ALL_METHODS.txt",
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

refresh_simulation_final_results() {
  echo
  echo "================================================================================"
  echo "BUILD CANONICAL SIMULATION FINAL RESULTS FROM FRESH ROUTE-2 BASELINE"
  echo "================================================================================"

  preserve_route_report_artifacts
  install_route2_dataset_as_canonical
  bash run/bus_real_data/reporting/run_refresh_final_results.sh --promote
  restore_route_report_artifacts
  install_density_report_artifacts

  test -s "$SIM_FINAL_DIR/01_BASELINE_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/01_FOV_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/02_MOTION_BLUR_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/03_RESOLUTION_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/04_LIGHTING_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/05_ROUTE_PATH_ALL_METHODS.txt"
  test -s "$SIM_FINAL_DIR/ablations/06_FRAME_DENSITY_ALL_METHODS.txt"
}

run_simulation() {
  run_route_ablation
  prepare_fov_route2_variants
  prepare_resolution_route2_variants
  prepare_motion_blur_route2_variants
  prepare_density_route2_variants
  prepare_lighting_route2_variants
  refresh_simulation_final_results
}

run_real() {
  echo
  echo "================================================================================"
  echo "REAL 0.5x FULL END-TO-END CALIBRATION"
  echo "================================================================================"

  rm -f "$REAL_FINAL_DIR/REAL_DATA_ALL_METHODS.txt"
  find "$REAL_FINAL_DIR" -maxdepth 1 -type f -iname '*cross*method*' -delete 2>/dev/null || true

  local pipeline_code=0
  bash run/real_vehicle_data/run_full_real_pipeline.sh --gpu "$GPU" || pipeline_code=$?

  local legacy_logs="$REAL_ROOT/_pipeline_logs"
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
  test -s "$final"
  if grep -Eqi 'cross[- ]method' "$final"; then
    echo "[ERROR] forbidden cross-method section remains in $final"
    exit 1
  fi
  echo "[OK] real final report without cross-method block: $final"
}

run_simulation_logged() {
  local log="$TMP_RUN_DIR/FULL_SIMULATION_RERUN.log"
  local code=0
  set +e
  (
    set -euo pipefail
    print_header "FINAL OVERNIGHT RERUN — ALL SIMULATION ABLATIONS"
    run_simulation
    echo
    echo "[OK] ALL SIMULATION ABLATIONS COMPLETE"
  ) 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  set -e
  copy_log_after_run "$log" "$SIM_FINAL_DIR/FULL_SIMULATION_RERUN.log"
  copy_log_after_run "$TMP_RUN_DIR/PREFLIGHT.log" "$SIM_FINAL_DIR/PREFLIGHT.log"
  return "$code"
}

run_real_logged() {
  local log="$TMP_RUN_DIR/FULL_REAL_RERUN.log"
  local code=0
  set +e
  (
    set -euo pipefail
    print_header "FINAL OVERNIGHT RERUN — REAL VEHICLE"
    run_real
    echo
    echo "[OK] REAL VEHICLE RERUN COMPLETE"
  ) 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  set -e
  copy_log_after_run "$log" "$REAL_FINAL_DIR/FULL_REAL_RERUN.log"
  copy_log_after_run "$TMP_RUN_DIR/PREFLIGHT.log" "$REAL_FINAL_DIR/PREFLIGHT.log"
  return "$code"
}

run_preflight_logged
case "$SECTION" in
  preflight)
    ;;
  simulation)
    run_simulation_logged
    ;;
  real)
    run_real_logged
    ;;
  all)
    run_simulation_logged
    run_real_logged
    ;;
esac

echo
echo "================================================================================"
echo "[OK] FINAL OVERNIGHT RERUN SECTION COMPLETE"
echo "section=$SECTION"
echo "simulation results: $SIM_FINAL_DIR"
echo "real results: $REAL_FINAL_DIR"
echo "================================================================================"
