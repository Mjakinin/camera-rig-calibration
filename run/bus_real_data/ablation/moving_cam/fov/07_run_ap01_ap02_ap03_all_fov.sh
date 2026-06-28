#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

VARIANTS=(
  "fov_40deg"
  "fov_50deg"
  "fov_60deg"
  "fov_69deg_baseline"
  "fov_80deg"
  "fov_90deg"
  "fov_100deg"
  "fov_110deg"
  "fov_120deg"
  "fov_140deg_extreme"
)

BASE_SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
BASE_RAW="$BASE_SHARED/raw_images"
BASE_OBS="$BASE_SHARED/aruco_observations"
BASE_META="$BASE_SHARED/metadata"

ABL_ROOT="results/bus_real_data/ablation/moving_cam/fov"
DATASET_ROOT="$ABL_ROOT/00_prepared_datasets"
OBS_ROOT="$ABL_ROOT/01_shared_observations"

AP01_RESULT="results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
AP02_RESULT="results/bus_real_data/02_ref_marker_graph_ba"
AP03_RESULT="results/bus_real_data/03_targetless_colmap_aruco_scale"
CMP_RESULT="results/bus_real_data/90_approach_comparison_ref_aruco"

AP01_OUT="$ABL_ROOT/02_ap01_results"
AP02_OUT="$ABL_ROOT/03_ap02_results"
AP03_OUT="$ABL_ROOT/04_ap03_results"

LOG_ROOT="$ABL_ROOT/_pipeline_logs"
BACKUP_ROOT="$ABL_ROOT/_baseline_backup_runtime"

mkdir -p "$AP01_OUT" "$AP02_OUT" "$AP03_OUT" "$LOG_ROOT" "$BACKUP_ROOT"

AP01_PIPELINE="run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh"
AP02_PIPELINE="run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh"
AP03_PIPELINE="run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh"

backup_baseline() {
  echo "[INFO] creating/restoring baseline backup..."

  rm -rf "$BACKUP_ROOT"
  mkdir -p "$BACKUP_ROOT"

  if [ -d "$BASE_RAW" ]; then
    cp -a "$BASE_RAW" "$BACKUP_ROOT/raw_images"
  fi
  if [ -d "$BASE_OBS" ]; then
    cp -a "$BASE_OBS" "$BACKUP_ROOT/aruco_observations"
  fi
  if [ -d "$BASE_META" ]; then
    cp -a "$BASE_META" "$BACKUP_ROOT/metadata"
  fi

  echo "[OK] backup created at: $BACKUP_ROOT"
}

restore_baseline() {
  set +e
  echo
  echo "[RESTORE] restoring original shared baseline..."

  if [ -d "$BACKUP_ROOT/raw_images" ]; then
    rm -rf "$BASE_RAW"
    mkdir -p "$BASE_SHARED"
    cp -aL "$BACKUP_ROOT/raw_images" "$BASE_RAW"
  fi

  if [ -d "$BACKUP_ROOT/aruco_observations" ]; then
    rm -rf "$BASE_OBS"
    mkdir -p "$BASE_SHARED"
    cp -aL "$BACKUP_ROOT/aruco_observations" "$BASE_OBS"
  fi

  if [ -d "$BACKUP_ROOT/metadata" ]; then
    rm -rf "$BASE_META"
    mkdir -p "$BASE_SHARED"
    cp -aL "$BACKUP_ROOT/metadata" "$BASE_META"
  fi

  echo "[OK] shared baseline restored."
}
trap restore_baseline EXIT

install_variant_as_shared_baseline() {
  local variant="$1"

  local variant_raw="$DATASET_ROOT/$variant/raw_images"
  local variant_obs="$OBS_ROOT/$variant"
  local variant_capture="$ABL_ROOT/00_captures/$variant"

  if [ ! -d "$variant_raw" ]; then
    echo "[ERROR] missing variant raw_images: $variant_raw"
    exit 1
  fi

  if [ ! -f "$variant_obs/shared_all_aruco_observations.csv" ]; then
    echo "[ERROR] missing variant shared observations: $variant_obs"
    exit 1
  fi

  echo "[INFO] installing variant into shared baseline:"
  echo "       variant: $variant"
  echo "       raw:     $variant_raw"
  echo "       obs:     $variant_obs"

  rm -rf "$BASE_RAW" "$BASE_OBS"
  mkdir -p "$BASE_SHARED"

  cp -aL "$variant_raw" "$BASE_RAW"
  cp -aL "$variant_obs" "$BASE_OBS"

  mkdir -p "$BASE_META"
  if [ -f "$variant_capture/route_commanded.csv" ]; then
    cp "$variant_capture/route_commanded.csv" "$BASE_META/route_commanded.csv"
    mkdir -p "$BASE_RAW/ap1_metadata"
    cp "$variant_capture/route_commanded.csv" "$BASE_RAW/ap1_metadata/route_commanded.csv"
  elif [ -f "$BACKUP_ROOT/metadata/route_commanded.csv" ]; then
    cp "$BACKUP_ROOT/metadata/route_commanded.csv" "$BASE_META/route_commanded.csv"
    mkdir -p "$BASE_RAW/ap1_metadata"
    cp "$BACKUP_ROOT/metadata/route_commanded.csv" "$BASE_RAW/ap1_metadata/route_commanded.csv"
  fi
}

snapshot_results() {
  local variant="$1"
  local approach="$2"

  case "$approach" in
    AP01)
      local out="$AP01_OUT/$variant"
      rm -rf "$out"
      mkdir -p "$out"
      if [ -d "$AP01_RESULT" ]; then cp -a "$AP01_RESULT" "$out/"; fi
      ;;
    AP02)
      local out="$AP02_OUT/$variant"
      rm -rf "$out"
      mkdir -p "$out"
      if [ -d "$AP02_RESULT" ]; then cp -a "$AP02_RESULT" "$out/"; fi
      if [ -d "$CMP_RESULT" ]; then cp -a "$CMP_RESULT" "$out/" || true; fi
      ;;
    AP03)
      local out="$AP03_OUT/$variant"
      rm -rf "$out"
      mkdir -p "$out"
      if [ -d "$AP03_RESULT" ]; then cp -a "$AP03_RESULT" "$out/"; fi
      if [ -d "$CMP_RESULT" ]; then cp -a "$CMP_RESULT" "$out/" || true; fi
      ;;
  esac
}

run_ap01() {
  local variant="$1"
  local log="$LOG_ROOT/${variant}_AP01.log"

  echo
  echo "================ AP01 $variant ================"

  # AP01 pipeline re-runs shared preprocessing on the currently installed shared baseline.
  RUN_SHARED_BASELINE=0 bash "$AP01_PIPELINE" 2>&1 | tee "$log"

  snapshot_results "$variant" "AP01"
}

run_ap02() {
  local variant="$1"
  local log="$LOG_ROOT/${variant}_AP02.log"

  echo
  echo "================ AP02 $variant ================"

  # AP02 also uses the currently installed shared baseline.
  SHARED_OBS="$BASE_OBS" bash "$AP02_PIPELINE" --skip-shared-baseline 2>&1 | tee "$log"

  snapshot_results "$variant" "AP02"
}

run_ap03() {
  local variant="$1"
  local log="$LOG_ROOT/${variant}_AP03.log"

  echo
  echo "================ AP03 $variant ================"

  # AP03 prepares COLMAP dataset from the currently installed shared baseline.
  bash "$AP03_PIPELINE" --min-area-px2 "${AP03_MIN_AREA_PX2:-100}" 2>&1 | tee "$log"

  # Optional: append current combined AP03 final if available.
  if [ -f "run/bus_real_data/approach3_targetless_colmap_aruco_scale/12_make_ap03_final_combined_result.py" ]; then
    PYTHONPATH=run/bus_real_data python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/12_make_ap03_final_combined_result.py       2>&1 | tee -a "$log" || true
  fi

  snapshot_results "$variant" "AP03"
}

main() {
  backup_baseline

  for variant in "${VARIANTS[@]}"; do
    echo
    echo "############################################################"
    echo "# MOVING CAMERA RESOLUTION VARIANT: $variant"
    echo "############################################################"

    install_variant_as_shared_baseline "$variant"

    run_ap01 "$variant"
    run_ap02 "$variant"
    run_ap03 "$variant"

    echo
    echo "[OK] finished all approaches for $variant"
  done

  restore_baseline
  trap - EXIT

  echo
  echo "[OK] all AP01/AP02/AP03 FOV ablation runs finished."
  echo "[INFO] results:"
  find "$ABL_ROOT" -maxdepth 4 -type f \( -name "*FINAL*.txt" -o -name "*FINAL*.csv" -o -name "*EVALUATION*.txt" -o -name "*EVALUATION*.csv" \) | sort || true
}

main "$@"
