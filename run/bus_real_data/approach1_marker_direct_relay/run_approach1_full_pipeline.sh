#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

# Do not use `set -u` before sourcing ROS setup files.
# ROS setup scripts may reference optional unset variables.
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

set -eo pipefail

RESULT_ROOT="results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
RUN_SHARED_BASELINE="${RUN_SHARED_BASELINE:-1}"
LOG_DIR="$RESULT_ROOT/_pipeline_logs"

mkdir -p "$LOG_DIR"

run_step () {
  STEP="$1"
  shift

  echo
  echo "================================================================================"
  echo "RUNNING $STEP: $*"
  echo "================================================================================"

  "$@" 2>&1 | tee "$LOG_DIR/${STEP}.log"
}

echo
echo "================================================================================"
echo "AP01: Marker direct relay / multimarker / multichain"
echo "Using shared baseline for raw images + ArUco detections."
echo "================================================================================"

if [ "$RUN_SHARED_BASELINE" = "1" ]; then
  run_step 00_shared_baseline \
    bash run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh
else
  echo
  echo "================================================================================"
  echo "SKIPPING 00_shared_baseline because RUN_SHARED_BASELINE=0"
  echo "================================================================================"
fi

run_step 01_prepare_ap1_adapter_cache \
  python3 run/bus_real_data/_shared/baseline/03_export_ap1_observations_from_shared.py \
    --static-out "$RESULT_ROOT/.ap01_compat_cache/static_observations" \
    --sequence "$RESULT_ROOT/.ap01_compat_cache/moving_observations" \
    --route-csv "results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/metadata/route_commanded.csv"

# -----------------------------------------------------------------------------
# Static branch first.
# These results do not depend on the moving-camera COLMAP reconstruction.
# -----------------------------------------------------------------------------

run_step 10_eval_direct_static_cam3_cam1 \
  python3 run/bus_real_data/approach1_marker_direct_relay/10_eval_direct_static_cam3_cam1.py

run_step 11_make_direct_static_report_cam3_cam1 \
  python3 run/bus_real_data/approach1_marker_direct_relay/11_make_direct_static_report_cam3_cam1.py

run_step 13_eval_direct_static_cam3_cam1_multimarker \
  python3 run/bus_real_data/approach1_marker_direct_relay/13_eval_direct_static_cam3_cam1_multimarker.py

# -----------------------------------------------------------------------------
# Moving branch is best-effort.
# Failure here must not delete the valid direct cam3-cam1 result.
# -----------------------------------------------------------------------------

COLMAP_OK=0
SCALE_OK=0

if run_step 06_run_colmap_moving_sequence \
  python3 run/bus_real_data/approach1_marker_direct_relay/06_run_colmap_moving_sequence.py
then
  COLMAP_OK=1
else
  echo "[WARN] AP01 moving COLMAP failed; retaining static direct results."
fi

if [ "$COLMAP_OK" = "1" ]; then
  run_step 07_evaluate_colmap_position_vs_gt \
    python3 run/bus_real_data/approach1_marker_direct_relay/07_evaluate_colmap_position_vs_gt.py \
    || echo "[WARN] AP01 COLMAP position diagnostic failed."

  run_step 08_make_colmap_error_tables \
    python3 run/bus_real_data/approach1_marker_direct_relay/08_make_colmap_error_tables.py \
    || echo "[WARN] AP01 COLMAP error-table diagnostic failed."

  run_step 09_evaluate_colmap_rotation_vs_gt \
    python3 run/bus_real_data/approach1_marker_direct_relay/09_evaluate_colmap_rotation_vs_gt.py \
    || echo "[WARN] AP01 COLMAP rotation diagnostic failed."

  if run_step 12_estimate_colmap_scale_from_aruco \
    python3 run/bus_real_data/approach1_marker_direct_relay/12_estimate_colmap_scale_from_aruco.py
  then
    SCALE_OK=1
  else
    echo "[WARN] AP01 moving metric scale failed; retaining static direct results."
  fi
fi

if [ "$SCALE_OK" = "1" ]; then
  run_step 14_eval_moving_relay_chains \
    python3 run/bus_real_data/approach1_marker_direct_relay/14_eval_moving_relay_chains.py \
    || echo "[WARN] AP01 moving relay evaluation incomplete."
else
  echo "[WARN] AP01 moving relay skipped because metric moving scale is unavailable."
fi

# Patched exporter catches missing relay rows and always exports the available subset.
run_step 15_export_final_extrinsics_cam3_reference \
  python3 run/bus_real_data/approach1_marker_direct_relay/15_export_final_extrinsics_cam3_reference.py

echo
echo "================================================================================"
echo "[OK] Full Approach 01 pipeline completed."
echo "================================================================================"

echo
echo "Final report:"
cat "$RESULT_ROOT/07_final_extrinsics_cam3_reference/FINAL_READABLE_REPORT.txt"
