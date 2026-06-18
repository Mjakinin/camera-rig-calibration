#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

# Do not use `set -u` before sourcing ROS setup files.
# ROS setup scripts may reference optional unset variables.
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

set -eo pipefail

RESULT_ROOT="results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
LOG_DIR="$RESULT_ROOT/_pipeline_logs"

mkdir -p "$LOG_DIR"

run_step () {
  STEP="$1"
  SCRIPT="$2"

  echo
  echo "================================================================================"
  echo "RUNNING $STEP: $SCRIPT"
  echo "================================================================================"

  python3 "$SCRIPT" 2>&1 | tee "$LOG_DIR/${STEP}.log"
}

run_step 01_detect_static_a4_markers run/bus_real_data/01_detect_static_a4_markers.py
run_step 02_generate_moving_camera_route run/bus_real_data/02_generate_moving_camera_route.py
run_step 03_preview_moving_camera_route run/bus_real_data/03_preview_moving_camera_route.py
run_step 04_capture_moving_camera_route run/bus_real_data/04_capture_moving_camera_route.py
run_step 05_detect_moving_a4_markers run/bus_real_data/05_detect_moving_a4_markers.py
run_step 06_run_colmap_moving_sequence run/bus_real_data/06_run_colmap_moving_sequence.py
run_step 07_evaluate_colmap_position_vs_gt run/bus_real_data/07_evaluate_colmap_position_vs_gt.py
run_step 08_make_colmap_error_tables run/bus_real_data/08_make_colmap_error_tables.py
run_step 09_evaluate_colmap_rotation_vs_gt run/bus_real_data/09_evaluate_colmap_rotation_vs_gt.py
run_step 10_eval_direct_static_cam3_cam1 run/bus_real_data/10_eval_direct_static_cam3_cam1.py
run_step 11_make_direct_static_report_cam3_cam1 run/bus_real_data/11_make_direct_static_report_cam3_cam1.py
run_step 12_estimate_colmap_scale_from_aruco run/bus_real_data/12_estimate_colmap_scale_from_aruco.py
run_step 13_eval_direct_static_cam3_cam1_multimarker run/bus_real_data/13_eval_direct_static_cam3_cam1_multimarker.py
run_step 14_eval_moving_relay_chains run/bus_real_data/14_eval_moving_relay_chains.py
run_step 15_export_final_extrinsics_cam3_reference run/bus_real_data/15_export_final_extrinsics_cam3_reference.py

echo
echo "================================================================================"
echo "[OK] Full Approach 01 pipeline completed."
echo "================================================================================"

echo
echo "Final report:"
cat "$RESULT_ROOT/07_final_extrinsics_cam3_reference/FINAL_READABLE_REPORT.txt"
