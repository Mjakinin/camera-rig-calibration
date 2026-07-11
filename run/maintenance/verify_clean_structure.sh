#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "================================================================================"
echo "FINAL CLEAN STRUCTURE AUDIT"
echo "================================================================================"

python3 run/real_vehicle_data/00_validate_and_prepare_shared_input.py

for path in \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT \
  results/bus_real_data/00_method_lock \
  results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/raw_images \
  results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/aruco_observations
do
  test -e "$path" || {
    echo "[ERROR] missing $path"
    exit 1
  }
done

test ! -e \
  results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/raw_images/raw_images \
  || {
    echo "[ERROR] nested raw_images/raw_images still exists"
    exit 1
  }

test ! -e docs || {
  echo "[ERROR] docs still exists"
  exit 1
}

test ! -e tools/project_cleanup || {
  echo "[ERROR] temporary project_cleanup tooling still exists"
  exit 1
}

report_count="$(
  find results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT -type f | wc -l
)"
test "$report_count" -gt 1 || {
  echo "[ERROR] final simulation report is empty"
  exit 1
}

echo
echo "=== Restored simulation result roots ==="
for path in \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT
do
  printf '%-85s files=%s\n' "$path" "$(find "$path" -type f | wc -l)"
done

echo
echo "=== Root directories ==="
find . -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

echo
echo "=== .github ==="
if test -d .github; then
  find .github -maxdepth 3 -type f -print
else
  echo "not present"
fi

echo
echo "=== src/calib_lab/real_vehicle_data ==="
if test -d src/calib_lab/real_vehicle_data; then
  find src/calib_lab/real_vehicle_data -maxdepth 3 -print
else
  echo "not present"
fi

echo
echo "=== Git status summary ==="
git status --short
echo
git diff --cached --stat

echo
echo "[OK] final clean structure audit passed"
