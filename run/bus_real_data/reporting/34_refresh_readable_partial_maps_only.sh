#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD/run/bus_real_data:${PYTHONPATH:-}"

FINAL="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

required=(
  "run/bus_real_data/ablation/world/route/02_write_readable_route_reports.py"
  "run/bus_real_data/ablation/moving_cam/density/05_write_readable_density_reports.py"
  "run/bus_real_data/reporting/33_write_ref14_available_maps.py"
  "run/bus_real_data/reporting/35_remove_legacy_bestfit_full_maps.py"
  "$FINAL/details/secondary/00_BASELINE_MAP_TO_GT.txt"
  "$FINAL/details/secondary/01_FOV_MAP_TO_GT.txt"
  "$FINAL/details/secondary/02_MOTION_BLUR_MAP_TO_GT.txt"
  "$FINAL/details/secondary/03_RESOLUTION_MAP_TO_GT.txt"
  "$FINAL/details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt"
  "$FINAL/details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

echo "================================================================================"
echo "REPORT-ONLY REFRESH"
echo "- no image capture"
echo "- no ArUco redetection"
echo "- no AP01/AP02/AP03 rerun"
echo "- rewrites readable route and density detail reports"
echo "- removes legacy bottom blocks and old GT-aligned marker-map tables"
echo "- installs exactly one REF14 map inside each non-lighting variant block"
echo "- intentionally skips lighting until corrected recapture is finished"
echo "================================================================================"

python3 run/bus_real_data/ablation/world/route/02_write_readable_route_reports.py
python3 run/bus_real_data/ablation/moving_cam/density/05_write_readable_density_reports.py
python3 run/bus_real_data/reporting/35_remove_legacy_bestfit_full_maps.py
python3 run/bus_real_data/reporting/33_write_ref14_available_maps.py --skip-lighting

declare -A expected_counts=(
  ["00_BASELINE_MAP_TO_GT.txt"]=1
  ["01_FOV_MAP_TO_GT.txt"]=4
  ["02_MOTION_BLUR_MAP_TO_GT.txt"]=4
  ["03_RESOLUTION_MAP_TO_GT.txt"]=4
  ["05_ROUTE_PATH_MAP_TO_GT.txt"]=2
  ["06_FRAME_DENSITY_MAP_TO_GT.txt"]=7
)

for filename in "${!expected_counts[@]}"; do
  report="$FINAL/details/secondary/$filename"
  count="$(
    grep -c '^=== AP02 REF14-ANCHORED AVAILABLE MAP BEGIN:' "$report" \
      || true
  )"

  if [[ "$count" -ne "${expected_counts[$filename]}" ]]; then
    echo "[ERROR] $filename contains $count inline maps; expected ${expected_counts[$filename]}"
    exit 1
  fi

  if grep -q '^=== AP02 REF14-ANCHORED AVAILABLE MAPS BEGIN ===' "$report"; then
    echo "[ERROR] Legacy duplicated bottom block remains in $filename"
    exit 1
  fi

  if grep -q '^AP02 OPTIONAL GT-ALIGNED FULL MAP$' "$report"; then
    echo "[ERROR] Old GT-aligned marker-map table remains in $filename"
    exit 1
  fi

  echo "[OK] $filename: $count inline map section(s), no old/duplicate map"
done

echo
echo "[INFO] Lighting was not modified because its current captures are invalid."
echo "[INFO] The corrected lighting resume writes one inline map per lighting variant."
echo
cat "$FINAL/AP02_REF14_AVAILABLE_MAP_AUDIT.txt"

echo
echo "[OK] Report-only readable inline-map refresh complete."
