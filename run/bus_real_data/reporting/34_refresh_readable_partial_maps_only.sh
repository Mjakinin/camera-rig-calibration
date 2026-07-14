#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD/run/bus_real_data:${PYTHONPATH:-}"

FINAL="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

required=(
  "run/bus_real_data/ablation/world/route/02_write_readable_route_reports.py"
  "run/bus_real_data/ablation/moving_cam/density/05_write_readable_density_reports.py"
  "run/bus_real_data/reporting/33_write_ref14_available_maps.py"
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
echo "- appends REF14-anchored complete/partial AP02 maps"
echo "- intentionally skips lighting until corrected recapture is finished"
echo "================================================================================"

python3 run/bus_real_data/ablation/world/route/02_write_readable_route_reports.py
python3 run/bus_real_data/ablation/moving_cam/density/05_write_readable_density_reports.py

python3 - <<'PY'
from pathlib import Path
import runpy

namespace = runpy.run_path(
    "run/bus_real_data/reporting/33_write_ref14_available_maps.py"
)

namespace["GROUPS"][:] = [
    group
    for group in namespace["GROUPS"]
    if Path(group["report"]).name != "04_LIGHTING_MAP_TO_GT.txt"
]

namespace["main"]()
PY

for report in \
  "$FINAL/details/secondary/00_BASELINE_MAP_TO_GT.txt" \
  "$FINAL/details/secondary/01_FOV_MAP_TO_GT.txt" \
  "$FINAL/details/secondary/02_MOTION_BLUR_MAP_TO_GT.txt" \
  "$FINAL/details/secondary/03_RESOLUTION_MAP_TO_GT.txt" \
  "$FINAL/details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt" \
  "$FINAL/details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt"
do
  if ! grep -q "AP02 REF14-ANCHORED AVAILABLE CAMERA + MARKER MAPS" "$report"; then
    echo "[ERROR] Partial-map section missing from $report"
    exit 1
  fi
  echo "[OK] partial-map section: $report"
done

echo
echo "[INFO] Lighting was not modified because its current captures are invalid."
echo "[INFO] The corrected lighting rerun writes the lighting partial maps after recapture."
echo
cat "$FINAL/AP02_REF14_AVAILABLE_MAP_AUDIT.txt"

echo
echo "[OK] Report-only readable partial-map refresh complete."
