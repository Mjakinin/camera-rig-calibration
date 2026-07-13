#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BA_SCRIPT="run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba_distortion.py"

echo "=== AP02 Phase 3: distortion-aware graph BA static_only ==="

STATIC_ONLY_STATUS="OK"

if ! python3 \
  "$BA_SCRIPT" \
  --mode static_only \
  --max-nfev 100
then
  STATIC_ONLY_STATUS="FAILED"
  echo "[WARN] AP02 static_only BA unavailable."
  echo "[WARN] Continuing with the main with_moving BA."
fi

echo
echo "=== AP02 Phase 3: distortion-aware marker-aware with_moving BA ==="

python3 -u \
  "$BA_SCRIPT" \
  --mode with_moving \
  --moving-selection smart \
  --top-per-marker 8 \
  --top-per-pair 4 \
  --max-moving-frames 0 \
  --max-nfev 160

echo
echo "=== AP02 BA static_only summary ==="

STATIC_SUMMARY="results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/static_only/ba_summary.txt"

if [[ -f "$STATIC_SUMMARY" ]]; then
  cat "$STATIC_SUMMARY"
else
  echo "[INFO] static_only summary unavailable: $STATIC_ONLY_STATUS"
fi

echo
echo "=== AP02 BA with_moving summary ==="

cat \
  results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/with_moving/ba_summary.txt

echo
echo "=== Selected AP02 moving keyframes ==="

python3 - <<'PY'
from pathlib import Path
import csv

path = Path(
    "results/bus_real_data/02_ref_marker_graph_ba/"
    "07_graph_ba/with_moving/moving_frame_selection.csv"
)
rows = list(csv.DictReader(path.open()))
print(f"selected frames: {len(rows)}")
ref_rows = [
    row
    for row in rows
    if str(row.get("reference_marker_seen", "")).lower() in {"true", "1"}
]
print(f"selected Ref14 frames: {len(ref_rows)}")
print("Ref14 frame IDs:", [row["observer_id"] for row in ref_rows])
PY

echo
echo "[OK] AP02 distortion-aware marker-aware graph BA complete."
