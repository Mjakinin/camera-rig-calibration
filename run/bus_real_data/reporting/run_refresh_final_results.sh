#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

BUS="results/bus_real_data"
FINAL="$BUS/99_FINAL_RESULTS_FOR_REPORT"
BUILD="$BUS/.final_results_build"
CORE="$BUILD/core"
GENERATED="$BUILD/generated"

REUSE_BASELINE=0
PROMOTE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse-baseline)
      REUSE_BASELINE=1
      shift
      ;;
    --promote)
      PROMOTE=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

rm -rf "$BUILD"
mkdir -p "$CORE"

cleanup() {
  rm -rf "$BUILD"
}

trap cleanup EXIT

if [[ "$REUSE_BASELINE" == "1" ]]; then
  echo "=== Reuse existing clean-baseline evaluation ==="

  python3 - "$FINAL" "$CORE" <<'PY'
from pathlib import Path
import shutil
import sys


final = Path(sys.argv[1])
core = Path(sys.argv[2])


sources = {
    "BASELINE_FINAL_PAIRWISE_SUMMARY.csv": [
        final / "data/BASELINE_PRIMARY.csv",
        final / "data/primary/BASELINE_SUMMARY.csv",
        final / "data/BASELINE_PAIRWISE_SUMMARY.csv",
        final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv",
    ],
    "BASELINE_FINAL_PAIRWISE_DETAIL.csv": [
        final / "data/BASELINE_PRIMARY_DETAIL.csv",
        final / "data/primary/BASELINE_DETAIL.csv",
        final / "data/BASELINE_PAIRWISE_DETAIL.csv",
        final / "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
    ],
    "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv": [
        final / "data/BASELINE_SECONDARY.csv",
        final
        / "data/secondary/"
        "BASELINE_ALIGNED_CAMERA_MAP_SUMMARY.csv",
        final / "data/BASELINE_SECONDARY_SUMMARY.csv",
        final
        / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv",
    ],
    "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv": [
        final / "data/BASELINE_SECONDARY_DETAIL.csv",
        final
        / "data/secondary/"
        "BASELINE_ALIGNED_CAMERA_MAP_DETAIL.csv",
        final / "data/BASELINE_SECONDARY_DETAIL.csv",
        final
        / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv",
    ],
}


for destination_name, candidates in sources.items():
    source = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file()
        ),
        None,
    )

    if source is None:
        raise SystemExit(
            "[ERROR] Cannot reuse baseline; missing "
            + destination_name
        )

    destination = core / destination_name
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    print(
        f"[COPY] {source} -> {destination}"
    )
PY

else
  echo "=== Recalculate clean-baseline primary and secondary evaluation ==="

  python3 \
    run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py \
    --final-root "$CORE"
fi

echo
echo "=== Build canonical final reports ==="

python3 \
  run/bus_real_data/reporting/build_final_results.py \
  --core-root "$CORE" \
  --output-root "$GENERATED" \
  --preserve-from "$FINAL"

echo
echo "=== Validate generated structure ==="

required=(
  "00_READ_ME_FIRST.txt"
  "01_BASELINE_ALL_METHODS.txt"
  "02_ALL_ABLATIONS_ALL_METHODS.txt"
  "03_AP01_RESULTS.txt"
  "04_AP02_RESULTS.txt"
  "05_AP03_RESULTS.txt"
  "details/primary/00_BASELINE_CAM_TO_CAM.txt"
  "details/secondary/00_BASELINE_MAP_TO_GT.txt"
  "data/primary/ALL_ABLATIONS_DETAIL.csv"
  "data/secondary/ALL_ABLATIONS_ALIGNED_CAMERA_MAP_DETAIL.csv"
  "MANIFEST.json"
)

for relative in "${required[@]}"; do
  if [[ ! -f "$GENERATED/$relative" ]]; then
    echo "[ERROR] Missing generated file: $relative"
    exit 1
  fi
done

if [[ -d "$GENERATED/ablations/by_variant" ]]; then
  echo "[ERROR] Unexpected by_variant directory"
  exit 1
fi

if [[ "$PROMOTE" != "1" ]]; then
  echo
  echo "[OK] Validation build completed."
  echo "[OK] Existing final results were not replaced."
  echo "[INFO] Use --promote only after reviewing the validation output."
  exit 0
fi

echo
echo "=== Promote generated result tree without interrupting live logs ==="

mkdir -p "$FINAL"

# Runtime files can be open and actively followed while the overnight run is
# still executing. Keep their inodes and paths intact while replacing only the
# generated report artifacts.
find "$FINAL" -mindepth 1 -maxdepth 1 \
  ! -name 'OVERNIGHT_LIVE.log' \
  ! -name 'FULL_SIMULATION_RERUN.log' \
  ! -name 'PREFLIGHT.log' \
  ! -name 'LIVE_STATUS.txt' \
  -exec rm -rf -- {} +

shopt -s dotglob nullglob
generated_entries=("$GENERATED"/*)
if [[ "${#generated_entries[@]}" -gt 0 ]]; then
  mv "${generated_entries[@]}" "$FINAL/"
fi
shopt -u dotglob nullglob
rmdir "$GENERATED"

mkdir -p \
  "$BUS/00_shared_baseline" \
  "$BUS/01_marker_direct_relay_multimarker_multichain" \
  "$BUS/02_ref_marker_graph_ba" \
  "$BUS/03_targetless_colmap_aruco_scale"

cp \
  "$FINAL/01_BASELINE_ALL_METHODS.txt" \
  "$BUS/00_shared_baseline/RESULTS_SUMMARY.txt"

cp \
  "$FINAL/03_AP01_RESULTS.txt" \
  "$BUS/01_marker_direct_relay_multimarker_multichain/RESULTS_SUMMARY.txt"

cp \
  "$FINAL/04_AP02_RESULTS.txt" \
  "$BUS/02_ref_marker_graph_ba/RESULTS_SUMMARY.txt"

cp \
  "$FINAL/05_AP03_RESULTS.txt" \
  "$BUS/03_targetless_colmap_aruco_scale/RESULTS_SUMMARY.txt"

trap - EXIT
rm -rf "$BUILD"

echo
echo "[OK] Canonical final results refreshed."
echo "[OK] Existing generated report files were replaced."
echo "[OK] Live status and runtime logs remained readable."
echo "[OK] No result rows were appended."
echo "[OK] Output: $FINAL"
