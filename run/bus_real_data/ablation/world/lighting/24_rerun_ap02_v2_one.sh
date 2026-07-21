#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VARIANT="${1:-}"

case "$VARIANT" in
    ceiling_dark_extreme|ceiling_low|ceiling_normal|ceiling_bright)
        ;;
    *)
        echo "Usage: $0 <lighting variant>"
        exit 2
        ;;
esac

ROOT="results/bus_real_data/ablation/world/lighting"
VAR_ROOT="$ROOT/$VARIANT"

SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"

AP02_RESULT="results/bus_real_data/02_ref_marker_graph_ba"
FINAL_RESULT="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

OUT="$VAR_ROOT/AP02_V2"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/_ap02_v2_runtime_backup_${STAMP}"

PRESERVE=(
  "$SHARED/raw_images"
  "$SHARED/aruco_observations"
  "$SHARED/metadata"
  "$AP02_RESULT"
  "$FINAL_RESULT"
)

BACKED_UP=()

restore() {
    set +e

    echo
    echo "=== Restore canonical repository results ==="

    for path in "${PRESERVE[@]}"; do
        rm -rf "$path"
    done

    for path in "${BACKED_UP[@]}"; do
        if [[ -e "$BACKUP/$path" ]]; then
            mkdir -p "$(dirname "$path")"
            mv "$BACKUP/$path" "$path"
            echo "[RESTORED] $path"
        fi
    done

    rm -rf "$BACKUP"
}

trap restore EXIT INT TERM

for required in \
  "$VAR_ROOT/raw_images" \
  "$VAR_ROOT/aruco_observations"
do
    if [[ ! -d "$required" ]]; then
        echo "[ERROR] Missing $required"
        exit 1
    fi
done

for path in "${PRESERVE[@]}"; do
    if [[ -e "$path" ]]; then
        mkdir -p "$BACKUP/$(dirname "$path")"
        mv "$path" "$BACKUP/$path"
        BACKED_UP+=("$path")
        echo "[BACKUP] $path"
    fi
done

mkdir -p "$SHARED"

cp -a \
  "$VAR_ROOT/raw_images" \
  "$SHARED/raw_images"

cp -a \
  "$VAR_ROOT/aruco_observations" \
  "$SHARED/aruco_observations"

if [[ -d "$VAR_ROOT/metadata" ]]; then
    cp -a \
      "$VAR_ROOT/metadata" \
      "$SHARED/metadata"
else
    mkdir -p "$SHARED/metadata"
fi

rm -rf "$OUT"
mkdir -p "$OUT"

echo
echo "================================================================================"
echo "AP02 V2: $VARIANT"
echo "================================================================================"

bash \
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
  --skip-shared-baseline

python3 \
  run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py

cp -a \
  "$AP02_RESULT" \
  "$OUT/AP02_INTERNAL"


if [[ -d "$FINAL_RESULT/AP02" ]]; then
    cp -a \
      "$FINAL_RESULT/AP02" \
      "$OUT/AP02"
fi

python3 - "$FINAL_RESULT" "$OUT" <<'PY'
import csv
import sys
from pathlib import Path


source_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])


def filter_method(
    source: Path,
    destination: Path,
    method: str,
) -> None:
    if not source.exists():
        raise RuntimeError(f"Missing evaluator file: {source}")

    with source.open(newline="", errors="replace") as file:
        rows = list(csv.DictReader(file))

    selected = [
        row
        for row in rows
        if row.get("method") == method
    ]

    if not selected:
        raise RuntimeError(
            f"No {method} rows in {source}"
        )

    fields = list(selected[0])

    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(selected)


filter_method(
    source_root / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv",
    output_root / "AP02_V2_PAIRWISE_SUMMARY.csv",
    "AP02",
)

filter_method(
    source_root / "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
    output_root / "AP02_V2_PAIRWISE_DETAIL.csv",
    "AP02",
)

print("[OK] wrote AP02 v2 evaluation files")
PY

echo
echo "=== AP02 V2 SUMMARY ==="

column -s, -t \
  < "$OUT/AP02_V2_PAIRWISE_SUMMARY.csv"

echo
echo "=== AP02 V2 PAIRS ==="

column -s, -t \
  < "$OUT/AP02_V2_PAIRWISE_DETAIL.csv"

echo
echo "[OK] AP02 v2 result stored at: $OUT"
