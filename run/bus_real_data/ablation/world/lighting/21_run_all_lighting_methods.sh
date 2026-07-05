#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ROOT="results/bus_real_data/ablation/world/lighting"

VARIANTS=(
  ceiling_dark_extreme
  ceiling_low
  ceiling_normal
  ceiling_bright
)

COMMON_RUNNER="run/bus_real_data/ablation/_shared/12_run_one_clean_variant_common.sh"

if ! command -v colmap >/dev/null 2>&1; then
    echo "[ERROR] colmap is not available."
    exit 127
fi

for variant in "${VARIANTS[@]}"; do
    for required in \
      "$ROOT/$variant/raw_images" \
      "$ROOT/$variant/aruco_observations"
    do
        if [[ ! -d "$required" ]]; then
            echo "[ERROR] Missing required directory: $required"
            exit 1
        fi
    done
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="results/bus_real_data/_runtime_backups/pre_lighting_${STAMP}"

PATHS_TO_PRESERVE=(
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/metadata
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain
  results/bus_real_data/02_ref_marker_graph_ba
  results/bus_real_data/03_targetless_colmap_aruco_scale
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT
)

BACKED_UP=()

restore_original_state() {
    set +e

    echo
    echo "================================================================================"
    echo "RESTORING PRE-LIGHTING CANONICAL STATE"
    echo "================================================================================"

    for path in "${PATHS_TO_PRESERVE[@]}"; do
        rm -rf "$path"
    done

    for path in "${BACKED_UP[@]}"; do
        mkdir -p "$(dirname "$path")"

        if [[ -e "$BACKUP/$path" ]]; then
            mv "$BACKUP/$path" "$path"
            echo "[RESTORED] $path"
        fi
    done

    rm -rf "$BACKUP"
}

trap restore_original_state EXIT INT TERM

for path in "${PATHS_TO_PRESERVE[@]}"; do
    if [[ -e "$path" ]]; then
        mkdir -p "$BACKUP/$(dirname "$path")"
        mv "$path" "$BACKUP/$path"
        BACKED_UP+=("$path")
        echo "[BACKED UP] $path"
    fi
done

for variant in "${VARIANTS[@]}"; do
    echo
    echo "################################################################################"
    echo "RUNNING LIGHTING VARIANT: $variant"
    echo "################################################################################"

    bash "$COMMON_RUNNER" \
      "$ROOT" \
      "PHYSICAL CEILING LIGHTING ABLATION" \
      "$variant"

    python3 \
      run/bus_real_data/ablation/22_reconcile_ablation_status.py \
      "$ROOT"
done

python3 \
  run/bus_real_data/ablation/21_collect_full_ablation_report.py \
  "$ROOT" \
  "physical_ceiling_lighting"

echo
echo "[OK] All lighting method runs completed."
echo "[OK] Original canonical baseline will now be restored."
