#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

ROOT="results/bus_real_data/ablation/world/lighting"
FINAL="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

# A failed older lighting backup could leave the tracked final-report tree absent.
# Recover only that tree from the current commit before touching method workspaces.
if [[ ! -d "$FINAL" ]]; then
  echo "[WARN] Missing $FINAL; restoring tracked canonical reports from HEAD."
  git restore --source=HEAD --worktree -- "$FINAL"
fi

if [[ ! -d "$FINAL" ]]; then
  echo "[ERROR] Could not restore canonical final-report tree: $FINAL"
  exit 1
fi

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

# The canonical final-report directory is deliberately NOT moved. On WSL/NTFS it
# can be open in Explorer or an editor, and moving it caused Permission denied.
# Variant runs are forced to skip canonical report refresh; the outer lighting
# driver refreshes reports once, after all four variants are complete.
PATHS_TO_PRESERVE=(
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/metadata
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain
  results/bus_real_data/02_ref_marker_graph_ba
  results/bus_real_data/03_targetless_colmap_aruco_scale
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

  REFRESH_CANONICAL_FINAL=0 \
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
echo "[OK] Canonical final reports were not moved or refreshed during variants."
echo "[OK] Original canonical method outputs will now be restored."
