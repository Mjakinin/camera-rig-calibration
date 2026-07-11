#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

ROOT="${1:?usage: $0 <root> <label> <variant>...}"
LABEL="${2:?usage: $0 <root> <label> <variant>...}"
shift 2
if [[ "$#" -eq 0 ]]; then
  echo "[ERROR] at least one variant is required"
  exit 2
fi
VARIANTS=("$@")

DETECTOR="run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"
COMMON="run/bus_real_data/ablation/_shared/12_run_one_clean_variant_common.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_LABEL="$(printf '%s' "$LABEL" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
BACKUP="results/bus_real_data/_runtime_backups/${SAFE_LABEL}_${STAMP}"
REUSE_EXISTING_OBSERVATIONS="${REUSE_EXISTING_OBSERVATIONS:-0}"

# The canonical final-report directory is intentionally not moved here.
# The common variant runner is called with REFRESH_CANONICAL_FINAL=0, so it
# cannot overwrite those reports. Keeping the directory in place allows the
# overnight live log and LIVE_STATUS.txt to remain readable throughout runs.
CANONICAL_PATHS=(
  results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain
  results/bus_real_data/02_ref_marker_graph_ba
  results/bus_real_data/03_targetless_colmap_aruco_scale
)
BACKED_UP=()

restore() {
  set +e
  for path in "${CANONICAL_PATHS[@]}"; do
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

for variant in "${VARIANTS[@]}"; do
  raw="$ROOT/$variant/raw_images"
  if [[ ! -d "$raw" ]]; then
    echo "[ERROR] missing raw_images for $variant: $raw"
    exit 1
  fi

  obs="$ROOT/$variant/aruco_observations"
  if [[ "$REUSE_EXISTING_OBSERVATIONS" == "1" && -s "$obs/shared_all_aruco_observations.csv" ]]; then
    echo "[REUSE] existing ArUco observations: $obs"
  else
    rm -rf "$obs"
    python3 "$DETECTOR" \
      --dataset "$raw" \
      --out "$obs" \
      --dictionary DICT_4X4_50
  fi
done

for path in "${CANONICAL_PATHS[@]}"; do
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
  echo "$LABEL: $variant"
  echo "################################################################################"
  REFRESH_CANONICAL_FINAL=0 bash "$COMMON" "$ROOT" "$LABEL" "$variant"
  python3 run/bus_real_data/ablation/22_reconcile_ablation_status.py "$ROOT"
done

python3 run/bus_real_data/ablation/21_collect_full_ablation_report.py \
  "$ROOT" "$SAFE_LABEL"

echo "[OK] completed group: $LABEL"
