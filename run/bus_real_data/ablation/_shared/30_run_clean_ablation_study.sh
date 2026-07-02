#!/usr/bin/env bash
set -u -o pipefail

# ABLATION_PREFLIGHT_COLMAP_GIT
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
git config --global --add safe.directory /workspaces/project >/dev/null 2>&1 || true

if ! git -C /workspaces/project rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "[ERROR] git repository not accessible; check safe.directory"
  exit 2
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found in PATH=$PATH"
  echo "Install with: apt-get update && apt-get install -y colmap"
  exit 127
fi


cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

ABL_ROOT="${1:?usage: $0 <ablation_root> <label> <summary_name> <variant...>}"
ABL_LABEL="${2:?usage: $0 <ablation_root> <label> <summary_name> <variant...>}"
SUMMARY_NAME="${3:?usage: $0 <ablation_root> <label> <summary_name> <variant...>}"
shift 3

mkdir -p "$ABL_ROOT/ABLATION_SUMMARY"

for variant in "$@"; do
  FINAL="$ABL_ROOT/$variant/FINAL_RESULTS"
  STATUS="$FINAL/RUN_STATUS.txt"

  if [ -f "$STATUS" ] && \
     grep -q "PAIRWISE_STATUS=.*OK" "$STATUS" && \
     grep -q "SECONDARY_STATUS=.*OK" "$STATUS"; then
    echo "[SKIP] $variant already evaluated"
    continue
  fi

  echo
  echo "================================================================================"
  echo "$ABL_LABEL $variant"
  echo "================================================================================"

  bash run/bus_real_data/ablation/_shared/12_run_one_clean_variant_common.sh \
    "$ABL_ROOT" "$ABL_LABEL" "$variant" \
    2>&1 | tee "$ABL_ROOT/$variant/RUN_FULL_AP01_AP02_AP03.log"
done

python3 run/bus_real_data/ablation/21_collect_full_ablation_report.py \
  "$ABL_ROOT" "$SUMMARY_NAME"

python3 run/bus_real_data/ablation/20_collect_ablation_final_summary.py \
  "$ABL_ROOT" "$SUMMARY_NAME" || true

echo
echo "[OK] full ablation study done: $ABL_ROOT"
echo "[OK] report: $ABL_ROOT/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt"
