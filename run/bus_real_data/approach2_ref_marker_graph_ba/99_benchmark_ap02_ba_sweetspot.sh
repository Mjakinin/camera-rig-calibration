#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

OUT="results/bus_real_data/02_ref_marker_graph_ba/99_ap02_ba_sweetspot"
AP02_BA_DIR="results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/with_moving"
AP02_FINAL="results/bus_real_data/02_ref_marker_graph_ba/08_final_results/ap02_with_moving_static_camera_poses_ref_marker.csv"

rm -rf "$OUT"
mkdir -p "$OUT"

cp -a "$AP02_BA_DIR" "$OUT/_backup_with_moving_before_sweetspot"
cp "$AP02_FINAL" "$OUT/_backup_ap02_final_before_sweetspot.csv"

SUMMARY="$OUT/AP02_BA_SWEETSPOT_SUMMARY.csv"
echo "variant,stride,max_frames,max_nfev,status,seconds,ba_success,ba_nfev,ba_final_mean_px,ba_final_median_px,ap02_mean_t_cm,ap02_mean_r_deg,ap02_worst_pair,ap02_worst_t_cm,ap02_worst_r_deg" > "$SUMMARY"

run_one () {
  variant="$1"
  stride="$2"
  max_frames="$3"
  max_nfev="$4"
  timeout_s="$5"

  echo
  echo "================================================================================"
  echo "[AP02 SWEETSPOT] $variant | stride=$stride max_frames=$max_frames max_nfev=$max_nfev timeout=${timeout_s}s"
  echo "================================================================================"

  rm -rf "$AP02_BA_DIR"

  start=$(date +%s)

  set +e
  timeout "$timeout_s" python3 -u run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py \
    --mode with_moving \
    --ref-marker-id 14 \
    --moving-stride "$stride" \
    --max-moving-frames "$max_frames" \
    --max-nfev "$max_nfev" \
    > "$OUT/${variant}_ba.log" 2>&1
  rc=$?
  set -e

  end=$(date +%s)
  seconds=$((end-start))

  if [ "$rc" -eq 124 ]; then
    status="TIMEOUT"
  elif [ "$rc" -eq 0 ]; then
    status="OK"
  else
    status="FAILED"
  fi

  mkdir -p "$OUT/$variant"

  if [ -d "$AP02_BA_DIR" ]; then
    cp -a "$AP02_BA_DIR" "$OUT/$variant/with_moving_ba"
  fi

  ba_success=""
  ba_nfev=""
  ba_mean=""
  ba_median=""

  if [ -f "$AP02_BA_DIR/ba_summary.txt" ]; then
    cp "$AP02_BA_DIR/ba_summary.txt" "$OUT/$variant/ba_summary.txt"
    ba_success=$(grep -E "^- success:" "$AP02_BA_DIR/ba_summary.txt" | sed 's/.*: //')
    ba_nfev=$(grep -E "^- nfev:" "$AP02_BA_DIR/ba_summary.txt" | sed 's/.*: //')
    ba_mean=$(grep -A5 "Final reprojection error" "$AP02_BA_DIR/ba_summary.txt" | grep "mean:" | sed 's/.*: //')
    ba_median=$(grep -A5 "Final reprojection error" "$AP02_BA_DIR/ba_summary.txt" | grep "median:" | sed 's/.*: //')
  fi

  ap02_mean_t=""
  ap02_mean_r=""
  ap02_worst_pair=""
  ap02_worst_t=""
  ap02_worst_r=""

  if [ "$status" = "OK" ] && [ -f "$AP02_BA_DIR/optimized_static_camera_poses_ref_marker.csv" ]; then
    python3 run/bus_real_data/approach2_ref_marker_graph_ba/08_export_ap02_final_results.py \
      > "$OUT/${variant}_export.log" 2>&1

    python3 run/bus_real_data/evaluation/10_eval_pairwise_static_camera_extrinsics.py \
      > "$OUT/${variant}_pairwise_eval.log" 2>&1

    cp results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP02/AP02_PAIRWISE_SUMMARY.csv \
      "$OUT/$variant/AP02_PAIRWISE_SUMMARY.csv" 2>/dev/null || true
    cp results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP02/AP02_PAIRWISE_RESULT.csv \
      "$OUT/$variant/AP02_PAIRWISE_RESULT.csv" 2>/dev/null || true

    vals=$(python3 - <<'PY'
import csv
from pathlib import Path
p = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP02/AP02_PAIRWISE_SUMMARY.csv")
r = next(csv.DictReader(p.open()))
print(",".join([
    r.get("mean_pair_t_cm",""),
    r.get("mean_pair_r_deg",""),
    r.get("worst_pair",""),
    r.get("worst_pair_t_cm",""),
    r.get("worst_pair_r_deg",""),
]))
PY
)
    ap02_mean_t=$(echo "$vals" | cut -d, -f1)
    ap02_mean_r=$(echo "$vals" | cut -d, -f2)
    ap02_worst_pair=$(echo "$vals" | cut -d, -f3)
    ap02_worst_t=$(echo "$vals" | cut -d, -f4)
    ap02_worst_r=$(echo "$vals" | cut -d, -f5)
  fi

  echo "$variant,$stride,$max_frames,$max_nfev,$status,$seconds,$ba_success,$ba_nfev,$ba_mean,$ba_median,$ap02_mean_t,$ap02_mean_r,$ap02_worst_pair,$ap02_worst_t,$ap02_worst_r" >> "$SUMMARY"

  echo
  echo "--- $variant summary row ---"
  tail -n 1 "$SUMMARY"
}

# timeout per candidate: 20 min. If one is too slow, it is skipped.
run_one "ap02_fast_40f_s5_n60"    5  40  60  1200
run_one "ap02_mid_70f_s4_n80"     4  70  80  1200
run_one "ap02_good_90f_s3_n100"   3  90 100  1200
run_one "ap02_heavy_120f_s2_n100" 2 120 100  1200

echo
echo "================================================================================"
echo "[AP02 SWEETSPOT SUMMARY]"
echo "================================================================================"
cat "$SUMMARY"

# Restore original AP02 final after benchmark.
rm -rf "$AP02_BA_DIR"
cp -a "$OUT/_backup_with_moving_before_sweetspot" "$AP02_BA_DIR"
cp "$OUT/_backup_ap02_final_before_sweetspot.csv" "$AP02_FINAL"

python3 run/bus_real_data/evaluation/10_eval_pairwise_static_camera_extrinsics.py \
  > "$OUT/recompute_after_restore.log" 2>&1

echo
echo "[OK] benchmark done. Original AP02 restored."
echo "[OK] results: $OUT"
