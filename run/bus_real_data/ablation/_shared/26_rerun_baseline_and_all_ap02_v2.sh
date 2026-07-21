#!/usr/bin/env bash

set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

LOG_ROOT="results/bus_real_data/_master_logs/AP02_V2_${STAMP}"
STATE_ROOT="results/bus_real_data/_runtime_backups/AP02_V2_${STAMP}"

mkdir -p \
  "$LOG_ROOT" \
  "$STATE_ROOT"

FAILURES="$LOG_ROOT/FAILURES.txt"
ROOT_LIST="$LOG_ROOT/ABLATION_ROOTS.txt"
VARIANT_LIST="$LOG_ROOT/VARIANTS.txt"

: > "$FAILURES"
: > "$ROOT_LIST"
: > "$VARIANT_LIST"

SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"

AP01_ROOT="results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
AP02_ROOT="results/bus_real_data/02_ref_marker_graph_ba"
AP03_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
FINAL_ROOT="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

SHARED_TAR="$STATE_ROOT/clean_shared_baseline.tar"
BASELINE_AP02_TAR="$STATE_ROOT/baseline_ap02_v2.tar"
BASELINE_FINAL_TAR="$STATE_ROOT/baseline_final_results.tar"

record_failure() {
    echo "$*" | tee -a "$FAILURES"
}

run_required() {
    label="$1"
    shift

    echo
    echo "================================================================================"
    echo "$label"
    echo "================================================================================"

    "$@" 2>&1 | tee "$LOG_ROOT/${label}.log"

    rc="${PIPESTATUS[0]}"

    if [[ "$rc" -ne 0 ]]; then
        record_failure \
          "[FATAL] $label failed with return code $rc"

        exit "$rc"
    fi
}

restore_baseline_state() {
    set +e

    echo
    echo "================================================================================"
    echo "RESTORING CLEAN BASELINE STATE"
    echo "================================================================================"

    if [[ -f "$SHARED_TAR" ]]; then
        rm -rf "$SHARED"

        mkdir -p "$(dirname "$SHARED")"

        tar \
          -C "$(dirname "$SHARED")" \
          -xf "$SHARED_TAR"

        echo "[RESTORED] $SHARED"
    fi

    if [[ -f "$BASELINE_AP02_TAR" ]]; then
        rm -rf "$AP02_ROOT"

        tar \
          -C results/bus_real_data \
          -xf "$BASELINE_AP02_TAR"

        echo "[RESTORED] $AP02_ROOT"
    fi

    if [[ -f "$BASELINE_FINAL_TAR" ]]; then
        rm -rf "$FINAL_ROOT"

        tar \
          -C results/bus_real_data \
          -xf "$BASELINE_FINAL_TAR"

        echo "[RESTORED] $FINAL_ROOT"
    fi
}

echo "================================================================================"
echo "AP02 V2 BASELINE + ALL ABLATIONS MASTER RUN"
echo "stamp: $STAMP"
echo "logs:  $LOG_ROOT"
echo "================================================================================"

###############################################################################
# Preflight.
###############################################################################

if ! command -v colmap >/dev/null 2>&1; then
    record_failure "[FATAL] colmap not found"
    exit 127
fi

python3 -m py_compile \
  run/bus_real_data/approach2_ref_marker_graph_ba/ap02_observation_quality.py \
  run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py \
  run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py \
  run/bus_real_data/ablation/_shared/25_merge_ap02_v2_variant_results.py

if ! grep -q \
  "05_initialize_ref_marker_pose_graph_v2.py" \
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh
then
    record_failure \
      "[FATAL] AP02 phase 2 is not configured for v2 initializer"

    exit 1
fi

if ! grep -q \
  -- "--moving-selection smart" \
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase3_graph_ba_fast.sh
then
    record_failure \
      "[FATAL] AP02 phase 3 is not configured for smart frame selection"

    exit 1
fi

STATIC_COUNT="$(
  find "$SHARED/raw_images/static" \
    -maxdepth 1 \
    -name 'cam_edge_*.png' \
    | wc -l
)"

MOVING_COUNT="$(
  find "$SHARED/raw_images/moving" \
    -maxdepth 1 \
    -name 'frame_*.png' \
    | wc -l
)"

INFO_COUNT="$(
  find "$SHARED/raw_images/camera_info" \
    -maxdepth 1 \
    -name '*.json' \
    | wc -l
)"

echo "Shared-baseline preflight:"
echo "- static:     $STATIC_COUNT / 4"
echo "- moving:     $MOVING_COUNT / 189"
echo "- cameraInfo: $INFO_COUNT / 5"

if [[ "$STATIC_COUNT" -ne 4 ]] \
  || [[ "$MOVING_COUNT" -ne 189 ]] \
  || [[ "$INFO_COUNT" -ne 5 ]]
then
    record_failure \
      "[FATAL] clean shared baseline is structurally incomplete"

    exit 1
fi

###############################################################################
# Full clean baseline rerun: AP01 + AP02 v2 + AP03.
###############################################################################

echo
echo "================================================================================"
echo "CLEAN BASELINE: RESET METHOD OUTPUTS"
echo "================================================================================"

rm -rf \
  "$AP01_ROOT" \
  "$AP02_ROOT" \
  "$AP03_ROOT" \
  "$FINAL_ROOT"

run_required \
  baseline_01_shared_detection \
  bash \
  run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh \
  --dataset "$SHARED/raw_images" \
  --out "$SHARED/aruco_observations"

run_required \
  baseline_02_ap01 \
  env \
  RUN_SHARED_BASELINE=0 \
  bash \
  run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh

run_required \
  baseline_03_ap02_v2 \
  bash \
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
  --skip-shared-baseline

run_required \
  baseline_04_ap03_prepare \
  python3 \
  run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
  --moving-stride 1 \
  --max-moving 0

run_required \
  baseline_05_ap03_colmap \
  bash \
  run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh

run_required \
  baseline_06_ap03_inspect \
  python3 \
  run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py

run_required \
  baseline_07_ap03_marker_scale \
  python3 \
  run/bus_real_data/approach3_targetless_colmap_aruco_scale/10_estimate_scale_from_marker_size_only.py \
  --marker-ids 0-14,16-20 \
  --min-area-px2 100 \
  --reproj-thresh-px 5 \
  --ransac-iters 1000 \
  --min-inliers 4

run_required \
  baseline_08_partial_aware_evaluation \
  python3 \
  run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py

echo
echo "================================================================================"
echo "FREEZE NEW CLEAN BASELINE RESULTS"
echo "================================================================================"

rm -rf "$SHARED/FINAL_RESULTS"

cp -a \
  "$FINAL_ROOT" \
  "$SHARED/FINAL_RESULTS"

mkdir -p \
  "$SHARED/BASELINE_METHOD_REPORTS"

cp -a \
  "$AP02_ROOT/08_final_results/." \
  "$SHARED/BASELINE_METHOD_REPORTS/AP02_V2/" \
  2>/dev/null || true

cp \
  "$AP01_ROOT/07_final_extrinsics_cam3_reference/FINAL_READABLE_REPORT.txt" \
  "$SHARED/BASELINE_METHOD_REPORTS/AP01_FINAL_READABLE_REPORT.txt" \
  2>/dev/null || true

cp \
  "$AP03_ROOT/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_REPORT.txt" \
  "$SHARED/BASELINE_METHOD_REPORTS/AP03_MARKER_SIZE_SCALE_ONLY_REPORT.txt" \
  2>/dev/null || true

tar \
  -C "$(dirname "$SHARED")" \
  -cf "$SHARED_TAR" \
  "$(basename "$SHARED")"

tar \
  -C results/bus_real_data \
  -cf "$BASELINE_AP02_TAR" \
  02_ref_marker_graph_ba

tar \
  -C results/bus_real_data \
  -cf "$BASELINE_FINAL_TAR" \
  99_FINAL_RESULTS_FOR_REPORT

trap restore_baseline_state EXIT INT TERM

###############################################################################
# Discover every complete ablation variant.
###############################################################################

python3 - <<'PY' > "$VARIANT_LIST"
from pathlib import Path

root = Path(
    "results/bus_real_data/ablation"
)

variants = []

for raw_images in root.rglob("raw_images"):
    variant = raw_images.parent

    try:
        relative_parts = variant.relative_to(root).parts
    except ValueError:
        continue

    if any(
        part.startswith("_")
        for part in relative_parts
    ):
        continue

    if not (
        variant / "aruco_observations"
    ).is_dir():
        continue

    if not (
        variant / "FINAL_RESULTS"
    ).is_dir():
        continue

    variants.append(variant)

for variant in sorted(set(variants)):
    print(variant)
PY

VARIANT_TOTAL="$(
  wc -l < "$VARIANT_LIST"
)"

echo
echo "================================================================================"
echo "DISCOVERED ABLATION VARIANTS: $VARIANT_TOTAL"
echo "================================================================================"

cat "$VARIANT_LIST"

###############################################################################
# AP02 v2 only for every variant.
###############################################################################

CURRENT=0

while IFS= read -r VAR_ROOT
do
    [[ -n "$VAR_ROOT" ]] || continue

    CURRENT=$((CURRENT + 1))

    GROUP_ROOT="$(dirname "$VAR_ROOT")"
    VARIANT="$(basename "$VAR_ROOT")"
    VAR_FINAL="$VAR_ROOT/FINAL_RESULTS"

    echo "$GROUP_ROOT" >> "$ROOT_LIST"

    VAR_LOG="$VAR_ROOT/AP02_V2_RERUN.log"

    echo
    echo "################################################################################"
    echo "AP02 V2 VARIANT $CURRENT / $VARIANT_TOTAL"
    echo "variant: $VARIANT"
    echo "root:    $VAR_ROOT"
    echo "################################################################################"

    if [[ ! -d "$VAR_ROOT/raw_images" ]] \
      || [[ ! -d "$VAR_ROOT/aruco_observations" ]]
    then
        record_failure \
          "[SKIP] $VAR_ROOT: missing raw_images or aruco_observations"

        continue
    fi

    rm -rf \
      "$SHARED/raw_images" \
      "$SHARED/aruco_observations" \
      "$SHARED/metadata"

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

    rm -rf \
      "$AP02_ROOT" \
      "$FINAL_ROOT"

    set +e

    bash \
      run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
      --skip-shared-baseline \
      --skip-report \
      2>&1 | tee "$VAR_LOG"

    AP02_RC="${PIPESTATUS[0]}"

    python3 \
      run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py \
      2>&1 | tee -a "$VAR_LOG"

    EVAL_RC="${PIPESTATUS[0]}"

    set -e

    if [[ "$EVAL_RC" -ne 0 ]]; then
        record_failure \
          "[FAILED EVALUATOR] $VAR_ROOT AP02_RC=$AP02_RC EVAL_RC=$EVAL_RC"

        python3 - "$VAR_FINAL/RUN_STATUS.txt" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
values = {}

if path.exists():
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

values["variant"] = values.get(
    "variant",
    path.parent.parent.name,
)

values["AP01_STATUS"] = values.get(
    "AP01_STATUS",
    "UNKNOWN",
)

values["AP02_STATUS"] = "FAILED_V2_EVALUATOR"

values["AP03_STATUS"] = values.get(
    "AP03_STATUS",
    "UNKNOWN",
)

values["PAIRWISE_STATUS"] = values.get(
    "PAIRWISE_STATUS",
    "UNKNOWN",
)

values["SECONDARY_STATUS"] = values.get(
    "SECONDARY_STATUS",
    "UNKNOWN",
)

order = [
    "variant",
    "AP01_STATUS",
    "AP02_STATUS",
    "AP03_STATUS",
    "PAIRWISE_STATUS",
    "SECONDARY_STATUS",
]

path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

path.write_text(
    "\n".join(
        f"{key}={values[key]}"
        for key in order
    )
    + "\n"
)
PY

        continue
    fi

    python3 \
      run/bus_real_data/ablation/_shared/25_merge_ap02_v2_variant_results.py \
      --variant-root "$VAR_ROOT" \
      --pipeline-rc "$AP02_RC" \
      2>&1 | tee -a "$VAR_LOG"

    MERGE_RC="${PIPESTATUS[0]}"

    if [[ "$MERGE_RC" -ne 0 ]]; then
        record_failure \
          "[FAILED MERGE] $VAR_ROOT merge_rc=$MERGE_RC"

        continue
    fi

    if [[ "$AP02_RC" -ne 0 ]]; then
        record_failure \
          "[AP02 PIPELINE NONZERO, RESULT RETAINED] $VAR_ROOT rc=$AP02_RC"
    fi

done < "$VARIANT_LIST"

###############################################################################
# Restore canonical baseline before rebuilding ablation summaries.
###############################################################################

restore_baseline_state
trap - EXIT INT TERM

###############################################################################
# Reconcile and rebuild every affected ablation summary.
###############################################################################

sort -u \
  "$ROOT_LIST" \
  -o "$ROOT_LIST"

while IFS= read -r GROUP_ROOT
do
    [[ -n "$GROUP_ROOT" ]] || continue

    LABEL="$(
      echo "$GROUP_ROOT" \
      | sed 's#results/bus_real_data/ablation/##' \
      | tr '/' '_'
    )"

    echo
    echo "================================================================================"
    echo "REBUILD ABLATION SUMMARY: $GROUP_ROOT"
    echo "================================================================================"

    python3 \
      run/bus_real_data/ablation/22_reconcile_ablation_status.py \
      "$GROUP_ROOT" \
      2>&1 | tee \
      "$LOG_ROOT/reconcile_${LABEL}.log"

    RECONCILE_RC="${PIPESTATUS[0]}"

    if [[ "$RECONCILE_RC" -ne 0 ]]; then
        record_failure \
          "[FAILED RECONCILE] $GROUP_ROOT rc=$RECONCILE_RC"
    fi

    python3 \
      run/bus_real_data/ablation/21_collect_full_ablation_report.py \
      "$GROUP_ROOT" \
      "${LABEL}_ap02_v2" \
      2>&1 | tee \
      "$LOG_ROOT/collect_${LABEL}.log"

    COLLECT_RC="${PIPESTATUS[0]}"

    if [[ "$COLLECT_RC" -ne 0 ]]; then
        record_failure \
          "[FAILED COLLECT] $GROUP_ROOT rc=$COLLECT_RC"
    fi

done < "$ROOT_LIST"

###############################################################################
# Build global AP02 v2 result and v1-v2 comparison tables.
###############################################################################

python3 - <<'PY'
from __future__ import annotations

import csv
import math
from pathlib import Path


ABLATION = Path(
    "results/bus_real_data/ablation"
)

OUT = (
    ABLATION
    / "AP02_V2_SUMMARY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def read_csv(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        errors="replace",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        output = float(value)
        return output if math.isfinite(output) else None
    except (TypeError, ValueError):
        return None


current_rows = []
comparison_rows = []

baseline_summary = Path(
    "results/bus_real_data/"
    "99_FINAL_RESULTS_FOR_REPORT/"
    "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
)

for row in read_csv(baseline_summary):
    if row.get("method") != "AP02":
        continue

    current_rows.append({
        "ablation": "baseline",
        "variant": "clean_shared_baseline",
        **row,
    })

for summary in sorted(
    ABLATION.rglob(
        "FINAL_RESULTS/"
        "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    )
):
    variant_root = summary.parent.parent

    try:
        relative = variant_root.relative_to(
            ABLATION
        )
    except ValueError:
        continue

    if any(
        part.startswith("_")
        for part in relative.parts
    ):
        continue

    current = next(
        (
            row
            for row in read_csv(summary)
            if row.get("method") == "AP02"
        ),
        None,
    )

    if current is None:
        continue

    ablation_name = "/".join(
        relative.parts[:-1]
    )

    variant_name = relative.parts[-1]

    current_rows.append({
        "ablation": ablation_name,
        "variant": variant_name,
        **current,
    })

    old_path = (
        summary.parent
        / "AP02_V1_ARCHIVE"
        / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    )

    old = next(
        (
            row
            for row in read_csv(old_path)
            if row.get("method") == "AP02"
        ),
        None,
    )

    if old is None:
        continue

    old_t = number(
        old.get("mean_pair_t_cm")
    )

    new_t = number(
        current.get("mean_pair_t_cm")
    )

    old_r = number(
        old.get("mean_pair_r_deg")
    )

    new_r = number(
        current.get("mean_pair_r_deg")
    )

    comparison_rows.append({
        "ablation": ablation_name,
        "variant": variant_name,
        "v1_status": old.get("status", ""),
        "v2_status": current.get("status", ""),
        "v1_mean_pair_t_cm": (
            "" if old_t is None else old_t
        ),
        "v2_mean_pair_t_cm": (
            "" if new_t is None else new_t
        ),
        "translation_change_cm": (
            ""
            if old_t is None or new_t is None
            else new_t - old_t
        ),
        "translation_improvement_percent": (
            ""
            if (
                old_t is None
                or new_t is None
                or abs(old_t) < 1e-12
            )
            else 100.0 * (old_t - new_t) / old_t
        ),
        "v1_mean_pair_r_deg": (
            "" if old_r is None else old_r
        ),
        "v2_mean_pair_r_deg": (
            "" if new_r is None else new_r
        ),
        "rotation_change_deg": (
            ""
            if old_r is None or new_r is None
            else new_r - old_r
        ),
        "rotation_improvement_percent": (
            ""
            if (
                old_r is None
                or new_r is None
                or abs(old_r) < 1e-12
            )
            else 100.0 * (old_r - new_r) / old_r
        ),
        "v1_worst_pair": old.get(
            "worst_pair",
            "",
        ),
        "v2_worst_pair": current.get(
            "worst_pair",
            "",
        ),
        "v1_worst_pair_t_cm": old.get(
            "worst_pair_t_cm",
            "",
        ),
        "v2_worst_pair_t_cm": current.get(
            "worst_pair_t_cm",
            "",
        ),
    })

write_csv(
    OUT / "AP02_V2_ALL_RESULTS.csv",
    current_rows,
)

write_csv(
    OUT / "AP02_V1_V2_COMPARISON.csv",
    comparison_rows,
)

report = [
    "AP02 V2 BASELINE AND ABLATION SUMMARY",
    "======================================",
    "",
    f"Current AP02 result rows: {len(current_rows)}",
    f"V1/V2 comparison rows: {len(comparison_rows)}",
    "",
    "Files:",
    f"- {OUT / 'AP02_V2_ALL_RESULTS.csv'}",
    f"- {OUT / 'AP02_V1_V2_COMPARISON.csv'}",
]

(
    OUT / "README.txt"
).write_text(
    "\n".join(report) + "\n"
)

print("\n".join(report))
PY

###############################################################################
# Final state summary.
###############################################################################

echo
echo "================================================================================"
echo "FINAL CLEAN BASELINE"
echo "================================================================================"

cat \
  "$FINAL_ROOT/BASELINE_FINAL_CLEAN_COMPARISON.txt" \
  2>/dev/null || true

echo
echo "================================================================================"
echo "MASTER RUN COMPLETE"
echo "================================================================================"

echo "Baseline final results:"
echo "  $FINAL_ROOT"

echo
echo "All AP02 v2 rows:"
echo "  results/bus_real_data/ablation/AP02_V2_SUMMARY/AP02_V2_ALL_RESULTS.csv"

echo
echo "V1 versus V2:"
echo "  results/bus_real_data/ablation/AP02_V2_SUMMARY/AP02_V1_V2_COMPARISON.csv"

echo
echo "Failures/warnings:"
echo "  $FAILURES"

echo
echo "Logs:"
echo "  $LOG_ROOT"

date -Iseconds \
  > "$LOG_ROOT/DONE.txt"

if [[ -s "$FAILURES" ]]; then
    echo
    echo "[WARN] Master run completed with recorded warnings/failures:"
    cat "$FAILURES"
else
    echo
    echo "[OK] Master run completed without recorded failures."
fi
