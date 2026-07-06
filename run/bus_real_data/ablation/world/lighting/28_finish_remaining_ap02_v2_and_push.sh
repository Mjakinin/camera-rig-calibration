#!/usr/bin/env bash

set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

LOG_ROOT="results/bus_real_data/_master_logs/AP02_V2_FINISH_${STAMP}"
FAILURES="$LOG_ROOT/FAILURES.txt"

mkdir -p "$LOG_ROOT"
: > "$FAILURES"

exec > >(tee "$LOG_ROOT/FULL_RUN.log") 2>&1

LIGHT_ROOT="results/bus_real_data/ablation/world/lighting"

SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"

AP01_ROOT="results/bus_real_data/01_marker_direct_relay_multimarker_multichain"
AP02_ROOT="results/bus_real_data/02_ref_marker_graph_ba"
AP03_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
FINAL_ROOT="results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"

MERGE_HELPER="run/bus_real_data/ablation/_shared/25_merge_ap02_v2_variant_results.py"

PHASE3_SCRIPT="run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase3_graph_ba_fast.sh"
BA_SCRIPT="run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py"
EXPORT_SCRIPT="run/bus_real_data/approach2_ref_marker_graph_ba/08_export_ap02_final_results.py"

OUT_SUMMARY="results/bus_real_data/ablation/AP02_V2_SUMMARY"

mkdir -p "$OUT_SUMMARY"

fail() {
    echo "[ERROR] $*" | tee -a "$FAILURES"
}

fatal() {
    fail "$*"
    exit 1
}

echo "================================================================================"
echo "FINISH REMAINING AP02 V2 LIGHTING VARIANTS"
echo "stamp: $STAMP"
echo "================================================================================"

###############################################################################
# Refuse to start if the old master is still genuinely active.
###############################################################################

ACTIVE_OLD="$(
  pgrep -af \
    '26_rerun_baseline_and_all_ap02_v2.sh|27_resume_fast_ap02_v2_all.sh' \
    2>/dev/null \
  | grep -v \
    '28_finish_remaining_ap02_v2_and_push.sh' \
  || true
)"

if [[ -n "$ACTIVE_OLD" ]]; then
    echo "$ACTIVE_OLD"
    fatal "An older AP02-v2 master process is still active."
fi

###############################################################################
# Validate required local implementation.
###############################################################################

for REQUIRED in \
  "$MERGE_HELPER" \
  "$BA_SCRIPT" \
  "$EXPORT_SCRIPT" \
  "$PHASE3_SCRIPT" \
  run/bus_real_data/approach2_ref_marker_graph_ba/ap02_observation_quality.py \
  run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py
do
    [[ -f "$REQUIRED" ]] \
      || fatal "Missing required file: $REQUIRED"
done

python3 -m py_compile \
  "$MERGE_HELPER" \
  "$BA_SCRIPT" \
  run/bus_real_data/approach2_ref_marker_graph_ba/ap02_observation_quality.py \
  run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py \
  || fatal "AP02 Python compile check failed."

grep -q \
  "05_initialize_ref_marker_pose_graph_v2.py" \
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh \
  || fatal "Phase 2 is not using AP02 v2 initialization."

grep -q \
  -- "--moving-selection smart" \
  "$PHASE3_SCRIPT" \
  || fatal "Phase 3 is not using marker-aware selection."

###############################################################################
# Locate one complete frozen clean-baseline state.
###############################################################################

STATE_DIR="$(
  find \
    results/bus_real_data/_runtime_backups \
    -type f \
    -name baseline_final_results.tar \
    -printf '%T@ %h\n' \
    2>/dev/null \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"

[[ -n "$STATE_DIR" ]] \
  || fatal "No frozen baseline runtime backup found."

CLEAN_SHARED_TAR="$STATE_DIR/clean_shared_baseline.tar"
BASELINE_AP02_TAR="$STATE_DIR/baseline_ap02_v2.tar"
BASELINE_FINAL_TAR="$STATE_DIR/baseline_final_results.tar"

for ARCHIVE in \
  "$CLEAN_SHARED_TAR" \
  "$BASELINE_AP02_TAR" \
  "$BASELINE_FINAL_TAR"
do
    [[ -f "$ARCHIVE" ]] \
      || fatal "Missing baseline archive: $ARCHIVE"
done

echo "[OK] baseline state: $STATE_DIR"

###############################################################################
# AP01/AP03 are needed only so that the evaluator can produce its temporary
# combined table. The merge helper retains the original per-variant AP01/AP03.
###############################################################################

[[ -f \
  "$AP01_ROOT/07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv" \
]] || fatal "Canonical AP01 baseline output is missing."

[[ -f \
  "$AP03_ROOT/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv" \
]] || fatal "Canonical AP03 baseline output is missing."

NEED_RESTORE=1

restore_clean_baseline() {
    set +e

    if [[ "$NEED_RESTORE" != "1" ]]; then
        return
    fi

    echo
    echo "================================================================================"
    echo "RESTORE CLEAN BASELINE"
    echo "================================================================================"

    chmod -R u+rwX "$SHARED" 2>/dev/null || true

    rm -rf "$SHARED"
    mkdir -p "$(dirname "$SHARED")"

    tar \
      -C "$(dirname "$SHARED")" \
      -xf "$CLEAN_SHARED_TAR"

    rm -rf "$AP02_ROOT"

    tar \
      -C results/bus_real_data \
      -xf "$BASELINE_AP02_TAR"

    rm -rf "$FINAL_ROOT"

    tar \
      -C results/bus_real_data \
      -xf "$BASELINE_FINAL_TAR"

    echo "[OK] clean shared baseline restored"
    echo "[OK] baseline AP02 restored"
    echo "[OK] baseline final tables restored"

    NEED_RESTORE=0
}

trap restore_clean_baseline EXIT INT TERM

###############################################################################
# Reconstruct any old per-variant table which was removed by the interrupted
# master. The existing aggregate contains its previous v1/AP01/AP03 rows.
###############################################################################

repair_variant_tables() {
    local VARIANT="$1"
    local VAR_FINAL="$LIGHT_ROOT/$VARIANT/FINAL_RESULTS"

    python3 - \
      "$LIGHT_ROOT" \
      "$VARIANT" \
      "$VAR_FINAL" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path


root = Path(sys.argv[1])
variant = sys.argv[2]
final = Path(sys.argv[3])

summary_root = root / "ABLATION_SUMMARY"

mapping = {
    "BASELINE_FINAL_PAIRWISE_SUMMARY.csv":
        summary_root / "PRIMARY_PAIRWISE_SUMMARY_ALL_VARIANTS.csv",

    "BASELINE_FINAL_PAIRWISE_DETAIL.csv":
        summary_root / "PRIMARY_PAIRWISE_DETAIL_ALL_VARIANTS.csv",

    "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv":
        summary_root / "SECONDARY_REF14_WORLD_SUMMARY_ALL_VARIANTS.csv",

    "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv":
        summary_root / "SECONDARY_REF14_WORLD_DETAIL_ALL_VARIANTS.csv",
}


def read_csv(path: Path):
    if not path.exists():
        return []

    with path.open(newline="", errors="replace") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows):
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


for target_name, aggregate in mapping.items():
    target = final / target_name

    if target.exists() and target.stat().st_size > 10:
        continue

    selected = []

    for row in read_csv(aggregate):
        if row.get("variant") != variant:
            continue

        restored = dict(row)
        restored.pop("variant", None)
        selected.append(restored)

    if not selected:
        raise RuntimeError(
            f"Cannot restore {target_name} for {variant} "
            f"from {aggregate}"
        )

    write_csv(target, selected)
    print(f"[RESTORED OLD TABLE] {target}")
PY

    local ARCHIVE="$VAR_FINAL/AP02_V1_ARCHIVE"

    if [[ -d "$ARCHIVE" ]] \
      && [[ ! -s "$ARCHIVE/BASELINE_FINAL_PAIRWISE_SUMMARY.csv" ]]
    then
        echo "[INFO] removing incomplete v1 archive: $ARCHIVE"
        rm -rf "$ARCHIVE"
    fi
}

variant_is_complete() {
    local VAR_FINAL="$1"

    python3 - "$VAR_FINAL" <<'PY'
import csv
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])

meta = root / "AP02_V2_METADATA.json"
summary = root / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
archive = (
    root
    / "AP02_V1_ARCHIVE"
    / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
)

if not (meta.exists() and summary.exists() and archive.exists()):
    raise SystemExit(1)

try:
    data = json.loads(meta.read_text())
except Exception:
    raise SystemExit(1)

if str(data.get("version", "")).lower() != "v2":
    raise SystemExit(1)

with summary.open(newline="", errors="replace") as file:
    rows = list(csv.DictReader(file))

if not any(row.get("method") == "AP02" for row in rows):
    raise SystemExit(1)

raise SystemExit(0)
PY
}

###############################################################################
# Run exactly the missing Lighting variants.
###############################################################################

VARIANTS=(
  ceiling_dark_extreme
  ceiling_low
  ceiling_normal
  ceiling_bright
)

for VARIANT in "${VARIANTS[@]}"
do
    VAR_ROOT="$LIGHT_ROOT/$VARIANT"
    VAR_FINAL="$VAR_ROOT/FINAL_RESULTS"
    VAR_LOG="$VAR_ROOT/AP02_V2_RERUN.log"

    echo
    echo "################################################################################"
    echo "VARIANT: $VARIANT"
    echo "################################################################################"

    [[ -d "$VAR_ROOT/raw_images" ]] \
      || fatal "Missing raw_images: $VAR_ROOT"

    [[ -d "$VAR_ROOT/aruco_observations" ]] \
      || fatal "Missing observations: $VAR_ROOT"

    mkdir -p "$VAR_FINAL"

    repair_variant_tables "$VARIANT"

    if variant_is_complete "$VAR_FINAL"; then
        echo "[SKIP] AP02 v2 already complete: $VARIANT"
        continue
    fi

    echo
    echo "=== Install variant into shared baseline ==="

    chmod -R u+rwX "$SHARED" 2>/dev/null || true

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

    echo
    echo "=== Run AP02 v2 ==="

    set +e

    bash \
      run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
      --skip-shared-baseline \
      --skip-report \
      2>&1 | tee "$VAR_LOG"

    AP02_RC="${PIPESTATUS[0]}"

    set -e

    WITH_MOVING_OUTPUT="$AP02_ROOT/07_graph_ba/with_moving/optimized_static_camera_poses_ref_marker.csv"

    if [[ "$AP02_RC" -ne 0 ]] \
      && [[ ! -s "$WITH_MOVING_OUTPUT" ]]
    then
        echo
        echo "[WARN] Full pipeline returned $AP02_RC."
        echo "[INFO] Retrying only the final with-moving BA with the exact"
        echo "       same v2 parameters used by the completed variants."

        set +e

        python3 -u "$BA_SCRIPT" \
          --mode with_moving \
          --moving-selection smart \
          --top-per-marker 8 \
          --top-per-pair 4 \
          --max-moving-frames 0 \
          --max-nfev 160 \
          2>&1 | tee -a "$VAR_LOG"

        FALLBACK_BA_RC="${PIPESTATUS[0]}"

        set -e

        if [[ "$FALLBACK_BA_RC" -eq 0 ]]; then
            set +e

            python3 "$EXPORT_SCRIPT" \
              2>&1 | tee -a "$VAR_LOG"

            EXPORT_RC="${PIPESTATUS[0]}"

            set -e

            if [[ "$EXPORT_RC" -eq 0 ]]; then
                AP02_RC=0
            fi
        fi
    fi

    echo
    echo "=== Evaluate current AP02 output ==="

    set +e

    python3 \
      run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py \
      2>&1 | tee -a "$VAR_LOG"

    EVAL_RC="${PIPESTATUS[0]}"

    set -e

    if [[ "$EVAL_RC" -ne 0 ]]; then
        fail "$VARIANT evaluator failed with rc=$EVAL_RC"
        continue
    fi

    echo
    echo "=== Merge only AP02 v2 into existing variant tables ==="

    set +e

    python3 "$MERGE_HELPER" \
      --variant-root "$VAR_ROOT" \
      --pipeline-rc "$AP02_RC" \
      2>&1 | tee -a "$VAR_LOG"

    MERGE_RC="${PIPESTATUS[0]}"

    set -e

    if [[ "$MERGE_RC" -ne 0 ]]; then
        fail "$VARIANT merge failed with rc=$MERGE_RC"
        continue
    fi

    if ! variant_is_complete "$VAR_FINAL"; then
        fail "$VARIANT did not produce a complete v2 metadata/table set"
        continue
    fi

    echo "[OK] AP02 v2 complete: $VARIANT"
done

###############################################################################
# Restore the clean canonical baseline before producing reports.
###############################################################################

restore_clean_baseline
trap - EXIT INT TERM

###############################################################################
# Rebuild Lighting aggregate.
###############################################################################

echo
echo "================================================================================"
echo "REBUILD LIGHTING AGGREGATE"
echo "================================================================================"

python3 \
  run/bus_real_data/ablation/22_reconcile_ablation_status.py \
  "$LIGHT_ROOT" \
  || fatal "Lighting status reconciliation failed."

python3 \
  run/bus_real_data/ablation/21_collect_full_ablation_report.py \
  "$LIGHT_ROOT" \
  "lighting_ap02_v2_final" \
  || fatal "Lighting report collection failed."

###############################################################################
# Build final baseline + all 16 AP02-v1/v2 comparison.
###############################################################################

echo
echo "================================================================================"
echo "BUILD FINAL 16/16 COMPARISON"
echo "================================================================================"

python3 - <<'PY'
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path


RESULTS = Path("results/bus_real_data")
ABLATION = RESULTS / "ablation"
OUT = ABLATION / "AP02_V2_SUMMARY"

OUT.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = OUT / "AP02_V1_V2_FINAL_COMPARISON.csv"
COMPARISON_TXT = OUT / "AP02_V1_V2_FINAL_COMPARISON.txt"
BASELINE_CSV = OUT / "BASELINE_ALL_METHODS_FINAL.csv"
STATUS_JSON = OUT / "AP02_V2_COMPLETION_STATUS.json"


def read_csv(path: Path):
    if not path.exists():
        return []

    with path.open(newline="", errors="replace") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows):
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def improve(old, new):
    if old is None or new is None or abs(old) < 1e-12:
        return None

    return 100.0 * (old - new) / old


def classify(old, new):
    if old is None and new is not None:
        return "RECOVERED"

    if old is not None and new is None:
        return "LOST"

    if old is None or new is None:
        return "NO_RESULT"

    change = improve(old, new)

    if change >= 5.0:
        return "BETTER"

    if change <= -5.0:
        return "WORSE"

    return "SIMILAR"


def method_row(path: Path, method: str):
    return next(
        (
            row
            for row in read_csv(path)
            if row.get("method") == method
        ),
        None,
    )


variants = []

for raw in ABLATION.rglob("raw_images"):
    variant = raw.parent

    try:
        relative = variant.relative_to(ABLATION)
    except ValueError:
        continue

    if any(part.startswith("_") for part in relative.parts):
        continue

    if not (variant / "aruco_observations").is_dir():
        continue

    if not (variant / "FINAL_RESULTS").is_dir():
        continue

    variants.append(variant)

variants = sorted(set(variants))

rows = []
incomplete = []

for variant in variants:
    relative = variant.relative_to(ABLATION)
    final = variant / "FINAL_RESULTS"

    metadata = final / "AP02_V2_METADATA.json"
    current_path = final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    old_path = (
        final
        / "AP02_V1_ARCHIVE"
        / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    )

    if not (
        metadata.exists()
        and current_path.exists()
        and old_path.exists()
    ):
        incomplete.append(str(relative))
        continue

    try:
        meta = json.loads(metadata.read_text())
    except Exception:
        incomplete.append(str(relative))
        continue

    if str(meta.get("version", "")).lower() != "v2":
        incomplete.append(str(relative))
        continue

    current = method_row(current_path, "AP02")
    old = method_row(old_path, "AP02")

    if current is None or old is None:
        incomplete.append(str(relative))
        continue

    old_t = num(old.get("mean_pair_t_cm"))
    new_t = num(current.get("mean_pair_t_cm"))

    old_r = num(old.get("mean_pair_r_deg"))
    new_r = num(current.get("mean_pair_r_deg"))

    rows.append({
        "ablation": "/".join(relative.parts[:-1]),
        "variant": relative.parts[-1],
        "comparison": classify(old_t, new_t),

        "v1_status": old.get("status", ""),
        "v2_status": current.get("status", ""),

        "v1_pair_count": old.get("pair_count_ok", ""),
        "v2_pair_count": current.get("pair_count_ok", ""),

        "v1_mean_pair_t_cm": old_t,
        "v2_mean_pair_t_cm": new_t,
        "translation_improvement_percent": improve(old_t, new_t),

        "v1_mean_pair_r_deg": old_r,
        "v2_mean_pair_r_deg": new_r,
        "rotation_improvement_percent": improve(old_r, new_r),

        "v1_worst_pair": old.get("worst_pair", ""),
        "v2_worst_pair": current.get("worst_pair", ""),

        "v1_worst_pair_t_cm": old.get("worst_pair_t_cm", ""),
        "v2_worst_pair_t_cm": current.get("worst_pair_t_cm", ""),
    })


baseline_path = (
    RESULTS
    / "99_FINAL_RESULTS_FOR_REPORT"
    / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
)

baseline_rows = read_csv(baseline_path)

write_csv(COMPARISON_CSV, rows)
write_csv(BASELINE_CSV, baseline_rows)

counts = {}

for row in rows:
    counts[row["comparison"]] = (
        counts.get(row["comparison"], 0) + 1
    )

lines = [
    "AP02 V1 VS AP02 V2 — FINAL COMPARISON",
    "=======================================",
    "",
    f"Variants discovered: {len(variants)}",
    f"Variants completed:  {len(rows)}",
    f"Incomplete:          {len(incomplete)}",
    "",
    "CLEAN BASELINE — ALL METHODS",
    "============================",
    "",
    (
        f"{'Method':8s}"
        f"{'Status':22s}"
        f"{'Mean t [cm]':15s}"
        f"{'Mean r [deg]':15s}"
        f"{'Worst pair'}"
    ),
    "-" * 80,
]

for row in baseline_rows:
    lines.append(
        f"{row.get('method', ''):8s}"
        f"{row.get('status', ''):22s}"
        f"{fmt(num(row.get('mean_pair_t_cm'))):15s}"
        f"{fmt(num(row.get('mean_pair_r_deg'))):15s}"
        f"{row.get('worst_pair', '-')}"
    )

lines += [
    "",
    "ALL ABLATIONS",
    "=============",
    "",
    (
        f"{'Ablation':31s}"
        f"{'Variant':29s}"
        f"{'Result':12s}"
        f"{'V1 t':>9s}"
        f"{'V2 t':>9s}"
        f"{'Δt %':>9s}"
        f"{'V1 r':>9s}"
        f"{'V2 r':>9s}"
        f"{'Δr %':>9s}"
        f"  {'Status V1 -> V2'}"
    ),
    "-" * 160,
]

for row in rows:
    lines.append(
        f"{row['ablation'][:31]:31s}"
        f"{row['variant'][:29]:29s}"
        f"{row['comparison']:12s}"
        f"{fmt(row['v1_mean_pair_t_cm']):>9s}"
        f"{fmt(row['v2_mean_pair_t_cm']):>9s}"
        f"{fmt(row['translation_improvement_percent'], 1):>9s}"
        f"{fmt(row['v1_mean_pair_r_deg']):>9s}"
        f"{fmt(row['v2_mean_pair_r_deg']):>9s}"
        f"{fmt(row['rotation_improvement_percent'], 1):>9s}"
        f"  {row['v1_status']} -> {row['v2_status']}"
    )

lines += [
    "",
    "COUNTS",
    "======",
    f"- BETTER:    {counts.get('BETTER', 0)}",
    f"- SIMILAR:   {counts.get('SIMILAR', 0)}",
    f"- WORSE:     {counts.get('WORSE', 0)}",
    f"- RECOVERED: {counts.get('RECOVERED', 0)}",
    f"- LOST:      {counts.get('LOST', 0)}",
    f"- NO_RESULT: {counts.get('NO_RESULT', 0)}",
]

if incomplete:
    lines += [
        "",
        "INCOMPLETE",
        "==========",
        *[f"- {item}" for item in incomplete],
    ]

COMPARISON_TXT.write_text(
    "\n".join(lines) + "\n"
)

STATUS_JSON.write_text(
    json.dumps({
        "checked_at": datetime.now().isoformat(),
        "expected_variants": 16,
        "discovered_variants": len(variants),
        "completed_variants": len(rows),
        "incomplete_variants": incomplete,
        "status": (
            "COMPLETE"
            if (
                len(variants) == 16
                and len(rows) == 16
                and not incomplete
            )
            else "INCOMPLETE"
        ),
    }, indent=2) + "\n"
)

print("\n".join(lines))

if (
    len(variants) != 16
    or len(rows) != 16
    or incomplete
):
    raise SystemExit(
        "[ERROR] Final AP02 v2 verification is not 16/16."
    )

print()
print("[OK] Final AP02-v2 verification: 16 / 16")
PY

###############################################################################
# Refuse to push if any operational failure was recorded.
###############################################################################

if [[ -s "$FAILURES" ]]; then
    echo
    echo "================================================================================"
    echo "FAILURES"
    echo "================================================================================"

    cat "$FAILURES"

    fatal "Results are incomplete; refusing to commit or push."
fi

###############################################################################
# Commit on a dedicated branch. Stage only implementation and compact reports.
###############################################################################

echo
echo "================================================================================"
echo "CREATE/SELECT GIT BRANCH"
echo "================================================================================"

BRANCH="${AP02_PUSH_BRANCH:-ap02-robust-graph-ba}"
CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    if git show-ref \
      --verify \
      --quiet \
      "refs/heads/$BRANCH"
    then
        git switch "$BRANCH"
    elif git ls-remote \
      --exit-code \
      --heads \
      origin \
      "$BRANCH" \
      >/dev/null 2>&1
    then
        git switch \
          --create "$BRANCH" \
          --track "origin/$BRANCH"
    else
        git switch \
          --create "$BRANCH"
    fi
fi

echo "[OK] branch: $(git branch --show-current)"

###############################################################################
# Explicit staging only. Do not use git add -A.
###############################################################################

CODE_FILES=(
  run/bus_real_data/approach2_ref_marker_graph_ba/ap02_observation_quality.py
  run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh
  run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py
  run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase3_graph_ba_fast.sh

  run/bus_real_data/_shared/baseline/01_recapture_clean_shared_baseline.sh

  run/bus_real_data/ablation/_shared/25_merge_ap02_v2_variant_results.py
  run/bus_real_data/ablation/_shared/26_rerun_baseline_and_all_ap02_v2.sh

  run/bus_real_data/ablation/world/lighting/28_finish_remaining_ap02_v2_and_push.sh
)

for FILE in "${CODE_FILES[@]}"
do
    if [[ -f "$FILE" ]]; then
        git add -- "$FILE"
    fi
done

if [[ -f \
  run/bus_real_data/ablation/_shared/27_resume_fast_ap02_v2_all.sh \
]]; then
    git add -- \
      run/bus_real_data/ablation/_shared/27_resume_fast_ap02_v2_all.sh
fi

# Canonical clean-baseline compact reports.
find \
  "$FINAL_ROOT" \
  -maxdepth 1 \
  -type f \
  \( \
    -name '*.csv' \
    -o -name '*.txt' \
    -o -name '*.json' \
  \) \
  -print0 \
| xargs -0 -r git add --

# Global AP02-v2 comparison.
find \
  "$OUT_SUMMARY" \
  -maxdepth 1 \
  -type f \
  \( \
    -name '*.csv' \
    -o -name '*.txt' \
    -o -name '*.json' \
  \) \
  -print0 \
| xargs -0 -r git add --

# Per-ablation aggregate tables.
find \
  results/bus_real_data/ablation \
  -path '*/ABLATION_SUMMARY/*' \
  -type f \
  \( \
    -name '*.csv' \
    -o -name '*.txt' \
    -o -name '*.json' \
  \) \
  -print0 \
| xargs -0 -r git add --

# Compact top-level per-variant results.
while IFS= read -r -d '' VAR_FINAL
do
    for NAME in \
      BASELINE_FINAL_PAIRWISE_SUMMARY.csv \
      BASELINE_FINAL_PAIRWISE_DETAIL.csv \
      BASELINE_FINAL_CLEAN_COMPARISON.txt \
      SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv \
      SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv \
      SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt \
      RUN_STATUS.txt \
      AP02_V2_METADATA.json
    do
        if [[ -f "$VAR_FINAL/$NAME" ]]; then
            git add -- "$VAR_FINAL/$NAME"
        fi
    done

    ARCHIVE="$VAR_FINAL/AP02_V1_ARCHIVE"

    if [[ -d "$ARCHIVE" ]]; then
        find \
          "$ARCHIVE" \
          -maxdepth 1 \
          -type f \
          \( \
            -name '*.csv' \
            -o -name '*.txt' \
            -o -name '*.json' \
          \) \
          -print0 \
        | xargs -0 -r git add --
    fi
done < <(
  find \
    results/bus_real_data/ablation \
    -type d \
    -name FINAL_RESULTS \
    -print0
)

###############################################################################
# Hard safety check against generated image/database/archive data.
###############################################################################

SUSPICIOUS="$(
  git diff \
    --cached \
    --name-only \
  | grep -E \
    '(^|/)(raw_images|debug_images|_capture_staging|_runtime_backups|_baseline_backups)/|\.(png|jpg|jpeg|tar|db|bin|npy|npz)$' \
  || true
)"

if [[ -n "$SUSPICIOUS" ]]; then
    echo "$SUSPICIOUS"

    while IFS= read -r FILE
    do
        [[ -n "$FILE" ]] || continue
        git restore --staged -- "$FILE" 2>/dev/null || true
    done <<< "$SUSPICIOUS"

    fatal "Suspicious generated data was staged; push cancelled."
fi

git diff --cached --check \
  || fatal "Git staged diff check failed."

echo
echo "=== STAGED FILES ==="
git diff --cached --stat

if git diff --cached --quiet; then
    echo "[INFO] No new staged changes; skipping commit."
else
    git commit \
      -m "Improve AP02 graph initialization and rerun ablations"
fi

echo
echo "================================================================================"
echo "PUSH"
echo "================================================================================"

git push \
  --set-upstream \
  origin \
  "$BRANCH"

echo
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"

echo "[OK] All 16 AP02-v2 variants complete."
echo "[OK] Clean baseline restored."
echo "[OK] Final comparison:"
echo "     $OUT_SUMMARY/AP02_V1_V2_FINAL_COMPARISON.txt"
echo "[OK] Branch pushed:"
echo "     $BRANCH"

date -Iseconds > "$LOG_ROOT/DONE.txt"
