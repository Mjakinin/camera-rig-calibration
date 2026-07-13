#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD/run/bus_real_data:${PYTHONPATH:-}"

set +u
source /opt/ros/humble/setup.bash
[[ -f install/setup.bash ]] && source install/setup.bash
set -u

STAMP="$(date +%Y%m%d_%H%M%S)"

RESULTS="results/bus_real_data"
LOG_ROOT="$RESULTS/_final_corrected_overnight/$STAMP"
MASTER_LOG="$LOG_ROOT/OVERNIGHT_MASTER.log"
STATUS_FILE="$LOG_ROOT/STATUS.txt"

mkdir -p "$LOG_ROOT"

exec > >(tee "$MASTER_LOG") 2>&1

SHARED="$RESULTS/00_shared_baseline/bus_real_data_ref_marker_v1"
ROUTE2="$RESULTS/ablation/world/route/route2"

FOV_ROOT="$RESULTS/ablation/moving_cam/fov"
DENSITY_ROOT="$RESULTS/ablation/moving_cam/density"

DENSITY_125="density_route2_125pct_recaptured"
DENSITY_OFFSET="density_stride_8_offset4"
DENSITY_STRIDE16="density_stride_16_6p25pct"

DENSITY_125_ROOT="$DENSITY_ROOT/$DENSITY_125"

CAPTURE_RUNNER="run/bus_real_data/ablation/moving_cam/density/03_capture_route2_125pct.sh"
SPARSE_GENERATOR="run/bus_real_data/ablation/moving_cam/density/04_make_extra_sparse_density_variants.py"

VARIANT_RUNNER="run/bus_real_data/ablation/_shared/30_run_existing_variant_group.sh"
AP02_ALL_RUNNER="run/bus_real_data/ablation/_shared/26_rerun_baseline_and_all_ap02_v2.sh"
BASELINE_HELPER="run/bus_real_data/ablation/_shared/31_reuse_route2_baseline.py"

AP02_PIPELINE="run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh"
AP02_MAP_EVALUATOR="run/bus_real_data/approach2_ref_marker_graph_ba/09_eval_ap02_gt_aligned_full_map.py"

SIM_FINAL="$RESULTS/99_FINAL_RESULTS_FOR_REPORT"
REAL_FINAL="results/real_vehicle_data/real_05x_4k_3hz_v1/99_FINAL_RESULTS"

phase() {
    echo
    echo "================================================================================"
    echo "$1"
    echo "time=$(date -Iseconds)"
    echo "================================================================================"

    {
        echo "status=RUNNING"
        echo "phase=$1"
        echo "updated=$(date -Iseconds)"
        echo "log=$MASTER_LOG"
    } > "$STATUS_FILE"
}

restore_route2() {
    set +e

    if [[ -d "$ROUTE2/raw_images" ]] \
      && [[ -d "$ROUTE2/aruco_observations" ]]
    then
        python3 "$BASELINE_HELPER" \
            --source "$ROUTE2" \
            --target "$SHARED" \
            --variant route2_nominal \
            --group canonical/route2 \
            --dataset-only \
            >/dev/null 2>&1
    fi
}

on_exit() {
    local code=$?

    restore_route2

    if [[ "$code" -ne 0 ]]; then
        echo
        echo "================================================================================"
        echo "[ERROR] OVERNIGHT RUN FAILED"
        echo "return_code=$code"
        echo "log=$MASTER_LOG"
        echo "================================================================================"

        {
            echo "status=FAILED"
            echo "return_code=$code"
            echo "finished=$(date -Iseconds)"
            echo "log=$MASTER_LOG"
        } > "$STATUS_FILE"
    fi

    exit "$code"
}

trap on_exit EXIT INT TERM

validate_125_capture() {
python3 - "$DENSITY_125_ROOT" <<'PY'
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])

images = sorted(
    (root / "raw_images" / "moving").glob("frame_*.png")
)

if len(images) != 236:
    raise SystemExit(1)

indices = [
    int(path.stem.rsplit("_", 1)[-1])
    for path in images
]

if indices != list(range(236)):
    raise SystemExit(1)

route = root / "metadata" / "route_commanded.csv"
observations = (
    root
    / "aruco_observations"
    / "shared_moving_aruco_observations.csv"
)

if not route.is_file() or not observations.is_file():
    raise SystemExit(1)

with route.open(
    newline="",
    encoding="utf-8",
    errors="replace",
) as handle:
    route_rows = list(csv.DictReader(handle))

with observations.open(
    newline="",
    encoding="utf-8",
    errors="replace",
) as handle:
    observation_rows = list(csv.DictReader(handle))

if len(route_rows) != 236 or not observation_rows:
    raise SystemExit(1)

unique_hashes = {
    hashlib.sha256(path.read_bytes()).hexdigest()
    for path in images
}

if len(unique_hashes) < 200:
    raise SystemExit(1)

print(
    "[OK] 125-percent capture:",
    f"images={len(images)}",
    f"unique_images={len(unique_hashes)}",
    f"moving_observations={len(observation_rows)}",
)
PY
}

validate_sparse_variant() {
    local variant="$1"
    local expected="$2"
    local root="$DENSITY_ROOT/$variant"

    local count
    count="$(
        find "$root/raw_images/moving" \
            -maxdepth 1 \
            -name 'frame_*.png' \
            2>/dev/null \
            | wc -l
    )"

    if [[ "$count" -ne "$expected" ]]; then
        echo "[ERROR] $variant: $count frames instead of $expected"
        return 1
    fi

    test -s \
        "$root/aruco_observations/shared_all_aruco_observations.csv"

    test -s \
        "$root/metadata/route_commanded.csv"

    echo "[OK] $variant: $count frames"
}

echo "================================================================================"
echo "FINAL CORRECTED OVERNIGHT RUN"
echo "stamp=$STAMP"
echo "commit=$(git rev-parse HEAD)"
echo "log=$MASTER_LOG"
echo "================================================================================"

###############################################################################
# 0. Preflight
###############################################################################

phase "0. PREFLIGHT"

command -v python3
command -v colmap
command -v ign
command -v ros2

test -d "$ROUTE2/raw_images"
test -d "$ROUTE2/aruco_observations"

test -f "$CAPTURE_RUNNER"
test -f "$SPARSE_GENERATOR"
test -f "$VARIANT_RUNNER"
test -f "$AP02_ALL_RUNNER"
test -f "$BASELINE_HELPER"

python3 -m py_compile \
    run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py \
    run/bus_real_data/ablation/moving_cam/density/02_make_route2_125pct_route.py \
    "$SPARSE_GENERATOR" \
    "$AP02_MAP_EVALUATOR"

bash -n \
    "$CAPTURE_RUNNER" \
    "$VARIANT_RUNNER" \
    "$AP02_ALL_RUNNER" \
    "$AP02_PIPELINE"

grep -q '^RUN_GT_FREE_GATE=0$' "$AP02_PIPELINE"
grep -q 'SIM_MARKER_IDS =' "$AP02_MAP_EVALUATOR"
grep -q 'ALIGNMENT_MARKER_IDS =' "$AP02_MAP_EVALUATOR"

grep -q 'qos_profile_sensor_data' \
    run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py

grep -q 'startup_timeout = max(args.timeout, 30.0)' \
    run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py

grep -q 'MOVING_COUNT.*189' "$AP02_ALL_RUNNER"
grep -q -- '--marker-ids 0-14,16-20' "$AP02_ALL_RUNNER"

ROUTE2_COUNT="$(
    find "$ROUTE2/raw_images/moving" \
        -maxdepth 1 \
        -name 'frame_*.png' \
        | wc -l
)"

if [[ "$ROUTE2_COUNT" -ne 189 ]]; then
    echo "[ERROR] Route 2 has $ROUTE2_COUNT instead of 189 frames"
    false
fi

echo "[OK] Route-2 baseline: 189 frames"

python3 - <<'PY'
import csv
import json
import statistics
from pathlib import Path

root = Path(
    "results/bus_real_data/ablation/moving_cam/fov"
)

expected = {
    "fov_40deg": 1758.386,
    "fov_69deg_baseline": 929.467,
    "fov_100deg": 537.024,
    "fov_140deg_extreme": 232.941,
}

for variant, target in expected.items():
    info_path = (
        root
        / variant
        / "raw_images"
        / "camera_info"
        / "moving_calib_camera.json"
    )

    observation_path = (
        root
        / variant
        / "aruco_observations"
        / "shared_moving_aruco_observations.csv"
    )

    info = json.loads(info_path.read_text())

    with observation_path.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))

    csv_fx = statistics.median(
        float(row["fx"])
        for row in rows
        if row.get("fx")
    )

    values = [
        float(info["k"][0]),
        float(info["K"][0]),
        csv_fx,
    ]

    if any(abs(value - target) >= 1.0 for value in values):
        raise SystemExit(
            f"[ERROR] inconsistent FOV input: {variant}"
        )

    print(
        f"[OK] {variant}: "
        f"k={values[0]:.3f} "
        f"K={values[1]:.3f} "
        f"csv={values[2]:.3f}"
    )
PY

###############################################################################
# 1. Capture Route-2 at 125 percent
###############################################################################

phase "1. ROUTE-2 125-PERCENT GAZEBO CAPTURE"

if validate_125_capture; then
    echo "[OK] Reusing existing valid 125-percent capture."
else
    rm -rf "$DENSITY_125_ROOT"

    CAPTURE_OK=0

    for attempt in 1 2 3; do
        echo
        echo "--------------------------------------------------------------------------------"
        echo "CAPTURE ATTEMPT $attempt / 3"
        echo "--------------------------------------------------------------------------------"

        pkill -f "ign gazebo" 2>/dev/null || true
        pkill -f "gz sim" 2>/dev/null || true
        pkill -f "ros_gz_bridge" 2>/dev/null || true
        pkill -f "parameter_bridge" 2>/dev/null || true

        sleep 5

        set +e
        bash "$CAPTURE_RUNNER"
        CAPTURE_RC=$?
        set -e

        if [[ "$CAPTURE_RC" -eq 0 ]] && validate_125_capture; then
            CAPTURE_OK=1
            break
        fi

        echo "[WARN] Capture attempt $attempt failed, rc=$CAPTURE_RC"

        find "$DENSITY_ROOT" \
            -maxdepth 1 \
            -type d \
            -name '_capture_staging_density_route2_125pct_recaptured_*' \
            -exec rm -rf {} +

        rm -rf "$DENSITY_125_ROOT"

        sleep 10
    done

    if [[ "$CAPTURE_OK" -ne 1 ]]; then
        echo "[ERROR] All three capture attempts failed."
        false
    fi
fi

validate_125_capture

###############################################################################
# 2. Additional density datasets
###############################################################################

phase "2. CREATE STRIDE-8 OFFSET-4 AND STRIDE-16"

python3 "$SPARSE_GENERATOR"

validate_sparse_variant "density_stride_8_offset4" 25
validate_sparse_variant "density_stride_16_6p25pct" 13

###############################################################################
# 3. Corrected FOV with all approaches
###############################################################################

phase "3. CORRECTED FOV — AP01 AP02 AP03"

REFRESH_CANONICAL_FINAL=0 \
REUSE_EXISTING_OBSERVATIONS=1 \
bash "$VARIANT_RUNNER" \
    "$FOV_ROOT" \
    moving_cam_fov_corrected \
    fov_40deg \
    fov_100deg \
    fov_140deg_extreme

###############################################################################
# 4. New density variants with all approaches
###############################################################################

phase "4. NEW DENSITY — AP01 AP02 AP03"

REFRESH_CANONICAL_FINAL=0 \
REUSE_EXISTING_OBSERVATIONS=1 \
bash "$VARIANT_RUNNER" \
    "$DENSITY_ROOT" \
    moving_cam_density_extended \
    "$DENSITY_125" \
    "$DENSITY_OFFSET" \
    "$DENSITY_STRIDE16"

###############################################################################
# 5. Restore Route 2
###############################################################################

phase "5. RESTORE ROUTE-2 CANONICAL BASELINE"

python3 "$BASELINE_HELPER" \
    --source "$ROUTE2" \
    --target "$SHARED" \
    --variant route2_nominal \
    --group canonical/route2 \
    --dataset-only

CANONICAL_COUNT="$(
    find "$SHARED/raw_images/moving" \
        -maxdepth 1 \
        -name 'frame_*.png' \
        | wc -l
)"

if [[ "$CANONICAL_COUNT" -ne 189 ]]; then
    echo "[ERROR] Canonical baseline has $CANONICAL_COUNT frames"
    false
fi

echo "[OK] canonical Route 2 restored"

###############################################################################
# 6. Baseline and every simulation AP02
###############################################################################

phase "6. BASELINE + AP02 FOR EVERY SIMULATION VARIANT"

bash "$AP02_ALL_RUNNER" "$STAMP"

###############################################################################
# 7. Refresh final simulation reports
###############################################################################

phase "7. REFRESH FINAL SIMULATION REPORTS"

python3 "$BASELINE_HELPER" \
    --source "$ROUTE2" \
    --target "$SHARED" \
    --variant route2_nominal \
    --group canonical/route2 \
    --dataset-only

bash \
    run/bus_real_data/reporting/run_refresh_final_results.sh \
    --reuse-baseline \
    --promote

mkdir -p \
    "$SIM_FINAL/ablations" \
    "$SIM_FINAL/data/primary" \
    "$SIM_FINAL/data/secondary" \
    "$SIM_FINAL/details/primary" \
    "$SIM_FINAL/details/secondary"

install_group() {
    local source="$1"
    local number="$2"
    local name="$3"

    test -s "$source/FULL_ABLATION_REPORT.txt"
    test -s "$source/RUN_STATUS_ALL_VARIANTS.csv"
    test -s "$source/PRIMARY_PAIRWISE_SUMMARY_ALL_VARIANTS.csv"
    test -s "$source/PRIMARY_PAIRWISE_DETAIL_ALL_VARIANTS.csv"
    test -s "$source/SECONDARY_REF14_WORLD_SUMMARY_ALL_VARIANTS.csv"
    test -s "$source/SECONDARY_REF14_WORLD_DETAIL_ALL_VARIANTS.csv"

    cp -f \
        "$source/FULL_ABLATION_REPORT.txt" \
        "$SIM_FINAL/ablations/${number}_${name}_ALL_METHODS.txt"

    cp -f \
        "$source/RUN_STATUS_ALL_VARIANTS.csv" \
        "$SIM_FINAL/data/${name}_RUN_STATUS.csv"

    cp -f \
        "$source/PRIMARY_PAIRWISE_SUMMARY_ALL_VARIANTS.csv" \
        "$SIM_FINAL/data/primary/${name}_ABLATION_SUMMARY.csv"

    cp -f \
        "$source/PRIMARY_PAIRWISE_DETAIL_ALL_VARIANTS.csv" \
        "$SIM_FINAL/data/primary/${name}_ABLATION_DETAIL.csv"

    cp -f \
        "$source/SECONDARY_REF14_WORLD_SUMMARY_ALL_VARIANTS.csv" \
        "$SIM_FINAL/data/secondary/${name}_ABLATION_SUMMARY.csv"

    cp -f \
        "$source/SECONDARY_REF14_WORLD_DETAIL_ALL_VARIANTS.csv" \
        "$SIM_FINAL/data/secondary/${name}_ABLATION_DETAIL.csv"
}

install_group \
    "$RESULTS/ablation/world/route/ABLATION_SUMMARY" \
    "05" \
    "ROUTE_PATH"

install_group \
    "$DENSITY_ROOT/ABLATION_SUMMARY" \
    "06" \
    "FRAME_DENSITY"

python3 - <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

final = Path(
    "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
)


def render_csv(source: Path, destination: Path, title: str) -> None:
    with source.open(
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        rows = list(csv.DictReader(handle))

    fields = list(rows[0]) if rows else []

    lines = [
        title,
        "=" * 120,
        "",
        f"Source: {source}",
        f"Rows: {len(rows)}",
        "",
    ]

    current_variant = None
    current_method = None

    for row in rows:
        variant = row.get("variant", "")
        method = row.get("method", "")

        if variant != current_variant:
            lines += [
                "",
                "#" * 120,
                f"VARIANT: {variant}",
                "#" * 120,
            ]
            current_variant = variant
            current_method = None

        if method != current_method:
            lines += [
                "",
                method or "UNSPECIFIED",
                "-" * 80,
            ]
            current_method = method

        values = [
            f"{field}={row.get(field, '')}"
            for field in fields
            if field not in {"variant", "method"}
        ]

        lines.append(" | ".join(values))

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


render_csv(
    final / "data/primary/FRAME_DENSITY_ABLATION_DETAIL.csv",
    final / "details/primary/06_FRAME_DENSITY_CAM_TO_CAM.txt",
    "MOVING-CAMERA FRAME DENSITY — CAMERA-TO-CAMERA DETAIL",
)

render_csv(
    final / "data/secondary/FRAME_DENSITY_ABLATION_DETAIL.csv",
    final / "details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt",
    "MOVING-CAMERA FRAME DENSITY — SECONDARY MAP DETAIL",
)

render_csv(
    final / "data/primary/ROUTE_PATH_ABLATION_DETAIL.csv",
    final / "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt",
    "MOVING-CAMERA ROUTE — CAMERA-TO-CAMERA DETAIL",
)

render_csv(
    final / "data/secondary/ROUTE_PATH_ABLATION_DETAIL.csv",
    final / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt",
    "MOVING-CAMERA ROUTE — SECONDARY MAP DETAIL",
)

density_root = Path(
    "results/bus_real_data/ablation/moving_cam/density"
)

manifest = {
    "frame_density_ablation": {
        "baseline": "density_stride_1_100pct",
        "dense_gazebo_recapture": (
            "density_route2_125pct_recaptured"
        ),
        "additional_variants": [
            "density_stride_8_offset4",
            "density_stride_16_6p25pct",
        ],
        "report": (
            "ablations/06_FRAME_DENSITY_ALL_METHODS.txt"
        ),
    }
}

(
    final / "EXTRA_ABLATIONS_MANIFEST.json"
).write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("[OK] installed readable route and density reports")
PY

###############################################################################
# 8. Real reports only
###############################################################################

phase "8. REFRESH REAL REPORTS ONLY"

bash \
    run/real_vehicle_data/run_full_real_pipeline.sh \
    --only report \
    --gpu 0

###############################################################################
# 9. AP02 marker-map audit
###############################################################################

phase "9. AP02 FULL-MAP COMPLETENESS AUDIT"

python3 - <<'PY'
from __future__ import annotations

import csv
from pathlib import Path

expected = set(range(0, 15)) | set(range(16, 21))

ablation = Path(
    "results/bus_real_data/ablation"
)

final_root = Path(
    "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
)

records = []

for final in sorted(ablation.rglob("FINAL_RESULTS")):
    variant = final.parent

    try:
        relative = variant.relative_to(ablation)
    except ValueError:
        continue

    if any(part.startswith("_") for part in relative.parts):
        continue

    map_csv = (
        final
        / "DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.csv"
    )

    if not map_csv.is_file():
        records.append({
            "variant": str(relative),
            "status": "NO_FULL_MAP_CSV",
            "available": [],
            "missing": sorted(expected),
        })
        continue

    with map_csv.open(
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        rows = list(csv.DictReader(handle))

    available = {
        int(float(row["marker_id"]))
        for row in rows
        if row.get("entity_type") == "aruco_marker"
        and row.get("marker_id") not in {"", None}
    }

    missing = sorted(expected - available)

    records.append({
        "variant": str(relative),
        "status": (
            "FULL_20_OF_20"
            if not missing
            else "PARTIAL_MARKER_MAP"
        ),
        "available": sorted(available),
        "missing": missing,
    })

report = (
    final_root
    / "AP02_FULL_MAP_COMPLETENESS_AUDIT.txt"
)

lines = [
    "AP02 FULL-MAP COMPLETENESS AUDIT",
    "=" * 120,
    "",
    f"Expected marker IDs: {sorted(expected)}",
    "",
]

for record in records:
    lines += [
        record["variant"],
        f"  status:    {record['status']}",
        f"  count:     {len(record['available'])} / 20",
        f"  available: {record['available']}",
        f"  missing:   {record['missing']}",
        "",
    ]

report.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(report.read_text(encoding="utf-8"))
print("[OK] wrote:", report)
PY

###############################################################################
# 10. Final verification
###############################################################################

phase "10. FINAL VERIFICATION"

test -s "$SIM_FINAL/01_BASELINE_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/01_FOV_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/02_MOTION_BLUR_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/03_RESOLUTION_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/04_LIGHTING_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/05_ROUTE_PATH_ALL_METHODS.txt"
test -s "$SIM_FINAL/ablations/06_FRAME_DENSITY_ALL_METHODS.txt"

test -s \
    "$SIM_FINAL/details/primary/06_FRAME_DENSITY_CAM_TO_CAM.txt"

test -s \
    "$SIM_FINAL/details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt"

test -s \
    "$SIM_FINAL/AP02_FULL_MAP_COMPLETENESS_AUDIT.txt"

test -s \
    "$DENSITY_125_ROOT/FINAL_RESULTS/RUN_STATUS.txt"

test -s \
    "$DENSITY_ROOT/$DENSITY_OFFSET/FINAL_RESULTS/RUN_STATUS.txt"

test -s \
    "$DENSITY_ROOT/$DENSITY_STRIDE16/FINAL_RESULTS/RUN_STATUS.txt"

test -s "$REAL_FINAL/REAL_DATA_ALL_METHODS.txt"
test -s "$REAL_FINAL/REAL_PRIMARY_CAM_TO_CAM.txt"
test -s "$REAL_FINAL/REAL_SECONDARY_AP02_MARKER_MAP_REF3.txt"
test -s "$REAL_FINAL/REAL_SECONDARY_AP02_MARKER_MAP_REF3.csv"

{
    echo "status=COMPLETE"
    echo "finished=$(date -Iseconds)"
    echo "log=$MASTER_LOG"
    echo "simulation_results=$SIM_FINAL"
    echo "real_results=$REAL_FINAL"
} > "$STATUS_FILE"

trap - EXIT INT TERM

restore_route2

echo
echo "================================================================================"
echo "[OK] FINAL CORRECTED OVERNIGHT RUN COMPLETE"
echo "log=$MASTER_LOG"
echo "simulation=$SIM_FINAL"
echo "real=$REAL_FINAL"
echo "================================================================================"
