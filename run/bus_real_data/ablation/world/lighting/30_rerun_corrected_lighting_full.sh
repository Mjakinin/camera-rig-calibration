#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD/run/bus_real_data:${PYTHONPATH:-}"

set +u
source /opt/ros/humble/setup.bash
[[ -f install/setup.bash ]] && source install/setup.bash
set -u

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="results/bus_real_data/ablation/world/lighting/_corrected_rerun_${STAMP}"
MASTER_LOG="$LOG_ROOT/CORRECTED_LIGHTING_RERUN.log"
mkdir -p "$LOG_ROOT"
exec > >(tee "$MASTER_LOG") 2>&1

VARIANTS=(
  ceiling_dark_extreme
  ceiling_low
  ceiling_normal
  ceiling_bright
)

cleanup() {
  pkill -f "ign gazebo" 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true
  pkill -f "ros_gz_bridge" 2>/dev/null || true
  pkill -f "parameter_bridge" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

phase() {
  echo
  echo "================================================================================"
  echo "$1"
  echo "time=$(date -Iseconds)"
  echo "================================================================================"
}

phase "1. REGENERATE LIGHTING WORLDS FROM CURRENT 20-MARKER BASE WORLD"
python3 run/bus_real_data/ablation/world/lighting/create_physical_lighting_worlds.py

python3 - <<'PY'
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root = Path("src/calib_lab/bus_real_data/worlds/lighting")
expected = set(range(0, 15)) | set(range(16, 21))
variants = (
    "ceiling_dark_extreme",
    "ceiling_low",
    "ceiling_normal",
    "ceiling_bright",
)

for variant in variants:
    path = root / f"bus_real_data_moving_camera_light_{variant}.sdf"
    world = ET.parse(path).getroot()
    found = set()
    for include in world.iter("include"):
        name = (include.findtext("name") or "").strip()
        match = re.fullmatch(r"marker_(\d{3})", name)
        if match:
            found.add(int(match.group(1)))
        elif name == "aruco_ref_floor_14":
            found.add(14)
    if found != expected:
        raise SystemExit(
            f"[ERROR] {variant}: marker set {sorted(found)} != {sorted(expected)}"
        )
    print(f"[OK] {variant}: exact marker set {sorted(found)}")
PY

phase "2. PREPARE LIGHTING VARIANT METADATA"
bash run/bus_real_data/ablation/world/lighting/19_prepare_lighting_datasets.sh

phase "3. RECAPTURE ALL FOUR LIGHTING VARIANTS"
for variant in "${VARIANTS[@]}"; do
  bash run/bus_real_data/ablation/world/lighting/18_capture_one_lighting_variant.sh "$variant"
done

phase "4. REDETECT ARUCO OBSERVATIONS"
bash run/bus_real_data/ablation/world/lighting/20_detect_lighting_variants.sh

phase "5. OBSERVATION AUDIT"
python3 - <<'PY'
from pathlib import Path
import csv

root = Path("results/bus_real_data/ablation/world/lighting")
variants = (
    "ceiling_dark_extreme",
    "ceiling_low",
    "ceiling_normal",
    "ceiling_bright",
)
for variant in variants:
    path = root / variant / "aruco_observations/shared_all_aruco_observations.csv"
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    ids = sorted({int(float(row["marker_id"])) for row in rows if row.get("marker_id")})
    print(f"[INFO] {variant}: detected marker IDs {ids}")
PY

phase "6. RERUN AP01, AP02 AND AP03 FOR ALL LIGHTING VARIANTS"
bash run/bus_real_data/ablation/world/lighting/21_run_all_lighting_methods.sh

phase "7. REFRESH ALL CANONICAL REPORTS"
bash run/bus_real_data/reporting/run_refresh_final_results.sh --reuse-baseline --promote

phase "8. FINAL LIGHTING AND REPORT VERIFICATION"
python3 - <<'PY'
from pathlib import Path
import csv

root = Path("results/bus_real_data/ablation/world/lighting")
final99 = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")
expected_world = set(range(0, 15)) | set(range(16, 21))

for variant in (
    "ceiling_dark_extreme",
    "ceiling_low",
    "ceiling_normal",
    "ceiling_bright",
):
    final = root / variant / "FINAL_RESULTS"
    status = final / "RUN_STATUS.txt"
    if not status.is_file():
        raise SystemExit(f"[ERROR] missing {status}")
    print()
    print(variant)
    print(status.read_text(errors="replace").strip())

    marker_map = (
        final
        / "AP02_V2_DIAGNOSTICS/08_final_results/"
        / "ap02_with_moving_marker_poses_ref_marker.csv"
    )
    if marker_map.is_file():
        with marker_map.open(newline="", errors="replace") as handle:
            rows = list(csv.DictReader(handle))
        ids = sorted({int(float(row["entity_id"])) for row in rows})
        print(f"AP02 estimated marker IDs: {ids}")
        print(f"AP02 missing marker IDs: {sorted(expected_world - set(ids))}")

report = final99 / "details/secondary/04_LIGHTING_MAP_TO_GT.txt"
count = sum(
    1
    for line in report.read_text(errors="replace").splitlines()
    if line.startswith("=== AP02 REF14-ANCHORED AVAILABLE MAP BEGIN:")
)
if count != 4:
    raise SystemExit(
        f"[ERROR] lighting report contains {count} inline maps instead of 4"
    )
print(f"[OK] lighting report contains {count} inline marker maps")
PY

echo
echo "================================================================================"
echo "[OK] CORRECTED LIGHTING RERUN COMPLETE"
echo "log=$MASTER_LOG"
echo "================================================================================"
