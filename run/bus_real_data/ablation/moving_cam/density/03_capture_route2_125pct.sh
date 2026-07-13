#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="$PWD/run/bus_real_data:${PYTHONPATH:-}"

set +u
source /opt/ros/humble/setup.bash
[[ -f install/setup.bash ]] && source install/setup.bash
set -u

command -v ign >/dev/null
command -v ros2 >/dev/null
command -v python3 >/dev/null

export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

ROOT="results/bus_real_data/ablation/moving_cam/density"
SOURCE="results/bus_real_data/ablation/world/route/route2"
VARIANT="density_route2_125pct_recaptured"
VAR_ROOT="$ROOT/$VARIANT"

ROUTE="src/calib_lab/bus_real_data/config/moving_camera_route2_density_125pct.json"
WORLD="src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"

CAPTURE_TOOL="run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"
DETECTOR="run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE="$ROOT/_capture_staging_${VARIANT}_${STAMP}"
CAPTURE_OUT="$STAGE/moving_capture"
DATASET="$STAGE/dataset"
LOGS="$STAGE/logs"

mkdir -p "$LOGS"

python3 \
  run/bus_real_data/ablation/moving_cam/density/02_make_route2_125pct_route.py

EXPECTED="$(
python3 - "$ROUTE" <<'PY'
import json
import sys
from pathlib import Path

print(len(json.loads(Path(sys.argv[1]).read_text())["frames"]))
PY
)"

if [[ "$EXPECTED" -ne 236 ]]; then
  echo "[ERROR] expected 236 poses, found $EXPECTED"
  exit 1
fi

pkill -f "ign gazebo" 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ros_gz_bridge" 2>/dev/null || true
pkill -f "parameter_bridge" 2>/dev/null || true
sleep 2

GZ_PID=""
BRIDGE_PID=""

cleanup() {
  set +e

  if [[ -n "$BRIDGE_PID" ]]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi

  if [[ -n "$GZ_PID" ]]; then
    kill "$GZ_PID" 2>/dev/null || true
    wait "$GZ_PID" 2>/dev/null || true
  fi

  pkill -f "parameter_bridge" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "================================================================================"
echo "ROUTE-2 DENSITY 125 PERCENT — REAL GAZEBO RECAPTURE"
echo "route:   $ROUTE"
echo "output:  $VAR_ROOT"
echo "frames:  $EXPECTED"
echo "================================================================================"

ign gazebo -r -s "$WORLD" \
  > "$LOGS/gazebo.log" 2>&1 &

GZ_PID=$!

WORLD_NAME=""

for _ in $(seq 1 60); do
  WORLD_NAME="$(
    ign service -l 2>/dev/null \
      | sed -n 's#^/world/\([^/]*\)/set_pose$#\1#p' \
      | head -n 1
  )"

  [[ -n "$WORLD_NAME" ]] && break
  sleep 1
done

if [[ -z "$WORLD_NAME" ]]; then
  echo "[ERROR] Gazebo set_pose service not found"
  tail -100 "$LOGS/gazebo.log" || true
  exit 1
fi

echo "[OK] Gazebo world: $WORLD_NAME"

ros2 run ros_gz_bridge parameter_bridge \
  "/bus_real_data/moving_calib_camera/image@sensor_msgs/msg/Image@gz.msgs.Image" \
  > "$LOGS/bridge.log" 2>&1 &

BRIDGE_PID=$!

TOPIC="/bus_real_data/moving_calib_camera/image"
FOUND=0

for _ in $(seq 1 60); do
  if ros2 topic list 2>/dev/null | grep -qx "$TOPIC"; then
    FOUND=1
    break
  fi
  sleep 1
done

if [[ "$FOUND" != "1" ]]; then
  echo "[ERROR] ROS image topic not found: $TOPIC"
  tail -100 "$LOGS/bridge.log" || true
  exit 1
fi

echo "[OK] ROS topic: $TOPIC"

python3 "$CAPTURE_TOOL" \
  --route "$ROUTE" \
  --world "$WORLD_NAME" \
  --name moving_calib_camera \
  --out "$CAPTURE_OUT" \
  --settle 0.35 \
  --post-pose-skip 5 \
  --timeout 3.0 \
  --clean \
  2>&1 | tee "$LOGS/moving_capture.log"

CAPTURED="$(
  find "$CAPTURE_OUT/images" \
    -maxdepth 1 \
    -name 'frame_*.png' \
    | wc -l
)"

if [[ "$CAPTURED" -ne "$EXPECTED" ]]; then
  echo "[ERROR] captured $CAPTURED / $EXPECTED moving frames"
  exit 1
fi

mkdir -p \
  "$DATASET/raw_images/moving" \
  "$DATASET/raw_images/ap1_metadata" \
  "$DATASET/metadata" \
  "$DATASET/capture_logs"

cp -a "$SOURCE/raw_images/static" \
  "$DATASET/raw_images/static"

cp -a "$SOURCE/raw_images/camera_info" \
  "$DATASET/raw_images/camera_info"

if [[ -d "$SOURCE/raw_images/static_multi" ]]; then
  cp -a "$SOURCE/raw_images/static_multi" \
    "$DATASET/raw_images/static_multi"
fi

cp -a "$CAPTURE_OUT/images/." \
  "$DATASET/raw_images/moving/"

if [[ -d "$SOURCE/metadata" ]]; then
  cp -a "$SOURCE/metadata/." \
    "$DATASET/metadata/"
fi

python3 - \
  "$CAPTURE_OUT/route_commanded.csv" \
  "$DATASET/metadata/route_commanded.csv" \
  "$DATASET/raw_images/ap1_metadata/route_commanded.csv" \
  "$VAR_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
metadata_output = Path(sys.argv[2])
ap1_output = Path(sys.argv[3])
final_root = Path(sys.argv[4])

with source.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 236:
    raise RuntimeError(
        f"Expected 236 route rows, found {len(rows)}"
    )

fields = list(rows[0])

for row in rows:
    frame = int(row["frame"])
    row["image"] = str(
        final_root
        / "raw_images"
        / "moving"
        / f"frame_{frame:04d}.png"
    )

for output in (metadata_output, ap1_output):
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] wrote:", output)
PY

cp "$ROUTE" \
  "$DATASET/metadata/moving_camera_route2_density_125pct.json"

python3 "$DETECTOR" \
  --dataset "$DATASET/raw_images" \
  --out "$DATASET/aruco_observations" \
  --dictionary DICT_4X4_50 \
  2>&1 | tee "$LOGS/aruco_detection.log"

python3 - \
  "$DATASET/VARIANT_METADATA.json" \
  "$DATASET/metadata/density_variant.json" \
  "$VAR_ROOT" \
  "$ROUTE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

variant_metadata = Path(sys.argv[1])
density_metadata = Path(sys.argv[2])
variant_root = Path(sys.argv[3])
route = Path(sys.argv[4])

payload = {
    "group": "moving_cam/density",
    "variant": "density_route2_125pct_recaptured",
    "parameter": "moving camera temporal-spatial frame density",
    "source_dataset": (
        "results/bus_real_data/ablation/world/route/route2"
    ),
    "source_frame_count": 189,
    "selected_frame_count": 236,
    "density_percent": 125.0,
    "density_factor": 1.25,
    "route_geometry": "same piecewise Route-2 trajectory",
    "route_endpoints_unchanged": True,
    "images_newly_rendered_in_gazebo": True,
    "image_interpolation_used": False,
    "static_images_unchanged": True,
    "camera_info_unchanged": True,
    "world_unchanged": True,
    "route_file": str(route),
    "raw_images": str(variant_root / "raw_images"),
    "aruco_observations": str(
        variant_root / "aruco_observations"
    ),
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "changed_factor_only": (
        "finer temporal-spatial sampling of the same Route-2 path"
    ),
}

for path in (variant_metadata, density_metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

print("[OK] wrote density metadata")
PY

cp -a "$LOGS/." "$DATASET/capture_logs/"

test -s \
  "$DATASET/aruco_observations/shared_moving_aruco_observations.csv"

FINAL_COUNT="$(
  find "$DATASET/raw_images/moving" \
    -maxdepth 1 \
    -name 'frame_*.png' \
    | wc -l
)"

if [[ "$FINAL_COUNT" -ne 236 ]]; then
  echo "[ERROR] staged dataset contains $FINAL_COUNT / 236 frames"
  exit 1
fi

rm -rf "$VAR_ROOT"
mv "$DATASET" "$VAR_ROOT"

echo
echo "[OK] installed real Gazebo 125-percent density variant:"
echo "     $VAR_ROOT"
echo "[OK] moving images: $FINAL_COUNT"
echo "[OK] capture logs: $VAR_ROOT/capture_logs"
