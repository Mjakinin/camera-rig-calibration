#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(
  "res_320x180_extreme"
  "res_640x360"
  "res_960x540"
  "res_1280x720_baseline"
  "res_1920x1080"
)

CAPTURE_ROOT="results/bus_real_data/ablation/moving_cam/res/00_captures"
PREP_ROOT="results/bus_real_data/ablation/moving_cam/res/00_prepared_datasets"

echo "============================================================"
echo "MOVING CAM RES: make all raw_images ready"
echo "============================================================"

echo
echo "=== 1/4 generate Gazebo world variants ==="
python3 run/bus_real_data/ablation/moving_cam/res/00_generate_res_world_variants.py

echo
echo "=== 2/4 capture missing variants ==="
for v in "${VARIANTS[@]}"; do
  img_dir="$CAPTURE_ROOT/$v/images"
  frame_count="$(find "$img_dir" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l || true)"

  if [ "${FORCE_CAPTURE:-0}" = "1" ] || [ "$frame_count" -lt 200 ]; then
    echo
    echo "------------------------------------------------------------"
    echo "[CAPTURE] $v  existing_frames=$frame_count"
    echo "------------------------------------------------------------"
    AUTO_CONFIRM=1 bash run/bus_real_data/ablation/moving_cam/res/02_capture_one_res_variant.sh "$v"
  else
    echo "[SKIP] $v already has $frame_count frames"
  fi
done

echo
echo "=== 3/4 prepare raw_images datasets ==="
python3 run/bus_real_data/ablation/moving_cam/res/03_prepare_res_raw_datasets.py

echo
echo "=== 4/4 validate prepared raw_images ==="
python3 - <<'PY'
from pathlib import Path
import cv2
import sys

root = Path("results/bus_real_data/ablation/moving_cam/res/00_prepared_datasets")
expected = {
    "res_640x360": (360, 640, 3),
    "res_960x540": (540, 960, 3),
    "res_1280x720_baseline": (720, 1280, 3),
    "res_1920x1080": (1080, 1920, 3),
}

ok = True

for variant, exp_shape in expected.items():
    print(f"\n=== {variant} ===")
    raw = root / variant / "raw_images"
    static_p = raw / "static" / "cam_edge_0.png"
    moving_p = raw / "moving" / "frame_0000.png"
    cam_info = raw / "camera_info" / "moving_calib_camera.json"

    for label, p in [("static", static_p), ("moving", moving_p)]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        shape = None if img is None else img.shape
        print(f"{label}: {p}")
        print(f"  exists={p.exists()} shape={shape}")
        if img is None:
            ok = False

    img = cv2.imread(str(moving_p), cv2.IMREAD_COLOR)
    if img is not None and img.shape != exp_shape:
        print(f"  [ERROR] moving shape expected {exp_shape}, got {img.shape}")
        ok = False

    frame_count = len(list((raw / "moving").glob("frame_*.png")))
    print(f"  moving frame_count={frame_count}")
    print(f"  camera_info exists={cam_info.exists()}")

    if frame_count < 200 or not cam_info.exists():
        ok = False

if not ok:
    print("\n[FAIL] some prepared datasets are incomplete")
    sys.exit(2)

print("\n[OK] all moving_cam/res raw_images datasets are ready")
PY

echo
echo "[DONE] moving_cam/res raw_images ready:"
echo "  $PREP_ROOT"
