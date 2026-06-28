#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(
  "fov_40deg"
  "fov_50deg"
  "fov_60deg"
  "fov_69deg_baseline"
  "fov_80deg"
  "fov_90deg"
  "fov_100deg"
  "fov_110deg"
  "fov_120deg"
  "fov_140deg_extreme"
)

CAPTURE_ROOT="results/bus_real_data/ablation/moving_cam/fov/00_captures"

echo "============================================================"
echo "MOVING CAM FOV: generate, capture, prepare, detect"
echo "============================================================"

echo
echo "=== 1/5 generate FOV world variants ==="
python3 run/bus_real_data/ablation/moving_cam/fov/00_generate_fov_world_variants.py

echo
echo "=== 2/5 capture missing FOV variants ==="
for v in "${VARIANTS[@]}"; do
  img_dir="$CAPTURE_ROOT/$v/images"
  frame_count="$(find "$img_dir" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l || true)"

  if [ "${FORCE_CAPTURE:-0}" = "1" ] || [ "$frame_count" -lt 200 ]; then
    echo
    echo "------------------------------------------------------------"
    echo "[CAPTURE] $v  existing_frames=$frame_count"
    echo "------------------------------------------------------------"
    AUTO_CONFIRM=1 bash run/bus_real_data/ablation/moving_cam/fov/02_capture_one_fov_variant.sh "$v"
  else
    echo "[SKIP] $v already has $frame_count frames"
  fi
done

echo
echo "=== 3/5 prepare FOV raw_images datasets ==="
python3 run/bus_real_data/ablation/moving_cam/fov/03_prepare_fov_raw_datasets.py

echo
echo "=== 4/5 run shared ArUco observations ==="
bash run/bus_real_data/ablation/moving_cam/fov/04_run_shared_observations_all_fov.sh

echo
echo "=== 5/5 summarize shared observations ==="
python3 run/bus_real_data/ablation/moving_cam/fov/05_summarize_shared_observations_fov.py

echo
echo "[DONE] moving_cam/fov ready."
