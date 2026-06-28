#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"
source /opt/ros/humble/setup.bash

python3 run/bus_real_data/approach2_ref_marker_graph_ba/01_capture_shared_raw_dataset.py \
  --out results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images \
  --moving-frames 204 \
  --moving-dt 0.10

cat > results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images/MANIFEST.txt <<'EOF'
Dataset: bus_real_data_ref_marker_v1
Purpose: Shared raw image dataset for AP02 reference-marker graph BA and AP03 targetless COLMAP + ArUco scale.
Raw images only. Approach-specific detections/results must be written into their own result folders.
EOF

echo "[OK] Shared raw dataset captured."
