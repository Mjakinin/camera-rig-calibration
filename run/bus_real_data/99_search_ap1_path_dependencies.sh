#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

echo "=== AP01/AP02/AP03 path dependency search ==="
echo

echo "[1] Hardcoded AP01 moving sequence references:"
grep -R "03_moving_camera_sequence" -n run src results/bus_real_data/*.md results/bus_real_data/*/README.md 2>/dev/null || true
echo

echo "[2] Hardcoded AP01 static detection references:"
grep -R "01_static_a4_marker_detection" -n run src results/bus_real_data/*.md results/bus_real_data/*/README.md 2>/dev/null || true
echo

echo "[3] Hardcoded AP01 result root references:"
grep -R "01_marker_direct_relay_multimarker_multichain" -n run src 2>/dev/null || true
echo

echo "[4] Shared raw image dataset references:"
grep -R "00_raw_images\|bus_real_data_ref_marker_v1" -n run src results/bus_real_data/*.md 2>/dev/null || true
echo

echo "[5] Stale AP02-internal raw dataset references:"
grep -R "02_ref_marker_graph_ba/01_dataset\|AP02_ROOT / \"01_dataset\"\|AP02_ROOT / '01_dataset'" -n run src 2>/dev/null || true
echo

echo "[6] Old stale run3/minimal refs:"
grep -R "05_moving_camera_sequence_run3\|06_colmap_moving_sequence_run3\|minimal_world\|beintelli_bus_model" -n run src results 2>/dev/null || true
echo

echo "[7] AP02 must not import AP01 internals directly:"
if [ -d run/bus_real_data/approach2_ref_marker_graph_ba ]; then
  grep -R "01_marker_direct_relay\|AP1_ROOT\|AP01_ROOT\|moving_detections.csv\|01_static_a4_marker_detection" -n run/bus_real_data/approach2_ref_marker_graph_ba || true
else
  echo "[INFO] AP02 folder not present yet."
fi
echo

echo "=== Done ==="
