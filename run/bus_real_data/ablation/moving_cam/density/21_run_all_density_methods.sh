#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

python3 run/bus_real_data/ablation/moving_cam/density/00_generate_density_variants.py

bash run/bus_real_data/ablation/_shared/30_run_existing_variant_group.sh \
  results/bus_real_data/ablation/moving_cam/density \
  "moving_cam_density" \
  density_stride_1_100pct \
  density_stride_2_50pct \
  density_stride_4_25pct \
  density_stride_8_12p5pct
