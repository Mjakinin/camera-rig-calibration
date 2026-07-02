#!/usr/bin/env bash
set -euo pipefail
bash run/bus_real_data/ablation/_shared/12_run_one_clean_variant_common.sh \
  results/bus_real_data/ablation/moving_cam/fov \
  "[FOV VARIANT]" \
  "$@"
