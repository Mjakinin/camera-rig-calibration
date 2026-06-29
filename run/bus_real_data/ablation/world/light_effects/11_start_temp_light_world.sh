#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

VARIANT="${1:-dim}"

case "$VARIANT" in
  baseline|baseline_copy)
    FILE="bus_real_data_moving_camera_light_test_baseline_copy.sdf"
    ;;
  dim)
    FILE="bus_real_data_moving_camera_light_test_dim.sdf"
    ;;
  side_sun)
    FILE="bus_real_data_moving_camera_light_test_side_sun.sdf"
    ;;
  glare)
    FILE="bus_real_data_moving_camera_light_test_glare.sdf"
    ;;
  *)
    echo "Usage:"
    echo "  bash run/bus_real_data/ablation/world/light_effects/11_start_temp_light_world.sh dim"
    echo "  bash run/bus_real_data/ablation/world/light_effects/11_start_temp_light_world.sh side_sun"
    echo "  bash run/bus_real_data/ablation/world/light_effects/11_start_temp_light_world.sh glare"
    echo "  bash run/bus_real_data/ablation/world/light_effects/11_start_temp_light_world.sh baseline"
    exit 1
    ;;
esac

export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

WORLD="$PWD/src/calib_lab/bus_real_data/worlds/_temp_light_tests/$FILE"

echo "[INFO] WORLD=$WORLD"
echo "[INFO] GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"

pkill -f "ign gazebo" || true
pkill -f "gz sim" || true
sleep 1

if command -v gz >/dev/null 2>&1; then
  gz sim -r "$WORLD"
else
  ign gazebo -r "$WORLD"
fi
