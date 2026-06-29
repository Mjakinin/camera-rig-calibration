#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

VARIANT="${1:-low_moderate}"

case "$VARIANT" in
  baseline)
    FILE="bus_real_data_moving_camera_light_v2_baseline.sdf"
    ;;
  low_moderate)
    FILE="bus_real_data_moving_camera_light_v2_low_moderate.sdf"
    ;;
  low_extreme)
    FILE="bus_real_data_moving_camera_light_v2_low_extreme.sdf"
    ;;
  side_sun)
    FILE="bus_real_data_moving_camera_light_v2_side_sun.sdf"
    ;;
  glare)
    FILE="bus_real_data_moving_camera_light_v2_glare.sdf"
    ;;
  *)
    echo "Usage:"
    echo "  bash run/bus_real_data/ablation/world/light_effects/13_start_temp_light_world_v2.sh baseline"
    echo "  bash run/bus_real_data/ablation/world/light_effects/13_start_temp_light_world_v2.sh low_moderate"
    echo "  bash run/bus_real_data/ablation/world/light_effects/13_start_temp_light_world_v2.sh low_extreme"
    echo "  bash run/bus_real_data/ablation/world/light_effects/13_start_temp_light_world_v2.sh side_sun"
    echo "  bash run/bus_real_data/ablation/world/light_effects/13_start_temp_light_world_v2.sh glare"
    exit 1
    ;;
esac

export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

WORLD="$PWD/src/calib_lab/bus_real_data/worlds/_temp_light_tests/$FILE"

echo "[INFO] starting: $VARIANT"
echo "[INFO] world: $WORLD"

pkill -f "ign gazebo" || true
pkill -f "gz sim" || true
sleep 1

if command -v gz >/dev/null 2>&1; then
  gz sim -r "$WORLD"
else
  ign gazebo -r "$WORLD"
fi
