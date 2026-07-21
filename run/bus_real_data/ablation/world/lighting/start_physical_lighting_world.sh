#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VARIANT="${1:-ceiling_normal}"

case "$VARIANT" in
    ceiling_dark_extreme|ceiling_low|ceiling_normal|ceiling_bright)
        ;;
    *)
        echo "Usage:"
        echo "  $0 ceiling_dark_extreme"
        echo "  $0 ceiling_low"
        echo "  $0 ceiling_normal"
        echo "  $0 ceiling_bright"
        exit 2
        ;;
esac

GENERATOR="run/bus_real_data/ablation/world/lighting/create_physical_lighting_worlds.py"

WORLD="$PWD/src/calib_lab/bus_real_data/worlds/lighting/bus_real_data_moving_camera_light_${VARIANT}.sdf"

if [[ ! -f "$WORLD" ]]; then
    python3 "$GENERATOR"
fi

export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$PWD/src/calib_lab/bus_real_data/models:$PWD/src/calib_lab/bus_real_data/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$PWD/src/calib_lab/bus_real_data/models:${GAZEBO_MODEL_PATH:-}"

echo "============================================================"
echo "Diffuse bus ceiling-light preview"
echo "variant: $VARIANT"
echo "world:   $WORLD"
echo "============================================================"

if command -v ign >/dev/null 2>&1; then
    exec ign gazebo -r "$WORLD"
fi

if command -v gz >/dev/null 2>&1; then
    exec gz sim -r "$WORLD"
fi

echo "[ERROR] Neither ign nor gz was found."
exit 127
