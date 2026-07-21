#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ROOT="results/bus_real_data/ablation/world/lighting"

VARIANTS=(
  ceiling_dark_extreme
  ceiling_low
  ceiling_normal
  ceiling_bright
)

mkdir -p "$ROOT"

for variant in "${VARIANTS[@]}"; do
    VAR_ROOT="$ROOT/$variant"

    WORLD="src/calib_lab/bus_real_data/worlds/lighting/bus_real_data_moving_camera_light_${variant}.sdf"

    if [[ ! -f "$WORLD" ]]; then
        echo "[ERROR] Missing world: $WORLD"
        exit 1
    fi

    mkdir -p \
      "$VAR_ROOT/raw_images" \
      "$VAR_ROOT/metadata"

    VARIANT="$variant" WORLD="$WORLD" VAR_ROOT="$VAR_ROOT" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

variant = os.environ["VARIANT"]
world = os.environ["WORLD"]
root = Path(os.environ["VAR_ROOT"])

metadata = {
    "variant": variant,
    "ablation": "physical_ceiling_light_intensity",
    "world": world,
    "camera_resolution": [1280, 720],
    "moving_camera_fov": "unchanged",
    "moving_camera_route": (
        "src/calib_lab/bus_real_data/config/"
        "moving_camera_route_interpolated.json"
    ),
    "lighting_geometry_fixed": True,
    "light_level": variant.removeprefix("ceiling_"),
}

(root / "VARIANT_METADATA.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

(root / "metadata" / "lighting_variant.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
PY

    echo "[OK] prepared $VAR_ROOT"
done
