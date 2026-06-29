#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ALIASES = {
    "baseline": "baseline",
    "low": "low_moderate",
    "low_moderate": "low_moderate",
    "extreme_low": "low_extreme",
    "low_extreme": "low_extreme",
    "side_sun": "side_sun",
    "glare": "glare",
}

if len(sys.argv) != 5:
    print("usage: python3 15_patch_temp_sun_direction.py <variant> <dx> <dy> <dz>")
    print("variants: baseline, low, low_moderate, extreme_low, low_extreme, side_sun, glare")
    print("example: python3 15_patch_temp_sun_direction.py low -1 0.08 -0.005")
    sys.exit(1)

variant_in = sys.argv[1]
variant = ALIASES.get(variant_in)
if variant is None:
    raise SystemExit(f"[ERROR] unknown variant: {variant_in}")

dx, dy, dz = sys.argv[2], sys.argv[3], sys.argv[4]

world = Path(f"src/calib_lab/bus_real_data/worlds/_temp_light_tests/bus_real_data_moving_camera_light_v2_{variant}.sdf")
if not world.exists():
    print("[ERROR] world not found:", world)
    print("[INFO] existing light_v2 worlds:")
    for p in sorted(Path("src/calib_lab/bus_real_data/worlds/_temp_light_tests").glob("bus_real_data_moving_camera_light_v2_*.sdf")):
        print(" ", p.name)
    raise SystemExit(1)

s = world.read_text()

pattern = r'(<light[^>]*type="directional"[^>]*>.*?<direction>)([^<]+)(</direction>)'
m = re.search(pattern, s, flags=re.S)
if not m:
    raise SystemExit("[ERROR] no directional light <direction> found")

new_dir = f"{dx} {dy} {dz}"
s2 = re.sub(pattern, rf'\1{new_dir}\3', s, count=1, flags=re.S)

world.write_text(s2)
print(f"[OK] patched {world}")
print(f"[OK] variant alias: {variant_in} -> {variant}")
print(f"[OK] new sun direction: {new_dir}")
