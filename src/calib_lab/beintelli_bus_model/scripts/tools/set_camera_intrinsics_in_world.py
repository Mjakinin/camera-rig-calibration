#!/usr/bin/env python3
# AUTO_IMPORT_COMMON_START
from pathlib import Path as _CalibLabPath
import sys as _CalibLabSys
for _p in _CalibLabPath(__file__).resolve().parents:
    if _p.name == "calib_lab":
        if str(_p) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_p))
        _common_scripts = _p / "common" / "scripts"
        if _common_scripts.exists() and str(_common_scripts) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_common_scripts))
        break
# AUTO_IMPORT_COMMON_END

import argparse
import math
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--world", required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--hfov_deg", type=float, required=True)
args = parser.parse_args()

world = Path(args.world)
text = world.read_text()

hfov_rad = math.radians(args.hfov_deg)

text = re.sub(
    r"<horizontal_fov>[^<]+</horizontal_fov>",
    f"<horizontal_fov>{hfov_rad:.8f}</horizontal_fov>",
    text,
)

text = re.sub(
    r"<width>[^<]+</width>",
    f"<width>{args.width}</width>",
    text,
)

text = re.sub(
    r"<height>[^<]+</height>",
    f"<height>{args.height}</height>",
    text,
)

world.write_text(text)

print(f"[OK] Updated {world}")
print(f"[OK] width={args.width}")
print(f"[OK] height={args.height}")
print(f"[OK] horizontal_fov={args.hfov_deg} deg = {hfov_rad:.8f} rad")
