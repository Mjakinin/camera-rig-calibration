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
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--world", required=True)
parser.add_argument("--pose", required=True, help='Example: "0 0 0 0 0 0"')
args = parser.parse_args()

world = Path(args.world)
text = world.read_text()

pattern = re.compile(
    r'(<include>\s*<uri>model://beintelli_bus</uri>\s*<name>beintelli_bus</name>\s*<pose>)([^<]+)(</pose>\s*</include>)',
    re.MULTILINE,
)

new_text, n = pattern.subn(
    rf'\g<1>{args.pose}\g<3>',
    text,
)

if n != 1:
    raise RuntimeError(f"Expected to replace exactly one beintelli_bus pose, replaced {n}")

world.write_text(new_text)
print(f"[OK] Updated bus pose in {world}")
print(f"[OK] New pose: {args.pose}")
