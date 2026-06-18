#!/usr/bin/env python3

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE_WORLD = ROOT / "worlds" / "bus_real_data_camera_layout.sdf"
PLACEMENTS = ROOT / "config" / "a4_marker_placements.json"
OUT_WORLD = ROOT / "worlds" / "bus_real_data_a4_markers.sdf"

def include_block(item):
    return f"""
    <include>
      <uri>model://{item['model']}</uri>
      <name>{item['name']}</name>
      <pose>{item['pose']}</pose>
    </include>
"""

def main():
    text = BASE_WORLD.read_text()
    placements = json.loads(PLACEMENTS.read_text())

    marker_blocks = "\n".join(include_block(p) for p in placements)

    if "</world>" not in text:
        raise RuntimeError("Could not find </world> in base world.")

    text = text.replace("</world>", marker_blocks + "\n  </world>")
    OUT_WORLD.write_text(text)

    print(f"[OK] wrote {OUT_WORLD}")
    print(f"[OK] inserted {len(placements)} A4 ArUco markers")

if __name__ == "__main__":
    main()
