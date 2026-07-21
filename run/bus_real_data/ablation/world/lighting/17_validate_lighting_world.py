#!/usr/bin/env python3

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("world")
    args = parser.parse_args()

    path = Path(args.world)

    if not path.is_file():
        raise SystemExit(f"[ERROR] Missing world: {path}")

    root = ET.parse(path).getroot()

    world = root.find("world")
    if world is None:
        world = next(root.iter("world"), None)

    if world is None:
        raise SystemExit(f"[ERROR] No <world> in {path}")

    lights = world.findall("light")

    if not lights:
        raise SystemExit(f"[ERROR] No lights found in {path}")

    bad = []

    for light in lights:
        name = light.get("name", "<unnamed>")
        light_type = light.get("type", "<unknown>")

        visualize = (
            light.findtext("visualize", default="")
            .strip()
            .lower()
        )

        if visualize != "false":
            bad.append(
                f"{name}: type={light_type}, "
                f"visualize={visualize or '<missing>'}"
            )

    if bad:
        print(f"[ERROR] Visible light helpers in {path}:")
        for item in bad:
            print(f"  - {item}")
        raise SystemExit(1)

    print(
        f"[OK] {path.name}: "
        f"{len(lights)} lights, all visualize=false"
    )


if __name__ == "__main__":
    main()
