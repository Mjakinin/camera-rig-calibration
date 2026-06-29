#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
OUT_DIR = Path("src/calib_lab/bus_real_data/worlds/_temp_light_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure(parent, tag, text=None):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    if text is not None:
        node.text = str(text)
    return node

def set_vec3(parent, tag, vals):
    ensure(parent, tag, " ".join(str(v) for v in vals))

def set_vec4(parent, tag, vals):
    ensure(parent, tag, " ".join(str(v) for v in vals))

def find_world(root):
    w = root.find("world")
    if w is not None:
        return w
    for e in root.iter("world"):
        return e
    raise RuntimeError("No <world> found")

def ensure_scene(world):
    scene = world.find("scene")
    if scene is None:
        scene = ET.SubElement(world, "scene")
    return scene

def find_or_create_sun(world):
    for l in world.findall("light"):
        if l.get("type") == "directional":
            return l
    l = ET.SubElement(world, "light", {"name": "temp_directional_sun", "type": "directional"})
    return l

def remove_temp_lights(world):
    for l in list(world.findall("light")):
        name = l.get("name", "")
        if name.startswith("temp_") and l.get("type") != "directional":
            world.remove(l)

def add_spot(world, name, pose, direction, diffuse, specular, inner, outer, falloff, range_m):
    for l in list(world.findall("light")):
        if l.get("name") == name:
            world.remove(l)

    l = ET.SubElement(world, "light", {"name": name, "type": "spot"})
    ensure(l, "cast_shadows", "true")
    ensure(l, "pose", pose)
    set_vec4(l, "diffuse", diffuse)
    set_vec4(l, "specular", specular)
    set_vec3(l, "direction", direction)

    att = ensure(l, "attenuation")
    ensure(att, "range", str(range_m))
    ensure(att, "constant", "0.7")
    ensure(att, "linear", "0.015")
    ensure(att, "quadratic", "0.0008")

    spot = ensure(l, "spot")
    ensure(spot, "inner_angle", str(inner))
    ensure(spot, "outer_angle", str(outer))
    ensure(spot, "falloff", str(falloff))

def write_variant(filename, config):
    tree = ET.parse(SRC)
    root = tree.getroot()
    world = find_world(root)
    scene = ensure_scene(world)
    sun = find_or_create_sun(world)

    remove_temp_lights(world)

    ensure(scene, "shadows", "true")
    set_vec4(scene, "ambient", config["ambient"])
    set_vec4(scene, "background", config["background"])

    ensure(sun, "cast_shadows", "true")
    set_vec4(sun, "diffuse", config["sun_diffuse"])
    set_vec4(sun, "specular", config["sun_specular"])
    set_vec3(sun, "direction", config["sun_direction"])

    if "spot" in config:
        add_spot(world, **config["spot"])

    ET.indent(tree, space="  ")
    out = OUT_DIR / filename
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {out}")

def main():
    if not SRC.exists():
        raise SystemExit(f"Missing source world: {SRC}")

    variants = {
        # Baseline-like, but explicit scene values.
        "bus_real_data_moving_camera_light_v2_baseline.sdf": {
            "ambient": [0.40, 0.40, 0.40, 1.0],
            "background": [0.70, 0.75, 0.85, 1.0],
            "sun_diffuse": [0.80, 0.80, 0.78, 1.0],
            "sun_specular": [0.20, 0.20, 0.20, 1.0],
            "sun_direction": [-0.5, -0.2, -1.0],
        },

        # Moderate low light: darker than baseline, but not black.
        "bus_real_data_moving_camera_light_v2_low_moderate.sdf": {
            "ambient": [0.22, 0.22, 0.22, 1.0],
            "background": [0.12, 0.12, 0.14, 1.0],
            "sun_diffuse": [0.42, 0.42, 0.40, 1.0],
            "sun_specular": [0.08, 0.08, 0.08, 1.0],
            "sun_direction": [-0.35, 0.05, -1.0],
        },

        # Current style, kept as stress test.
        "bus_real_data_moving_camera_light_v2_low_extreme.sdf": {
            "ambient": [0.08, 0.08, 0.08, 1.0],
            "background": [0.02, 0.02, 0.02, 1.0],
            "sun_diffuse": [0.28, 0.28, 0.28, 1.0],
            "sun_specular": [0.04, 0.04, 0.04, 1.0],
            "sun_direction": [-0.35, 0.05, -1.0],
        },

        # More realistic side sun: keep fill light high enough.
        "bus_real_data_moving_camera_light_v2_side_sun.sdf": {
            "ambient": [0.38, 0.38, 0.36, 1.0],
            "background": [0.72, 0.78, 0.88, 1.0],
            "sun_diffuse": [1.45, 1.35, 1.15, 1.0],
            "sun_specular": [0.35, 0.33, 0.28, 1.0],
            "sun_direction": [-1.0, 0.05, -0.015],
        },

        # Normal-ish bus + strong local glare source.
        "bus_real_data_moving_camera_light_v2_glare.sdf": {
            "ambient": [0.32, 0.32, 0.31, 1.0],
            "background": [0.65, 0.70, 0.78, 1.0],
            "sun_diffuse": [0.75, 0.72, 0.68, 1.0],
            "sun_specular": [0.20, 0.20, 0.18, 1.0],
            "sun_direction": [-1.0, 0.03, -0.005],
            "spot": {
                "name": "temp_window_glare_spot",
                "pose": "5.8 1.4 2.25 0 0 2.95",
                "direction": [-1.0, -0.10, -0.02],
                "diffuse": [1.0, 0.97, 0.88, 1.0],
                "specular": [0.45, 0.45, 0.40, 1.0],
                "inner": 0.25,
                "outer": 0.75,
                "falloff": 0.9,
                "range_m": 30.0,
            },
        },
    }

    for filename, config in variants.items():
        write_variant(filename, config)

    readme = OUT_DIR / "README_TEMP_LIGHT_TESTS_V2.txt"
    readme.write_text(
        "Temporary V2 light test worlds. Safe to delete:\n\n"
        "rm -rf src/calib_lab/bus_real_data/worlds/_temp_light_tests\n\n"
        "Recommended final candidates:\n"
        "- light_v2_low_moderate\n"
        "- light_v2_side_sun\n"
        "- light_v2_glare\n"
        "- optional light_v2_low_extreme as stress test\n"
    )
    print(f"[OK] wrote {readme}")

if __name__ == "__main__":
    main()
