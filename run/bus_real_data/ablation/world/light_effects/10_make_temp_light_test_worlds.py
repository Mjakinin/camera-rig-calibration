#!/usr/bin/env python3
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
OUT_DIR = Path("src/calib_lab/bus_real_data/worlds/_temp_light_tests")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def child(parent, tag):
    return parent.find(tag)

def ensure(parent, tag, text=None):
    c = parent.find(tag)
    if c is None:
        c = ET.SubElement(parent, tag)
    if text is not None:
        c.text = text
    return c

def set_vec4(parent, tag, vals):
    ensure(parent, tag, " ".join(str(v) for v in vals))

def set_vec3(parent, tag, vals):
    ensure(parent, tag, " ".join(str(v) for v in vals))

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def find_world(root):
    w = root.find("world")
    if w is not None:
        return w
    for c in root:
        if c.tag.endswith("world"):
            return c
    raise RuntimeError("No <world> found")

def ensure_scene(world):
    sc = world.find("scene")
    if sc is None:
        sc = ET.SubElement(world, "scene")
    return sc

def find_first_directional(world):
    for l in world.findall("light"):
        if l.get("type") == "directional":
            return l
    # if none exists, create one
    l = ET.SubElement(world, "light", {"name": "temp_directional_sun", "type": "directional"})
    ensure(l, "cast_shadows", "true")
    set_vec4(l, "diffuse", [0.8, 0.8, 0.8, 1.0])
    set_vec4(l, "specular", [0.2, 0.2, 0.2, 1.0])
    set_vec3(l, "direction", [-0.5, 0.1, -1.0])
    ensure(l, "attenuation")
    return l

def add_spot(world, name, pose, direction,
             diffuse=(3.5, 3.2, 2.8, 1.0),
             specular=(0.3, 0.3, 0.3, 1.0),
             inner=0.35, outer=0.75, falloff=0.8,
             range_m=25.0, constant=0.8, linear=0.02, quadratic=0.001):
    # remove existing with same name
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
    ensure(att, "constant", str(constant))
    ensure(att, "linear", str(linear))
    ensure(att, "quadratic", str(quadratic))

    spot = ensure(l, "spot")
    ensure(spot, "inner_angle", str(inner))
    ensure(spot, "outer_angle", str(outer))
    ensure(spot, "falloff", str(falloff))
    return l

def make_variant(name, modifier):
    tree = ET.parse(SRC)
    root = tree.getroot()
    world = find_world(root)
    scene = ensure_scene(world)
    sun = find_first_directional(world)

    # remove any prior temp lights if present
    for l in list(world.findall("light")):
        if l.get("name", "").startswith("temp_") and l is not sun:
            world.remove(l)

    modifier(root, world, scene, sun)

    indent(root)
    out = OUT_DIR / name
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {out}")

def mod_baseline_copy(root, world, scene, sun):
    ensure(scene, "shadows", "true")

def mod_dim(root, world, scene, sun):
    # darker overall
    ensure(scene, "shadows", "true")
    set_vec4(scene, "ambient", [0.08, 0.08, 0.08, 1.0])
    set_vec4(scene, "background", [0.02, 0.02, 0.02, 1.0])

    ensure(sun, "cast_shadows", "true")
    set_vec4(sun, "diffuse", [0.28, 0.28, 0.28, 1.0])
    set_vec4(sun, "specular", [0.08, 0.08, 0.08, 1.0])
    set_vec3(sun, "direction", [-0.35, 0.05, -1.0])

def mod_side_sun(root, world, scene, sun):
    # stronger side sunlight
    ensure(scene, "shadows", "true")
    set_vec4(scene, "ambient", [0.18, 0.18, 0.18, 1.0])
    set_vec4(scene, "background", [0.65, 0.72, 0.82, 1.0])

    ensure(sun, "cast_shadows", "true")
    set_vec4(sun, "diffuse", [1.65, 1.55, 1.35, 1.0])
    set_vec4(sun, "specular", [0.35, 0.35, 0.30, 1.0])
    set_vec3(sun, "direction", [-1.0, -0.25, -0.55])

def mod_glare(root, world, scene, sun):
    # normal-ish ambient + strong local spotlight
    ensure(scene, "shadows", "true")
    set_vec4(scene, "ambient", [0.15, 0.15, 0.15, 1.0])
    set_vec4(scene, "background", [0.50, 0.55, 0.60, 1.0])

    ensure(sun, "cast_shadows", "true")
    set_vec4(sun, "diffuse", [0.75, 0.72, 0.68, 1.0])
    set_vec4(sun, "specular", [0.20, 0.20, 0.18, 1.0])
    set_vec3(sun, "direction", [-0.8, -0.1, -0.9])

    # Spotlight from one side into the bus area.
    # If this misses, we can tweak after your visual feedback.
    add_spot(
        world,
        name="temp_window_glare_spot",
        pose="1.5 2.6 2.8 0 0 -1.57",
        direction="0 -1 -0.25",
        diffuse=(4.8, 4.5, 4.1, 1.0),
        specular=(0.5, 0.5, 0.45, 1.0),
        inner=0.28,
        outer=0.85,
        falloff=0.9,
        range_m=30.0,
        constant=0.7,
        linear=0.015,
        quadratic=0.0008,
    )

def write_readme():
    txt = """Temporary light test worlds
===========================

These worlds are TEMP ONLY.
Delete folder if the idea is bad:

  rm -rf src/calib_lab/bus_real_data/worlds/_temp_light_tests

Files:
- bus_real_data_moving_camera_light_test_baseline_copy.sdf
- bus_real_data_moving_camera_light_test_dim.sdf
- bus_real_data_moving_camera_light_test_side_sun.sdf
- bus_real_data_moving_camera_light_test_glare.sdf
"""
    (OUT_DIR / "README_TEMP_LIGHT_TESTS.txt").write_text(txt)

def main():
    if not SRC.exists():
        raise SystemExit(f"Missing source world: {SRC}")

    make_variant("bus_real_data_moving_camera_light_test_baseline_copy.sdf", mod_baseline_copy)
    make_variant("bus_real_data_moving_camera_light_test_dim.sdf", mod_dim)
    make_variant("bus_real_data_moving_camera_light_test_side_sun.sdf", mod_side_sun)
    make_variant("bus_real_data_moving_camera_light_test_glare.sdf", mod_glare)
    write_readme()

if __name__ == "__main__":
    main()
