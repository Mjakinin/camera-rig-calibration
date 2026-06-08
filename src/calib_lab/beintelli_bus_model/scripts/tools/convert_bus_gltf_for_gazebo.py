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

import sys
from pathlib import Path
import bpy

argv = sys.argv

if "--" not in argv:
    raise RuntimeError(
        "Usage: blender --background --python convert_bus_gltf_for_gazebo.py -- input.gltf output_dir"
    )

args = argv[argv.index("--") + 1:]

if len(args) != 2:
    raise RuntimeError("Expected: input.gltf output_dir")

input_gltf = Path(args[0]).resolve()
output_dir = Path(args[1]).resolve()

output_dir.mkdir(parents=True, exist_ok=True)

obj_path = output_dir / "beintelli_erklarbus.obj"
stl_path = output_dir / "beintelli_erklarbus.stl"

print(f"[INFO] Input glTF: {input_gltf}")
print(f"[INFO] Output dir: {output_dir}")

# Enable exporters for Blender 3.x
try:
    bpy.ops.preferences.addon_enable(module="io_scene_obj")
    print("[INFO] Enabled OBJ exporter add-on.")
except Exception as e:
    print(f"[WARN] Could not enable OBJ exporter add-on: {e}")

try:
    bpy.ops.preferences.addon_enable(module="io_mesh_stl")
    print("[INFO] Enabled STL exporter add-on.")
except Exception as e:
    print(f"[WARN] Could not enable STL exporter add-on: {e}")

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import glTF
print("[INFO] Importing glTF...")
bpy.ops.import_scene.gltf(filepath=str(input_gltf))

mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
print(f"[INFO] Mesh objects found: {len(mesh_objects)}")

if not mesh_objects:
    raise RuntimeError("No mesh objects found after importing glTF.")

# Apply transforms and triangulate
for obj in mesh_objects:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"[INFO] Processing mesh: {obj.name}")

    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    except Exception as e:
        print(f"[WARN] transform_apply failed for {obj.name}: {e}")

    try:
        triangulate = obj.modifiers.new(name="triangulate_for_gazebo", type="TRIANGULATE")
        bpy.ops.object.modifier_apply(modifier=triangulate.name)
    except Exception as e:
        print(f"[WARN] triangulate failed for {obj.name}: {e}")

# Select all mesh objects for export
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = mesh_objects[0]

print(f"[INFO] Exporting OBJ: {obj_path}")

try:
    bpy.ops.export_scene.obj(
        filepath=str(obj_path),
        use_selection=True,
        use_materials=True,
        path_mode="COPY",
    )
    print("[OK] OBJ export finished.")
except Exception as e:
    print(f"[ERROR] OBJ export failed: {e}")

print(f"[INFO] Exporting STL: {stl_path}")

try:
    bpy.ops.export_mesh.stl(
        filepath=str(stl_path),
        use_selection=True,
    )
    print("[OK] STL export finished.")
except Exception as e:
    print(f"[ERROR] STL export failed: {e}")

print("[DONE] Conversion finished.")
