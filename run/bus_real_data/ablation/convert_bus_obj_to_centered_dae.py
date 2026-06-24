import bpy
from pathlib import Path
from mathutils import Vector

repo = Path.home() / "app-ras" / "camera-rig-calibration"

obj_path = repo / "src/calib_lab/bus_real_data/models/beintelli_bus/meshes/obj/beintelli_erklarbus.obj"
dae_path = repo / "src/calib_lab/bus_real_data/models/beintelli_bus/meshes/obj/beintelli_erklarbus_centered.dae"

# Clean scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import OBJ with MTL/textures
bpy.ops.import_scene.obj(filepath=str(obj_path))

objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

if not objects:
    raise RuntimeError("No mesh objects imported from OBJ.")

# Calculate global bounding box
min_v = Vector((float("inf"), float("inf"), float("inf")))
max_v = Vector((float("-inf"), float("-inf"), float("-inf")))

for obj in objects:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        min_v.x = min(min_v.x, world_corner.x)
        min_v.y = min(min_v.y, world_corner.y)
        min_v.z = min(min_v.z, world_corner.z)
        max_v.x = max(max_v.x, world_corner.x)
        max_v.y = max(max_v.y, world_corner.y)
        max_v.z = max(max_v.z, world_corner.z)

center_x = (min_v.x + max_v.x) / 2.0
center_y = (min_v.y + max_v.y) / 2.0
bottom_z = min_v.z

offset = Vector((-center_x, -center_y, -bottom_z))

print("Old min:", min_v)
print("Old max:", max_v)
print("Applying offset:", offset)

# Apply offset to all mesh objects
for obj in objects:
    obj.location += offset

# Select all mesh objects for export
bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = objects[0]

# Export Collada / DAE
bpy.ops.wm.collada_export(
    filepath=str(dae_path),
    selected=True,
    apply_modifiers=True,
    triangulate=True,
)

print("Wrote:", dae_path)

