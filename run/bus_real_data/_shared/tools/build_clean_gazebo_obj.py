import bpy
import bmesh
import math
import re
import shutil
import sys
from pathlib import Path


if "--" not in sys.argv:
    raise RuntimeError("Arguments after -- are missing")

args = sys.argv[sys.argv.index("--") + 1:]

if len(args) != 3:
    raise RuntimeError(
        "Usage: blender --background --python script.py -- "
        "<input.glb> <source_texture_dir> <output_dir>"
    )

source_glb = Path(args[0]).resolve()
source_texture_dir = Path(args[1]).resolve()
output_dir = Path(args[2]).resolve()

output_obj = output_dir / "bus_innenraum_clean.obj"
output_mtl = output_dir / "bus_innenraum_clean.mtl"
output_texture_dir = output_dir / "textures"

if not source_glb.is_file():
    raise RuntimeError(f"GLB missing: {source_glb}")

if not source_texture_dir.is_dir():
    raise RuntimeError(
        f"Source texture directory missing: {source_texture_dir}"
    )

output_dir.mkdir(parents=True, exist_ok=True)
output_texture_dir.mkdir(parents=True, exist_ok=True)


def find_principled(material):
    if not material.use_nodes:
        return None

    if material.node_tree is None:
        return None

    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node

    return None


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ------------------------------------------------------------
# Clear Blender scene
# ------------------------------------------------------------

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


# ------------------------------------------------------------
# Import original Sketchfab GLB
# ------------------------------------------------------------

bpy.ops.import_scene.gltf(
    filepath=str(source_glb)
)

mesh_objects = [
    obj
    for obj in bpy.context.scene.objects
    if obj.type == "MESH"
]

if not mesh_objects:
    raise RuntimeError("No mesh objects imported from GLB")

print("[INFO] Mesh objects:", len(mesh_objects))
print(
    "[INFO] Faces:",
    sum(len(obj.data.polygons) for obj in mesh_objects),
)
print("[INFO] Materials:", len(bpy.data.materials))
print("[INFO] Blender images:", len(bpy.data.images))


# ------------------------------------------------------------
# Copy already extracted GLB textures
# ------------------------------------------------------------

available_textures = {}

for source in sorted(source_texture_dir.glob("Image_*.*")):
    if source.suffix.casefold() not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        continue

    target = output_texture_dir / source.name
    shutil.copy2(source, target)

    available_textures[source.stem] = target

    print(
        f"[TEXTURE] {source.name} -> "
        f"textures/{target.name}"
    )

if len(available_textures) < 9:
    raise RuntimeError(
        "Expected at least 9 extracted GLB textures, "
        f"found {len(available_textures)}"
    )


# ------------------------------------------------------------
# Redirect Blender image datablocks to persistent textures
# ------------------------------------------------------------

redirected_images = 0

for image in bpy.data.images:
    image_name = Path(image.name).stem

    match = re.search(
        r"Image_(\d+)",
        image_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        continue

    canonical_name = f"Image_{int(match.group(1))}"

    target = available_textures.get(canonical_name)

    if target is None:
        print(
            "[WARN] No persistent texture for image:",
            image.name,
        )
        continue

    image.filepath = str(target)
    image.filepath_raw = str(target)

    redirected_images += 1

    print(
        f"[IMAGE] {image.name} -> {target.name}"
    )

print("[INFO] Redirected images:", redirected_images)


# ------------------------------------------------------------
# Simplify materials for Gazebo Fortress
# ------------------------------------------------------------

for material in bpy.data.materials:
    principled = find_principled(material)

    if principled is not None:
        metallic = principled.inputs.get("Metallic")
        roughness = principled.inputs.get("Roughness")
        specular = principled.inputs.get("Specular")
        alpha = principled.inputs.get("Alpha")
        emission = principled.inputs.get("Emission")
        emission_strength = principled.inputs.get(
            "Emission Strength"
        )

        if metallic is not None:
            metallic.default_value = 0.0

        if roughness is not None:
            roughness.default_value = 0.82

        if specular is not None:
            specular.default_value = 0.06

        if alpha is not None:
            alpha.default_value = 1.0

        if emission is not None:
            emission.default_value = (
                0.0,
                0.0,
                0.0,
                1.0,
            )

        if emission_strength is not None:
            emission_strength.default_value = 0.0

    # Fortress handles these complex transparency materials poorly.
    material.blend_method = "OPAQUE"

    if hasattr(material, "use_screen_refraction"):
        material.use_screen_refraction = False

    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False


# ------------------------------------------------------------
# Geometry cleanup without duplicating or merging UV seams
# ------------------------------------------------------------

removed_degenerate_faces = 0

for index, obj in enumerate(mesh_objects):
    world_matrix = obj.matrix_world.copy()

    obj.parent = None
    obj.matrix_world = world_matrix

    bpy.ops.object.select_all(action="DESELECT")

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.transform_apply(
        location=True,
        rotation=True,
        scale=True,
    )

    mesh = obj.data
    faces_before = len(mesh.polygons)

    bm = bmesh.new()
    bm.from_mesh(mesh)

    bm.edges.ensure_lookup_table()

    if bm.edges:
        bmesh.ops.dissolve_degenerate(
            bm,
            edges=list(bm.edges),
            dist=1.0e-10,
        )

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    mesh.validate(
        verbose=False,
        clean_customdata=False,
    )

    # Smooth curved areas but retain hard CAD transitions.
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.radians(35.0)

    mesh.update()

    removed_degenerate_faces += (
        faces_before - len(mesh.polygons)
    )

    if index % 20 == 0:
        print(
            f"[CLEAN] {index + 1}/"
            f"{len(mesh_objects)}: {obj.name}"
        )

print(
    "[INFO] Removed degenerate faces:",
    removed_degenerate_faces,
)


# ------------------------------------------------------------
# Remove cameras, lights and helper nodes
# ------------------------------------------------------------

for obj in list(bpy.context.scene.objects):
    if obj.type != "MESH":
        bpy.data.objects.remove(
            obj,
            do_unlink=True,
        )


# ------------------------------------------------------------
# Export OBJ
# ------------------------------------------------------------

bpy.ops.object.select_all(action="DESELECT")

for obj in mesh_objects:
    if obj.name not in bpy.context.scene.objects:
        continue

    obj.hide_set(False)
    obj.hide_render = False
    obj.select_set(True)

bpy.context.view_layer.objects.active = mesh_objects[0]

try:
    bpy.ops.preferences.addon_enable(
        module="io_scene_obj"
    )
except Exception:
    pass

bpy.ops.export_scene.obj(
    filepath=str(output_obj),
    use_selection=True,
    use_mesh_modifiers=True,
    use_edges=False,
    use_smooth_groups=True,
    use_smooth_groups_bitflags=False,
    use_normals=True,
    use_uvs=True,
    use_materials=True,
    use_triangles=True,
    path_mode="RELATIVE",
    axis_forward="-Z",
    axis_up="Y",
)

if not output_obj.is_file():
    raise RuntimeError("OBJ export failed")

generated_mtl = output_obj.with_suffix(".mtl")

if not generated_mtl.is_file():
    raise RuntimeError(
        f"MTL export failed: {generated_mtl}"
    )


# ------------------------------------------------------------
# Make MTL conservative and Fortress-compatible
# ------------------------------------------------------------

cleaned_lines = []

for line in generated_mtl.read_text(
    errors="replace"
).splitlines():
    stripped = line.strip()

    if stripped.startswith("Ka "):
        values = stripped.split()

        if len(values) == 4:
            rgb = [
                clamp(float(value), 0.03, 0.35)
                for value in values[1:]
            ]

            cleaned_lines.append(
                "Ka "
                + " ".join(
                    f"{value:.6f}"
                    for value in rgb
                )
            )
        else:
            cleaned_lines.append(line)

    elif stripped.startswith("Kd "):
        values = stripped.split()

        if len(values) == 4:
            rgb = [
                clamp(float(value), 0.035, 0.85)
                for value in values[1:]
            ]

            cleaned_lines.append(
                "Kd "
                + " ".join(
                    f"{value:.6f}"
                    for value in rgb
                )
            )
        else:
            cleaned_lines.append(line)

    elif stripped.startswith("Ks "):
        cleaned_lines.append(
            "Ks 0.015000 0.015000 0.015000"
        )

    elif stripped.startswith("Ke "):
        cleaned_lines.append(
            "Ke 0.000000 0.000000 0.000000"
        )

    elif stripped.startswith("Ns "):
        cleaned_lines.append("Ns 4.000000")

    elif stripped.startswith("Ni "):
        cleaned_lines.append("Ni 1.000000")

    elif stripped.startswith("d "):
        cleaned_lines.append("d 1.000000")

    elif stripped.startswith("Tr "):
        cleaned_lines.append("Tr 0.000000")

    elif stripped.startswith("illum "):
        cleaned_lines.append("illum 2")

    elif stripped.startswith(
        (
            "Pr ",
            "Pm ",
            "Ps ",
            "Pc ",
            "Pcr ",
            "aniso ",
            "anisor ",
            "map_Pr ",
            "map_Pm ",
            "map_Ps ",
            "map_Ke ",
            "map_Bump ",
            "bump ",
            "norm ",
            "refl ",
        )
    ):
        # Discard complex PBR maps for this controlled test.
        continue

    elif stripped.startswith("map_Kd "):
        raw_path = stripped.split(maxsplit=1)[1]
        filename = Path(raw_path).name

        source_candidate = output_texture_dir / filename

        if source_candidate.is_file():
            cleaned_lines.append(
                f"map_Kd textures/{filename}"
            )
        else:
            print(
                "[WARN] Exported material references "
                f"missing diffuse texture: {filename}"
            )

    else:
        cleaned_lines.append(line)

output_mtl.write_text(
    "\n".join(cleaned_lines) + "\n"
)

if generated_mtl != output_mtl:
    generated_mtl.unlink(missing_ok=True)

obj_text = output_obj.read_text(
    errors="replace"
)

obj_text, count = re.subn(
    r"^mtllib\s+.+$",
    f"mtllib {output_mtl.name}",
    obj_text,
    count=1,
    flags=re.MULTILINE,
)

if count != 1:
    raise RuntimeError(
        "Could not normalize OBJ mtllib reference"
    )

output_obj.write_text(obj_text)

print()
print("[OK] Clean OBJ:", output_obj)
print("[OK] Clean MTL:", output_mtl)
print("[OK] OBJ bytes:", output_obj.stat().st_size)
print("[OK] MTL bytes:", output_mtl.stat().st_size)
