# BeIntelli Bus Meshes

This folder contains mesh representations and textures for the BeIntelli bus model.

## Subfolders

```text
dae/
```

COLLADA mesh assets.

```text
gltf/
```

glTF scene and texture assets.

```text
obj/
```

OBJ/MTL mesh assets and texture files.

## Usage

The Gazebo model references these mesh files from `model.sdf`. Keep relative paths stable unless you also update the model SDF.

## Git/LFS note

Large mesh and texture assets may be tracked through Git LFS. Avoid duplicating large binary files unless the model genuinely needs another representation.
