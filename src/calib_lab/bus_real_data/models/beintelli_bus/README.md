# BeIntelli Bus Model

This folder contains the Gazebo model definition and mesh assets for the BeIntelli/Intellibus-style vehicle used in the benchmark world.

## Contents

```text
model.config
model.sdf
```

Gazebo model metadata and SDF definition.

```text
meshes/
```

Geometry and texture assets used by the model.

```text
ATTRIBUTION.txt
```

Attribution/licensing information for the model assets.

## Usage

The bus model is referenced by the main worlds under:

```text
src/calib_lab/bus_real_data/worlds/
```

The model is part of the rendered scene and provides physical/visual context for camera placement. It is not itself a calibration algorithm.

## Editing rule

Keep mesh and attribution files together. If model paths change, update the relevant SDF worlds and verify that Gazebo still resolves all mesh URIs.
