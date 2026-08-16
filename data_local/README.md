# Local real-data input (`data_local`)

`data_local/` is the landing zone for **local real-world recordings** and validated local simulation-route files.

Raw acquisition data below this directory is intentionally ignored by Git. Put the data here, let `rigcal` discover it, confirm the proposed roles in the Wizard, and publish the resulting experiment under `results/`.

## Simplest real-world workflow

Create one directory per acquisition:

```text
data_local/<experiment_name>/
```

Then start:

```bash
rigcal
```

and choose:

```text
Start a new calibration
  → Real data
  → select the experiment
  → confirm detected inputs/intrinsics
  → add AP01/AP02/AP03
  → review
  → run
```

`rigcal` scans each experiment directory recursively. Exact filenames and subfolders are flexible.

## Recommended layout

A clean layout is:

```text
data_local/vehicle_exterior_day_01/
├── static/
│   ├── front_left.png
│   ├── front_right.png
│   └── ...
├── moving_frames/
│   ├── frame_000001.png
│   └── ...
├── camera_info/
├── intrinsics/
└── checkerboard/
```

A flatter directory is also valid, for example:

```text
data_local/vehicle_exterior_day_01/
├── moving_camera.mp4
├── checkerboard.mp4
├── front_left.png
├── front_left_intrinsics.yaml
├── front_right.png
├── front_right_meta.yaml
└── optional_all_cameras.mcap
```

Typical supported inputs include:

- static images or static-camera videos;
- a moving-camera video or extracted moving frames;
- camera-info JSON/YAML files;
- intrinsics files;
- checkerboard video/images;
- ROS 2 bags (`.mcap` or `.db3`).

The Wizard proposes roles and asks for confirmation before saving a configuration.

## Moving-camera intrinsics

Moving frames and moving-camera intrinsics are independent selections.

A checkerboard recording can be used to create/recalculate a reusable managed profile:

```text
rigcal
  → Start a new calibration
  → Real data
  → Create or recalculate moving-camera intrinsics
```

Managed profiles are published below:

```text
config/intrinsics/<profile-id>/<profile-hash>/
```

A profile may then be selected for compatible moving frames. Camera/lens identity is shown explicitly; resolution alone is not treated as sufficient identity.

## Local simulation routes

Validated local moving-camera routes belong below:

```text
data_local/simulation_routes/
```

The simulation Wizard discovers valid routes from there. Route validation is performed before Gazebo capture.

## What not to put here

- Published calibration results belong in `results/`.
- Temporary execution state belongs in `workspace/`.
- Managed reusable intrinsics belong in `config/intrinsics/`.

Do not commit large raw acquisitions merely to make the Wizard see them; local files in `data_local/` are the intended workflow.
