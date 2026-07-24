# Local real-data landing zone

Create one directory per recording below this directory and put all available
inputs anywhere inside it. Subfolders and exact filenames are optional.

Example:

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

`rigcal` scans each dataset directory recursively. It proposes roles for moving
videos or frames, static image/intrinsic pairs, checkerboard videos, and ROS 2
camera topics, then asks for confirmation before saving a configuration. Raw
files below this directory remain ignored by Git.

The checkerboard video does not need to remain coupled to one frame dataset.
Use `rigcal` → Start a new calibration → Real data → Create or recalculate
moving-camera intrinsics to publish a reusable profile. The profile can then
be selected for any existing or new moving frames with the same exact image
resolution. Camera/lens identity is shown explicitly and is never guessed from
resolution alone.
