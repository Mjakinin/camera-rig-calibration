# Registering a Gazebo world

Gazebo worlds are data-driven. No Python change is required.

1. Copy `config/simulation_worlds/_template.yaml` to
   `config/simulation_worlds/<world-id>.yaml`.
2. Set the SDF, Gazebo resource directories, camera model/sensor names, ROS
   image and CameraInfo topics, and at least one route JSON.
3. Mark exactly one route with `baseline: true`.
4. List only parameters the world supports in `capabilities`.
5. Restart `rigcal`. The world appears in the simulation-world table.

Paths in a manifest are relative to the manifest file. Validation checks the
named SDF world, every static and moving camera sensor, topics, route files and
resource paths. Moving-camera baseline resolution and horizontal FOV are read
from the selected SDF sensor. Static camera XML is never rewritten by the
simulation parameter composer.

Example:

```yaml
schema_version: 1
id: warehouse
display_name: Warehouse calibration rig
sdf: ../../gazebo/warehouse.sdf
resource_paths:
  - ../../gazebo/models
static_cameras:
  - id: front_left
    model_name: front_left
    sensor_name: color_sensor
    image_topic: /front_left/image
    camera_info_topic: /front_left/camera_info
moving_camera:
  id: calibration_camera
  model_name: calibration_camera
  sensor_name: color_sensor
  image_topic: /calibration_camera/image
  camera_info_topic: /calibration_camera/camera_info
routes:
  - id: survey
    path: ../../gazebo/routes/survey.json
    baseline: true
capabilities:
  - route
  - density
  - resolution
  - fov
  - motion_blur
  - capture
```

The Wizard's guided import creates the same manifest from an SDF, route,
camera IDs and topics. To add optional lighting control, include `lighting` in
`capabilities` and declare `lighting_profiles`.
