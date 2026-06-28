AP02 Gazebo Graph Visualization
===============================

Purpose
-------
This folder contains Gazebo SDF overlay models for visual debugging of AP02
inside the real bus Gazebo world.

Main idea
---------
AP02 estimates everything in the reference ArUco coordinate frame:

  aruco_marker_14 / aruco_ref_floor_14

For visualization, the script converts the AP02 result back into the Gazebo
world frame and spawns a visual-only overlay model.

Files
-----
- ap02_graph_clean_overlay_model.sdf
  Clean presentation overlay:
  Ref-ArUco, static cameras, marker map, ref-to-node edges, camera frustums.

- ap02_graph_full_debug_overlay_model.sdf
  Full debug overlay:
  everything from clean overlay, plus selected moving frames, moving trajectory,
  observation edges, GT nodes, and red estimated-vs-GT error vectors.

- spawn_clean_overlay.sh
  Spawns the clean overlay into the currently running Gazebo world.

- spawn_full_debug_overlay.sh
  Spawns the full debug overlay into the currently running Gazebo world.

Legend
------
- yellow sphere/plane:
  reference ArUco marker 14

- blue spheres/frustums:
  estimated static camera poses

- green spheres/planes:
  estimated marker-map poses

- thin blue/green lines:
  distance edges from Ref14 to cameras/markers

- purple nodes/line:
  selected moving-camera frames and trajectory

- orange transparent lines:
  AP02 observation edges used in BA

- black ghost nodes:
  GT camera/marker positions

- red lines:
  estimated-to-GT error vectors

How to use
----------
Terminal 1: start the bus world normally.

Terminal 2:
  cd /workspaces/project
  bash results/bus_real_data/90_approach_comparison_ref_aruco/91_gazebo_ap02_graph_debug/spawn_clean_overlay.sh

or:

  bash results/bus_real_data/90_approach_comparison_ref_aruco/91_gazebo_ap02_graph_debug/spawn_full_debug_overlay.sh

Recommended for presentation
----------------------------
Use clean overlay first. It is easier to explain.

Use full debug overlay when explaining why the moving camera helps:
the orange observation edges and purple trajectory show how the graph connects
markers and cameras that do not directly see Ref14.

