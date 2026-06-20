#!/usr/bin/env python3
from pathlib import Path

RESULTS_ROOT = Path("results/bus_real_data")

SHARED_RAW_ROOT = RESULTS_ROOT / "00_raw_images" / "bus_real_data_ref_marker_v1"
SHARED_STATIC_DIR = SHARED_RAW_ROOT / "static"
SHARED_MOVING_DIR = SHARED_RAW_ROOT / "moving"
SHARED_CAMERA_INFO_DIR = SHARED_RAW_ROOT / "camera_info"

WORLD_SDF_MOVING_CAMERA = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

ARUCO_DICT_NAME = "DICT_4X4_50"
REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"
MARKER_LENGTH_M = 0.170
