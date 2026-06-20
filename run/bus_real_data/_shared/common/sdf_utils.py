#!/usr/bin/env python3
import math
import xml.etree.ElementTree as ET
import numpy as np

from _shared.common.geometry import rpy_to_R, make_T, invT


R_MODEL_FROM_OPENCV_MARKER = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=np.float64)


def parse_pose_text(text):
    vals = [float(x) for x in text.split()]
    if len(vals) < 6:
        raise RuntimeError(f"Invalid SDF pose: {text}")
    x, y, z, roll, pitch, yaw = vals[:6]
    return make_T(rpy_to_R(roll, pitch, yaw), [x, y, z]), vals[:6]


def parse_world_poses(world_sdf):
    tree = ET.parse(world_sdf)
    root = tree.getroot()

    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "").strip()
        pose_el = model.find("pose")
        if not name or pose_el is None or not pose_el.text:
            continue
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None or not name_el.text or not pose_el.text:
            continue
        name = name_el.text.strip()
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    return poses


def get_R_opt_to_link():
    return rpy_to_R(0.0, -math.pi / 2.0, math.pi / 2.0)


def sdf_model_pose_to_optical(T_W_model):
    T_opt_to_link = make_T(get_R_opt_to_link(), np.zeros(3))
    return T_W_model @ invT(T_opt_to_link)


def sdf_marker_model_to_opencv_frame(T_W_model):
    T_model_cvmarker = make_T(R_MODEL_FROM_OPENCV_MARKER, np.zeros(3))
    return T_W_model @ T_model_cvmarker


def gt_static_camera_poses_ref_aruco(world_sdf, static_cameras, ref_marker_entity):
    poses = parse_world_poses(world_sdf)

    if ref_marker_entity not in poses:
        raise RuntimeError(f"Missing reference marker in SDF: {ref_marker_entity}")

    T_W_ref_cv = sdf_marker_model_to_opencv_frame(poses[ref_marker_entity]["T_W_model"])
    T_ref_W = invT(T_W_ref_cv)

    gt = {}
    for cam in static_cameras:
        if cam not in poses:
            raise RuntimeError(f"Missing camera in SDF: {cam}")
        T_W_cam = sdf_model_pose_to_optical(poses[cam]["T_W_model"])
        gt[cam] = T_ref_W @ T_W_cam

    return gt
