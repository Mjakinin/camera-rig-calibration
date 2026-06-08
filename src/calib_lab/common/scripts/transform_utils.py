#!/usr/bin/env python3

import math
import cv2
import numpy as np


def rvec_tvec_to_matrix(rvec, tvec):
    """
    Convert OpenCV solvePnP output into a 4x4 homogeneous transform.

    solvePnP returns the pose of the target/object in the camera frame.
    Therefore this returns T_camera_target.
    """
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    return T


def invert_transform(T):
    """
    Invert a rigid 4x4 transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def relative_transform_from_common_target(T_cam_a_target, T_cam_b_target):
    """
    Compute relative camera transform from two observations of the same target.

    Given:
        T_cam_a_target
        T_cam_b_target

    We compute:
        T_cam_a_cam_b = T_cam_a_target * inverse(T_cam_b_target)

    This maps points from camera_b coordinates into camera_a coordinates.

    For the first sanity check, the most important value is the baseline norm.
    It should be close to the known physical camera distance.
    """
    return T_cam_a_target @ invert_transform(T_cam_b_target)


def translation_norm(T):
    return float(np.linalg.norm(T[:3, 3]))


def rotation_angle_deg_from_matrix(R):
    """
    Compute the rotation angle in degrees represented by a rotation matrix.
    """
    trace_value = np.trace(R)
    cos_theta = (trace_value - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def relative_rotation_angle_deg(T):
    return rotation_angle_deg_from_matrix(T[:3, :3])


def format_vector(v):
    v = np.asarray(v).reshape(-1)
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
