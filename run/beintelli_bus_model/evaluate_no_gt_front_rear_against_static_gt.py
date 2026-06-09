#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np

R_OPT_LINK = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)

def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out

def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def rpy_to_R_world_link(roll, pitch, yaw):
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def optical_T_world_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)

def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_est_csv", required=True)
    ap.add_argument("--output_txt", required=True)
    args = ap.parse_args()

    T_est = np.loadtxt(args.T_est_csv, delimiter=",")

    T_front_world_gt = optical_T_world_from_gazebo_pose(
        -3.90, 0.0, 2.85,
        0.0, 0.69813170, 0.0,
    )
    T_rear_world_gt = optical_T_world_from_gazebo_pose(
        5.70, 0.0, 2.85,
        0.0, 0.69813170, math.pi,
    )

    T_gt = T_front_world_gt @ inv_T(T_rear_world_gt)

    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    baseline_est = float(np.linalg.norm(t_est))
    baseline_gt = float(np.linalg.norm(t_gt))
    baseline_error_cm = (baseline_est - baseline_gt) * 100.0
    translation_error_cm = float(np.linalg.norm(t_est - t_gt) * 100.0)
    rot_error = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])

    text = f"""NO-GT RESULT EVALUATION AGAINST STATIC GAZEBO GT
================================================

Important:
  This script evaluates the final result only.
  It must not be used inside the estimation pipeline.

Input:
  T_est_csv: {args.T_est_csv}

Metrics:
  baseline_est_m:       {baseline_est:.6f}
  baseline_gt_m:        {baseline_gt:.6f}
  baseline_error_cm:    {baseline_error_cm:.2f}
  translation_error_cm: {translation_error_cm:.2f}
  rotation_error_deg:   {rot_error:.2f}

T_est:
{T_est}

T_gt:
{T_gt}
"""

    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_txt).write_text(text)
    print(text)

if __name__ == "__main__":
    main()
