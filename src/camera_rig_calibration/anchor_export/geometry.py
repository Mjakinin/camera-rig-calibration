from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RobustPoseResult:
    transform: np.ndarray
    inlier_indices: tuple[int, ...]
    translation_residuals_m: tuple[float, ...]
    rotation_residuals_deg: tuple[float, ...]
    translation_threshold_m: float
    rotation_threshold_deg: float


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return validate_transform(transform)


def validate_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("SE(3) transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("SE(3) transform has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("SE(3) rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("SE(3) rotation determinant is not +1")
    return value.copy()


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = validate_transform(transform)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def rvec_to_rotation(vector: Iterable[float]) -> np.ndarray:
    rvec = np.asarray(tuple(vector), dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-14:
        return np.eye(3, dtype=np.float64)
    axis = rvec / angle
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0 or (
        abs(float(quaternion[3])) < 1e-15
        and next((value for value in quaternion[:3] if abs(value) > 1e-15), 1.0)
        < 0.0
    ):
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def quaternion_to_rotation(quaternion: Iterable[float]) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(tuple(quaternion), dtype=np.float64).reshape(4)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = math.atan2(
        -float(matrix[2, 0]),
        math.hypot(float(matrix[0, 0]), float(matrix[1, 0])),
    )
    roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return roll, pitch, yaw


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = min(1.0, max(-1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def rigid_fit(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Rigid fit requires matching Nx3 point arrays")
    if source.shape[0] < 3:
        raise ValueError("Rigid fit requires at least three points")
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    if np.linalg.matrix_rank(source_centered, tol=1e-10) < 2:
        raise ValueError("Anchor corner geometry is degenerate")
    covariance = target_centered.T @ source_centered
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[2, 2] = -1.0
    rotation = left @ correction @ right_t
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    transform = make_transform(rotation, translation)
    predicted = (rotation @ source.T).T + translation
    rmse = math.sqrt(float(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    return transform, rmse


def _weighted_rotation_mean(
    rotations: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for rotation, weight in zip(rotations, weights, strict=True):
        quaternion = np.asarray(rotation_to_quaternion(rotation), dtype=np.float64)
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, -1]
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion_to_rotation(quaternion)


def robust_pose_average(
    transforms: Iterable[np.ndarray],
    weights: Iterable[float],
) -> RobustPoseResult:
    poses = [validate_transform(value) for value in transforms]
    weight_values = np.asarray(tuple(weights), dtype=np.float64)
    if not poses or len(poses) != len(weight_values):
        raise ValueError("Pose aggregation requires matching non-empty inputs")
    weight_values = np.maximum(weight_values, 1e-9)
    translations = np.vstack([pose[:3, 3] for pose in poses])
    initial_translation = np.median(translations, axis=0)
    initial_rotation = _weighted_rotation_mean(
        [pose[:3, :3] for pose in poses], weight_values
    )
    translation_residuals = np.linalg.norm(
        translations - initial_translation, axis=1
    )
    rotation_residuals = np.asarray(
        [
            rotation_error_deg(
                make_transform(initial_rotation, np.zeros(3)),
                make_transform(pose[:3, :3], np.zeros(3)),
            )
            for pose in poses
        ]
    )

    def threshold(values: np.ndarray, floor: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 3.0 * 1.4826 * mad)

    translation_threshold = threshold(translation_residuals, 0.20)
    rotation_threshold = threshold(rotation_residuals, 5.0)
    inliers = tuple(
        index
        for index, (translation, rotation) in enumerate(
            zip(translation_residuals, rotation_residuals, strict=True)
        )
        if translation <= translation_threshold and rotation <= rotation_threshold
    )
    if not inliers:
        raise ValueError("Robust pose aggregation rejected every candidate")
    selected_weights = weight_values[list(inliers)]
    selected_weights /= selected_weights.sum()
    translation = np.average(translations[list(inliers)], axis=0, weights=selected_weights)
    rotation = _weighted_rotation_mean(
        [poses[index][:3, :3] for index in inliers],
        selected_weights,
    )
    return RobustPoseResult(
        transform=make_transform(rotation, translation),
        inlier_indices=inliers,
        translation_residuals_m=tuple(float(value) for value in translation_residuals),
        rotation_residuals_deg=tuple(float(value) for value in rotation_residuals),
        translation_threshold_m=translation_threshold,
        rotation_threshold_deg=rotation_threshold,
    )


def pose_payload(transform: np.ndarray) -> dict[str, object]:
    value = validate_transform(transform)
    roll, pitch, yaw = rotation_to_rpy(value[:3, :3])
    qx, qy, qz, qw = rotation_to_quaternion(value[:3, :3])
    return {
        "x_m": float(value[0, 3]),
        "y_m": float(value[1, 3]),
        "z_m": float(value[2, 3]),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "yaw_rad": float(yaw),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
        "matrix": [[float(item) for item in row] for row in value],
    }
