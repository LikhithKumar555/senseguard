"""Rigid-body transforms used everywhere in SenseGuard.

Conventions
-----------
- All angles are in **radians** inside the code. Convert at the edges.
- A 4x4 matrix named ``T_a_from_b`` takes a point written in frame ``b``
  and returns the same point written in frame ``a``:  p_a = T_a_from_b @ p_b.
  With this naming, chains read naturally:
  T_cam_from_radar = T_cam_from_vehicle @ T_vehicle_from_radar.
- Vehicle and radar frames: x forward, y left, z up.
- Camera frame (KITTI / OpenCV style): x right, y down, z forward.
"""

from __future__ import annotations

import numpy as np

# Rotation that turns vehicle-style axes (x fwd, y left, z up) into
# camera-style axes (x right, y down, z fwd), for a camera looking straight ahead.
R_CAM_FROM_VEHICLE_AXES = np.array(
    [
        [0.0, -1.0, 0.0],  # cam x (right) = -vehicle y (left)
        [0.0, 0.0, -1.0],  # cam y (down)  = -vehicle z (up)
        [1.0, 0.0, 0.0],   # cam z (fwd)   =  vehicle x (fwd)
    ]
)


def rot_x(angle: float) -> np.ndarray:
    """Rotation about the x axis (roll)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about the y axis (pitch)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about the z axis (yaw). Positive yaw turns x towards y (left)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw to a rotation matrix, applied as R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a 3-vector."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, using R^T instead of a general matrix inverse."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points. Returns (N, 3)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    return points @ T[:3, :3].T + T[:3, 3]


def rotate_vectors(T_or_R: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) direction or velocity vectors. Translation is ignored."""
    R = np.asarray(T_or_R, dtype=float)[:3, :3]
    vectors = np.asarray(vectors, dtype=float)
    return vectors @ R.T


def box_corners(center, size, yaw: float) -> np.ndarray:
    """The 8 corners of a 3D box that is rotated only about z.

    center: (x, y, z) of the box centre.
    size:   (length along x, width along y, height along z) before rotation.
    yaw:    heading in radians.
    Returns an (8, 3) array. The first 4 corners are the bottom face.
    """
    length, width, height = np.asarray(size, dtype=float)
    dx, dy, dz = length / 2.0, width / 2.0, height / 2.0
    local = np.array(
        [
            [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz],
            [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        ]
    )
    return local @ rot_z(yaw).T + np.asarray(center, dtype=float)
