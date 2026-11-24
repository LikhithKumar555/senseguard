"""Pinhole camera projection, and projecting radar points and 3D boxes into the image.

The camera model has no lens distortion. VoD images are rectified, so this
should be enough there too, but check it when the dataset arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senseguard.geometry.transforms import box_corners, transform_points


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus the image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @classmethod
    def from_matrix(cls, K, width: int, height: int) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=float)
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=width, height=height)


def project_points(points_cam: np.ndarray, intrinsics: CameraIntrinsics):
    """Project (N, 3) points in the camera frame to pixels.

    Returns
    -------
    uv:    (N, 2) pixel coordinates (u = column, v = row). Points at or behind
           the camera get NaN, so they can never be mistaken for valid pixels.
    depth: (N,) the z coordinate in the camera frame (metres).
    """
    points_cam = np.asarray(points_cam, dtype=float)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must have shape (N, 3), got {points_cam.shape}")
    depth = points_cam[:, 2].copy()
    uv = np.full((len(points_cam), 2), np.nan)
    front = depth > 1e-6
    x, y, z = points_cam[front].T
    uv[front, 0] = intrinsics.fx * x / z + intrinsics.cx
    uv[front, 1] = intrinsics.fy * y / z + intrinsics.cy
    return uv, depth


def in_image_mask(uv: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics,
                  min_depth: float = 0.1) -> np.ndarray:
    """True for points that are in front of the camera and inside the image."""
    with np.errstate(invalid="ignore"):
        return (
            (depth > min_depth)
            & (uv[:, 0] >= 0) & (uv[:, 0] < intrinsics.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < intrinsics.height)
        )


def project_radar_to_image(points_radar: np.ndarray, T_cam_from_radar: np.ndarray,
                           intrinsics: CameraIntrinsics):
    """Radar points (N, 3) -> pixels. Returns (uv, depth, mask) like the functions above."""
    points_cam = transform_points(T_cam_from_radar, points_radar)
    uv, depth = project_points(points_cam, intrinsics)
    return uv, depth, in_image_mask(uv, depth, intrinsics)


def project_box_to_2d(center, size, yaw: float, T_cam_from_box_frame: np.ndarray,
                      intrinsics: CameraIntrinsics):
    """Project a 3D box to an axis-aligned 2D box [u_min, v_min, u_max, v_max].

    The box is clipped to the image. Returns None if any corner is behind the
    camera (a partly-behind box would give a wrong 2D box) or if the clipped
    box has no area.
    """
    corners = box_corners(center, size, yaw)
    uv, depth = project_points(transform_points(T_cam_from_box_frame, corners), intrinsics)
    if np.any(depth <= 0.1):
        return None
    u_min = np.clip(uv[:, 0].min(), 0, intrinsics.width)
    u_max = np.clip(uv[:, 0].max(), 0, intrinsics.width)
    v_min = np.clip(uv[:, 1].min(), 0, intrinsics.height)
    v_max = np.clip(uv[:, 1].max(), 0, intrinsics.height)
    if u_max - u_min < 1.0 or v_max - v_min < 1.0:
        return None
    return np.array([u_min, v_min, u_max, v_max])
