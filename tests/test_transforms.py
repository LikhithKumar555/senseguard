import numpy as np
import pytest

from senseguard.geometry.transforms import (
    R_CAM_FROM_VEHICLE_AXES,
    box_corners,
    invert_transform,
    make_transform,
    rot_x,
    rot_y,
    rot_z,
    rotate_vectors,
    rpy_to_matrix,
    transform_points,
)


def test_rot_z_90_turns_x_into_y():
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_rot_x_90_turns_y_into_z():
    np.testing.assert_allclose(rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_rot_y_90_turns_z_into_x():
    np.testing.assert_allclose(rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-12)


@pytest.mark.parametrize("rpy", [(0.1, -0.2, 0.3), (1.0, 0.5, -2.0), (0, 0, 0)])
def test_rotations_are_proper(rpy):
    R = rpy_to_matrix(*rpy)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rpy_order_is_yaw_after_pitch_after_roll():
    r, p, y = 0.1, 0.2, 0.3
    np.testing.assert_allclose(rpy_to_matrix(r, p, y), rot_z(y) @ rot_y(p) @ rot_x(r))


def test_transform_points_hand_computed():
    # Rotate 90 deg about z, then move by (1, 2, 3): (1, 0, 0) -> (0, 1, 0) -> (1, 3, 3)
    T = make_transform(rot_z(np.pi / 2), [1, 2, 3])
    np.testing.assert_allclose(transform_points(T, np.array([[1.0, 0, 0]])), [[1, 3, 3]], atol=1e-12)


def test_invert_transform_round_trip():
    T = make_transform(rpy_to_matrix(0.3, -0.1, 1.2), [0.5, -2.0, 1.0])
    np.testing.assert_allclose(invert_transform(T) @ T, np.eye(4), atol=1e-12)
    pts = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(transform_points(invert_transform(T), transform_points(T, pts)), pts,
                               atol=1e-12)


def test_rotate_vectors_ignores_translation():
    T = make_transform(np.eye(3), [10, 10, 10])
    np.testing.assert_allclose(rotate_vectors(T, np.array([[1.0, 2, 3]])), [[1, 2, 3]])


def test_transform_points_rejects_bad_shape():
    with pytest.raises(ValueError):
        transform_points(np.eye(4), np.zeros(3))


def test_vehicle_to_camera_axes():
    # Forward in the vehicle is +z in the camera; left is -x; up is -y.
    np.testing.assert_allclose(R_CAM_FROM_VEHICLE_AXES @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(R_CAM_FROM_VEHICLE_AXES @ [0, 1, 0], [-1, 0, 0])
    np.testing.assert_allclose(R_CAM_FROM_VEHICLE_AXES @ [0, 0, 1], [0, -1, 0])


def test_box_corners_axis_aligned():
    corners = box_corners(center=[10, 0, 1], size=[4, 2, 2], yaw=0.0)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [8, -1, 0])
    np.testing.assert_allclose(corners.max(axis=0), [12, 1, 2])


def test_box_corners_rotated_90():
    # Turned 90 deg, the 4 m length now lies along y.
    corners = box_corners(center=[0, 0, 0], size=[4, 2, 2], yaw=np.pi / 2)
    np.testing.assert_allclose(corners.max(axis=0), [1, 2, 1], atol=1e-12)
