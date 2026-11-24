import numpy as np
import pytest

from senseguard.geometry.projection import (
    CameraIntrinsics,
    in_image_mask,
    project_box_to_2d,
    project_points,
    project_radar_to_image,
)
from senseguard.geometry.transforms import R_CAM_FROM_VEHICLE_AXES, make_transform

K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=600.0, width=1920, height=1200)


def test_intrinsics_matrix_round_trip():
    np.testing.assert_allclose(K.matrix, [[1000, 0, 960], [0, 1000, 600], [0, 0, 1]])
    assert CameraIntrinsics.from_matrix(K.matrix, 1920, 1200) == K


def test_project_hand_computed_point():
    # u = 1000 * 1 / 10 + 960 = 1060 ; v = 1000 * 0.5 / 10 + 600 = 650
    uv, depth = project_points(np.array([[1.0, 0.5, 10.0]]), K)
    np.testing.assert_allclose(uv, [[1060.0, 650.0]])
    np.testing.assert_allclose(depth, [10.0])


def test_point_on_optical_axis_hits_principal_point():
    uv, _ = project_points(np.array([[0.0, 0.0, 7.0]]), K)
    np.testing.assert_allclose(uv, [[K.cx, K.cy]])


def test_points_behind_camera_are_nan_and_masked_out():
    uv, depth = project_points(np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]), K)
    assert np.all(np.isnan(uv[0]))
    np.testing.assert_array_equal(in_image_mask(uv, depth, K), [False, True])


def test_points_outside_image_are_masked_out():
    # u = 1000 * 20 / 10 + 960 = 2960 > width
    uv, depth = project_points(np.array([[20.0, 0.0, 10.0]]), K)
    assert not in_image_mask(uv, depth, K)[0]


def test_radar_point_straight_ahead_projects_to_principal_point():
    # Radar and camera at the same spot; radar axes x fwd, y left, z up.
    T_cam_from_radar = make_transform(R_CAM_FROM_VEHICLE_AXES, [0, 0, 0])
    uv, depth, mask = project_radar_to_image(np.array([[15.0, 0.0, 0.0]]), T_cam_from_radar, K)
    np.testing.assert_allclose(uv, [[K.cx, K.cy]])
    np.testing.assert_allclose(depth, [15.0])
    assert mask[0]


def test_radar_point_to_the_left_lands_left_of_centre():
    T_cam_from_radar = make_transform(R_CAM_FROM_VEHICLE_AXES, [0, 0, 0])
    uv, _, _ = project_radar_to_image(np.array([[10.0, 1.0, 0.0]]), T_cam_from_radar, K)
    np.testing.assert_allclose(uv, [[960.0 - 100.0, 600.0]])


def test_project_box_to_2d_hand_computed():
    # Box in camera frame, centre 10 m ahead, 2 m wide, 2 m tall, 0.001 m deep:
    # corners at x = +-1, y = +-1  ->  u = 960 +- 100, v = 600 +- 100 (depth ~10).
    T = np.eye(4)
    box = project_box_to_2d(center=[0, 0, 10], size=[2, 2, 0.001], yaw=0.0,
                            T_cam_from_box_frame=T, intrinsics=K)
    # In camera frame, box_corners' "length" is along x, "width" along y, "height" along z.
    np.testing.assert_allclose(box, [860, 500, 1060, 700], atol=0.1)


def test_project_box_behind_camera_is_none():
    assert project_box_to_2d([0, 0, -10], [2, 2, 2], 0.0, np.eye(4), K) is None


def test_project_points_rejects_bad_shape():
    with pytest.raises(ValueError):
        project_points(np.zeros((3,)), K)
