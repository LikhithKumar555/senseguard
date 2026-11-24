from dataclasses import replace

import numpy as np

from senseguard.data.synthetic import (
    RADAR_COLUMNS,
    SceneConfig,
    generate_scene,
    sensor_velocity,
)
from senseguard.geometry.projection import project_radar_to_image
from senseguard.geometry.transforms import invert_transform, transform_points


def test_same_seed_gives_identical_scene():
    a, b = generate_scene(seed=42), generate_scene(seed=42)
    np.testing.assert_array_equal(a.radar.to_array(), b.radar.to_array())
    for oa, ob in zip(a.objects, b.objects):
        np.testing.assert_array_equal(oa.center, ob.center)


def test_different_seed_gives_different_scene():
    a, b = generate_scene(seed=1), generate_scene(seed=2)
    assert a.radar.to_array().shape != b.radar.to_array().shape or not np.allclose(
        a.radar.to_array(), b.radar.to_array())


def test_array_layout():
    scene = generate_scene(seed=0)
    arr = scene.radar.to_array()
    assert arr.shape == (len(scene.radar), len(RADAR_COLUMNS))
    assert len(scene.objects) == SceneConfig().n_objects
    assert scene.radar.is_static.sum() >= SceneConfig().n_static * 0.9


def test_points_are_inside_field_of_view():
    cfg = SceneConfig(position_noise_std=0.0)
    pts = generate_scene(cfg, seed=3).radar.points
    r = np.linalg.norm(pts, axis=1)
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    assert r.min() >= cfg.min_range and r.max() <= cfg.max_range
    assert np.abs(az).max() <= cfg.fov_azimuth_deg


def test_static_points_follow_doppler_law_without_noise():
    # Straight driving, no yaw offset: v_r = -v * x / r for static points.
    cfg = SceneConfig(vr_noise_std=0.0, position_noise_std=0.0, ego_speed=8.0)
    radar = generate_scene(cfg, seed=5).radar
    s = radar.is_static
    expected = -8.0 * radar.points[s, 0] / np.linalg.norm(radar.points[s], axis=1)
    np.testing.assert_allclose(radar.v_r[s], expected, atol=1e-9)


def test_compensated_velocity_is_zero_for_static_points_when_calibrated():
    cfg = SceneConfig(vr_noise_std=0.0, position_noise_std=0.0, yaw_rate=0.2)
    radar = generate_scene(cfg, seed=6).radar
    np.testing.assert_allclose(radar.v_r_compensated[radar.is_static], 0.0, atol=1e-9)


def test_yaw_offset_leaves_residual_in_compensated_velocity():
    # The car compensates with the wrong (nominal) mount, so static points no longer read 0.
    cfg = SceneConfig(vr_noise_std=0.0, position_noise_std=0.0, radar_yaw_offset_deg=3.0)
    radar = generate_scene(cfg, seed=6).radar
    assert np.abs(radar.v_r_compensated[radar.is_static]).max() > 0.1


def test_velocity_bias_shifts_every_point():
    base = SceneConfig(vr_noise_std=0.0, position_noise_std=0.0)
    a = generate_scene(base, seed=7).radar
    b = generate_scene(replace(base, velocity_bias=0.5), seed=7).radar
    np.testing.assert_allclose(b.v_r - a.v_r, 0.5, atol=1e-12)


def test_sensor_velocity_includes_turning():
    np.testing.assert_allclose(sensor_velocity(10.0, 0.0, 3.6, 0.0), [10, 0, 0])
    # Turning left at 0.5 rad/s, a sensor 2 m ahead also moves 1 m/s to the left.
    np.testing.assert_allclose(sensor_velocity(10.0, 0.5, 2.0, 0.0), [10, 1, 0])


def test_true_calibration_projects_object_points_onto_their_objects():
    scene = generate_scene(SceneConfig(position_noise_std=0.0), seed=11)
    radar = scene.radar
    pts_vehicle = transform_points(scene.T_vehicle_from_radar_true, radar.points)
    for i, obj in enumerate(scene.objects):
        mine = radar.object_id == i
        if not mine.any():
            continue
        # Back in the vehicle frame, every object point lies inside its box (with a small margin).
        local = pts_vehicle[mine] - obj.center
        c, s = np.cos(-obj.yaw), np.sin(-obj.yaw)
        local_xy = local[:, :2] @ np.array([[c, -s], [s, c]]).T
        assert np.all(np.abs(local_xy) <= obj.size[:2] / 2 + 1e-6)
    # And the camera chain is consistent.
    np.testing.assert_allclose(scene.T_cam_from_radar_true,
                               invert_transform(scene.T_vehicle_from_cam) @ scene.T_vehicle_from_radar_true)
    _, _, mask = project_radar_to_image(radar.points, scene.T_cam_from_radar_true, scene.intrinsics)
    assert mask.any()


def test_yaw_offset_moves_projected_points():
    cfg = SceneConfig(position_noise_std=0.0, radar_yaw_offset_deg=2.0)
    scene = generate_scene(cfg, seed=4)
    uv_true, _, m1 = project_radar_to_image(scene.radar.points, scene.T_cam_from_radar_true, scene.intrinsics)
    uv_nom, _, m2 = project_radar_to_image(scene.radar.points, scene.T_cam_from_radar_nominal, scene.intrinsics)
    both = m1 & m2
    shift = np.abs(uv_true[both, 0] - uv_nom[both, 0])
    # 2 degrees at fx = 1200 is roughly 1200 * tan(2 deg) = 42 px of horizontal shift.
    assert np.median(shift) > 30
