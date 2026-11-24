from dataclasses import replace

import numpy as np
import pytest

from senseguard.data.synthetic import SceneConfig, generate_scene
from senseguard.health.radar_health import (
    estimate_mounting_yaw,
    fit_radar_ego_motion,
    median_yaw,
    wrap_angle,
)

BASE = SceneConfig()


def _yaw_from_scene(scene, fit_bias=False):
    cfg = scene.config
    fit = fit_radar_ego_motion(scene.radar.points, scene.radar.v_r, fit_bias=fit_bias, seed=scene.seed)
    assert fit is not None
    m = cfg.radar_mount
    return fit, estimate_mounting_yaw(fit, cfg.ego_speed, cfg.yaw_rate, m.x, m.y)


def _sequence_yaw_error_deg(cfg, n_frames, trial, fit_bias=False):
    yaws = [_yaw_from_scene(generate_scene(cfg, seed=trial * 1000 + k), fit_bias)[1]
            for k in range(n_frames)]
    return np.degrees(median_yaw(yaws)) - cfg.radar_yaw_offset_deg


# ---------- exact cases, no noise, static world only ----------
# With road users present the fit is NOT exact: slow movers (pedestrians at
# 0.2-1.3 m/s) can fall inside the inlier band and pull the fit. See
# test_slow_movers_contaminate_single_frame below.

EXACT = replace(BASE, vr_noise_std=0.0, position_noise_std=0.0, n_objects=0)

@pytest.mark.parametrize("offset_deg", [-3.0, -1.0, 0.0, 0.5, 2.0, 3.0])
def test_recovers_yaw_offset_exactly_without_noise(offset_deg):
    cfg = replace(EXACT, radar_yaw_offset_deg=offset_deg)
    fit, yaw = _yaw_from_scene(generate_scene(cfg, seed=0))
    assert np.degrees(yaw) == pytest.approx(offset_deg, abs=1e-6)
    assert fit.speed == pytest.approx(cfg.ego_speed, abs=1e-6)


def test_recovers_yaw_while_turning_without_noise():
    cfg = replace(EXACT, radar_yaw_offset_deg=2.0, yaw_rate=0.4)
    _, yaw = _yaw_from_scene(generate_scene(cfg, seed=1))
    assert np.degrees(yaw) == pytest.approx(2.0, abs=1e-6)


def test_recovers_velocity_bias_without_noise():
    cfg = replace(EXACT, radar_yaw_offset_deg=1.0, velocity_bias=0.5)
    fit, yaw = _yaw_from_scene(generate_scene(cfg, seed=2), fit_bias=True)
    assert fit.bias == pytest.approx(0.5, abs=1e-6)
    assert np.degrees(yaw) == pytest.approx(1.0, abs=1e-6)


def test_moving_objects_are_rejected_as_outliers():
    cfg = replace(BASE, vr_noise_std=0.0, position_noise_std=0.0, n_objects=8)
    scene = generate_scene(cfg, seed=3)
    fit, _ = _yaw_from_scene(scene)
    r = scene.radar
    # All static points are inliers. Moving points are outliers unless their
    # Doppler happens to look static (e.g. crossing traffic), so only check most.
    assert fit.inliers[r.is_static].all()
    moving = ~r.is_static
    if moving.sum() > 0:
        assert (~fit.inliers[moving]).mean() > 0.5


def test_slow_movers_contaminate_single_frame():
    # Documents a known limitation: same scene, noise-free, but with road users.
    # Slow pedestrians become inliers, so the single-frame yaw is off by ~0.1 deg.
    cfg = replace(EXACT, n_objects=4, radar_yaw_offset_deg=2.0)
    scene = generate_scene(cfg, seed=0)
    fit, yaw = _yaw_from_scene(scene)
    assert (fit.inliers & ~scene.radar.is_static).sum() > 0
    assert 0.01 < abs(np.degrees(yaw) - 2.0) < 0.2


# ---------- the milestone criterion, with noise ----------

@pytest.mark.parametrize("offset_deg", [-3.0, 0.5, 2.0])
def test_yaw_within_0p2_deg_using_10_frame_median(offset_deg):
    """Milestone 'done' check: yaw offset recovered within 0.2 deg on synthetic data.

    Per frame the error is larger (95th percentile about 0.25 deg at 10 m/s with
    0.1 m/s Doppler noise), so the check uses a median over 10 frames (~0.8 s at 13 Hz).
    """
    cfg = replace(BASE, radar_yaw_offset_deg=offset_deg)
    errors = [_sequence_yaw_error_deg(cfg, n_frames=10, trial=t) for t in range(20)]
    assert np.max(np.abs(errors)) < 0.2


def test_yaw_within_0p2_deg_while_turning_with_bias():
    cfg = replace(BASE, radar_yaw_offset_deg=2.0, yaw_rate=0.3, velocity_bias=0.5)
    errors = [_sequence_yaw_error_deg(cfg, 10, t, fit_bias=True) for t in range(20)]
    assert np.max(np.abs(errors)) < 0.2


def test_single_frame_estimate_is_unbiased():
    cfg = replace(BASE, radar_yaw_offset_deg=2.0)
    errors = [np.degrees(_yaw_from_scene(generate_scene(cfg, seed=s))[1]) - 2.0 for s in range(100)]
    assert abs(np.mean(errors)) < 0.05


# ---------- behaviour and edge cases ----------

def test_fit_is_deterministic_with_seed():
    scene = generate_scene(seed=9)
    a = fit_radar_ego_motion(scene.radar.points, scene.radar.v_r, seed=123)
    b = fit_radar_ego_motion(scene.radar.points, scene.radar.v_r, seed=123)
    np.testing.assert_array_equal(a.velocity, b.velocity)
    np.testing.assert_array_equal(a.inliers, b.inliers)


def test_too_few_points_returns_none():
    pts = np.array([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, -1.0, 0.0]])
    assert fit_radar_ego_motion(pts, np.array([-10.0, -9.9, -9.9])) is None


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        fit_radar_ego_motion(np.zeros((5, 2)), np.zeros(5))


def test_standstill_raises():
    scene = generate_scene(seed=0)
    fit = fit_radar_ego_motion(scene.radar.points, scene.radar.v_r)
    with pytest.raises(ValueError):
        estimate_mounting_yaw(fit, ego_speed=0.1)


def test_median_yaw_handles_wraparound():
    near_pi = [np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02]
    assert abs(wrap_angle(median_yaw(near_pi) - (np.pi - 0.01))) < 1e-9


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle([0.0, 3 * np.pi / 2, -3 * np.pi / 2]),
                               [0.0, -np.pi / 2, np.pi / 2])
