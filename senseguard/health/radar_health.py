"""Radar health checks.

For now this holds the **physics check with stationary points**.

The idea in plain words
-----------------------
A point that does not move in the world (a wall, a pole) still has a Doppler
velocity, because the radar itself moves. If the radar moves with velocity
(vx, vy) in its own frame, a static point in direction u = p / |p| has

    v_r = -(vx * ux + vy * uy)        (+ c, if the radar has a velocity bias)

So with enough static points we can fit (vx, vy) by least squares. Moving road
users do not follow this curve, so RANSAC throws them out as outliers.

The fitted velocity tells us in which direction the radar *thinks* it moves.
We know in which direction the car really moves (forward, plus a small sideways
part when it turns). The angle between the two is the radar's **mounting yaw**.
If that angle drifts away from the calibrated value, the radar has rotated.
A non-zero constant c means a velocity bias.

Assumptions and limits
----------------------
- The radar does not move vertically (vz = 0). A pitch error is not detected here.
- Static points must be the majority. A traffic jam where most points move
  together could fool RANSAC. That is a real limitation to test later on VoD.
- One frame is noisy. Combine many frames (for example a median) for a stable estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EgoMotionFit:
    """Result of fitting the radar's own motion from Doppler."""

    velocity: np.ndarray    # (2,) radar velocity in the radar frame, (vx, vy), m/s
    bias: float             # constant v_r offset c, m/s (0 if not fitted)
    inliers: np.ndarray     # (N,) bool, points that fit the static-world model
    rms_residual: float     # m/s, over the inliers

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def heading(self) -> float:
        """Direction the radar moves in, inside its own frame (rad). 0 = straight ahead."""
        return float(np.arctan2(self.velocity[1], self.velocity[0]))


def _design_matrix(points: np.ndarray, fit_bias: bool) -> np.ndarray:
    unit = points[:, :2] / np.linalg.norm(points, axis=1, keepdims=True)
    cols = [-unit[:, 0], -unit[:, 1]]
    if fit_bias:
        cols.append(np.ones(len(points)))
    return np.column_stack(cols)


def fit_radar_ego_motion(points, v_r, *, fit_bias: bool = False, inlier_threshold: float = 0.3,
                         n_iterations: int = 200, min_inliers: int = 10,
                         seed: int = 0) -> EgoMotionFit | None:
    """Fit the radar's own velocity from one frame with RANSAC + least squares.

    points: (N, 3) radar points in the radar frame.
    v_r:    (N,) measured radial velocities (raw, NOT ego-compensated).
    fit_bias: also fit a constant v_r offset. Needs a wide field of view to be
        reliable, because near the centre a bias looks like extra forward speed.
    inlier_threshold: max |residual| in m/s for a point to count as static.
    Returns None if fewer than ``min_inliers`` points fit.
    """
    points = np.asarray(points, dtype=float)
    v_r = np.asarray(v_r, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(v_r):
        raise ValueError("points must be (N, 3) and v_r must be (N,)")

    n_params = 3 if fit_bias else 2
    if len(points) < max(min_inliers, n_params):
        return None

    A = _design_matrix(points, fit_bias)
    rng = np.random.default_rng(seed)

    best_inliers = None
    best_count = 0
    for _ in range(n_iterations):
        sample = rng.choice(len(points), size=n_params, replace=False)
        A_s = A[sample]
        if np.linalg.matrix_rank(A_s) < n_params:
            continue
        params = np.linalg.solve(A_s, v_r[sample])
        inliers = np.abs(A @ params - v_r) < inlier_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    # Refine: least squares on the inliers, then re-select inliers once and refit.
    inliers = best_inliers
    for _ in range(2):
        params, *_ = np.linalg.lstsq(A[inliers], v_r[inliers], rcond=None)
        inliers = np.abs(A @ params - v_r) < inlier_threshold
        if inliers.sum() < min_inliers:
            return None

    residual = A[inliers] @ params - v_r[inliers]
    bias = float(params[2]) if fit_bias else 0.0
    return EgoMotionFit(velocity=params[:2].copy(), bias=bias, inliers=inliers,
                        rms_residual=float(np.sqrt(np.mean(residual**2))))


def wrap_angle(angle):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def estimate_mounting_yaw(fit: EgoMotionFit, ego_speed: float, yaw_rate: float = 0.0,
                          mount_x: float = 0.0, mount_y: float = 0.0) -> float:
    """Radar mounting yaw in the vehicle frame (rad), from an ego-motion fit.

    The radar really moves in direction alpha in the vehicle frame
    (alpha = 0 when driving straight; non-zero when turning, because the radar
    sits ahead of the rear axle). The fit says it moves in direction phi in its
    own frame. A radar rotated by yaw delta sees direction alpha - delta, so
    delta = alpha - phi.

    ego_speed and yaw_rate come from wheel odometry / IMU.
    mount_x, mount_y: radar position in the vehicle frame (from calibration).
    """
    if ego_speed <= 0.5:
        raise ValueError("ego_speed too low: the yaw estimate is meaningless near standstill")
    alpha = np.arctan2(yaw_rate * mount_x, ego_speed - yaw_rate * mount_y)
    return float(wrap_angle(alpha - fit.heading))


def median_yaw(yaw_estimates) -> float:
    """Robust combination of per-frame yaw estimates (rad).

    Takes the median of the offsets from the first estimate, so angles near
    +-pi do not break it. One frame is noisy; on synthetic data a median over
    about 10 frames brings the error well under 0.2 degrees
    (see scripts/yaw_fit_report.py).
    """
    yaws = np.asarray(list(yaw_estimates), dtype=float)
    if len(yaws) == 0:
        raise ValueError("need at least one yaw estimate")
    ref = yaws[0]
    return float(wrap_angle(ref + np.median(wrap_angle(yaws - ref))))
