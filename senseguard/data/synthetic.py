"""Small synthetic scene generator for developing and testing code before real data arrives.

One call to ``generate_scene`` makes one frame:
- the ego vehicle drives with a given speed and yaw rate,
- a 3+1D radar sees static clutter (walls, poles, parked things) and a few moving
  road users (car / pedestrian / cyclist boxes),
- every radar point gets x, y, z, RCS, a radial (Doppler) velocity v_r, and an
  ego-compensated v_r,
- a pinhole camera looks forward, so boxes and radar points can be projected.

The radar can be mounted with a **yaw offset**: the *true* mount differs from the
*nominal* mount that the car believes in. That is exactly the calibration fault the
ego-motion check should find.

This is a toy world. The numbers (intrinsics, noise, RCS) are made up to be roughly
plausible. They are NOT View-of-Delft values.

Frames: vehicle frame origin on the ground under the rear axle, x forward, y left,
z up. Radar frame: same axis directions, rotated by the radar mount. Camera frame:
x right, y down, z forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from senseguard.geometry.projection import CameraIntrinsics
from senseguard.geometry.transforms import (
    R_CAM_FROM_VEHICLE_AXES,
    invert_transform,
    make_transform,
    rotate_vectors,
    rpy_to_matrix,
    transform_points,
)

# Class name -> (length, width, height) in metres, speed range in m/s, RCS mean and std in dBsm.
OBJECT_CLASSES = {
    "car": {"size": (4.2, 1.8, 1.5), "speed": (0.0, 15.0), "rcs": (10.0, 4.0)},
    "pedestrian": {"size": (0.6, 0.6, 1.75), "speed": (0.0, 2.0), "rcs": (-5.0, 3.0)},
    "cyclist": {"size": (1.8, 0.6, 1.7), "speed": (2.0, 7.0), "rcs": (0.0, 3.0)},
}
CLUTTER_RCS = (0.0, 5.0)

# Column order of RadarFrame.to_array(). This is what I expect VoD to use;
# check it against VoD's SENSORS_AND_DATA document when access arrives.
RADAR_COLUMNS = ("x", "y", "z", "rcs", "v_r", "v_r_compensated", "time")


@dataclass(frozen=True)
class SensorMount:
    """Where a sensor sits on the vehicle: position in metres, angles in radians."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def T_vehicle_from_sensor(self) -> np.ndarray:
        """For radar-style sensors whose axes are x fwd, y left, z up."""
        return make_transform(rpy_to_matrix(self.roll, self.pitch, self.yaw),
                              (self.x, self.y, self.z))

    def with_yaw_offset(self, offset_rad: float) -> "SensorMount":
        return SensorMount(self.x, self.y, self.z, self.roll, self.pitch, self.yaw + offset_rad)


@dataclass(frozen=True)
class SceneConfig:
    """Everything that defines a synthetic frame, apart from the random seed."""

    # Ego motion
    ego_speed: float = 10.0          # m/s, forward
    yaw_rate: float = 0.0            # rad/s, positive = turning left

    # Scene content
    n_static: int = 150              # static clutter points
    n_objects: int = 4               # road users
    points_per_object: tuple[int, int] = (5, 20)

    # Radar
    radar_mount: SensorMount = SensorMount(x=3.6, y=0.0, z=0.5)   # nominal mount
    radar_yaw_offset_deg: float = 0.0   # true mount = nominal + this offset
    velocity_bias: float = 0.0          # m/s, added to every v_r (a radar fault)
    vr_noise_std: float = 0.1           # m/s
    position_noise_std: float = 0.05    # m
    max_range: float = 50.0             # m
    min_range: float = 2.0              # m
    fov_azimuth_deg: float = 60.0       # half-angle
    fov_elevation_deg: float = 15.0     # half-angle

    # Camera
    camera_mount: SensorMount = SensorMount(x=1.8, y=0.0, z=1.4)
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            fx=1200.0, fy=1200.0, cx=968.0, cy=608.0, width=1936, height=1216
        )
    )


@dataclass
class SceneObject:
    """A road user. All values in the vehicle frame."""

    cls: str
    center: np.ndarray      # (3,)
    size: np.ndarray        # (3,) length, width, height
    yaw: float              # heading, rad
    velocity: np.ndarray    # (3,) world velocity expressed in the vehicle frame, m/s


@dataclass
class RadarFrame:
    """Radar points of one frame. Positions are in the TRUE radar frame (what the sensor measures)."""

    points: np.ndarray            # (N, 3) x, y, z
    rcs: np.ndarray               # (N,) dBsm
    v_r: np.ndarray               # (N,) radial velocity, m/s, positive = moving away
    v_r_compensated: np.ndarray   # (N,) v_r with ego motion removed (using the NOMINAL mount)
    is_static: np.ndarray         # (N,) bool, ground truth: point does not move in the world
    object_id: np.ndarray         # (N,) int, index into Scene.objects, -1 for clutter

    def __len__(self) -> int:
        return len(self.points)

    def to_array(self) -> np.ndarray:
        """(N, 7) array in RADAR_COLUMNS order. Time is 0 for a single frame."""
        return np.column_stack([self.points, self.rcs, self.v_r, self.v_r_compensated,
                                np.zeros(len(self))])


@dataclass
class Scene:
    config: SceneConfig
    seed: int
    radar: RadarFrame
    objects: list[SceneObject]
    T_vehicle_from_radar_true: np.ndarray
    T_vehicle_from_radar_nominal: np.ndarray
    T_vehicle_from_cam: np.ndarray

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.config.intrinsics

    @property
    def T_cam_from_vehicle(self) -> np.ndarray:
        return invert_transform(self.T_vehicle_from_cam)

    @property
    def T_cam_from_radar_true(self) -> np.ndarray:
        """The real geometry. Projecting with this puts radar points on their objects."""
        return self.T_cam_from_vehicle @ self.T_vehicle_from_radar_true

    @property
    def T_cam_from_radar_nominal(self) -> np.ndarray:
        """What the car believes. Differs from the true one when there is a yaw offset."""
        return self.T_cam_from_vehicle @ self.T_vehicle_from_radar_nominal

    @property
    def sensor_velocity_vehicle(self) -> np.ndarray:
        """Velocity of the radar itself (w.r.t. the ground), in the vehicle frame."""
        return sensor_velocity(self.config.ego_speed, self.config.yaw_rate,
                               self.config.radar_mount.x, self.config.radar_mount.y)


def sensor_velocity(ego_speed: float, yaw_rate: float, mount_x: float, mount_y: float) -> np.ndarray:
    """Velocity of a point fixed on the vehicle at (mount_x, mount_y), in the vehicle frame.

    v = v_ego + omega x r, with v_ego = (ego_speed, 0, 0) and omega = (0, 0, yaw_rate).
    This assumes no side slip, which is fine for normal urban driving.
    """
    return np.array([ego_speed - yaw_rate * mount_y, yaw_rate * mount_x, 0.0])


def _in_fov(points_radar: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    r = np.linalg.norm(points_radar, axis=1)
    az = np.degrees(np.arctan2(points_radar[:, 1], points_radar[:, 0]))
    el = np.degrees(np.arcsin(np.clip(points_radar[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    return ((r >= cfg.min_range) & (r <= cfg.max_range)
            & (np.abs(az) <= cfg.fov_azimuth_deg) & (np.abs(el) <= cfg.fov_elevation_deg))


def _sample_clutter(rng, cfg: SceneConfig, T_vehicle_from_radar: np.ndarray) -> np.ndarray:
    """Static points spread over the radar's field of view, above the ground. Vehicle frame."""
    out = np.empty((0, 3))
    for _ in range(50):  # rejection sampling; almost always finishes in 1-2 rounds
        n = 4 * cfg.n_static
        r = cfg.min_range + (cfg.max_range - cfg.min_range) * np.sqrt(rng.random(n))
        az = np.radians(rng.uniform(-cfg.fov_azimuth_deg, cfg.fov_azimuth_deg, n))
        el = np.radians(rng.uniform(-cfg.fov_elevation_deg, cfg.fov_elevation_deg, n))
        pts_radar = np.column_stack([r * np.cos(el) * np.cos(az),
                                     r * np.cos(el) * np.sin(az),
                                     r * np.sin(el)])
        pts_vehicle = transform_points(T_vehicle_from_radar, pts_radar)
        keep = (pts_vehicle[:, 2] > 0.05) & (pts_vehicle[:, 2] < 4.0)
        out = np.vstack([out, pts_vehicle[keep]])
        if len(out) >= cfg.n_static:
            return out[: cfg.n_static]
    raise RuntimeError("Could not sample enough clutter points; check the field of view settings.")


def _sample_objects(rng, cfg: SceneConfig) -> list[SceneObject]:
    names = list(OBJECT_CLASSES)
    objects = []
    for _ in range(cfg.n_objects):
        cls = names[rng.integers(len(names))]
        spec = OBJECT_CLASSES[cls]
        size = np.array(spec["size"])
        x = rng.uniform(6.0, 0.8 * cfg.max_range)
        y = rng.uniform(-0.5, 0.5) * x * np.tan(np.radians(cfg.fov_azimuth_deg))
        heading = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(*spec["speed"])
        velocity = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
        objects.append(SceneObject(cls, np.array([x, y, size[2] / 2]), size, heading, velocity))
    return objects


def _sample_points_in_box(rng, obj: SceneObject, n: int) -> np.ndarray:
    local = (rng.random((n, 3)) - 0.5) * obj.size
    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + obj.center


def generate_scene(config: SceneConfig | None = None, seed: int = 0) -> Scene:
    """Make one synthetic frame. Same config + same seed -> exactly the same scene."""
    cfg = config or SceneConfig()
    rng = np.random.default_rng(seed)

    nominal = cfg.radar_mount
    true = nominal.with_yaw_offset(np.radians(cfg.radar_yaw_offset_deg))
    T_vehicle_from_radar_true = true.T_vehicle_from_sensor
    T_vehicle_from_radar_nominal = nominal.T_vehicle_from_sensor
    T_radar_from_vehicle_true = invert_transform(T_vehicle_from_radar_true)

    cm = cfg.camera_mount
    R_vehicle_from_cam = rpy_to_matrix(cm.roll, cm.pitch, cm.yaw) @ R_CAM_FROM_VEHICLE_AXES.T
    T_vehicle_from_cam = make_transform(R_vehicle_from_cam, (cm.x, cm.y, cm.z))

    # --- Points in the vehicle frame, with their world velocities ---
    clutter = _sample_clutter(rng, cfg, T_vehicle_from_radar_true)
    positions = [clutter]
    velocities = [np.zeros_like(clutter)]
    rcs = [rng.normal(*CLUTTER_RCS, len(clutter))]
    obj_ids = [np.full(len(clutter), -1)]

    objects = _sample_objects(rng, cfg)
    for i, obj in enumerate(objects):
        n = int(rng.integers(cfg.points_per_object[0], cfg.points_per_object[1] + 1))
        pts = _sample_points_in_box(rng, obj, n)
        positions.append(pts)
        velocities.append(np.tile(obj.velocity, (n, 1)))
        rcs.append(rng.normal(*OBJECT_CLASSES[obj.cls]["rcs"], n))
        obj_ids.append(np.full(n, i))

    pos_vehicle = np.vstack(positions)
    vel_vehicle = np.vstack(velocities)
    rcs = np.concatenate(rcs)
    obj_ids = np.concatenate(obj_ids)

    # --- Into the TRUE radar frame, keep what the radar can see ---
    pos_radar = transform_points(T_radar_from_vehicle_true, pos_vehicle)
    keep = _in_fov(pos_radar, cfg)
    pos_radar, vel_vehicle, rcs, obj_ids = pos_radar[keep], vel_vehicle[keep], rcs[keep], obj_ids[keep]

    # --- Doppler: v_r = u . (v_point - v_sensor), everything in the true radar frame ---
    v_sensor_vehicle = sensor_velocity(cfg.ego_speed, cfg.yaw_rate, true.x, true.y)
    v_rel_radar = rotate_vectors(T_radar_from_vehicle_true, vel_vehicle - v_sensor_vehicle)
    unit = pos_radar / np.linalg.norm(pos_radar, axis=1, keepdims=True)
    v_r = np.sum(unit * v_rel_radar, axis=1)
    v_r = v_r + cfg.velocity_bias + rng.normal(0.0, cfg.vr_noise_std, len(v_r))

    # --- Ego compensation as the car would do it: with the NOMINAL mount ---
    v_sensor_nominal_radar = rotate_vectors(invert_transform(T_vehicle_from_radar_nominal),
                                            v_sensor_vehicle[None, :])[0]
    v_r_comp = v_r + unit @ v_sensor_nominal_radar

    # Position noise last: the radar measures noisy positions of the true points.
    pos_radar = pos_radar + rng.normal(0.0, cfg.position_noise_std, pos_radar.shape)

    is_static = np.linalg.norm(vel_vehicle, axis=1) < 1e-9
    radar = RadarFrame(pos_radar, rcs, v_r, v_r_comp, is_static, obj_ids)
    return Scene(cfg, seed, radar, objects, T_vehicle_from_radar_true,
                 T_vehicle_from_radar_nominal, T_vehicle_from_cam)
