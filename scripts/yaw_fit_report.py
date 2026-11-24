"""Measure how well the radar ego-motion fit recovers a mounting yaw offset on SYNTHETIC data.

Usage:
    python scripts/yaw_fit_report.py                 # prints a table
    python scripts/yaw_fit_report.py --csv experiments/synthetic_yaw_fit.csv

Every row is a scenario. For each one we run many independent trials and report
the absolute yaw error, both for a single frame and for a median over N frames.
These are synthetic numbers only. They say the math and code are right; they do
NOT say how well this works on a real radar.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np

from senseguard.data.synthetic import SceneConfig, generate_scene
from senseguard.health.radar_health import estimate_mounting_yaw, fit_radar_ego_motion, median_yaw

SCENARIOS = {
    "straight, 2 deg offset": dict(radar_yaw_offset_deg=2.0),
    "straight, -3 deg offset": dict(radar_yaw_offset_deg=-3.0),
    "turning 0.3 rad/s, 2 deg": dict(radar_yaw_offset_deg=2.0, yaw_rate=0.3),
    "slow 3 m/s, 2 deg": dict(radar_yaw_offset_deg=2.0, ego_speed=3.0),
    "bias 0.5 m/s, 2 deg (bias fitted)": dict(radar_yaw_offset_deg=2.0, velocity_bias=0.5),
    "bias 0.5 m/s, 2 deg (bias NOT fitted)": dict(radar_yaw_offset_deg=2.0, velocity_bias=0.5),
    "noise 0.25 m/s, 2 deg": dict(radar_yaw_offset_deg=2.0, vr_noise_std=0.25),
}


def frame_yaw(cfg: SceneConfig, seed: int, fit_bias: bool) -> float:
    scene = generate_scene(cfg, seed)
    fit = fit_radar_ego_motion(scene.radar.points, scene.radar.v_r, fit_bias=fit_bias, seed=seed)
    if fit is None:
        return np.nan
    m = cfg.radar_mount
    return estimate_mounting_yaw(fit, cfg.ego_speed, cfg.yaw_rate, m.x, m.y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--frames", type=int, default=10, help="frames per median")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for name, overrides in SCENARIOS.items():
        cfg = replace(SceneConfig(), **overrides)
        fit_bias = "bias fitted" in name
        truth = cfg.radar_yaw_offset_deg
        single, multi = [], []
        for t in range(args.trials):
            yaws = [frame_yaw(cfg, t * 1000 + k, fit_bias) for k in range(args.frames)]
            yaws = [y for y in yaws if np.isfinite(y)]
            single.append(np.degrees(yaws[0]) - truth)
            multi.append(np.degrees(median_yaw(yaws)) - truth)
        single, multi = np.abs(single), np.abs(multi)
        rows.append({
            "scenario": name,
            "trials": args.trials,
            "single_frame_p95_deg": round(float(np.percentile(single, 95)), 3),
            "single_frame_max_deg": round(float(single.max()), 3),
            f"median{args.frames}_p95_deg": round(float(np.percentile(multi, 95)), 3),
            f"median{args.frames}_max_deg": round(float(multi.max()), 3),
            "within_0p2_deg": bool(multi.max() < 0.2),
        })

    keys = list(rows[0])
    widths = [max(len(k), *(len(str(r[k])) for r in rows)) for k in keys]
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w) for k, w in zip(keys, widths)))
    print("\nSynthetic data only. Not a result on a real radar.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
