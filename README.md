# SenseGuard

[![CI](https://github.com/LikhithKumar555/senseguard/actions/workflows/ci.yml/badge.svg)](https://github.com/LikhithKumar555/senseguard/actions/workflows/ci.yml)

**A camera + radar perception system that notices when its own sensors are failing, and keeps the car safe when they do.**

> **Status: early development.** This is the repo skeleton. There are no results yet.
> Every number that appears here later will come from a script in `scripts/` and a CSV in `experiments/`.

## The problem

Cameras and radars fail in everyday driving. Snow or mud blocks the radar. Dirt or glare blinds the camera. A bump shifts a sensor so the calibration is wrong. One sensor's data arrives late.

A driver-assistance system must **notice** this, **tell** the rest of the system which sensor is unreliable, and **switch to a safer mode**. These ideas are central to ISO 26262 (functional safety) and ISO 21448 (SOTIF). This project borrows those ideas; it does **not** claim compliance with either standard.

## What SenseGuard will do

- **Detect faults** in a camera and a 4D radar: blockage, a dirty or blinded lens, calibration drift, and time delay between sensors.
- **Explain them**: a health score from 0 to 1 per sensor, plus the fault type.
- **Use them**: a fusion step that trusts each sensor according to its health, and a safety state machine (NOMINAL / DEGRADED / MINIMAL RISK).
- **Benchmark them**: a seeded, reproducible fault library on top of the View-of-Delft dataset.

```
camera frames ──► camera health checks ───────┐
      │                                       │
      └──► YOLO 2D detection ─┐               ▼
                              ├─► fusion ◄── health monitor ──► safety mode
radar points ──► clustering ──┘     ▲           ▲
      │                             │           │
      └──► radar health checks ─────┼───────────┘
                                    │
           cross-sensor checks (radar ↔ camera reprojection, timing)
```

## Data

Main dataset: [View-of-Delft](https://viewofdelft-dataset.tudelft.nl/) (TU Delft). It is free for non-commercial research only, so **no dataset files are stored in this repo**. You need to request access yourself. Until access arrives, all code is developed and tested on a small synthetic generator.

## Repo layout

```
senseguard/
├── senseguard/
│   ├── data/          # dataset loaders and synthetic generator
│   ├── geometry/      # projection and transforms
│   ├── faults/        # fault injection
│   ├── perception/    # YOLO, radar clustering, fusion
│   ├── health/        # health checks and monitor
│   ├── safety/        # safety state machine
│   └── eval/          # metrics and benchmark
├── ros2_ws/src/senseguard_ros/   # ROS 2 C++ health-monitor node (later)
├── scripts/           # command-line entry points
├── tests/             # pytest, one test file per module
├── experiments/       # CSV results (no raw data)
└── media/             # figures and demo video
```

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/LikhithKumar555/senseguard.git
cd senseguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[vision,radar]" # optional: OpenCV, YOLO, scikit-learn
pytest
```

## Roadmap

- [x] Repo skeleton with CI
- [x] Synthetic data generator, projection code, radar ego-motion and yaw-offset fit
- [ ] View-of-Delft loader and radar-on-image visualiser
- [ ] Baseline perception: YOLO, radar clustering, late fusion
- [ ] Fault library and benchmark configs
- [ ] Health monitor (rules, then a learned camera classifier)
- [ ] Health-aware fusion and safety modes
- [ ] ROS 2 C++ node, bag replay demo, video

## Results

No results on real data yet.

**Synthetic sanity check (not a real-radar result).** The radar ego-motion fit
recovers a mounting yaw offset. With 10 m/s ego speed and 0.1 m/s Doppler noise,
a median over 10 frames stays within 0.2° in every one of 100 trials
(worst 0.13°). It does **not** meet 0.2° at 3 m/s ego speed, at 0.25 m/s
Doppler noise, or when a velocity bias is present but not modelled.
Full table: [`experiments/synthetic_yaw_fit.csv`](experiments/synthetic_yaw_fit.csv),
made by `python scripts/yaw_fit_report.py --csv experiments/synthetic_yaw_fit.csv`.

## Limitations

- The yaw fit assumes the radar does not move vertically, so it cannot see a pitch error.
  Slow road users (for example pedestrians) can be mistaken for static points and
  pull a single-frame estimate by about 0.1°.
- All faults in the benchmark are **simulated**. Real snow on a radar or real mud on a lens can look different from the fault models used here. Every result will be labelled as coming from simulated faults.

## License

Code: MIT (see `LICENSE`). Datasets keep their own licenses and are not included.
