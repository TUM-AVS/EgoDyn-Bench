#!/usr/bin/env python3
"""
Analyze dynamics distributions from CARLA Frenetix replay logs.

Reads all *_logs.csv files from the Frenetix log directory, computes the
same dynamics features used in the nuScenes benchmark pipeline, and
compares distributions against the current benchmark thresholds.

Only clips with duration >= MIN_DURATION_S are included (default: 3.0 s),
matching the nuScenes clip length.

Usage:
    python scripts/analyze_carla_distributions.py [--logs-dir PATH] [--min-duration 3.0]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root so we can import the dynamics processor
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.generation.dynamics_features import DynamicsProcessor


# ---------------------------------------------------------------------------
# Benchmark thresholds (from questions_template.yaml)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "braking_emergency":   {"feature": "min_accel",           "op": "<",  "val": -3.924, "label": ("yes", "no")},
    "braking_moderate":    {"feature": "min_accel",           "op": "<",  "val": -1.962, "label": ("yes", "no")},
    "braking_low":         {"feature": "min_accel",           "op": "<",  "val": -0.981, "label": ("yes", "no")},
    "accel_high":          {"feature": "max_accel",           "op": ">",  "val":  3.924, "label": ("yes", "no")},
    "accel_moderate":      {"feature": "max_accel",           "op": ">",  "val":  1.962, "label": ("yes", "no")},
    "accel_low":           {"feature": "max_accel",           "op": ">",  "val":  0.981, "label": ("yes", "no")},
    "driving_smoothness":  {"feature": "max_abs_jerk",        "op": ">",  "val":  7.09,  "label": ("aggressive", "smooth")},
    "emergency_jerk":      {"feature": "max_abs_jerk",        "op": ">",  "val": 20.0,   "label": ("yes", "no")},  # full rule also ORs with braking_emergency
    "high_lateral_accel":  {"feature": "max_lateral_accel",   "op": ">",  "val":  2.0,   "label": ("yes", "no")},
    "mean_speed_low":      {"feature": "mean_speed",          "op": "<",  "val":  5.0,   "label": ("yes", "no")},
    "heading_change_15":   {"feature": "total_heading_change","op": ">",  "val":  0.2618,"label": ("yes", "no")},
    "speed_trend_accel":   {"feature": "mean_accel",          "op": ">",  "val":  0.25,  "label": ("accel", "-")},
    "speed_trend_decel":   {"feature": "mean_accel",          "op": "<",  "val": -0.25,  "label": ("decel", "-")},
    "turn_detected":       {"feature": "max_abs_yaw_rate",    "op": ">",  "val":  0.04,  "label": ("turning", "straight")},
}


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------
def parse_frenetix_log(csv_path: str) -> dict | None:
    """
    Parse a single Frenetix *_logs.csv and return ego trajectory arrays.

    Returns None if the file cannot be parsed or has insufficient data.
    """
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows = list(reader)
    except Exception:
        return None

    if len(rows) < 2:
        return None

    n = len(rows)

    # Ego position at each replanning step
    try:
        xs = np.array([float(r["x_position_vehicle_m"]) for r in rows])
        ys = np.array([float(r["y_position_vehicle_m"]) for r in rows])
    except (KeyError, ValueError):
        return None

    # Current state = first element of the planned trajectory arrays
    try:
        speeds = np.array([float(r["velocities_mps"].split(",")[0]) for r in rows])
        thetas = np.array([float(r["theta_orientations_rad"].split(",")[0]) for r in rows])
    except (KeyError, ValueError, IndexError):
        return None

    # Each CSV row is one simulation step at dt=0.1s (10 Hz).
    # (replanning_frequency=5 means new plans every 5th step, but all steps are logged)
    dt_step = 0.1
    timestamps = np.arange(n) * dt_step

    positions = np.column_stack([xs, ys])

    return {
        "timestamps": timestamps,
        "positions": positions,
        "yaws": thetas,
        "speeds_raw": speeds,
        "n_steps": n,
        "duration_s": (n - 1) * dt_step,
    }


def window_trajectory(traj: dict, window_s: float = 3.0) -> list[dict]:
    """
    Split a parsed trajectory into non-overlapping windows of fixed duration.

    At 10 Hz (dt=0.1s), a 3.0s window = 31 samples (matching nuScenes clips).
    The last window is kept only if it meets the full window length.

    Returns a list of trajectory dicts, each with the same structure as
    parse_frenetix_log output.  Returns an empty list if the trajectory is
    shorter than one window.
    """
    dt_step = traj["timestamps"][1] - traj["timestamps"][0] if len(traj["timestamps"]) > 1 else 0.1
    samples_per_window = int(round(window_s / dt_step)) + 1  # +1 because fencepost

    n = traj["n_steps"]
    windows = []

    start = 0
    while start + samples_per_window <= n:
        end = start + samples_per_window
        w = {
            "timestamps": traj["timestamps"][start:end] - traj["timestamps"][start],
            "positions": traj["positions"][start:end],
            "yaws": traj["yaws"][start:end],
            "speeds_raw": traj["speeds_raw"][start:end],
            "n_steps": samples_per_window,
            "duration_s": window_s,
        }
        windows.append(w)
        start = end - 1  # -1: next window starts at last sample of previous (no gap)

    return windows


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------
def compute_features(traj: dict) -> dict | None:
    """
    Run the benchmark DynamicsProcessor on a parsed trajectory to compute
    the same features used in the nuScenes pipeline.
    """
    processor = DynamicsProcessor(sampling_hz=10.0, smooth_window=5, smooth_polyorder=2)

    try:
        dynamics = processor.process_ego_trajectory(
            timestamps=traj["timestamps"],
            positions=traj["positions"],
            yaws=traj["yaws"],
        )
    except Exception:
        return None

    features = processor.compute_summary_features(
        speed=dynamics["speed"],
        accel=dynamics["accel"],
        yaw_rate=dynamics["yaw_rate"],
        jerk=dynamics["jerk"],
        yaws=dynamics["yaw"],
    )

    features["duration_s"] = traj["duration_s"]

    return features


# ---------------------------------------------------------------------------
# Distribution Analysis
# ---------------------------------------------------------------------------
def print_distribution(name: str, values: np.ndarray, percentiles=(5, 25, 50, 75, 90, 95)):
    """Print distribution summary for a single feature."""
    print(f"  {name:25s}  mean={np.mean(values):9.4f}  std={np.std(values):8.4f}  "
          f"min={np.min(values):9.4f}  max={np.max(values):9.4f}")
    pct_str = "  " + " ".join(f"p{p}={np.percentile(values, p):8.4f}" for p in percentiles)
    print(f"  {'':25s}{pct_str}")


def print_threshold_analysis(all_features: list[dict]):
    """Apply benchmark thresholds and print hit rates."""
    n = len(all_features)

    print(f"\n{'='*80}")
    print(f"BENCHMARK THRESHOLD ANALYSIS (n={n})")
    print(f"{'='*80}")
    print(f"{'Question':30s} {'Threshold':>12s} {'Hit%':>8s} {'Count':>8s}  Label")
    print("-" * 80)

    for qname, spec in THRESHOLDS.items():
        feat = spec["feature"]
        vals = np.array([f[feat] for f in all_features])
        if spec["op"] == ">":
            hits = np.sum(vals > spec["val"])
        else:
            hits = np.sum(vals < spec["val"])
        pct = 100.0 * hits / n
        label = spec["label"][0]
        print(f"  {qname:28s} {spec['op']} {spec['val']:>8.4f}  {pct:7.1f}%  {hits:7d}  → {label}")


def print_behavior_breakdown(all_records: list[dict]):
    """Show per-behavior statistics."""
    behaviors = sorted(set(r["behavior"] for r in all_records))

    print(f"\n{'='*80}")
    print("PER-BEHAVIOR SUMMARY")
    print(f"{'='*80}")
    print(f"{'Behavior':25s} {'Count':>6s} {'MeanSpd':>8s} {'MinAcc':>8s} {'MaxJrk':>8s} {'MaxLatA':>8s} {'MaxYawR':>8s}")
    print("-" * 80)

    for beh in behaviors:
        recs = [r for r in all_records if r["behavior"] == beh]
        feats = [r["features"] for r in recs]
        n = len(feats)
        mean_spd = np.mean([f["mean_speed"] for f in feats])
        min_acc = np.mean([f["min_accel"] for f in feats])
        max_jrk = np.mean([f["max_abs_jerk"] for f in feats])
        max_lat = np.mean([f["max_lateral_accel"] for f in feats])
        max_yaw = np.mean([f["max_abs_yaw_rate"] for f in feats])
        print(f"  {beh:23s} {n:6d} {mean_spd:8.2f} {min_acc:8.2f} {max_jrk:8.2f} {max_lat:8.2f} {max_yaw:8.4f}")


def print_nuscenes_comparison(carla_features: list[dict], nuscenes_index_path: str | None):
    """Side-by-side comparison with nuScenes distributions if available."""
    if nuscenes_index_path is None or not os.path.exists(nuscenes_index_path):
        print("\n  (nuScenes index not found — skipping comparison)")
        return

    # Load nuScenes features
    ns_features = []
    with open(nuscenes_index_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if "features" in rec:
                ns_features.append(rec["features"])

    if not ns_features:
        print("\n  (nuScenes index has no features — skipping comparison)")
        return

    feature_keys = [
        ("mean_speed",          "Mean Speed (m/s)"),
        ("max_speed",           "Max Speed (m/s)"),
        ("min_accel",           "Min Accel (m/s²)"),
        ("max_abs_jerk",        "Max |Jerk| (m/s³)"),
        ("max_lateral_accel",   "Max Lat Accel (m/s²)"),
        ("max_abs_yaw_rate",    "Max |Yaw Rate| (rad/s)"),
        ("total_heading_change","Total Heading (rad)"),
    ]

    print(f"\n{'='*80}")
    print("CARLA vs nuScenes DISTRIBUTION COMPARISON")
    print(f"{'='*80}")
    print(f"  CARLA clips: {len(carla_features):,d}    nuScenes clips: {len(ns_features):,d}")
    print()
    print(f"  {'Feature':28s} {'':4s} {'mean':>9s} {'p25':>9s} {'p50':>9s} {'p75':>9s} {'p95':>9s}")
    print("  " + "-" * 76)

    for key, label in feature_keys:
        c_vals = np.array([f[key] for f in carla_features if key in f])
        n_vals = np.array([f[key] for f in ns_features if key in f])

        if len(c_vals) == 0 or len(n_vals) == 0:
            continue

        print(f"  {label:28s} {'CAR':4s} {np.mean(c_vals):9.4f} "
              f"{np.percentile(c_vals,25):9.4f} {np.percentile(c_vals,50):9.4f} "
              f"{np.percentile(c_vals,75):9.4f} {np.percentile(c_vals,95):9.4f}")
        print(f"  {'':28s} {'nuS':4s} {np.mean(n_vals):9.4f} "
              f"{np.percentile(n_vals,25):9.4f} {np.percentile(n_vals,50):9.4f} "
              f"{np.percentile(n_vals,75):9.4f} {np.percentile(n_vals,95):9.4f}")
        print()

    # Threshold hit-rate comparison
    print(f"  {'Threshold Hit Rates':28s} {'CARLA':>10s} {'nuScenes':>10s} {'Delta':>10s}")
    print("  " + "-" * 60)
    for qname, spec in THRESHOLDS.items():
        feat = spec["feature"]
        c_vals = np.array([f[feat] for f in carla_features if feat in f])
        n_vals = np.array([f[feat] for f in ns_features if feat in f])

        if len(c_vals) == 0 or len(n_vals) == 0:
            continue

        if spec["op"] == ">":
            c_pct = 100.0 * np.sum(c_vals > spec["val"]) / len(c_vals)
            n_pct = 100.0 * np.sum(n_vals > spec["val"]) / len(n_vals)
        else:
            c_pct = 100.0 * np.sum(c_vals < spec["val"]) / len(c_vals)
            n_pct = 100.0 * np.sum(n_vals < spec["val"]) / len(n_vals)

        delta = c_pct - n_pct
        marker = "<<<" if abs(delta) > 5 else ""
        print(f"  {qname:28s} {c_pct:9.1f}% {n_pct:9.1f}% {delta:+9.1f}% {marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=str,
                        default=None,
                        help="Path to Frenetix logs directory")
    parser.add_argument("--nuscenes-index", type=str,
                        default=str(PROJECT_ROOT / "output" / "pipeline_all" / "clips" / "clips_index.jsonl"),
                        help="Path to nuScenes clips_index.jsonl for comparison")
    parser.add_argument("--min-duration", type=float, default=3.0,
                        help="Minimum clip duration in seconds (default: 3.0)")
    args = parser.parse_args()

    logs_dir = args.logs_dir
    min_dur = args.min_duration

    if not os.path.isdir(logs_dir):
        print(f"ERROR: Logs directory not found: {logs_dir}")
        sys.exit(1)

    # Discover all scene directories
    scene_dirs = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])
    print(f"Found {len(scene_dirs)} scene directories in {logs_dir}")

    # Process all CSV logs
    all_records = []   # list of {scene, behavior, features}
    n_total = 0
    n_skipped_short = 0
    n_skipped_error = 0

    for i, scene_name in enumerate(scene_dirs):
        scene_path = os.path.join(logs_dir, scene_name)
        csv_files = [f for f in os.listdir(scene_path) if f.endswith("_logs.csv")]

        for csv_file in csv_files:
            n_total += 1
            behavior = csv_file.replace("_logs.csv", "")
            csv_path = os.path.join(scene_path, csv_file)

            # Parse full trajectory
            traj = parse_frenetix_log(csv_path)
            if traj is None:
                n_skipped_error += 1
                continue

            # Window into 3-second clips (matching nuScenes clip length)
            windows = window_trajectory(traj, window_s=min_dur)
            if not windows:
                n_skipped_short += 1
                continue

            for w_idx, window in enumerate(windows):
                features = compute_features(window)
                if features is None:
                    n_skipped_error += 1
                    continue

                all_records.append({
                    "scene": scene_name,
                    "behavior": behavior,
                    "window": w_idx,
                    "features": features,
                })

        # Progress
        if (i + 1) % 200 == 0 or (i + 1) == len(scene_dirs):
            print(f"  Processed {i+1}/{len(scene_dirs)} scenes "
                  f"({len(all_records)} clips kept, "
                  f"{n_skipped_short} too short, "
                  f"{n_skipped_error} errors)")

    # Summary
    print(f"\n{'='*80}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*80}")
    print(f"  Total CSV files scanned:  {n_total:,d}")
    print(f"  Skipped (< {min_dur}s):       {n_skipped_short:,d}")
    print(f"  Skipped (parse error):    {n_skipped_error:,d}")
    print(f"  Clips retained (>= {min_dur}s): {len(all_records):,d}")

    if not all_records:
        print("\nNo clips to analyze. Exiting.")
        sys.exit(0)

    all_features = [r["features"] for r in all_records]

    # Duration distribution
    durations = np.array([f["duration_s"] for f in all_features])
    print(f"\n{'='*80}")
    print(f"DURATION DISTRIBUTION (n={len(all_features)})")
    print(f"{'='*80}")
    print_distribution("duration_s", durations)

    # Feature distributions
    feature_names = [
        "max_speed", "mean_speed", "min_accel", "max_abs_jerk",
        "max_lateral_accel", "max_abs_yaw_rate", "total_heading_change", "mean_accel",
    ]
    print(f"\n{'='*80}")
    print(f"FEATURE DISTRIBUTIONS (n={len(all_features)})")
    print(f"{'='*80}")
    for fname in feature_names:
        vals = np.array([f[fname] for f in all_features])
        print_distribution(fname, vals)

    # Threshold analysis
    print_threshold_analysis(all_features)

    # Per-behavior breakdown
    print_behavior_breakdown(all_records)

    # nuScenes comparison
    print_nuscenes_comparison(all_features, args.nuscenes_index)

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
