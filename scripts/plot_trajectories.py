#!/usr/bin/env python3
"""
Plot all benchmark trajectories in a single normalized overlay.

Each trajectory is translated to start at the origin and rotated so the
initial heading points along the positive x-axis.  This allows direct
visual comparison of trajectory shapes across all clips.

Usage:
    python scripts/plot_trajectories.py \
        --selected selected_clips.json \
        --output-dir assets/figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_carla_distributions import parse_frenetix_log, window_trajectory
from scripts.chunk_carla_videos import clip_id_to_parts

# ---------------------------------------------------------------------------
# TUM corporate colors
# ---------------------------------------------------------------------------
COLOR_NUSCENES = "#e37222"  # TUM orange
COLOR_CARLA = "#0065bd"     # TUM blue

CLIPS_DIR = PROJECT_ROOT / "output" / "pipeline_all" / "clips"
# Override with EGODYN_CARLA_LOGS_DIR or pass --carla-logs explicitly.
CARLA_LOGS_DIR = Path(
    os.environ.get("EGODYN_CARLA_LOGS_DIR", "./data/carla/frenetix_logs")
)


def normalize_trajectory(positions: np.ndarray, yaw_0: float) -> np.ndarray:
    """
    Translate to origin and rotate so initial heading points along +x.

    Args:
        positions: (N, 2) array of (x, y) positions
        yaw_0: initial heading in radians

    Returns:
        (N, 2) normalized positions
    """
    # Translate to origin
    pos = positions - positions[0]

    # Rotate so initial heading = 0 (pointing right)
    cos_a = np.cos(-yaw_0)
    sin_a = np.sin(-yaw_0)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pos = pos @ rot.T

    return pos


def load_nuscenes_trajectory(clip_id: str) -> np.ndarray | None:
    """Load and normalize a nuScenes trajectory from the NPZ file."""
    clips_index = CLIPS_DIR / "clips_index.jsonl"
    if not clips_index.exists():
        return None

    # Find the clip in the index
    with open(clips_index) as f:
        for line in f:
            rec = json.loads(line)
            if rec["clip_id"] == clip_id:
                arr_path = CLIPS_DIR / rec["array_ref"]
                if not arr_path.exists():
                    return None
                npz = np.load(arr_path)
                pos = npz["position"]  # (31, 2)
                yaw = npz["yaw"]      # (31,)
                return normalize_trajectory(pos, yaw[0])
    return None


def load_carla_trajectory(clip_id: str) -> np.ndarray | None:
    """Load and normalize a CARLA trajectory from Frenetix logs."""
    parts = clip_id_to_parts(clip_id)
    scene = parts["scene"]
    behavior = parts["behavior"]
    w_idx = parts["window_idx"]

    csv_path = CARLA_LOGS_DIR / scene / f"{behavior}_logs.csv"
    if not csv_path.exists():
        return None

    traj = parse_frenetix_log(str(csv_path))
    if traj is None:
        return None

    windows = window_trajectory(traj, window_s=3.0)
    if w_idx >= len(windows):
        return None

    window = windows[w_idx]
    pos = window["positions"]  # (N, 2)
    yaw = window["yaws"]      # (N,)

    return normalize_trajectory(pos, yaw[0])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selected", type=str, default="selected_clips.json")
    parser.add_argument("--output-dir", type=str, default="assets/figures")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Limit number of clips per source (for faster testing)")
    args = parser.parse_args()

    with open(args.selected) as f:
        clips = json.load(f)

    ns_clips = [c for c in clips if c["source"] == "nuscenes"]
    ca_clips = [c for c in clips if c["source"] == "carla"]

    if args.max_clips:
        ns_clips = ns_clips[:args.max_clips]
        ca_clips = ca_clips[:args.max_clips]

    print(f"Loading trajectories: {len(ns_clips)} nuScenes + {len(ca_clips)} CARLA")

    # Build index lookup for nuScenes (faster than scanning per clip)
    print("  Building nuScenes index...")
    ns_index = {}
    clips_index_path = CLIPS_DIR / "clips_index.jsonl"
    if clips_index_path.exists():
        with open(clips_index_path) as f:
            for line in f:
                rec = json.loads(line)
                ns_index[rec["clip_id"]] = rec["array_ref"]

    # Load all trajectories
    ns_trajs = []
    print("  Loading nuScenes trajectories...")
    for i, c in enumerate(ns_clips):
        arr_ref = ns_index.get(c["id"])
        if arr_ref is None:
            continue
        arr_path = CLIPS_DIR / arr_ref
        if not arr_path.exists():
            continue
        npz = np.load(arr_path)
        pos = normalize_trajectory(npz["position"], npz["yaw"][0])
        ns_trajs.append(pos)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(ns_clips)}")

    ca_trajs = []
    print("  Loading CARLA trajectories...")
    # Cache parsed logs to avoid re-reading the same CSV
    _log_cache = {}
    for i, c in enumerate(ca_clips):
        parts = clip_id_to_parts(c["id"])
        cache_key = (parts["scene"], parts["behavior"])

        if cache_key not in _log_cache:
            csv_path = CARLA_LOGS_DIR / parts["scene"] / f"{parts['behavior']}_logs.csv"
            if csv_path.exists():
                traj = parse_frenetix_log(str(csv_path))
                if traj is not None:
                    _log_cache[cache_key] = window_trajectory(traj, window_s=3.0)
                else:
                    _log_cache[cache_key] = []
            else:
                _log_cache[cache_key] = []

        windows = _log_cache[cache_key]
        w_idx = parts["window_idx"]
        if w_idx < len(windows):
            window = windows[w_idx]
            pos = normalize_trajectory(window["positions"], window["yaws"][0])
            ca_trajs.append(pos)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(ca_clips)}")

    print(f"\n  Loaded: {len(ns_trajs)} nuScenes + {len(ca_trajs)} CARLA trajectories")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Fixed alpha — visible for human readers
    alpha = 0.35
    lw = 1.0

    # Plot CARLA first (background), then nuScenes on top
    for pos in ca_trajs:
        ax.plot(pos[:, 0], pos[:, 1], color=COLOR_CARLA, alpha=alpha,
                linewidth=lw, solid_capstyle="round")

    for pos in ns_trajs:
        ax.plot(pos[:, 0], pos[:, 1], color=COLOR_NUSCENES, alpha=alpha,
                linewidth=lw, solid_capstyle="round")

    # Origin marker
    ax.plot(0, 0, "ko", markersize=3, zorder=10)

    # Remove all axes — pure trajectory visualization
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Tight crop around the data
    ax.margins(0.02)

    # No legend — colors described in LaTeX figure caption

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "trajectories_overlay.pdf"),
                dpi=300)
    fig.savefig(os.path.join(args.output_dir, "trajectories_overlay.png"),
                dpi=300)
    plt.close(fig)
    print(f"\n  Saved trajectories_overlay.pdf/png")

    # ── Also export trajectory stats for TikZ if desired ──────────────────
    # Export endpoint scatter data (final position of each trajectory)
    endpoints_path = os.path.join(args.output_dir, "..", "tikz_data",
                                  "trajectory_endpoints.csv")
    os.makedirs(os.path.dirname(endpoints_path), exist_ok=True)
    with open(endpoints_path, "w") as f:
        f.write("source,x_end,y_end,distance\n")
        for pos in ns_trajs:
            d = np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2)
            f.write(f"nuscenes,{pos[-1,0]:.4f},{pos[-1,1]:.4f},{d:.4f}\n")
        for pos in ca_trajs:
            d = np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2)
            f.write(f"carla,{pos[-1,0]:.4f},{pos[-1,1]:.4f},{d:.4f}\n")
    print(f"  Saved trajectory_endpoints.csv (for TikZ scatter plot)")


if __name__ == "__main__":
    main()
