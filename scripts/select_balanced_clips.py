#!/usr/bin/env python3
"""
Select a balanced subset of clips from nuScenes + CARLA for the benchmark.

Loads all clip features from both sources, applies the benchmark labeling
rules to approximate question answers, then uses a greedy algorithm to
select N clips that minimize answer-class imbalance across all questions.

Usage:
    python scripts/select_balanced_clips.py [--target 3000] [--seed 42]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_carla_distributions import parse_frenetix_log, window_trajectory, compute_features
from scripts.chunk_carla_videos import find_video_path, get_window_frame_range, BEHAVIOR_VIDEO_MAP


# ---------------------------------------------------------------------------
# Labeling rules (mirror questions_template.yaml using summary features)
# ---------------------------------------------------------------------------
def label_clip(features: dict) -> dict:
    """
    Apply threshold-based labeling rules to a clip's summary features.

    Returns a dict of {question_id: answer} for the questions that can be
    determined from summary features alone.  Complex temporal/sequential
    rules (brake_then_turn, stop_and_go, speed_peak_half, contrastive,
    dominant_axis) are omitted — they depend on the full time series and
    will be computed later in the full pipeline.
    """
    labels = {}

    # 1. braking_intensity  (multi-threshold on min_accel, calibrated ~25% per class)
    min_accel = features["min_accel"]
    if min_accel < -1.59:
        labels["braking_intensity"] = "emergency"
    elif min_accel < -0.89:
        labels["braking_intensity"] = "moderate"
    elif min_accel < -0.18:
        labels["braking_intensity"] = "low"
    else:
        labels["braking_intensity"] = "none"

    # 2. driving_smoothness (3-class based on mean |jerk|, calibrated ~33% per class)
    jerk_val = features.get("mean_abs_jerk", features.get("max_abs_jerk", 0.0))
    if jerk_val <= 1.25:
        labels["driving_smoothness"] = "smooth"
    elif jerk_val <= 2.15:
        labels["driving_smoothness"] = "moderate"
    else:
        labels["driving_smoothness"] = "aggressive"

    # 3. extreme_maneuver  (jerk > 20 m/s³ OR hard braking < -0.4g)
    jerk_emergency_val = features.get("max_abs_jerk", 0.0)
    labels["extreme_maneuver"] = "yes" if (
        jerk_emergency_val > 20.0 or features["min_accel"] < -3.924
    ) else "no"

    # 4. high_lateral_accel  (max_lateral_accel > 2.0)
    labels["high_lateral_accel"] = "yes" if features["max_lateral_accel"] > 2.0 else "no"

    # 5. mean_speed_low  (mean_speed < 5.0)
    labels["mean_speed_low"] = "yes" if features["mean_speed"] < 5.0 else "no"

    # 6. significant_heading_change  (total_heading_change > 0.2618 rad)
    labels["significant_heading_change"] = "yes" if features["total_heading_change"] > 0.2618 else "no"

    # 7. yaw_rate_turn_direction  (signed_max_yaw_rate with deadzone ±0.04 rad/s)
    signed_yr = features.get("signed_max_yaw_rate", 0.0)
    if abs(signed_yr) < 0.04:
        labels["yaw_rate_turn_direction"] = "straight"
    elif signed_yr > 0:
        labels["yaw_rate_turn_direction"] = "left"
    else:
        labels["yaw_rate_turn_direction"] = "right"

    # 8. speed_trend  (mean_accel thresholds)
    ma = features.get("mean_accel", 0.0)
    if ma > 0.25:
        labels["speed_trend"] = "accelerating"
    elif ma < -0.25:
        labels["speed_trend"] = "decelerating"
    else:
        labels["speed_trend"] = "steady"

    # 9. dominant_motion_axis  (longitudinal vs lateral accel proxy)
    rms_lon = (features.get("min_accel", 0.0)**2 + features.get("max_accel", 0.0)**2) ** 0.5
    peak_lat = features.get("max_lateral_accel", 0.0)
    if rms_lon < 0.5 and peak_lat < 0.5:
        labels["dominant_motion_axis"] = "none"
    elif peak_lat / max(rms_lon, 1e-6) > 1.0:
        labels["dominant_motion_axis"] = "lateral"
    else:
        labels["dominant_motion_axis"] = "longitudinal"

    return labels


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_nuscenes_clips(index_path: str) -> list[dict]:
    """Load clip features from the nuScenes clips_index.jsonl."""
    clips = []
    with open(index_path) as f:
        for line in f:
            rec = json.loads(line)
            if "features" not in rec:
                continue
            clips.append({
                "id": rec["clip_id"],
                "source": "nuscenes",
                "features": rec["features"],
            })
    return clips


def load_carla_clips(
    logs_dir: str,
    min_duration: float = 3.0,
    video_dir: str | None = None,
) -> list[dict]:
    """Load CARLA clips, windowed into fixed-length segments matching nuScenes.

    If video_dir is provided, only include clips whose corresponding FPV
    video exists and has enough frames for the window.  This ensures every
    selected clip can actually be chunked later.
    """
    import cv2

    clips = []
    n_no_video = 0
    n_short_video = 0

    scene_dirs = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])

    # Pre-cache video frame counts per (scene, behavior) to avoid
    # opening the same video file repeatedly for each window.
    _video_frames_cache: dict[tuple[str, str], int | None] = {}

    def _get_video_frames(scene: str, behavior: str) -> int | None:
        key = (scene, behavior)
        if key not in _video_frames_cache:
            if video_dir is None:
                _video_frames_cache[key] = None  # no check
                return None
            vpath = find_video_path(scene, behavior, video_dir)
            if vpath is None:
                _video_frames_cache[key] = -1  # missing
            else:
                cap = cv2.VideoCapture(vpath)
                _video_frames_cache[key] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        val = _video_frames_cache[key]
        return None if val is None else (val if val >= 0 else -1)

    for i, scene_name in enumerate(scene_dirs):
        scene_path = os.path.join(logs_dir, scene_name)
        for csv_file in os.listdir(scene_path):
            if not csv_file.endswith("_logs.csv"):
                continue
            behavior = csv_file.replace("_logs.csv", "")
            traj = parse_frenetix_log(os.path.join(scene_path, csv_file))
            if traj is None:
                continue

            # Check video availability once per scene+behavior
            total_video_frames = _get_video_frames(scene_name, behavior)
            if total_video_frames is not None and total_video_frames < 0:
                n_no_video += 1
                continue

            # Window into 3s clips (same as nuScenes clip length)
            windows = window_trajectory(traj, window_s=min_duration)
            for w_idx, window in enumerate(windows):
                # Check if this specific window fits within the video
                if total_video_frames is not None:
                    _, end_frame = get_window_frame_range(w_idx, window_s=min_duration)
                    if end_frame > total_video_frames:
                        n_short_video += 1
                        continue

                features = compute_features(window)
                if features is None:
                    continue
                clips.append({
                    "id": f"{scene_name}__{behavior}__w{w_idx}",
                    "source": "carla",
                    "features": features,
                })
        if (i + 1) % 500 == 0:
            print(f"  CARLA: {i+1}/{len(scene_dirs)} scenes, {len(clips)} clips")

    if video_dir is not None:
        print(f"  CARLA video filter: {n_no_video} behaviors skipped (no video), "
              f"{n_short_video} windows skipped (video too short)")

    return clips


# ---------------------------------------------------------------------------
# Greedy balanced selection
# ---------------------------------------------------------------------------
def compute_imbalance(counts: dict, target_frac: dict) -> float:
    """
    Compute imbalance score for one question.

    Returns the maximum deviation of any answer class from its target fraction.
    Higher = more imbalanced.
    """
    total = sum(counts.values())
    if total == 0:
        return 1.0
    max_dev = 0.0
    for cls, target in target_frac.items():
        actual = counts.get(cls, 0) / total
        max_dev = max(max_dev, abs(actual - target))
    return max_dev


def greedy_select(
    pool: list[dict],
    target_n: int,
    seed: int = 42,
    min_nuscenes_frac: float = 0.0,
    min_carla_frac: float = 0.0,
) -> list[int]:
    """
    Greedily select clips to minimize answer imbalance.

    Algorithm:
        1. Pre-label all clips.
        2. For each slot, find the question with the worst imbalance.
        3. Find which answer class is most underrepresented.
        4. From the pool, pick a random clip with that answer — preferring
           clips that also help secondary imbalances.
        5. Add it to the selection.

    Source-ratio constraints:
        When min_nuscenes_frac or min_carla_frac > 0, the algorithm checks
        whether either source is falling behind its minimum.  If so,
        candidates are restricted to clips from the underrepresented source.

    Returns list of indices into the pool.
    """
    rng = np.random.default_rng(seed)

    # Pre-label all clips
    all_labels = []
    for clip in pool:
        all_labels.append(label_clip(clip["features"]))

    # Determine questions and target distributions
    # Collect all answer classes per question from the pool
    question_classes = defaultdict(set)
    for labels in all_labels:
        for q, ans in labels.items():
            question_classes[q].add(ans)

    # Target: uniform distribution across classes for each question
    targets = {}
    for q, classes in question_classes.items():
        n_classes = len(classes)
        targets[q] = {cls: 1.0 / n_classes for cls in classes}

    questions = sorted(targets.keys())
    print(f"\n  Balancing on {len(questions)} questions: {questions}")
    for q in questions:
        print(f"    {q}: target {targets[q]}")

    # Pre-group: for each question and answer, which pool indices have that answer
    q_ans_indices = defaultdict(lambda: defaultdict(list))
    for idx, labels in enumerate(all_labels):
        for q, ans in labels.items():
            q_ans_indices[q][ans].append(idx)

    # Pre-group by source for fast filtering
    source_indices = defaultdict(set)
    for idx, clip in enumerate(pool):
        source_indices[clip["source"]].add(idx)

    # Source-ratio constraint info
    if min_nuscenes_frac > 0 or min_carla_frac > 0:
        print(f"\n  Source constraints: min nuScenes >= {min_nuscenes_frac:.0%}, "
              f"min CARLA >= {min_carla_frac:.0%}")

    # Selection loop
    selected = []
    selected_set = set()
    # Running counts per question
    running_counts = {q: Counter() for q in questions}
    # Running source counts
    source_counts = Counter()

    n_pool = len(pool)
    print(f"\n  Selecting {target_n} clips from pool of {n_pool}...")

    for step in range(target_n):
        # 1. Find the most imbalanced question
        worst_q = None
        worst_imb = -1
        for q in questions:
            imb = compute_imbalance(running_counts[q], targets[q])
            if imb > worst_imb:
                worst_imb = imb
                worst_q = q

        # 2. Find the most underrepresented class for that question
        total = max(sum(running_counts[worst_q].values()), 1)
        worst_cls = None
        worst_deficit = float("inf")
        for cls, target_f in targets[worst_q].items():
            actual_f = running_counts[worst_q].get(cls, 0) / total
            deficit = actual_f - target_f  # negative = underrepresented
            if deficit < worst_deficit:
                worst_deficit = deficit
                worst_cls = cls

        # 3. Find candidate clips with that answer, not yet selected
        candidates = [
            idx for idx in q_ans_indices[worst_q][worst_cls]
            if idx not in selected_set
        ]

        if not candidates:
            # Fallback: pick any unselected clip
            candidates = [i for i in range(n_pool) if i not in selected_set]
            if not candidates:
                break

        # 3b. Apply source-ratio constraint via hard caps.
        #     Once a source hits its maximum allowed count, it is excluded
        #     from all future selections — even if no clip from the remaining
        #     source has the ideal answer.  This guarantees the source ratio
        #     but may reduce answer balance for rare-event questions.
        ns_cap = target_n - int(np.ceil(min_carla_frac * target_n))   # max nuScenes
        ca_cap = target_n - int(np.ceil(min_nuscenes_frac * target_n))  # max CARLA
        excluded_sources = set()
        if source_counts.get("nuscenes", 0) >= ns_cap:
            excluded_sources.add("nuscenes")
        if source_counts.get("carla", 0) >= ca_cap:
            excluded_sources.add("carla")

        if excluded_sources:
            source_candidates = [
                idx for idx in candidates
                if pool[idx]["source"] not in excluded_sources
            ]
            if source_candidates:
                candidates = source_candidates
            else:
                # No clips from allowed sources have the needed answer.
                # Fall back to any unselected clip from allowed sources
                # (sacrifices answer balance to maintain source ratio).
                candidates = [
                    i for i in range(n_pool)
                    if i not in selected_set
                    and pool[i]["source"] not in excluded_sources
                ]
                if not candidates:
                    break

        # 4. Score candidates by secondary benefit (how many other underrepresented
        #    classes they contribute to)
        if len(candidates) > 500:
            # Subsample for speed
            candidates = list(rng.choice(candidates, size=500, replace=False))

        best_idx = None
        best_secondary = -1

        for idx in candidates:
            secondary = 0
            labels = all_labels[idx]
            for q in questions:
                if q == worst_q:
                    continue
                ans = labels.get(q)
                if ans is None:
                    continue
                q_total = max(sum(running_counts[q].values()), 1)
                actual_f = running_counts[q].get(ans, 0) / q_total
                target_f = targets[q].get(ans, 0.5)
                if actual_f < target_f:
                    # This clip helps an underrepresented class
                    secondary += (target_f - actual_f)
            if secondary > best_secondary:
                best_secondary = secondary
                best_idx = idx

        if best_idx is None:
            best_idx = rng.choice(candidates)

        # 5. Add to selection
        selected.append(best_idx)
        selected_set.add(best_idx)
        source_counts[pool[best_idx]["source"]] += 1
        labels = all_labels[best_idx]
        for q, ans in labels.items():
            running_counts[q][ans] += 1

        if (step + 1) % 500 == 0 or step == target_n - 1:
            print(f"    Step {step+1}/{target_n}")

    return selected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_selection(pool: list[dict], selected_indices: list[int]):
    """Print distribution report for the selected subset."""
    selected = [pool[i] for i in selected_indices]
    n = len(selected)

    # Source breakdown
    n_ns = sum(1 for c in selected if c["source"] == "nuscenes")
    n_ca = sum(1 for c in selected if c["source"] == "carla")
    print(f"\n{'='*80}")
    print(f"SELECTED {n} CLIPS: {n_ns} nuScenes ({100*n_ns/n:.1f}%) + {n_ca} CARLA ({100*n_ca/n:.1f}%)")
    print(f"{'='*80}")

    # Label all selected clips
    all_labels = [label_clip(c["features"]) for c in selected]
    questions = sorted(set(q for labels in all_labels for q in labels))

    print(f"\n  {'Question':28s} {'Class':15s} {'Count':>7s} {'Pct':>7s}")
    print("  " + "-" * 60)
    for q in questions:
        answers = [labels.get(q) for labels in all_labels if q in labels]
        counts = Counter(answers)
        total = sum(counts.values())
        for cls, cnt in sorted(counts.items()):
            print(f"  {q:28s} {cls:15s} {cnt:7d} {100*cnt/total:6.1f}%")
        print()

    # Feature distribution summary
    feature_keys = [
        "mean_speed", "max_speed", "min_accel", "mean_abs_jerk",
        "max_lateral_accel", "max_abs_yaw_rate", "total_heading_change",
    ]
    print(f"  {'Feature':28s} {'mean':>9s} {'p25':>9s} {'p50':>9s} {'p75':>9s} {'p95':>9s}")
    print("  " + "-" * 68)
    for fk in feature_keys:
        vals = np.array([c["features"][fk] for c in selected if fk in c["features"]])
        print(f"  {fk:28s} {np.mean(vals):9.4f} {np.percentile(vals,25):9.4f} "
              f"{np.percentile(vals,50):9.4f} {np.percentile(vals,75):9.4f} "
              f"{np.percentile(vals,95):9.4f}")

    # CARLA behavior breakdown
    carla_clips = [c for c in selected if c["source"] == "carla"]
    if carla_clips:
        behaviors = Counter(c["id"].split("__")[1] for c in carla_clips)
        print(f"\n  CARLA behavior breakdown:")
        for beh, cnt in sorted(behaviors.items()):
            print(f"    {beh:25s} {cnt:5d}")

    # CARLA scene coverage
    if carla_clips:
        scenes = set(c["id"].split("__")[0] for c in carla_clips)
        print(f"\n  CARLA unique scenes used: {len(scenes)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=3000,
                        help="Target number of clips to select (default: 3000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--min-duration", type=float, default=3.0,
                        help="Minimum CARLA clip duration in seconds (default: 3.0)")
    parser.add_argument("--nuscenes-index", type=str,
                        default=str(PROJECT_ROOT / "output" / "nuscenes_clips" / "clips_index.jsonl"))
    parser.add_argument("--carla-logs", type=str,
                        default=None)
    parser.add_argument("--carla-video-dir", type=str,
                        default=None,
                        help="CARLA video directory; used to filter out clips without matching video")
    parser.add_argument("--min-nuscenes-frac", type=float, default=0.0,
                        help="Minimum fraction of nuScenes clips (e.g. 0.6 for 60%%)")
    parser.add_argument("--min-carla-frac", type=float, default=0.0,
                        help="Minimum fraction of CARLA clips (e.g. 0.2 for 20%%)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save selected clip IDs as JSON (optional)")
    args = parser.parse_args()

    # Load data
    print("Loading nuScenes clips...")
    ns_clips = load_nuscenes_clips(args.nuscenes_index)
    print(f"  {len(ns_clips):,d} nuScenes clips loaded")

    print("Loading CARLA clips (with video availability check)...")
    carla_clips = load_carla_clips(
        args.carla_logs, args.min_duration, video_dir=args.carla_video_dir,
    )
    print(f"  {len(carla_clips):,d} CARLA clips loaded (>= {args.min_duration}s, video verified)")

    pool = ns_clips + carla_clips
    print(f"\nTotal pool: {len(pool):,d} clips")

    # Run selection
    selected_indices = greedy_select(
        pool, args.target, args.seed,
        min_nuscenes_frac=args.min_nuscenes_frac,
        min_carla_frac=args.min_carla_frac,
    )

    # Report
    report_selection(pool, selected_indices)

    # Save
    if args.output:
        selected_clips = []
        for idx in selected_indices:
            c = pool[idx]
            selected_clips.append({
                "id": c["id"],
                "source": c["source"],
                "features": c["features"],
            })
        with open(args.output, "w") as f:
            json.dump(selected_clips, f, indent=2)
        print(f"\nSaved {len(selected_clips)} selected clips to {args.output}")


if __name__ == "__main__":
    main()
