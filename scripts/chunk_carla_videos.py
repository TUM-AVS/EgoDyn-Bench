#!/usr/bin/env python3
"""
Chunk CARLA videos into 3-second clips matching the selected trajectory windows.

For each selected CARLA clip ID (e.g., "DEU_Weimar-10_1_T-4__Default__w0"),
this script:
  1. Locates the source video (FPV view)
  2. Extracts the correct frame range for the window index
  3. Resizes to 1280x720 (Cosmos Transfer 2.5 input resolution)
  4. Writes the chunked clip as MP4

The window-to-frame mapping mirrors the windowing logic in
analyze_carla_distributions.py (non-overlapping 3s windows at 10 Hz,
next window starts at last frame of previous).

Usage:
    python scripts/chunk_carla_videos.py \
        --selected selected_clips.json \
        --output-dir output/carla_chunks \
        [--cosmos-ready]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Behavior name mapping: log names → video file names
# ---------------------------------------------------------------------------
BEHAVIOR_VIDEO_MAP = {
    "Balanced": "Balanced",
    "Comfort": "Comfort",
    "Default": "Default",
    "Efficiency-Sporty": "Sporty",
    "Safety-Conservative": "Safety",
    # New data uses lowercase behavior names in log filenames
    "balanced": "Balanced",
    "comfort": "Comfort",
    "default": "Default",
    "sporty": "Sporty",
    "safety": "Safety",
}

# Video source parameters (verified from CARLA data)
SOURCE_FPS = 10.0
SOURCE_WIDTH = 1600
SOURCE_HEIGHT = 900

# Cosmos Transfer 2.5 target specs
COSMOS_WIDTH = 1280
COSMOS_HEIGHT = 720


def clip_id_to_parts(clip_id: str) -> dict:
    """Parse a CARLA clip ID into scene, behavior, and window index."""
    parts = clip_id.split("__")
    return {
        "scene": parts[0],
        "behavior": parts[1],
        "window_idx": int(parts[2].replace("w", "")),
    }


def get_window_frame_range(
    window_idx: int,
    window_s: float = 3.0,
    fps: float = 10.0,
) -> tuple[int, int]:
    """
    Compute the start and end frame for a given window index.

    Mirrors the windowing logic in analyze_carla_distributions.window_trajectory:
    - Each window is window_s seconds = (window_s * fps) + 1 frames (fencepost)
    - Non-overlapping: next window starts at last frame of previous
    """
    samples_per_window = int(round(window_s / (1.0 / fps))) + 1  # 31 for 3s@10Hz
    stride = samples_per_window - 1  # 30 frames between window starts
    start_frame = window_idx * stride
    end_frame = start_frame + samples_per_window  # exclusive
    return start_frame, end_frame


def find_video_path(
    scene: str,
    behavior: str,
    video_dir: str,
) -> str | None:
    """Locate the FPV video file for a given scene and behavior."""
    video_behavior = BEHAVIOR_VIDEO_MAP.get(behavior, behavior)
    scene_dir = os.path.join(video_dir, scene)
    if not os.path.isdir(scene_dir):
        return None

    # Expected pattern: {scene}_{behavior}_FPV.mp4
    expected = f"{scene}_{video_behavior}_FPV.mp4"
    path = os.path.join(scene_dir, expected)
    if os.path.isfile(path):
        return path

    # Fallback: search for any matching FPV file
    for f in os.listdir(scene_dir):
        if video_behavior in f and "FPV" in f and f.endswith(".mp4"):
            return os.path.join(scene_dir, f)

    return None


def extract_chunk(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    target_width: int = COSMOS_WIDTH,
    target_height: int = COSMOS_HEIGHT,
) -> bool:
    """
    Extract a frame range from a video, resize, and write as MP4.

    Returns True on success.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame > total_frames:
        cap.release()
        return False

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            writer.release()
            cap.release()
            return False
        resized = cv2.resize(frame, (target_width, target_height))
        writer.write(resized)

    writer.release()
    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--selected", type=str, required=True,
        help="Path to selected_clips.json from select_balanced_clips.py",
    )
    parser.add_argument(
        "--video-dir", type=str,
        default=None,
        help="Root directory of CARLA FPV videos",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/carla_chunks",
        help="Output directory for chunked clips",
    )
    parser.add_argument(
        "--cosmos-ready", action="store_true",
        help="Resize to 1280x720 for Cosmos Transfer 2.5 (default: keep 1600x900)",
    )
    args = parser.parse_args()

    # Load selected clips
    with open(args.selected) as f:
        all_clips = json.load(f)

    carla_clips = [c for c in all_clips if c["source"] == "carla"]
    print(f"Selected clips: {len(all_clips)} total, {len(carla_clips)} CARLA")

    # Set target resolution
    if args.cosmos_ready:
        target_w, target_h = COSMOS_WIDTH, COSMOS_HEIGHT
        print(f"Output resolution: {target_w}x{target_h} (Cosmos Transfer 2.5)")
    else:
        target_w, target_h = SOURCE_WIDTH, SOURCE_HEIGHT
        print(f"Output resolution: {target_w}x{target_h} (original)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Group clips by scene+behavior to minimize video re-opens
    groups = defaultdict(list)
    for clip in carla_clips:
        parts = clip_id_to_parts(clip["id"])
        key = (parts["scene"], parts["behavior"])
        groups[key].append((clip["id"], parts["window_idx"]))

    print(f"Processing {len(groups)} unique scene-behavior combinations...")

    n_success = 0
    n_fail = 0
    n_missing_video = 0

    for i, ((scene, behavior), windows) in enumerate(sorted(groups.items())):
        video_path = find_video_path(scene, behavior, args.video_dir)
        if video_path is None:
            n_missing_video += len(windows)
            continue

        for clip_id, w_idx in sorted(windows, key=lambda x: x[1]):
            start_frame, end_frame = get_window_frame_range(w_idx)
            output_path = os.path.join(args.output_dir, f"{clip_id}.mp4")

            ok = extract_chunk(
                video_path, start_frame, end_frame, output_path,
                target_w, target_h,
            )
            if ok:
                n_success += 1
            else:
                n_fail += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(groups)} groups, "
                  f"{n_success} success, {n_fail} failed, "
                  f"{n_missing_video} missing video")

    print(f"\nDone: {n_success} chunks written, "
          f"{n_fail} failed, {n_missing_video} missing video")
    print(f"Output: {args.output_dir}/")

    # Write manifest for Cosmos batch processing
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = []
    for clip in carla_clips:
        output_path = os.path.join(args.output_dir, f"{clip['id']}.mp4")
        if os.path.isfile(output_path):
            manifest.append({
                "clip_id": clip["id"],
                "input_path": f"{clip['id']}.mp4",
                "features": clip["features"],
            })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
