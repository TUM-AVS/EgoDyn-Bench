#!/usr/bin/env python3
"""
CLI script to extract 3-second clips from CARLA Frenetix replay logs.

Produces the same output format as extract_nuscenes_clips.py, enabling
full QA generation (all 14 question types) for CARLA clips.

Usage:
    python dataset/scripts/extract_carla_clips.py \\
        --carla_logs /path/to/frenetix_logs \\
        --output_dir output/carla_clips/ \\
        --carla_video_dir /path/to/videos \\
        --require_video

Output:
    - clips_index.jsonl: One record per clip with metadata and features
    - arrays/<clip_id>.npz: Numeric arrays for each clip
    - metadata.json: Dataset-level metadata
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_carla_distributions import parse_frenetix_log, window_trajectory
from scripts.chunk_carla_videos import find_video_path, get_window_frame_range
from dataset.generation.dynamics_features import DynamicsProcessor, validate_dynamics_arrays
from dataset.scripts.extract_nuscenes_clips import save_clip_arrays


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_carla_clip_record(
    clip_id: str,
    scene: str,
    behavior: str,
    window_idx: int,
    arrays_ref: str,
    features: Dict[str, float],
    video_path: str | None = None,
) -> dict:
    """Create a JSONL record for a CARLA clip."""
    record = {
        "clip_id": clip_id,
        "scene": scene,
        "behavior": behavior,
        "window_idx": window_idx,
        "t_end": 3.0,
        "array_ref": arrays_ref,
        "split": "unsplit",
        "source": "carla",
        "features": features,
    }
    if video_path is not None:
        record["video_path"] = video_path
    return record


def process_carla_clips(
    logs_dir: Path,
    output_dir: Path,
    processor: DynamicsProcessor,
    video_dir: Path | None = None,
    require_video: bool = False,
    behaviors: list[str] | None = None,
    scenes: list[str] | None = None,
    max_clips: int | None = None,
    max_accel_ms2: float | None = None,
) -> int:
    """
    Extract, process, and save CARLA clips.

    Returns:
        Number of successfully processed clips.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "clips_index.jsonl"

    # Discover scene directories (sorted for determinism)
    scene_dirs = sorted(
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    )
    if scenes is not None:
        scene_set = set(scenes)
        scene_dirs = [d for d in scene_dirs if d in scene_set]

    logger.info(f"Found {len(scene_dirs)} scene directories")

    # Video frame cache: (scene, behavior) -> (frame_count, video_path)
    # frame_count = -1 means video missing
    _video_cache: dict[tuple[str, str], tuple[int, str | None]] = {}

    def _get_video_info(scene: str, behavior: str) -> tuple[int, str | None]:
        key = (scene, behavior)
        if key not in _video_cache:
            if video_dir is None:
                _video_cache[key] = (-1, None)
            else:
                import cv2
                vpath = find_video_path(scene, behavior, str(video_dir))
                if vpath is None:
                    _video_cache[key] = (-1, None)
                else:
                    cap = cv2.VideoCapture(vpath)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    _video_cache[key] = (frames, vpath)
        return _video_cache[key]

    successful = 0
    failed = 0
    skipped_video = 0
    skipped_implausible = 0

    with open(index_path, 'w') as f:
        for scene_name in tqdm(scene_dirs, desc="Processing scenes"):
            scene_path = os.path.join(logs_dir, scene_name)
            csv_files = sorted(
                cf for cf in os.listdir(scene_path)
                if cf.endswith("_logs.csv") or
                (cf.startswith("logs_") and cf.endswith(".csv"))
            )

            for csv_file in csv_files:
                # Support both naming conventions:
                #   old: {Behavior}_logs.csv  (e.g. Balanced_logs.csv)
                #   new: logs_{behavior}.csv  (e.g. logs_balanced.csv)
                if csv_file.endswith("_logs.csv"):
                    behavior = csv_file.replace("_logs.csv", "")
                else:
                    behavior = csv_file.removeprefix("logs_").removesuffix(".csv")
                if behaviors is not None and behavior not in behaviors:
                    continue

                csv_path = os.path.join(scene_path, csv_file)
                traj = parse_frenetix_log(csv_path)
                if traj is None:
                    failed += 1
                    continue

                windows = window_trajectory(traj, window_s=3.0)
                if not windows:
                    continue

                # Check video once per scene+behavior
                total_frames, vpath = _get_video_info(scene_name, behavior)

                if total_frames < 0 and video_dir is not None and require_video:
                    skipped_video += len(windows)
                    continue

                for w_idx, window in enumerate(windows):
                    if max_clips is not None and successful >= max_clips:
                        break

                    clip_id = f"{scene_name}__{behavior}__w{w_idx}"

                    # Video availability check for this specific window
                    resolved_video_path = None
                    if video_dir is not None and total_frames >= 0:
                        _, end_frame = get_window_frame_range(w_idx)
                        if end_frame > total_frames:
                            if require_video:
                                skipped_video += 1
                                continue
                        else:
                            resolved_video_path = vpath

                    try:
                        arrays = processor.process_ego_trajectory(
                            timestamps=window["timestamps"],
                            positions=window["positions"],
                            yaws=window["yaws"],
                        )

                        is_valid, errors = validate_dynamics_arrays(arrays)
                        if not is_valid:
                            logger.warning(
                                f"Clip {clip_id} validation failed: {errors}"
                            )
                            failed += 1
                            continue

                        features = processor.compute_summary_features(
                            speed=arrays['speed'],
                            accel=arrays['accel'],
                            yaw_rate=arrays['yaw_rate'],
                            jerk=arrays['jerk'],
                            yaws=arrays['yaw'],
                        )

                        # Plausibility filter: reject clips exceeding
                        # physical acceleration limits for road vehicles
                        if max_accel_ms2 is not None:
                            peak_lon = max(
                                abs(features["min_accel"]),
                                abs(features["max_accel"]),
                            )
                            peak_lat = features["max_lateral_accel"]
                            if peak_lon > max_accel_ms2 or peak_lat > max_accel_ms2:
                                skipped_implausible += 1
                                continue

                        arrays_ref = save_clip_arrays(
                            output_dir, clip_id, arrays
                        )

                        record = create_carla_clip_record(
                            clip_id=clip_id,
                            scene=scene_name,
                            behavior=behavior,
                            window_idx=w_idx,
                            arrays_ref=arrays_ref,
                            features=features,
                            video_path=resolved_video_path,
                        )
                        f.write(json.dumps(record) + '\n')
                        successful += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to process clip {clip_id}: {e}",
                            exc_info=True,
                        )
                        failed += 1

                if max_clips is not None and successful >= max_clips:
                    break
            if max_clips is not None and successful >= max_clips:
                break

    logger.info(
        f"Successfully processed {successful} clips, "
        f"{failed} failed, {skipped_video} skipped (no video), "
        f"{skipped_implausible} skipped (implausible dynamics)"
    )
    return successful


def save_carla_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_clips: int,
) -> None:
    """Save dataset metadata."""
    metadata = {
        "source": "carla",
        "clip_seconds": 3.0,
        "sampling_hz": args.sampling_hz,
        "seed": args.seed,
        "num_clips": num_clips,
        "carla_logs": args.carla_logs,
        "carla_video_dir": args.carla_video_dir,
        "require_video": args.require_video,
        "max_accel_g": args.max_accel_g,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3-second clips from CARLA Frenetix replay logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--carla_logs", type=str, required=True,
        help="Path to Frenetix logs directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for extracted clips",
    )
    parser.add_argument(
        "--carla_video_dir", type=str, default=None,
        help="CARLA video directory for video path resolution",
    )
    parser.add_argument(
        "--require_video", action="store_true",
        help="Skip clips without matching video",
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Maximum number of clips to extract (default: all)",
    )
    parser.add_argument(
        "--sampling_hz", type=float, default=10.0,
        help="Resampling frequency in Hz",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--behaviors", type=str, nargs="+", default=None,
        help="Filter to specific behaviors (e.g. Default Comfort)",
    )
    parser.add_argument(
        "--scenes", type=str, nargs="+", default=None,
        help="Filter to specific scene names",
    )
    parser.add_argument(
        "--max_accel_g", type=float, default=1.0,
        help="Reject clips where peak accel (longitudinal or lateral) "
             "exceeds this many g (1g = 9.81 m/s²). Set to 0 to disable.",
    )

    args = parser.parse_args()

    logs_dir = Path(args.carla_logs)
    if not logs_dir.exists():
        logger.error(f"CARLA logs directory does not exist: {logs_dir}")
        return 1

    output_dir = Path(args.output_dir)
    video_dir = Path(args.carla_video_dir) if args.carla_video_dir else None

    processor = DynamicsProcessor(sampling_hz=args.sampling_hz)

    max_accel_ms2 = args.max_accel_g * 9.81 if args.max_accel_g > 0 else None
    if max_accel_ms2 is not None:
        logger.info(
            f"Plausibility filter: rejecting clips with peak accel > "
            f"{args.max_accel_g:.1f}g ({max_accel_ms2:.2f} m/s²)"
        )

    num_clips = process_carla_clips(
        logs_dir=logs_dir,
        output_dir=output_dir,
        processor=processor,
        video_dir=video_dir,
        require_video=args.require_video,
        behaviors=args.behaviors,
        scenes=args.scenes,
        max_clips=args.max_clips,
        max_accel_ms2=max_accel_ms2,
    )

    save_carla_metadata(output_dir, args, num_clips)

    logger.info(f"Done! Extracted {num_clips} clips to {output_dir}")
    logger.info(f"Index file: {output_dir / 'clips_index.jsonl'}")
    logger.info(f"Arrays directory: {output_dir / 'arrays'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
