#!/usr/bin/env python3
"""
CLI script to extract 3-second clips from nuScenes dataset.

Usage:
    python extract_nuscenes_clips.py \\
        --nuscenes_root /path/to/nuscenes \\
        --output_dir /path/to/output \\
        --num_clips 100 \\
        --seed 42

Output:
    - clips_index.jsonl: One record per clip with metadata and features
    - arrays/clip_XXXXX.npz: Numeric arrays for each clip
    - metadata.json: Dataset-level metadata
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import sys

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.nuscenes_extract import NuScenesClipExtractor
from dataset.generation.dynamics_features import DynamicsProcessor, validate_dynamics_arrays


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_clip_arrays(
    output_dir: Path,
    clip_id: str,
    arrays: Dict[str, np.ndarray],
) -> str:
    """
    Save clip arrays to NPZ file.

    Args:
        output_dir: Output directory
        clip_id: Clip identifier
        arrays: Dictionary of arrays to save

    Returns:
        Relative path to saved file
    """
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{clip_id}.npz"
    filepath = arrays_dir / filename

    np.savez_compressed(filepath, **arrays)

    return f"arrays/{filename}"


def create_clip_record(
    clip_data,
    arrays_ref: str,
    features: Dict[str, float],
) -> dict:
    """
    Create a JSONL record for a clip.

    Args:
        clip_data: ClipData object
        arrays_ref: Reference to stored arrays
        features: Summary features dictionary

    Returns:
        Dictionary ready for JSONL serialization
    """
    return {
        "clip_id": clip_data.clip_id,
        "scene_token": clip_data.scene_token,
        "sample_token": clip_data.sample_token,
        "t_start": clip_data.t_start,
        "t_end": clip_data.t_end,
        "camera": clip_data.camera,
        "frame_tokens": clip_data.frames.tokens,
        "frame_paths": clip_data.frames.paths,
        "num_frames": len(clip_data.frames.tokens),
        "array_ref": arrays_ref,
        "split": "unsplit",
        "features": features,
    }


def process_and_save_clips(
    extractor: NuScenesClipExtractor,
    processor: DynamicsProcessor,
    output_dir: Path,
    max_clips: int = None,
) -> int:
    """
    Extract, process, and save clips.

    Args:
        extractor: Configured NuScenesClipExtractor
        processor: Configured DynamicsProcessor
        output_dir: Output directory
        max_clips: Maximum number of clips to extract

    Returns:
        Number of successfully processed clips
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract clips
    logger.info("Extracting clips from nuScenes...")
    clips = extractor.extract_clips(max_clips=max_clips)

    if not clips:
        logger.warning("No clips extracted!")
        return 0

    # Open JSONL file for writing
    index_path = output_dir / "clips_index.jsonl"
    logger.info(f"Processing and saving {len(clips)} clips...")

    successful_clips = 0
    failed_clips = 0

    with open(index_path, 'w') as f:
        for clip_data in tqdm(clips, desc="Processing clips"):
            try:
                # Extract raw data
                timestamps = np.array([pose.timestamp for pose in clip_data.ego_poses])
                positions = np.array([[pose.x, pose.y] for pose in clip_data.ego_poses])
                yaws = np.array([pose.yaw for pose in clip_data.ego_poses])

                # Calculate target duration from clip window
                target_duration = clip_data.t_end - clip_data.t_start

                # Process dynamics with target duration to ensure exact clip length
                arrays = processor.process_ego_trajectory(
                    timestamps, positions, yaws, target_duration=target_duration
                )

                # Validate arrays (jerk is already computed and smoothed
                # inside process_ego_trajectory)
                is_valid, errors = validate_dynamics_arrays(arrays)
                if not is_valid:
                    logger.warning(f"Clip {clip_data.clip_id} validation failed: {errors}")
                    failed_clips += 1
                    continue

                # Compute summary features using the smoothed arrays
                features = processor.compute_summary_features(
                    speed=arrays['speed'],
                    accel=arrays['accel'],
                    yaw_rate=arrays['yaw_rate'],
                    jerk=arrays['jerk'],
                    yaws=arrays['yaw'],
                )

                # Save arrays
                arrays_ref = save_clip_arrays(output_dir, clip_data.clip_id, arrays)

                # Create and write record
                record = create_clip_record(clip_data, arrays_ref, features)
                f.write(json.dumps(record) + '\n')

                successful_clips += 1

            except Exception as e:
                logger.error(f"Failed to process clip {clip_data.clip_id}: {e}", exc_info=True)
                failed_clips += 1
                continue

    logger.info(f"Successfully processed {successful_clips} clips")
    if failed_clips > 0:
        logger.warning(f"Failed to process {failed_clips} clips")

    return successful_clips


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_clips: int,
):
    """
    Save dataset metadata.

    Args:
        output_dir: Output directory
        args: Command line arguments
        num_clips: Number of clips successfully extracted
    """
    metadata = {
        "nuscenes_version": args.nuscenes_version,
        "clip_seconds": args.clip_seconds,
        "sampling_hz": args.sampling_hz,
        "min_frames": args.min_frames,
        "camera": args.camera,
        "seed": args.seed,
        "num_clips": num_clips,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3-second clips from nuScenes dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        required=True,
        help="Path to nuScenes dataset root directory",
    )
    parser.add_argument(
        "--nuscenes_version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted clips",
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=None,
        help="Maximum number of clips to extract (default: all)",
    )
    parser.add_argument(
        "--clip_seconds",
        type=float,
        default=3.0,
        help="Clip duration in seconds",
    )
    parser.add_argument(
        "--sampling_hz",
        type=float,
        default=10.0,
        help="Resampling frequency in Hz for uniform time series",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=20,
        help="Minimum number of frames required per clip",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera sensor to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate paths
    nuscenes_root = Path(args.nuscenes_root)
    if not nuscenes_root.exists():
        logger.error(f"nuScenes root does not exist: {nuscenes_root}")
        return 1

    output_dir = Path(args.output_dir)

    # Initialize extractor and processor
    logger.info("Initializing extractor...")
    extractor = NuScenesClipExtractor(
        nuscenes_root=str(nuscenes_root),
        version=args.nuscenes_version,
        clip_seconds=args.clip_seconds,
        min_frames=args.min_frames,
        camera=args.camera,
    )

    processor = DynamicsProcessor(sampling_hz=args.sampling_hz)

    # Process and save clips
    num_clips = process_and_save_clips(
        extractor=extractor,
        processor=processor,
        output_dir=output_dir,
        max_clips=args.num_clips,
    )

    # Save metadata
    save_metadata(output_dir, args, num_clips)

    logger.info(f"Done! Extracted {num_clips} clips to {output_dir}")
    logger.info(f"Index file: {output_dir / 'clips_index.jsonl'}")
    logger.info(f"Arrays directory: {output_dir / 'arrays'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
