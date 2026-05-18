#!/usr/bin/env python3
"""
Validation suite for extracted nuScenes clips.

Verifies:
- Clip duration and timestamp consistency
- Frame references exist and are properly ordered
- Numeric arrays have no NaNs/Infs and correct shapes
- Alignment between frames and ego time series
- Distribution summaries for debugging
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClipValidator:
    """Validates extracted clips for correctness and consistency."""

    def __init__(
        self,
        clips_dir: Path,
        nuscenes_root: Path = None,
        timestamp_tolerance: float = 0.1,
    ):
        """
        Initialize validator.

        Args:
            clips_dir: Directory containing clips_index.jsonl and arrays/
            nuscenes_root: Optional path to nuScenes root for frame validation
            timestamp_tolerance: Tolerance for timestamp alignment checks (seconds)
        """
        self.clips_dir = Path(clips_dir)
        self.nuscenes_root = Path(nuscenes_root) if nuscenes_root else None
        self.timestamp_tolerance = timestamp_tolerance

        self.index_path = self.clips_dir / "clips_index.jsonl"
        self.arrays_dir = self.clips_dir / "arrays"

        # Statistics accumulators
        self.all_speeds = []
        self.all_accels = []
        self.all_yaw_rates = []
        self.all_jerks = []

    def validate_all(self) -> Tuple[int, int, List[str]]:
        """
        Validate all clips in the dataset.

        Returns:
            Tuple of (num_valid, num_invalid, error_messages)
        """
        if not self.index_path.exists():
            return 0, 0, [f"Index file not found: {self.index_path}"]

        if not self.arrays_dir.exists():
            return 0, 0, [f"Arrays directory not found: {self.arrays_dir}"]

        logger.info(f"Validating clips in {self.clips_dir}")

        num_valid = 0
        num_invalid = 0
        all_errors = []

        # Read all clips
        clips = []
        with open(self.index_path, 'r') as f:
            for line in f:
                clips.append(json.loads(line))

        logger.info(f"Found {len(clips)} clips to validate")

        # Validate each clip
        for clip in tqdm(clips, desc="Validating clips"):
            is_valid, errors = self._validate_clip(clip)

            if is_valid:
                num_valid += 1
            else:
                num_invalid += 1
                all_errors.extend(errors)

        # Print distribution summaries
        self._print_distribution_summary()

        return num_valid, num_invalid, all_errors

    def _validate_clip(self, clip: dict) -> Tuple[bool, List[str]]:
        """
        Validate a single clip.

        Args:
            clip: Clip record from JSONL

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        clip_id = clip['clip_id']

        # 1. Validate clip duration
        duration = clip['t_end'] - clip['t_start']
        if abs(duration - 3.0) > self.timestamp_tolerance:
            errors.append(
                f"{clip_id}: Invalid duration {duration:.3f}s (expected ~3.0s)"
            )

        # 2. Validate timestamps are ordered
        if clip['t_end'] <= clip['t_start']:
            errors.append(
                f"{clip_id}: t_end ({clip['t_end']}) <= t_start ({clip['t_start']})"
            )

        # 3. Validate frame references
        if len(clip['frame_tokens']) == 0:
            errors.append(f"{clip_id}: No frames in clip")

        if len(clip['frame_tokens']) != len(clip['frame_paths']):
            errors.append(
                f"{clip_id}: Mismatch between frame_tokens ({len(clip['frame_tokens'])}) "
                f"and frame_paths ({len(clip['frame_paths'])})"
            )

        # 4. Validate frame count
        num_frames = clip.get('num_frames', len(clip['frame_tokens']))
        if num_frames < 20:
            errors.append(f"{clip_id}: Only {num_frames} frames (min: 20)")

        # 5. Check if frames exist (if nuscenes_root provided)
        if self.nuscenes_root:
            for frame_path in clip['frame_paths'][:5]:  # Check first 5 for speed
                full_path = self.nuscenes_root / frame_path
                if not full_path.exists():
                    errors.append(f"{clip_id}: Frame not found: {frame_path}")
                    break

        # 6. Validate arrays
        array_ref = clip.get('array_ref')
        if not array_ref:
            errors.append(f"{clip_id}: Missing array_ref")
        else:
            array_path = self.clips_dir / array_ref
            if not array_path.exists():
                errors.append(f"{clip_id}: Array file not found: {array_ref}")
            else:
                array_errors = self._validate_arrays(clip_id, array_path, clip)
                errors.extend(array_errors)

        # 7. Validate features
        features = clip.get('features', {})
        required_features = [
            'max_speed', 'mean_speed', 'min_accel',
            'max_abs_yaw_rate', 'max_abs_jerk', 'max_lateral_accel',
            'total_heading_change',
        ]
        for feat in required_features:
            if feat not in features:
                errors.append(f"{clip_id}: Missing feature: {feat}")
            elif not np.isfinite(features[feat]):
                errors.append(f"{clip_id}: Feature {feat} is not finite: {features[feat]}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _validate_arrays(
        self,
        clip_id: str,
        array_path: Path,
        clip: dict,
    ) -> List[str]:
        """
        Validate numeric arrays for a clip.

        Args:
            clip_id: Clip identifier
            array_path: Path to NPZ file
            clip: Clip record from JSONL

        Returns:
            List of error messages
        """
        errors = []

        try:
            # Load arrays
            data = np.load(array_path)

            # Check required keys
            required_keys = ['timestamps', 'position', 'yaw', 'speed', 'accel', 'yaw_rate']
            for key in required_keys:
                if key not in data:
                    errors.append(f"{clip_id}: Missing array: {key}")
                    return errors

            # Extract arrays
            timestamps = data['timestamps']
            position = data['position']
            yaw = data['yaw']
            speed = data['speed']
            accel = data['accel']
            yaw_rate = data['yaw_rate']
            jerk = data.get('jerk')

            # Check for NaN/Inf
            for key, arr in [('timestamps', timestamps), ('position', position),
                            ('yaw', yaw), ('speed', speed), ('accel', accel),
                            ('yaw_rate', yaw_rate)]:
                if not np.all(np.isfinite(arr)):
                    errors.append(f"{clip_id}: Array {key} contains NaN or Inf")

            # Check shapes
            n_samples = len(timestamps)

            if position.shape != (n_samples, 2):
                errors.append(
                    f"{clip_id}: Position shape {position.shape} "
                    f"doesn't match expected ({n_samples}, 2)"
                )

            for key, arr in [('yaw', yaw), ('speed', speed),
                            ('accel', accel), ('yaw_rate', yaw_rate)]:
                if arr.shape != (n_samples,):
                    errors.append(
                        f"{clip_id}: Array {key} shape {arr.shape} "
                        f"doesn't match expected ({n_samples},)"
                    )

            # Check timestamps are monotonic
            if not np.all(np.diff(timestamps) > 0):
                errors.append(f"{clip_id}: Timestamps are not strictly increasing")

            # Check timestamp alignment with clip bounds
            # Timestamps should be relative to t_start (starting at ~0)
            if abs(timestamps[0]) > self.timestamp_tolerance:
                errors.append(
                    f"{clip_id}: First timestamp {timestamps[0]:.3f} "
                    f"not close to 0 (tolerance: {self.timestamp_tolerance}s)"
                )

            expected_duration = clip['t_end'] - clip['t_start']
            actual_duration = timestamps[-1] - timestamps[0]
            if abs(actual_duration - expected_duration) > self.timestamp_tolerance:
                errors.append(
                    f"{clip_id}: Array duration {actual_duration:.3f}s "
                    f"doesn't match clip duration {expected_duration:.3f}s "
                    f"(tolerance: {self.timestamp_tolerance}s)"
                )

            # Accumulate for distribution summary (only if valid)
            if len(errors) == 0:
                self.all_speeds.extend(speed.tolist())
                self.all_accels.extend(accel.tolist())
                self.all_yaw_rates.extend(yaw_rate.tolist())
                if jerk is not None:
                    self.all_jerks.extend(jerk.tolist())

        except Exception as e:
            errors.append(f"{clip_id}: Failed to load/validate arrays: {e}")

        return errors

    def _print_distribution_summary(self):
        """Print distribution summaries for speed, accel, yaw_rate, jerk."""
        logger.info("\n" + "="*60)
        logger.info("DISTRIBUTION SUMMARY")
        logger.info("="*60)

        def print_percentiles(name: str, values: List[float]):
            if not values:
                logger.info(f"{name}: No data")
                return

            arr = np.array(values)
            logger.info(f"\n{name}:")
            logger.info(f"  Count: {len(arr)}")
            logger.info(f"  Mean:  {np.mean(arr):.3f}")
            logger.info(f"  Std:   {np.std(arr):.3f}")
            logger.info(f"  Min:   {np.min(arr):.3f}")
            logger.info(f"  P05:   {np.percentile(arr, 5):.3f}")
            logger.info(f"  P25:   {np.percentile(arr, 25):.3f}")
            logger.info(f"  P50:   {np.percentile(arr, 50):.3f}")
            logger.info(f"  P75:   {np.percentile(arr, 75):.3f}")
            logger.info(f"  P95:   {np.percentile(arr, 95):.3f}")
            logger.info(f"  Max:   {np.max(arr):.3f}")

        print_percentiles("Speed (m/s)", self.all_speeds)
        print_percentiles("Acceleration (m/s²)", self.all_accels)
        print_percentiles("Yaw Rate (rad/s)", self.all_yaw_rates)
        print_percentiles("Jerk (m/s³)", self.all_jerks)

        logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted nuScenes clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        required=True,
        help="Directory containing clips_index.jsonl and arrays/",
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default=None,
        help="Optional path to nuScenes root for frame validation",
    )
    parser.add_argument(
        "--timestamp_tolerance",
        type=float,
        default=0.1,
        help="Tolerance for timestamp alignment checks (seconds)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all errors (not just summary)",
    )

    args = parser.parse_args()

    # Resolve clips_dir: if the user passes a .jsonl file, use its parent
    clips_dir = Path(args.clips_dir)
    if clips_dir.is_file() and clips_dir.suffix == ".jsonl":
        logger.info(f"Detected JSONL file; using parent directory: {clips_dir.parent}")
        clips_dir = clips_dir.parent

    # Initialize validator
    validator = ClipValidator(
        clips_dir=clips_dir,
        nuscenes_root=args.nuscenes_root,
        timestamp_tolerance=args.timestamp_tolerance,
    )

    # Run validation
    num_valid, num_invalid, errors = validator.validate_all()

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Valid clips:   {num_valid}")
    logger.info(f"Invalid clips: {num_invalid}")
    logger.info(f"Total errors:  {len(errors)}")

    if errors and args.verbose:
        logger.info("\n" + "="*60)
        logger.info("ERRORS")
        logger.info("="*60)
        for error in errors[:100]:  # Limit to first 100 errors
            logger.info(f"  {error}")
        if len(errors) > 100:
            logger.info(f"  ... and {len(errors) - 100} more errors")

    # Exit with error code if validation failed or no clips were found
    if num_invalid > 0 or len(errors) > 0:
        logger.error("Validation failed!")
        return 1

    if num_valid == 0:
        logger.error("Validation failed: no clips were found to validate")
        return 1

    logger.info("Validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
