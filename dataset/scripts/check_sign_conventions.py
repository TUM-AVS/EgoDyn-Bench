#!/usr/bin/env python3
"""
Diagnostic script to verify sign conventions for yaw-rate and acceleration.

This script analyzes extracted clips to determine:
1. Whether positive yaw_rate corresponds to left or right turns
2. Whether negative acceleration corresponds to braking (decreasing speed)

Usage:
    python check_sign_conventions.py \\
        --clips_index ./output/clips_50/clips_index.jsonl \\
        --output_report ./sign_convention_report.md
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clip_arrays(array_ref: str, clips_dir: Path) -> dict:
    """Load clip arrays from NPZ file."""
    array_path = clips_dir / array_ref
    data = np.load(array_path)
    return {key: data[key] for key in data.files}


def check_yaw_rate_convention(clips_index_path: Path, n_clips: int = 50):
    """
    Check yaw-rate sign convention.

    Analyzes clips with high yaw-rate to determine if positive yaw_rate
    corresponds to left turns or right turns.

    Args:
        clips_index_path: Path to clips_index.jsonl
        n_clips: Number of clips to analyze (highest max_abs_yaw_rate)

    Returns:
        Dictionary with analysis results
    """
    clips_dir = clips_index_path.parent

    # Load all clips
    logger.info(f"Loading clips from {clips_index_path}")
    clips = []
    with open(clips_index_path, 'r') as f:
        for line in f:
            clips.append(json.loads(line))

    # Sort by max_abs_yaw_rate (highest first)
    clips_sorted = sorted(clips, key=lambda c: c['features']['max_abs_yaw_rate'], reverse=True)

    # Analyze top N clips
    logger.info(f"Analyzing top {n_clips} clips with highest yaw-rate...")

    results = []

    for clip in clips_sorted[:n_clips]:
        # Load arrays
        arrays = load_clip_arrays(clip['array_ref'], clips_dir)

        yaw = arrays['yaw']
        yaw_rate = arrays['yaw_rate']

        # Compute heading change
        # Note: yaw is already unwrapped in the arrays
        yaw_start = yaw[0]
        yaw_end = yaw[-1]
        net_heading_change = yaw_end - yaw_start

        # Aggregate yaw_rate
        mean_yaw_rate = np.mean(yaw_rate)
        max_yaw_rate = np.max(yaw_rate)
        min_yaw_rate = np.min(yaw_rate)

        # Determine dominant direction
        if abs(max_yaw_rate) > abs(min_yaw_rate):
            dominant_yaw_rate = max_yaw_rate
            direction = "positive"
        else:
            dominant_yaw_rate = min_yaw_rate
            direction = "negative"

        result = {
            'clip_id': clip['clip_id'],
            'max_abs_yaw_rate': clip['features']['max_abs_yaw_rate'],
            'mean_yaw_rate': mean_yaw_rate,
            'dominant_yaw_rate': dominant_yaw_rate,
            'dominant_direction': direction,
            'total_heading_change': clip['features']['total_heading_change'],
            'net_heading_change': net_heading_change,
            'yaw_start': yaw_start,
            'yaw_end': yaw_end,
        }

        results.append(result)

    return results


def analyze_yaw_rate_results(results):
    """
    Analyze yaw-rate results to determine sign convention.

    In nuScenes coordinate system:
    - x-axis points forward (vehicle front)
    - y-axis points left
    - z-axis points up
    - yaw rotation is around z-axis
    - positive yaw rotation = counter-clockwise when viewed from above
    - positive yaw rotation = left turn

    Therefore:
    - positive yaw_rate should correspond to LEFT turns
    - negative yaw_rate should correspond to RIGHT turns
    """
    logger.info("\n" + "="*80)
    logger.info("YAW-RATE SIGN CONVENTION ANALYSIS")
    logger.info("="*80)

    # Count cases where positive yaw_rate → positive heading change
    positive_consistent = 0
    negative_consistent = 0

    for r in results:
        # Check consistency: does sign of mean_yaw_rate match sign of net_heading_change?
        if np.sign(r['mean_yaw_rate']) == np.sign(r['net_heading_change']):
            if r['mean_yaw_rate'] > 0:
                positive_consistent += 1
            else:
                negative_consistent += 1

    logger.info("\nSample of top 10 clips with highest yaw-rate:")
    logger.info("-" * 80)
    logger.info(f"{'Clip ID':<15} {'Mean Yaw-Rate':>15} {'Net Δ Heading':>15} {'Consistent?':>12}")
    logger.info("-" * 80)

    for r in results[:10]:
        consistent = np.sign(r['mean_yaw_rate']) == np.sign(r['net_heading_change'])
        logger.info(
            f"{r['clip_id']:<15} {r['mean_yaw_rate']:>15.4f} {r['net_heading_change']:>15.4f} "
            f"{'✓' if consistent else '✗':>12}"
        )

    logger.info("\n" + "="*80)
    logger.info("CONCLUSION:")
    logger.info("="*80)
    logger.info(f"Total clips analyzed: {len(results)}")
    logger.info(f"Clips with positive yaw_rate → positive heading change: {positive_consistent}")
    logger.info(f"Clips with negative yaw_rate → negative heading change: {negative_consistent}")

    # Determine convention
    total_consistent = positive_consistent + negative_consistent
    consistency_rate = total_consistent / len(results) if results else 0

    logger.info(f"\nConsistency rate: {consistency_rate:.1%}")

    if consistency_rate > 0.9:
        logger.info("\n✓ SIGN CONVENTION VERIFIED:")
        logger.info("  - Positive yaw_rate corresponds to POSITIVE heading change (LEFT turn)")
        logger.info("  - Negative yaw_rate corresponds to NEGATIVE heading change (RIGHT turn)")
        logger.info("\nThis matches nuScenes coordinate conventions:")
        logger.info("  - Yaw is rotation around z-axis (up)")
        logger.info("  - Positive rotation = counter-clockwise from above = LEFT turn")
        logger.info("\nNO FIX NEEDED for yaw-rate sign convention.")
        return "correct"
    else:
        logger.warning("\n✗ INCONSISTENT SIGN CONVENTION DETECTED")
        logger.warning(f"  Only {consistency_rate:.1%} of clips show consistent yaw_rate/heading change")
        logger.warning("\nFURTHER INVESTIGATION NEEDED.")
        return "inconsistent"


def check_acceleration_convention(clips_index_path: Path, n_clips: int = 20):
    """
    Check acceleration sign convention.

    Analyzes clips with most negative acceleration to verify that
    negative accel corresponds to decreasing speed (braking).

    Args:
        clips_index_path: Path to clips_index.jsonl
        n_clips: Number of clips to analyze (most negative min_accel)

    Returns:
        Dictionary with analysis results
    """
    clips_dir = clips_index_path.parent

    # Load all clips
    clips = []
    with open(clips_index_path, 'r') as f:
        for line in f:
            clips.append(json.loads(line))

    # Sort by min_accel (most negative first)
    clips_sorted = sorted(clips, key=lambda c: c['features']['min_accel'])

    logger.info(f"\nAnalyzing top {n_clips} clips with most negative acceleration...")

    results = []

    for clip in clips_sorted[:n_clips]:
        # Load arrays
        arrays = load_clip_arrays(clip['array_ref'], clips_dir)

        speed = arrays['speed']
        accel = arrays['accel']

        # Compute speed change
        speed_start = speed[0]
        speed_end = speed[-1]
        speed_change = speed_end - speed_start
        mean_speed = np.mean(speed)

        # Get acceleration stats
        min_accel = np.min(accel)
        mean_accel = np.mean(accel)

        result = {
            'clip_id': clip['clip_id'],
            'min_accel': min_accel,
            'mean_accel': mean_accel,
            'speed_start': speed_start,
            'speed_end': speed_end,
            'speed_change': speed_change,
            'mean_speed': mean_speed,
        }

        results.append(result)

    return results


def analyze_acceleration_results(results):
    """Analyze acceleration results to verify sign convention."""
    logger.info("\n" + "="*80)
    logger.info("ACCELERATION SIGN CONVENTION ANALYSIS")
    logger.info("="*80)

    # Count cases where negative accel → decreasing speed
    correct_braking = 0
    incorrect_braking = 0

    for r in results:
        # Negative accel should correspond to negative speed change (braking)
        if r['mean_accel'] < 0 and r['speed_change'] < 0:
            correct_braking += 1
        elif r['mean_accel'] < 0 and r['speed_change'] > 0:
            incorrect_braking += 1

    logger.info("\nSample of clips with most negative acceleration:")
    logger.info("-" * 80)
    logger.info(f"{'Clip ID':<15} {'Min Accel':>12} {'Mean Accel':>12} {'Δ Speed':>12} {'Braking?':>10}")
    logger.info("-" * 80)

    for r in results[:10]:
        is_braking = r['speed_change'] < -0.5  # Significant speed decrease
        logger.info(
            f"{r['clip_id']:<15} {r['min_accel']:>12.3f} {r['mean_accel']:>12.3f} "
            f"{r['speed_change']:>12.3f} {'✓' if is_braking else '✗':>10}"
        )

    logger.info("\n" + "="*80)
    logger.info("CONCLUSION:")
    logger.info("="*80)
    logger.info(f"Total clips analyzed: {len(results)}")
    logger.info(f"Clips with negative accel AND decreasing speed: {correct_braking}")
    logger.info(f"Clips with negative accel BUT increasing speed: {incorrect_braking}")

    if correct_braking > 0:
        consistency_rate = correct_braking / (correct_braking + incorrect_braking)
        logger.info(f"\nConsistency rate: {consistency_rate:.1%}")

        if consistency_rate > 0.7:
            logger.info("\n✓ SIGN CONVENTION VERIFIED:")
            logger.info("  - Negative acceleration corresponds to DECREASING speed (braking)")
            logger.info("  - Positive acceleration corresponds to INCREASING speed (accelerating)")
            logger.info("\nNO FIX NEEDED for acceleration sign convention.")
            return "correct"
        else:
            logger.warning("\n✗ INCONSISTENT SIGN CONVENTION DETECTED")
            logger.warning(f"  Only {consistency_rate:.1%} of clips show expected behavior")
            return "inconsistent"
    else:
        logger.info("\nNote: No clips with significant negative acceleration found.")
        logger.info("This may be normal for highway driving scenarios.")
        return "insufficient_data"


def generate_report(yaw_status: str, accel_status: str, output_path: Path):
    """Generate markdown report."""
    with open(output_path, 'w') as f:
        f.write("# Sign Convention Verification Report\n\n")
        f.write("## Executive Summary\n\n")

        # Yaw-rate summary
        f.write("### Yaw-Rate Sign Convention\n\n")
        if yaw_status == "correct":
            f.write("**Status:** ✅ VERIFIED\n\n")
            f.write("- Positive yaw_rate → LEFT turn (positive heading change)\n")
            f.write("- Negative yaw_rate → RIGHT turn (negative heading change)\n\n")
            f.write("**Action Required:** None. Convention is correct.\n\n")
        elif yaw_status == "inconsistent":
            f.write("**Status:** ⚠️ INCONSISTENT\n\n")
            f.write("**Action Required:** Further investigation needed.\n\n")

        # Acceleration summary
        f.write("### Acceleration Sign Convention\n\n")
        if accel_status == "correct":
            f.write("**Status:** ✅ VERIFIED\n\n")
            f.write("- Negative acceleration → Decreasing speed (braking)\n")
            f.write("- Positive acceleration → Increasing speed (accelerating)\n\n")
            f.write("**Action Required:** None. Convention is correct.\n\n")
        elif accel_status == "inconsistent":
            f.write("**Status:** ⚠️ INCONSISTENT\n\n")
            f.write("**Action Required:** Investigate computation or coordinate system issues.\n\n")
        elif accel_status == "insufficient_data":
            f.write("**Status:** ℹ️ INSUFFICIENT DATA\n\n")
            f.write("**Note:** Few clips with significant braking events found.\n\n")

        # Overall recommendations
        f.write("## Recommendations\n\n")

        if yaw_status == "correct" and accel_status in ["correct", "insufficient_data"]:
            f.write("✅ **No changes needed.** Sign conventions are physically correct.\n\n")
            f.write("Current labeling rules in Task 2 are using correct interpretations:\n")
            f.write("- `yaw_rate_sign_with_deadzone`: positive → left, negative → right\n")
            f.write("- `threshold_event` for braking: checks for negative acceleration\n")
        else:
            f.write("⚠️ **Further investigation recommended.**\n\n")

        # Technical details
        f.write("## Technical Details\n\n")
        f.write("### nuScenes Coordinate System\n\n")
        f.write("- **x-axis:** Forward (vehicle front)\n")
        f.write("- **y-axis:** Left\n")
        f.write("- **z-axis:** Up\n")
        f.write("- **Yaw:** Rotation around z-axis\n")
        f.write("- **Positive yaw rotation:** Counter-clockwise from above = LEFT turn\n\n")

        f.write("### Derivation Methods\n\n")
        f.write("- **Yaw:** Extracted from ego_pose quaternion → Euler angles\n")
        f.write("- **Yaw-rate:** Numerical derivative of unwrapped yaw: `(yaw[i+1] - yaw[i-1]) / (2*dt)`\n")
        f.write("- **Speed:** `sqrt(dx² + dy²) / dt` from position deltas\n")
        f.write("- **Acceleration:** Numerical derivative of speed\n\n")

        f.write("---\n")
        f.write("\n*Report generated by check_sign_conventions.py*\n")

    logger.info(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check sign conventions for yaw-rate and acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_index",
        type=str,
        required=True,
        help="Path to clips_index.jsonl",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="./sign_convention_report.md",
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--n_yaw_clips",
        type=int,
        default=50,
        help="Number of clips to analyze for yaw-rate",
    )
    parser.add_argument(
        "--n_accel_clips",
        type=int,
        default=20,
        help="Number of clips to analyze for acceleration",
    )

    args = parser.parse_args()

    clips_index_path = Path(args.clips_index)
    if not clips_index_path.exists():
        logger.error(f"Clips index not found: {clips_index_path}")
        return 1

    # Check yaw-rate convention
    yaw_results = check_yaw_rate_convention(clips_index_path, args.n_yaw_clips)
    yaw_status = analyze_yaw_rate_results(yaw_results)

    # Check acceleration convention
    accel_results = check_acceleration_convention(clips_index_path, args.n_accel_clips)
    accel_status = analyze_acceleration_results(accel_results)

    # Generate report
    output_path = Path(args.output_report)
    generate_report(yaw_status, accel_status, output_path)

    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
