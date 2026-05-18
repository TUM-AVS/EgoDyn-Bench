#!/usr/bin/env python3
"""
CLI script to build train/val splits from clips and QA data.

Usage:
    python build_splits.py \\
        --clips_index ./output/clips/clips_index.jsonl \\
        --qa_jsonl ./output/qa.jsonl \\
        --output_dir ./output/splits \\
        --seed 42 \\
        --num_val_clips 500 \\
        --num_train_clips 3000

Output:
    - train_clips.jsonl, val_clips.jsonl
    - train_qa.jsonl, val_qa.jsonl
    - split_metadata.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.split_builder import SplitBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build train/val splits from clips and QA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_index",
        type=str,
        required=True,
        help="Path to clips_index.jsonl",
    )
    parser.add_argument(
        "--qa_jsonl",
        type=str,
        required=True,
        help="Path to qa.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for split files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_val_clips",
        type=int,
        default=500,
        help="Target number of validation clips",
    )
    parser.add_argument(
        "--num_train_clips",
        type=int,
        default=3000,
        help="Target number of training clips",
    )
    parser.add_argument(
        "--min_per_bin",
        type=int,
        default=10,
        help="Minimum clips per stratification bin (for validation sampling)",
    )
    parser.add_argument(
        "--balance_strategy",
        type=str,
        default="qa_tags_stratified",
        help="Balancing strategy (currently only 'qa_tags_stratified' supported)",
    )
    parser.add_argument(
        "--tag_turn_qid",
        type=str,
        default="yaw_rate_turn_direction",
        help="Question ID for turning tag",
    )
    parser.add_argument(
        "--tag_braking_qid",
        type=str,
        default="braking_intensity",
        help="Question ID for braking tag",
    )
    parser.add_argument(
        "--tag_aggressive_qid",
        type=str,
        default="driving_smoothness",
        help="Question ID for aggressive driving tag",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Validation ratio (e.g. 0.2 for 80/20 split). "
             "Overrides --num_val_clips and --num_train_clips.",
    )
    parser.add_argument(
        "--exclude_scenes_from",
        type=str,
        default=None,
        help="Path to a directory of CARLA logs or a clips_index.jsonl to "
             "extract scene names from. Clips from these scenes will be "
             "excluded (e.g. to avoid overlap with benchmark data).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print statistics without writing files",
    )

    args = parser.parse_args()

    # Validate paths
    clips_index_path = Path(args.clips_index)
    if not clips_index_path.exists():
        logger.error(f"Clips index not found: {clips_index_path}")
        return 1

    qa_jsonl_path = Path(args.qa_jsonl)
    if not qa_jsonl_path.exists():
        logger.error(f"QA file not found: {qa_jsonl_path}")
        return 1

    output_dir = Path(args.output_dir)

    # Validate balance strategy
    if args.balance_strategy != "qa_tags_stratified":
        logger.error(f"Unknown balance strategy: {args.balance_strategy}")
        logger.error("Only 'qa_tags_stratified' is currently supported")
        return 1

    # Resolve scenes to exclude
    exclude_scenes = None
    if args.exclude_scenes_from:
        exclude_path = Path(args.exclude_scenes_from)
        exclude_scenes = set()

        if exclude_path.is_dir():
            # Directory of scene subdirectories (e.g. frenetix_logs/)
            exclude_scenes = {
                d for d in os.listdir(exclude_path)
                if os.path.isdir(os.path.join(exclude_path, d))
            }
        elif exclude_path.suffix == '.jsonl':
            # clips_index.jsonl — extract unique scene names
            with open(exclude_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if 'scene' in rec:
                        exclude_scenes.add(rec['scene'])
        elif exclude_path.suffix == '.json':
            # selected_clips.json — extract scene names from clip IDs
            with open(exclude_path) as f:
                data = json.load(f)
                for item in data:
                    clip_id = item['id'] if isinstance(item, dict) else item
                    # Scene is everything before the first "__"
                    scene = clip_id.split("__")[0]
                    exclude_scenes.add(scene)
        else:
            logger.error(f"Unsupported exclude format: {exclude_path}")
            return 1

        logger.info(f"Will exclude {len(exclude_scenes)} scenes from: {exclude_path}")

    # Validate ratio
    if args.val_ratio is not None:
        if not 0.0 < args.val_ratio < 1.0:
            logger.error(f"--val_ratio must be between 0 and 1, got {args.val_ratio}")
            return 1

    # Build tag question IDs
    tag_question_ids = {
        'turn': args.tag_turn_qid,
        'braking': args.tag_braking_qid,
        'aggressive': args.tag_aggressive_qid
    }

    # Initialize split builder
    logger.info("Initializing split builder...")
    builder = SplitBuilder(
        seed=args.seed,
        tag_question_ids=tag_question_ids
    )

    # Load data
    try:
        clips, qa_items = builder.load_data(
            clips_index_path=str(clips_index_path),
            qa_jsonl_path=str(qa_jsonl_path)
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Build splits
    try:
        split_data = builder.build_splits(
            clips=clips,
            qa_items=qa_items,
            num_val_clips=args.num_val_clips,
            num_train_clips=args.num_train_clips,
            min_per_bin=args.min_per_bin,
            val_ratio=args.val_ratio,
            exclude_scenes=exclude_scenes,
        )
    except Exception as e:
        logger.error(f"Failed to build splits: {e}", exc_info=True)
        return 1

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SPLIT STATISTICS")
    logger.info("="*80)

    metadata = split_data['metadata']
    counts = metadata['actual_counts']

    logger.info("\nInput:")
    logger.info(f"  Total clips: {counts['total_clips']}")
    logger.info(f"  Total QA items: {counts['total_qa']}")
    logger.info(f"  Eligible clips (with QA): {counts['eligible_clips']}")
    logger.info(f"  Dropped (no QA): {counts['dropped_clips_no_qa']}")

    logger.info("\nOutput:")
    logger.info(f"  Validation: {counts['val_clips']} clips, {counts['val_qa']} QA items")
    logger.info(f"  Training: {counts['train_clips']} clips, {counts['train_qa']} QA items")

    logger.info("\nStratification bins:")
    for bin_key, bin_data in sorted(metadata['stratification_bins'].items()):
        logger.info(f"  Bin {bin_key}:")
        logger.info(f"    Total: {bin_data['total']} clips")
        logger.info(f"    Sampled (min): {bin_data['sampled_min']}")
        logger.info(f"    Sampled (proportional): {bin_data['sampled_proportional']}")
        logger.info(f"    Sampled (total): {bin_data['sampled_total']}")

    logger.info("="*80)

    # Write files (unless dry run)
    if args.dry_run:
        logger.info("\nDRY RUN: Not writing files")
        logger.info("Remove --dry_run flag to write split files")
    else:
        try:
            builder.write_splits(output_dir, split_data)
            logger.info(f"\nSplit files written to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to write splits: {e}", exc_info=True)
            return 1

    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
