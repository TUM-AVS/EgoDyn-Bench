#!/usr/bin/env python3
"""
Validation suite for dataset splits.

Verifies:
- No clip_id overlap between train and val
- All QA items assigned to exactly one split
- Distribution summaries (clips, QA, answers, dynamics tags)
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SplitValidator:
    """Validates dataset splits for correctness and balance."""

    def __init__(self, splits_dir: Path):
        """
        Initialize split validator.

        Args:
            splits_dir: Directory containing split files
        """
        self.splits_dir = Path(splits_dir)

        # Expected file paths
        self.train_clips_path = self.splits_dir / 'train_clips.jsonl'
        self.val_clips_path = self.splits_dir / 'val_clips.jsonl'
        self.train_qa_path = self.splits_dir / 'train_qa.jsonl'
        self.val_qa_path = self.splits_dir / 'val_qa.jsonl'
        self.metadata_path = self.splits_dir / 'split_metadata.json'

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Run all validation checks.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        logger.info(f"Validating splits in {self.splits_dir}")

        errors = []

        # Check all files exist
        required_files = [
            self.train_clips_path,
            self.val_clips_path,
            self.train_qa_path,
            self.val_qa_path,
            self.metadata_path
        ]

        for file_path in required_files:
            if not file_path.exists():
                errors.append(f"Missing required file: {file_path}")

        if errors:
            return False, errors

        # Load data
        try:
            train_clips = self._load_jsonl(self.train_clips_path)
            val_clips = self._load_jsonl(self.val_clips_path)
            train_qa = self._load_jsonl(self.train_qa_path)
            val_qa = self._load_jsonl(self.val_qa_path)

            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            errors.append(f"Failed to load split files: {e}")
            return False, errors

        # Run validation checks
        errors.extend(self._check_clip_overlap(train_clips, val_clips))
        errors.extend(self._check_qa_assignment(train_qa, val_qa))
        errors.extend(self._check_split_field(train_clips, val_clips, train_qa, val_qa))
        errors.extend(self._validate_metadata(metadata, train_clips, val_clips, train_qa, val_qa))

        # Print distribution summaries
        if not errors:
            self._print_distribution_summary(
                train_clips, val_clips, train_qa, val_qa, metadata
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def _load_jsonl(self, path: Path) -> List[dict]:
        """Load JSONL file into list of records."""
        records = []
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def _check_clip_overlap(
        self,
        train_clips: List[dict],
        val_clips: List[dict]
    ) -> List[str]:
        """Check for clip_id overlap between train and val."""
        errors = []

        train_clip_ids = set(c['clip_id'] for c in train_clips)
        val_clip_ids = set(c['clip_id'] for c in val_clips)

        overlap = train_clip_ids & val_clip_ids

        if overlap:
            errors.append(
                f"Found {len(overlap)} clip_id(s) in both train and val: "
                f"{list(overlap)[:5]}"
            )

        logger.info(f"Clip overlap check: {len(overlap)} overlapping clips")
        return errors

    def _check_qa_assignment(
        self,
        train_qa: List[dict],
        val_qa: List[dict]
    ) -> List[str]:
        """Check that all QA items are assigned to exactly one split."""
        errors = []

        # Check for duplicate qa_id
        train_qa_ids = set(qa['qa_id'] for qa in train_qa)
        val_qa_ids = set(qa['qa_id'] for qa in val_qa)

        overlap = train_qa_ids & val_qa_ids

        if overlap:
            errors.append(
                f"Found {len(overlap)} qa_id(s) in both train and val: "
                f"{list(overlap)[:5]}"
            )

        # Check for duplicate qa_id within each split
        train_duplicates = len(train_qa) - len(train_qa_ids)
        val_duplicates = len(val_qa) - len(val_qa_ids)

        if train_duplicates > 0:
            errors.append(f"Found {train_duplicates} duplicate qa_id(s) in train split")

        if val_duplicates > 0:
            errors.append(f"Found {val_duplicates} duplicate qa_id(s) in val split")

        logger.info(
            f"QA assignment check: {len(overlap)} overlapping QA, "
            f"{train_duplicates} train dups, {val_duplicates} val dups"
        )
        return errors

    def _check_split_field(
        self,
        train_clips: List[dict],
        val_clips: List[dict],
        train_qa: List[dict],
        val_qa: List[dict]
    ) -> List[str]:
        """Check that split field is set correctly."""
        errors = []

        # Check train clips have split='train'
        train_clip_wrong_split = [
            c['clip_id'] for c in train_clips if c.get('split') != 'train'
        ]
        if train_clip_wrong_split:
            errors.append(
                f"Found {len(train_clip_wrong_split)} train clips with wrong split field: "
                f"{train_clip_wrong_split[:5]}"
            )

        # Check val clips have split='val'
        val_clip_wrong_split = [
            c['clip_id'] for c in val_clips if c.get('split') != 'val'
        ]
        if val_clip_wrong_split:
            errors.append(
                f"Found {len(val_clip_wrong_split)} val clips with wrong split field: "
                f"{val_clip_wrong_split[:5]}"
            )

        # Check train QA have split='train'
        train_qa_wrong_split = [
            qa['qa_id'] for qa in train_qa if qa.get('split') != 'train'
        ]
        if train_qa_wrong_split:
            errors.append(
                f"Found {len(train_qa_wrong_split)} train QA with wrong split field: "
                f"{train_qa_wrong_split[:5]}"
            )

        # Check val QA have split='val'
        val_qa_wrong_split = [
            qa['qa_id'] for qa in val_qa if qa.get('split') != 'val'
        ]
        if val_qa_wrong_split:
            errors.append(
                f"Found {len(val_qa_wrong_split)} val QA with wrong split field: "
                f"{val_qa_wrong_split[:5]}"
            )

        logger.info(
            f"Split field check: {len(train_clip_wrong_split)} train clip errors, "
            f"{len(val_clip_wrong_split)} val clip errors, "
            f"{len(train_qa_wrong_split)} train QA errors, "
            f"{len(val_qa_wrong_split)} val QA errors"
        )
        return errors

    def _validate_metadata(
        self,
        metadata: dict,
        train_clips: List[dict],
        val_clips: List[dict],
        train_qa: List[dict],
        val_qa: List[dict]
    ) -> List[str]:
        """Validate metadata counts match actual data."""
        errors = []

        actual_counts = metadata.get('actual_counts', {})

        # Check clip counts
        actual_train_clips = len(train_clips)
        actual_val_clips = len(val_clips)

        if actual_counts.get('train_clips') != actual_train_clips:
            errors.append(
                f"Metadata train_clips count mismatch: "
                f"{actual_counts.get('train_clips')} != {actual_train_clips}"
            )

        if actual_counts.get('val_clips') != actual_val_clips:
            errors.append(
                f"Metadata val_clips count mismatch: "
                f"{actual_counts.get('val_clips')} != {actual_val_clips}"
            )

        # Check QA counts
        actual_train_qa = len(train_qa)
        actual_val_qa = len(val_qa)

        if actual_counts.get('train_qa') != actual_train_qa:
            errors.append(
                f"Metadata train_qa count mismatch: "
                f"{actual_counts.get('train_qa')} != {actual_train_qa}"
            )

        if actual_counts.get('val_qa') != actual_val_qa:
            errors.append(
                f"Metadata val_qa count mismatch: "
                f"{actual_counts.get('val_qa')} != {actual_val_qa}"
            )

        logger.info(f"Metadata validation: {len(errors)} errors")
        return errors

    def _print_distribution_summary(
        self,
        train_clips: List[dict],
        val_clips: List[dict],
        train_qa: List[dict],
        val_qa: List[dict],
        metadata: dict
    ):
        """Print comprehensive distribution summaries."""
        logger.info("\n" + "="*80)
        logger.info("DISTRIBUTION SUMMARY")
        logger.info("="*80)

        # Overall counts
        logger.info(f"\n{'Split':<10} {'Clips':<10} {'QA Items':<12} {'QA/Clip':<10}")
        logger.info("-" * 80)

        train_qa_per_clip = len(train_qa) / len(train_clips) if train_clips else 0
        val_qa_per_clip = len(val_qa) / len(val_clips) if val_clips else 0

        logger.info(f"{'Train':<10} {len(train_clips):<10} {len(train_qa):<12} {train_qa_per_clip:<10.1f}")
        logger.info(f"{'Val':<10} {len(val_clips):<10} {len(val_qa):<12} {val_qa_per_clip:<10.1f}")
        logger.info(f"{'Total':<10} {len(train_clips)+len(val_clips):<10} {len(train_qa)+len(val_qa):<12}")

        # QA by category
        logger.info("\n" + "-"*80)
        logger.info("QA DISTRIBUTION BY CATEGORY")
        logger.info("-"*80)

        train_categories = Counter(qa['category'] for qa in train_qa)
        val_categories = Counter(qa['category'] for qa in val_qa)
        all_categories = sorted(set(train_categories.keys()) | set(val_categories.keys()))

        logger.info(f"\n{'Category':<30} {'Train':<12} {'Val':<12} {'Total':<12}")
        logger.info("-" * 80)

        for cat in all_categories:
            train_count = train_categories.get(cat, 0)
            val_count = val_categories.get(cat, 0)
            total_count = train_count + val_count

            logger.info(f"{cat:<30} {train_count:<12} {val_count:<12} {total_count:<12}")

        # QA by question_id
        logger.info("\n" + "-"*80)
        logger.info("QA DISTRIBUTION BY QUESTION_ID")
        logger.info("-"*80)

        train_questions = Counter(qa['question_id'] for qa in train_qa)
        val_questions = Counter(qa['question_id'] for qa in val_qa)
        all_questions = sorted(set(train_questions.keys()) | set(val_questions.keys()))

        logger.info(f"\n{'Question ID':<35} {'Train':<12} {'Val':<12} {'Total':<12}")
        logger.info("-" * 80)

        for qid in all_questions:
            train_count = train_questions.get(qid, 0)
            val_count = val_questions.get(qid, 0)
            total_count = train_count + val_count

            logger.info(f"{qid:<35} {train_count:<12} {val_count:<12} {total_count:<12}")

        # Answer distributions per question_id
        logger.info("\n" + "-"*80)
        logger.info("ANSWER DISTRIBUTIONS (Train vs Val)")
        logger.info("-"*80)

        train_answers_by_question = defaultdict(list)
        val_answers_by_question = defaultdict(list)

        for qa in train_qa:
            train_answers_by_question[qa['question_id']].append(qa['answer'])

        for qa in val_qa:
            val_answers_by_question[qa['question_id']].append(qa['answer'])

        for qid in sorted(all_questions):
            train_answers = train_answers_by_question.get(qid, [])
            val_answers = val_answers_by_question.get(qid, [])

            # Get answer_type
            answer_type = None
            for qa in train_qa + val_qa:
                if qa['question_id'] == qid:
                    answer_type = qa['answer_type']
                    break

            logger.info(f"\n  {qid} ({answer_type}):")

            if answer_type == "numeric":
                # Show numeric statistics
                if train_answers:
                    train_numeric = [a for a in train_answers if isinstance(a, (int, float))]
                    if train_numeric:
                        logger.info(f"    Train: n={len(train_numeric)}, "
                                  f"mean={np.mean(train_numeric):.2f}, "
                                  f"std={np.std(train_numeric):.2f}, "
                                  f"range=[{np.min(train_numeric):.2f}, {np.max(train_numeric):.2f}]")

                if val_answers:
                    val_numeric = [a for a in val_answers if isinstance(a, (int, float))]
                    if val_numeric:
                        logger.info(f"    Val:   n={len(val_numeric)}, "
                                  f"mean={np.mean(val_numeric):.2f}, "
                                  f"std={np.std(val_numeric):.2f}, "
                                  f"range=[{np.min(val_numeric):.2f}, {np.max(val_numeric):.2f}]")
            else:
                # Show categorical distribution
                train_counts = Counter(train_answers)
                val_counts = Counter(val_answers)
                all_answers = sorted(set(train_counts.keys()) | set(val_counts.keys()))

                for answer in all_answers:
                    train_cnt = train_counts.get(answer, 0)
                    val_cnt = val_counts.get(answer, 0)
                    train_pct = 100 * train_cnt / len(train_answers) if train_answers else 0
                    val_pct = 100 * val_cnt / len(val_answers) if val_answers else 0

                    logger.info(
                        f"    {answer}: "
                        f"train={train_cnt} ({train_pct:.1f}%), "
                        f"val={val_cnt} ({val_pct:.1f}%)"
                    )

        # Dynamics tag coverage
        logger.info("\n" + "-"*80)
        logger.info("DYNAMICS TAG COVERAGE")
        logger.info("-"*80)

        if 'tag_question_ids' in metadata:
            tag_qids = metadata['tag_question_ids']

            logger.info(f"\nTag question IDs:")
            logger.info(f"  Turn:       {tag_qids.get('turn', 'N/A')}")
            logger.info(f"  Braking:    {tag_qids.get('braking', 'N/A')}")
            logger.info(f"  Aggressive: {tag_qids.get('aggressive', 'N/A')}")

            # Derive tags from QA
            train_tags = self._derive_tags_from_qa(train_qa, tag_qids)
            val_tags = self._derive_tags_from_qa(val_qa, tag_qids)

            logger.info(f"\n{'Tag':<20} {'Train':<15} {'Val':<15}")
            logger.info("-" * 80)

            for tag_name in ['has_turn', 'has_braking', 'has_aggressive']:
                train_count = sum(1 for t in train_tags.values() if t.get(tag_name, False))
                val_count = sum(1 for t in val_tags.values() if t.get(tag_name, False))

                train_pct = 100 * train_count / len(train_clips) if train_clips else 0
                val_pct = 100 * val_count / len(val_clips) if val_clips else 0

                logger.info(
                    f"{tag_name:<20} {train_count} ({train_pct:.1f}%)"
                    f"{'':<3} {val_count} ({val_pct:.1f}%)"
                )

        # Stratification bin coverage
        if 'stratification_bins' in metadata:
            logger.info("\n" + "-"*80)
            logger.info("STRATIFICATION BIN COVERAGE")
            logger.info("-"*80)

            bins = metadata['stratification_bins']
            logger.info(f"\n{'Bin (T,B,A)':<20} {'Total':<10} {'Sampled (min)':<15} "
                       f"{'Sampled (prop)':<18} {'Sampled (total)':<15}")
            logger.info("-" * 80)

            for bin_key in sorted(bins.keys()):
                bin_data = bins[bin_key]
                logger.info(
                    f"{bin_key:<20} {bin_data['total']:<10} "
                    f"{bin_data['sampled_min']:<15} {bin_data['sampled_proportional']:<18} "
                    f"{bin_data['sampled_total']:<15}"
                )

        logger.info("\n" + "="*80)

    def _derive_tags_from_qa(
        self,
        qa_items: List[dict],
        tag_question_ids: Dict[str, str]
    ) -> Dict[str, Dict[str, bool]]:
        """Derive clip-level tags from QA answers."""
        clip_tags = defaultdict(lambda: {
            'has_turn': False,
            'has_braking': False,
            'has_aggressive': False
        })

        for qa in qa_items:
            clip_id = qa['clip_id']
            question_id = qa['question_id']
            answer = qa['answer']

            # Check for turning
            if question_id == tag_question_ids.get('turn'):
                if answer in ['left', 'right']:
                    clip_tags[clip_id]['has_turn'] = True

            # Check for braking
            elif question_id == tag_question_ids.get('braking'):
                if answer == 'yes':
                    clip_tags[clip_id]['has_braking'] = True

            # Check for aggressive driving
            elif question_id == tag_question_ids.get('aggressive'):
                if answer == 'aggressive':
                    clip_tags[clip_id]['has_aggressive'] = True

        return dict(clip_tags)


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing split files",
    )

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    if not splits_dir.exists():
        logger.error(f"Splits directory not found: {splits_dir}")
        return 1

    # Initialize validator
    validator = SplitValidator(splits_dir)

    # Run validation
    is_valid, errors = validator.validate_all()

    # Print results
    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)

    if is_valid:
        logger.info("✓ All validation checks passed")
        logger.info("\nSplit dataset is valid and ready for use.")
        return 0
    else:
        logger.error(f"✗ Validation failed with {len(errors)} error(s):")
        for error in errors:
            logger.error(f"  - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
