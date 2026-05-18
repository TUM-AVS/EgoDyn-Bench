#!/usr/bin/env python3
"""
Validation suite for generated QA dataset.

Verifies:
- Schema compliance
- Reference integrity (clip_id existence)
- Answer validity
- Distribution summaries
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


class QAValidator:
    """Validates QA dataset for correctness and consistency."""

    REQUIRED_FIELDS = [
        "qa_id", "clip_id", "question_id", "category", "question",
        "answer", "answer_type", "label_source", "rule_name", "split", "t_end"
    ]
    VALID_ANSWER_TYPES = ["binary", "multiclass", "numeric"]

    def __init__(
        self,
        qa_jsonl_path: Path,
        clips_index_path: Path = None,
    ):
        """
        Initialize QA validator.

        Args:
            qa_jsonl_path: Path to qa.jsonl
            clips_index_path: Optional path to clips_index.jsonl for reference checking
        """
        self.qa_jsonl_path = qa_jsonl_path
        self.clips_index_path = clips_index_path

        # Load valid clip IDs if provided
        self.valid_clip_ids: Set[str] = set()
        if clips_index_path and clips_index_path.exists():
            self._load_clip_ids()

    def _load_clip_ids(self):
        """Load valid clip IDs from clips index."""
        logger.info(f"Loading clip IDs from {self.clips_index_path}")
        with open(self.clips_index_path, 'r') as f:
            for line in f:
                clip = json.loads(line)
                self.valid_clip_ids.add(clip['clip_id'])
        logger.info(f"Loaded {len(self.valid_clip_ids)} valid clip IDs")

    def validate_all(self) -> Tuple[int, int, List[str]]:
        """
        Validate all QA items.

        Returns:
            Tuple of (num_valid, num_invalid, error_messages)
        """
        if not self.qa_jsonl_path.exists():
            return 0, 0, [f"QA file not found: {self.qa_jsonl_path}"]

        logger.info(f"Validating QA in {self.qa_jsonl_path}")

        num_valid = 0
        num_invalid = 0
        all_errors = []

        # Track for global checks
        seen_qa_ids = set()
        qa_items = []

        # Read and validate each QA item
        with open(self.qa_jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    qa_item = json.loads(line)
                    qa_items.append(qa_item)

                    # Validate item
                    is_valid, errors = self._validate_qa_item(qa_item, line_num, seen_qa_ids)

                    if is_valid:
                        num_valid += 1
                    else:
                        num_invalid += 1
                        all_errors.extend(errors)

                except json.JSONDecodeError as e:
                    error = f"Line {line_num}: Invalid JSON - {e}"
                    all_errors.append(error)
                    num_invalid += 1

        # Print distribution summaries
        if qa_items:
            self._print_distribution_summary(qa_items)

        return num_valid, num_invalid, all_errors

    def _validate_qa_item(
        self,
        qa_item: dict,
        line_num: int,
        seen_qa_ids: set
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single QA item.

        Args:
            qa_item: QA item dict
            line_num: Line number in file
            seen_qa_ids: Set of already-seen QA IDs

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        prefix = f"Line {line_num}"

        # 1. Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in qa_item:
                errors.append(f"{prefix}: Missing required field '{field}'")

        if errors:
            return False, errors

        qa_id = qa_item["qa_id"]

        # 2. Check qa_id uniqueness
        if qa_id in seen_qa_ids:
            errors.append(f"{prefix}: Duplicate qa_id '{qa_id}'")
        seen_qa_ids.add(qa_id)

        # 3. Check answer_type validity
        answer_type = qa_item["answer_type"]
        if answer_type not in self.VALID_ANSWER_TYPES:
            errors.append(
                f"{prefix} ({qa_id}): Invalid answer_type '{answer_type}'. "
                f"Must be one of {self.VALID_ANSWER_TYPES}"
            )

        # 4. Check answer validity based on type
        answer = qa_item["answer"]
        choices = qa_item.get("choices")

        if answer_type == "multiclass":
            if not choices:
                errors.append(f"{prefix} ({qa_id}): multiclass must have non-empty 'choices'")
            elif answer not in choices:
                errors.append(
                    f"{prefix} ({qa_id}): Answer '{answer}' not in choices {choices}"
                )

        elif answer_type == "binary":
            if choices and answer not in choices:
                errors.append(
                    f"{prefix} ({qa_id}): Answer '{answer}' not in binary choices {choices}"
                )

        elif answer_type == "numeric":
            if not isinstance(answer, (int, float)):
                errors.append(
                    f"{prefix} ({qa_id}): Numeric answer must be int or float, got {type(answer)}"
                )

        # 5. Check reference integrity (if clips index provided)
        if self.valid_clip_ids:
            clip_id = qa_item["clip_id"]
            if clip_id not in self.valid_clip_ids:
                errors.append(f"{prefix} ({qa_id}): Invalid clip_id '{clip_id}'")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _print_distribution_summary(self, qa_items: List[dict]):
        """Print distribution summaries for QA dataset."""
        logger.info("\n" + "="*60)
        logger.info("DISTRIBUTION SUMMARY")
        logger.info("="*60)

        # Total QA items
        logger.info(f"\nTotal QA items: {len(qa_items)}")

        # QA items by category
        categories = Counter(qa["category"] for qa in qa_items)
        logger.info("\nQA items by category:")
        for cat, count in sorted(categories.items()):
            pct = 100 * count / len(qa_items)
            logger.info(f"  {cat}: {count} ({pct:.1f}%)")

        # QA items by question_id
        question_ids = Counter(qa["question_id"] for qa in qa_items)
        logger.info(f"\nUnique question types: {len(question_ids)}")
        logger.info("\nQA items by question_id (top 10):")
        for qid, count in question_ids.most_common(10):
            logger.info(f"  {qid}: {count}")

        # QA items by answer_type
        answer_types = Counter(qa["answer_type"] for qa in qa_items)
        logger.info("\nQA items by answer_type:")
        for atype, count in sorted(answer_types.items()):
            pct = 100 * count / len(qa_items)
            logger.info(f"  {atype}: {count} ({pct:.1f}%)")

        # Clips coverage
        unique_clips = set(qa["clip_id"] for qa in qa_items)
        logger.info(f"\nUnique clips: {len(unique_clips)}")

        # QAs per clip distribution
        qas_per_clip = Counter(qa["clip_id"] for qa in qa_items)
        qas_per_clip_values = list(qas_per_clip.values())
        logger.info(f"\nQAs per clip:")
        logger.info(f"  Mean:   {np.mean(qas_per_clip_values):.1f}")
        logger.info(f"  Median: {np.median(qas_per_clip_values):.1f}")
        logger.info(f"  Min:    {np.min(qas_per_clip_values)}")
        logger.info(f"  Max:    {np.max(qas_per_clip_values)}")

        # Answer distributions per question_id
        logger.info("\nAnswer distributions:")
        answers_by_question = defaultdict(list)
        for qa in qa_items:
            answers_by_question[qa["question_id"]].append(qa["answer"])

        for qid in sorted(answers_by_question.keys()):
            answers = answers_by_question[qid]
            answer_counts = Counter(answers)

            # Get answer_type for this question
            answer_type = next(qa["answer_type"] for qa in qa_items if qa["question_id"] == qid)

            logger.info(f"\n  {qid} ({answer_type}):")

            if answer_type == "numeric":
                # Show numeric statistics
                numeric_values = [a for a in answers if isinstance(a, (int, float))]
                if numeric_values:
                    logger.info(f"    Count: {len(numeric_values)}")
                    logger.info(f"    Mean:  {np.mean(numeric_values):.2f}")
                    logger.info(f"    Std:   {np.std(numeric_values):.2f}")
                    logger.info(f"    Min:   {np.min(numeric_values):.2f}")
                    logger.info(f"    Max:   {np.max(numeric_values):.2f}")
            else:
                # Show categorical distribution
                for answer, count in sorted(answer_counts.items()):
                    pct = 100 * count / len(answers)
                    logger.info(f"    {answer}: {count} ({pct:.1f}%)")

        logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated QA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_jsonl",
        type=str,
        required=True,
        help="Path to qa.jsonl",
    )
    parser.add_argument(
        "--clips_index",
        type=str,
        default=None,
        help="Optional path to clips_index.jsonl for reference checking",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all errors (not just summary)",
    )

    args = parser.parse_args()

    # Initialize validator
    qa_jsonl_path = Path(args.qa_jsonl)
    clips_index_path = Path(args.clips_index) if args.clips_index else None

    validator = QAValidator(
        qa_jsonl_path=qa_jsonl_path,
        clips_index_path=clips_index_path,
    )

    # Run validation
    num_valid, num_invalid, errors = validator.validate_all()

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Valid QA items:   {num_valid}")
    logger.info(f"Invalid QA items: {num_invalid}")
    logger.info(f"Total errors:     {len(errors)}")

    if errors and args.verbose:
        logger.info("\n" + "="*60)
        logger.info("ERRORS")
        logger.info("="*60)
        for error in errors[:100]:  # Limit to first 100
            logger.info(f"  {error}")
        if len(errors) > 100:
            logger.info(f"  ... and {len(errors) - 100} more errors")

    # Exit with error code if validation failed
    if num_invalid > 0:
        logger.error("Validation failed!")
        return 1

    logger.info("Validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
