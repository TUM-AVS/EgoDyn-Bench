#!/usr/bin/env python3
"""
CLI script to generate QA dataset from clips and question templates.

Usage:
    python generate_qa.py \\
        --clips_index ./output/clips/clips_index.jsonl \\
        --questions_config ./dataset/configs/questions_template.yaml \\
        --output_qa_jsonl ./output/qa.jsonl \\
        --seed 42

Output:
    - qa.jsonl: One JSON record per QA item
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.config_loader import QuestionConfigLoader
from dataset.generation.qa_generator import generate_qa_dataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_qa_dataset(qa_items: list, output_path: Path):
    """
    Save QA items to JSONL file.

    Args:
        qa_items: List of QA item dictionaries
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(qa_items)} QA items to {output_path}")

    with open(output_path, 'w') as f:
        for qa_item in qa_items:
            f.write(json.dumps(qa_item) + '\n')

    logger.info(f"Saved QA dataset to {output_path}")


def print_summary(qa_items: list):
    """Print summary statistics of generated QA dataset."""
    logger.info("\n" + "="*60)
    logger.info("QA GENERATION SUMMARY")
    logger.info("="*60)

    # Total QA items
    logger.info(f"Total QA items: {len(qa_items)}")

    # Count by category
    categories = {}
    for qa in qa_items:
        cat = qa.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("\nQA items by category:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")

    # Count by question_id
    questions = {}
    for qa in qa_items:
        qid = qa.get("question_id", "unknown")
        questions[qid] = questions.get(qid, 0) + 1

    logger.info(f"\nUnique question types: {len(questions)}")
    logger.info("\nQA items by question_id:")
    for qid, count in sorted(questions.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {qid}: {count}")

    # Count unique clips
    unique_clips = set(qa["clip_id"] for qa in qa_items)
    logger.info(f"\nClips with QA: {len(unique_clips)}")

    if len(unique_clips) > 0:
        avg_qa_per_clip = len(qa_items) / len(unique_clips)
        logger.info(f"Average QA per clip: {avg_qa_per_clip:.1f}")

    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA dataset from clips and question templates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips_index",
        type=str,
        required=True,
        help="Path to clips_index.jsonl",
    )
    parser.add_argument(
        "--questions_config",
        type=str,
        required=True,
        help="Path to questions configuration YAML file",
    )
    parser.add_argument(
        "--output_qa_jsonl",
        type=str,
        required=True,
        help="Output path for qa.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_qas_per_clip",
        type=int,
        default=None,
        help="Maximum QA items per clip (None = all questions)",
    )
    parser.add_argument(
        "--category_filter",
        type=str,
        default=None,
        help="Filter questions by category (e.g., 'direct_dynamics')",
    )
    parser.add_argument(
        "--no_evidence",
        action="store_true",
        help="Exclude evidence from QA items",
    )

    args = parser.parse_args()

    # Validate paths
    clips_index_path = Path(args.clips_index)
    if not clips_index_path.exists():
        logger.error(f"Clips index not found: {clips_index_path}")
        return 1

    questions_config_path = Path(args.questions_config)
    if not questions_config_path.exists():
        logger.error(f"Questions config not found: {questions_config_path}")
        return 1

    output_path = Path(args.output_qa_jsonl)

    # Load question configuration
    logger.info("Loading question configuration...")
    try:
        config_loader = QuestionConfigLoader(str(questions_config_path))
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Get questions (with optional category filter)
    questions = config_loader.get_questions(category_filter=args.category_filter)
    defaults = config_loader.get_defaults()

    if not questions:
        logger.error("No questions found matching the criteria")
        return 1

    logger.info(f"Loaded {len(questions)} question templates")
    if args.category_filter:
        logger.info(f"Filtered by category: {args.category_filter}")

    # Override evidence setting if flag is set
    include_evidence = not args.no_evidence
    if args.no_evidence:
        logger.info("Evidence will be excluded from QA items")

    # Generate QA dataset
    logger.info("Generating QA dataset...")
    try:
        qa_items = generate_qa_dataset(
            clips_index_path=str(clips_index_path),
            questions=questions,
            defaults=defaults,
            include_evidence=include_evidence,
            max_qas_per_clip=args.max_qas_per_clip,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Failed to generate QA dataset: {e}", exc_info=True)
        return 1

    if not qa_items:
        logger.error("No QA items generated")
        return 1

    # Save QA dataset
    save_qa_dataset(qa_items, output_path)

    # Print summary
    print_summary(qa_items)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
