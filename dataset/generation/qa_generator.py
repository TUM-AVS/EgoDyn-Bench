"""
QA generation logic for creating question-answer pairs from clips.

Generates QA items by applying labeling rules to clip data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from dataset.generation.labeling_rules import apply_rule


logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates QA items from clips using question templates."""

    def __init__(
        self,
        include_evidence: bool = True,
        max_qas_per_clip: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize QA generator.

        Args:
            include_evidence: Whether to include evidence in QA items
            max_qas_per_clip: Maximum QAs per clip (None = unlimited)
            seed: Random seed for sampling
        """
        self.include_evidence = include_evidence
        self.max_qas_per_clip = max_qas_per_clip
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._qa_counter = 0

    def generate_qa_for_clip(
        self,
        clip_record: Dict[str, Any],
        clip_arrays: Dict[str, np.ndarray],
        questions: List[Dict[str, Any]],
        defaults: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate QA items for a single clip.

        Args:
            clip_record: Clip metadata record from clips_index.jsonl
            clip_arrays: Loaded clip arrays (timestamps, position, etc.)
            questions: List of question templates
            defaults: Default configuration values

        Returns:
            List of QA item dictionaries
        """
        clip_features = clip_record["features"]
        clip_id = clip_record["clip_id"]

        qa_items = []

        # Optionally sample questions if max_qas_per_clip is set
        selected_questions = questions
        if self.max_qas_per_clip and len(questions) > self.max_qas_per_clip:
            indices = self.rng.choice(len(questions), self.max_qas_per_clip, replace=False)
            selected_questions = [questions[i] for i in indices]
            logger.debug(
                f"Sampled {self.max_qas_per_clip} questions from {len(questions)} "
                f"for clip {clip_id}"
            )

        # Generate QA for each selected question
        for question in selected_questions:
            try:
                qa_item = self._generate_single_qa(
                    clip_record=clip_record,
                    clip_features=clip_features,
                    clip_arrays=clip_arrays,
                    question=question,
                    defaults=defaults
                )
                qa_items.append(qa_item)

            except Exception as e:
                logger.error(
                    f"Failed to generate QA for question '{question['question_id']}' "
                    f"on clip {clip_id}: {e}"
                )
                continue

        return qa_items

    def _generate_single_qa(
        self,
        clip_record: Dict[str, Any],
        clip_features: Dict[str, float],
        clip_arrays: Dict[str, np.ndarray],
        question: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a single QA item."""
        # Apply labeling rule
        rule_name = question["rule"]["name"]
        rule_params = question["rule"]["params"]

        answer, evidence = apply_rule(
            rule_name=rule_name,
            clip_features=clip_features,
            clip_arrays=clip_arrays,
            params=rule_params
        )

        # Validate answer against declared choices (binary/multiclass only)
        answer_type = question["answer_type"]
        choices = question.get("choices")
        if choices and answer_type in ("binary", "multiclass"):
            if answer not in choices:
                raise ValueError(
                    f"Rule '{rule_name}' returned answer '{answer}' "
                    f"not in declared choices {choices}"
                )

        # Format question text (with parameter substitution)
        question_text = self._format_question_text(question["question_text"], rule_params)

        # Create QA item
        qa_id = f"qa_{self._qa_counter:08d}"
        self._qa_counter += 1

        qa_item = {
            "qa_id": qa_id,
            "clip_id": clip_record["clip_id"],
            "question_id": question["question_id"],
            "category": question["category"],
            "question": question_text,
            "answer": answer,
            "answer_type": question["answer_type"],
            "choices": question.get("choices", None),
            "units": question.get("units", None),
            "label_source": defaults.get("label_source", "sensor_rule"),
            "rule_name": rule_name,
            "split": clip_record.get("split", "unsplit"),
            "t_end": clip_record["t_end"],
        }

        # Add evidence if enabled
        include_evidence_global = defaults.get("include_evidence", True)
        if self.include_evidence and include_evidence_global:
            qa_item["evidence"] = evidence
        else:
            qa_item["evidence"] = None

        # Add metadata
        if "metadata" in question:
            qa_item["metadata"] = question["metadata"]
        else:
            qa_item["metadata"] = None

        return qa_item

    def _format_question_text(
        self,
        text: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Format question text with parameter substitution.

        Args:
            text: Question template text
            params: Rule parameters for substitution

        Returns:
            Formatted question text
        """
        # Simple template variable substitution: {param_name}
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))

        return text


def load_clip_arrays(
    array_ref: str,
    clips_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Load clip arrays from NPZ file.

    Args:
        array_ref: Relative path to array file (from clips_index.jsonl)
        clips_dir: Base directory containing clips (for resolving array_ref)

    Returns:
        Dictionary of arrays
    """
    if clips_dir:
        array_path = clips_dir / array_ref
    else:
        array_path = Path(array_ref)

    if not array_path.exists():
        raise FileNotFoundError(f"Array file not found: {array_path}")

    # Load NPZ file and eagerly copy arrays so the file handle is released
    with np.load(array_path) as data:
        arrays = {key: data[key] for key in data.files}

    return arrays


def generate_qa_dataset(
    clips_index_path: str,
    questions: List[Dict[str, Any]],
    defaults: Dict[str, Any],
    include_evidence: bool = True,
    max_qas_per_clip: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate QA dataset from clips index and question templates.

    Args:
        clips_index_path: Path to clips_index.jsonl
        questions: List of question templates
        defaults: Default configuration values
        include_evidence: Whether to include evidence
        max_qas_per_clip: Maximum QAs per clip
        seed: Random seed

    Returns:
        List of QA items
    """
    import json

    clips_index_path = Path(clips_index_path)
    clips_dir = clips_index_path.parent

    # Load clips index
    logger.info(f"Loading clips from {clips_index_path}")
    clips = []
    with open(clips_index_path, 'r') as f:
        for line in f:
            clips.append(json.loads(line))

    logger.info(f"Loaded {len(clips)} clips")
    logger.info(f"Generating QA with {len(questions)} question templates")

    # Initialize generator
    generator = QAGenerator(
        include_evidence=include_evidence,
        max_qas_per_clip=max_qas_per_clip,
        seed=seed
    )

    # Generate QA for each clip
    all_qa_items = []

    for clip_idx, clip_record in enumerate(clips):
        # Load clip arrays
        try:
            clip_arrays = load_clip_arrays(clip_record["array_ref"], clips_dir)
        except Exception as e:
            logger.error(f"Failed to load arrays for clip {clip_record['clip_id']}: {e}")
            continue

        # Generate QA items
        qa_items = generator.generate_qa_for_clip(
            clip_record=clip_record,
            clip_arrays=clip_arrays,
            questions=questions,
            defaults=defaults
        )

        all_qa_items.extend(qa_items)

        if (clip_idx + 1) % 100 == 0:
            logger.info(f"Processed {clip_idx + 1}/{len(clips)} clips, {len(all_qa_items)} QA items")

    logger.info(f"Generated {len(all_qa_items)} QA items from {len(clips)} clips")
    logger.info(f"Average {len(all_qa_items) / len(clips):.1f} QA items per clip")

    return all_qa_items
