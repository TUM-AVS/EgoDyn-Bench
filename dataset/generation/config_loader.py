"""
Question template configuration loader and validator.

Loads YAML configuration files defining question templates and labeling rules.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from dataset.generation.labeling_rules import LabelingRuleRegistry


logger = logging.getLogger(__name__)


class QuestionConfigValidator:
    """Validates question template configuration."""

    VALID_ANSWER_TYPES = ["binary", "multiclass", "numeric"]
    REQUIRED_QUESTION_FIELDS = ["question_id", "category", "question_text", "answer_type", "rule"]
    REQUIRED_RULE_FIELDS = ["name", "params"]

    KNOWN_AGGREGATIONS = {"mean", "min", "max", "rms", "abs_max", "max_abs"}
    KNOWN_OPERATORS = {"less_than", "greater_than", "abs_greater_than"}
    NUMERIC_PARAM_SUBSTRINGS = ["threshold", "factor", "_seconds"]
    NUMERIC_PARAM_EXACT = {"rounding", "min_stops", "min_peak_value"}

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration structure.

        Args:
            config: Loaded configuration dict

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check top-level fields
        if "questions" not in config:
            errors.append("Missing required field: 'questions'")
            return False, errors

        if not isinstance(config["questions"], list):
            errors.append("Field 'questions' must be a list")
            return False, errors

        # Validate each question
        seen_ids = set()
        for idx, question in enumerate(config["questions"]):
            q_errors = QuestionConfigValidator._validate_question(question, idx, seen_ids)
            errors.extend(q_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def _validate_question(
        question: Dict[str, Any],
        idx: int,
        seen_ids: set
    ) -> List[str]:
        """Validate a single question template."""
        errors = []
        prefix = f"Question[{idx}]"

        # Check required fields
        for field in QuestionConfigValidator.REQUIRED_QUESTION_FIELDS:
            if field not in question:
                errors.append(f"{prefix}: Missing required field '{field}'")

        if errors:
            return errors

        # Check question_id uniqueness
        qid = question["question_id"]
        if qid in seen_ids:
            errors.append(f"{prefix}: Duplicate question_id '{qid}'")
        seen_ids.add(qid)

        # Validate answer_type
        answer_type = question["answer_type"]
        if answer_type not in QuestionConfigValidator.VALID_ANSWER_TYPES:
            errors.append(
                f"{prefix}: Invalid answer_type '{answer_type}'. "
                f"Must be one of {QuestionConfigValidator.VALID_ANSWER_TYPES}"
            )

        # Validate choices for multiclass
        if answer_type == "multiclass":
            if "choices" not in question or not question["choices"]:
                errors.append(f"{prefix}: multiclass questions must have non-empty 'choices'")
            elif not isinstance(question["choices"], list):
                errors.append(f"{prefix}: 'choices' must be a list")
            elif len(question["choices"]) < 2:
                errors.append(f"{prefix}: multiclass questions must have at least 2 choices")

        # Validate choices for binary
        if answer_type == "binary":
            if "choices" not in question or not question["choices"]:
                errors.append(f"{prefix}: binary questions must have 'choices'")
            elif not isinstance(question["choices"], list):
                errors.append(f"{prefix}: 'choices' must be a list")
            elif len(question["choices"]) != 2:
                errors.append(
                    f"{prefix}: binary questions must have exactly 2 choices, "
                    f"got {len(question['choices'])}. "
                    f"Consider using answer_type 'multiclass' if more choices are needed."
                )

        # Validate rule
        if "rule" in question:
            rule_errors = QuestionConfigValidator._validate_rule(question["rule"], prefix)
            errors.extend(rule_errors)

        return errors

    @staticmethod
    def _validate_rule(rule: Dict[str, Any], prefix: str) -> List[str]:
        """Validate rule specification."""
        errors = []

        for field in QuestionConfigValidator.REQUIRED_RULE_FIELDS:
            if field not in rule:
                errors.append(f"{prefix}.rule: Missing required field '{field}'")

        if "params" in rule and not isinstance(rule["params"], dict):
            errors.append(f"{prefix}.rule: 'params' must be a dictionary")

        # Stop if basic structure is invalid — can't validate further
        if errors:
            return errors

        # Check rule name against registry
        rule_name = rule["name"]
        registered_rules = LabelingRuleRegistry.list_rules()
        if rule_name not in registered_rules:
            errors.append(
                f"{prefix}.rule: Unknown rule name '{rule_name}'. "
                f"Registered rules: {registered_rules}"
            )

        # Validate common param patterns
        param_errors = QuestionConfigValidator._validate_params(
            rule["params"], f"{prefix}.rule.params"
        )
        errors.extend(param_errors)

        return errors

    @staticmethod
    def _validate_params(params: Dict[str, Any], path: str) -> List[str]:
        """Validate common param type patterns.

        Checks:
          - Params whose name contains 'threshold', 'factor', or '_seconds'
            (and a few exact names) must be numeric.
          - 'aggregation' must be a known method or percentile (p05..p99).
          - 'operator' must be a known comparison operator.

        Recurses into nested dicts (e.g. sequential_event's first_event/second_event).
        """
        errors = []

        for key, value in params.items():
            # Recurse into nested dicts
            if isinstance(value, dict):
                errors.extend(
                    QuestionConfigValidator._validate_params(value, f"{path}.{key}")
                )
                continue

            # Recurse into lists (e.g. conditions list, thresholds list)
            if isinstance(value, list):
                should_be_numeric = (
                    key in QuestionConfigValidator.NUMERIC_PARAM_EXACT
                    or any(sub in key for sub in QuestionConfigValidator.NUMERIC_PARAM_SUBSTRINGS)
                )
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        errors.extend(
                            QuestionConfigValidator._validate_params(
                                item, f"{path}.{key}[{i}]"
                            )
                        )
                    elif should_be_numeric and (isinstance(item, bool) or not isinstance(item, (int, float))):
                        errors.append(
                            f"{path}.{key}[{i}]: Expected numeric value, "
                            f"got {type(item).__name__} '{item}'"
                        )
                continue

            # Check numeric params
            should_be_numeric = (
                key in QuestionConfigValidator.NUMERIC_PARAM_EXACT
                or any(sub in key for sub in QuestionConfigValidator.NUMERIC_PARAM_SUBSTRINGS)
            )
            if should_be_numeric and (isinstance(value, bool) or not isinstance(value, (int, float))):
                errors.append(
                    f"{path}.{key}: Expected numeric value, "
                    f"got {type(value).__name__} '{value}'"
                )

            # Check aggregation
            if key == "aggregation":
                if not QuestionConfigValidator._is_valid_aggregation(value):
                    errors.append(
                        f"{path}.aggregation: Unknown method '{value}'. "
                        f"Must be one of {sorted(QuestionConfigValidator.KNOWN_AGGREGATIONS)} "
                        f"or a percentile like 'p50', 'p95'"
                    )

            # Check operator
            if key == "operator":
                if value not in QuestionConfigValidator.KNOWN_OPERATORS:
                    errors.append(
                        f"{path}.operator: Unknown operator '{value}'. "
                        f"Must be one of {sorted(QuestionConfigValidator.KNOWN_OPERATORS)}"
                    )

        return errors

    @staticmethod
    def _is_valid_aggregation(value: Any) -> bool:
        """Check if an aggregation method string is valid."""
        if not isinstance(value, str):
            return False
        if value in QuestionConfigValidator.KNOWN_AGGREGATIONS:
            return True
        # Percentile pattern: p05, p50, p95, etc.
        if re.match(r'^p\d+$', value):
            percentile = int(value[1:])
            return 0 <= percentile <= 100
        return False


class QuestionConfigLoader:
    """Loads and manages question template configurations."""

    def __init__(self, config_path: str):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        self._load()

    def _load(self):
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading question config from {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Validate
        is_valid, errors = QuestionConfigValidator.validate_config(self.config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        logger.info(f"Loaded {len(self.config['questions'])} question templates")

    def get_questions(
        self,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get question templates, optionally filtered by category.

        Args:
            category_filter: Optional category to filter by

        Returns:
            List of question template dictionaries
        """
        questions = self.config["questions"]

        if category_filter:
            questions = [q for q in questions if q.get("category") == category_filter]
            logger.info(f"Filtered to {len(questions)} questions in category '{category_filter}'")

        return questions

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return self.config.get("defaults", {})

    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific question template by ID.

        Args:
            question_id: Question template ID

        Returns:
            Question template dict or None if not found
        """
        for question in self.config["questions"]:
            if question["question_id"] == question_id:
                return question
        return None

    def format_question_text(self, question: Dict[str, Any], params: Dict[str, Any]) -> str:
        """
        Format question text with parameter substitution.

        Args:
            question: Question template dict
            params: Rule parameters for substitution

        Returns:
            Formatted question text
        """
        text = question["question_text"]

        # Simple template variable substitution: {param_name}
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))

        return text
