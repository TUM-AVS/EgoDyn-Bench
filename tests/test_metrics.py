"""Unit tests for evaluation.metrics."""

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.metrics import (
    semantic_accuracy, macro_f1, confusion_matrix, evaluate,
    _compute_group_metrics, _check_rule, compute_consistency,
    CONSISTENCY_RULES,
)
from evaluation.parsers import load_question_config


# ── semantic_accuracy ─────────────────────────────────────────────────────

class TestSemanticAccuracy:
    def test_perfect(self):
        assert semantic_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_all_wrong(self):
        assert semantic_accuracy(["a", "b", "c"], ["x", "y", "z"]) == 0.0

    def test_partial(self):
        assert semantic_accuracy(["a", "b", "c", "d"], ["a", "x", "c", "x"]) == 0.5

    def test_empty(self):
        assert semantic_accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            semantic_accuracy(["a", "b", "c"], ["a", "b"])


# ── macro_f1 ──────────────────────────────────────────────────────────────

class TestMacroF1:
    def test_perfect(self):
        oracle = ["a", "b", "a", "b"]
        pred = ["a", "b", "a", "b"]
        assert macro_f1(oracle, pred) == 1.0

    def test_all_wrong_binary(self):
        oracle = ["a", "a", "b", "b"]
        pred = ["b", "b", "a", "a"]
        assert macro_f1(oracle, pred) == 0.0

    def test_imbalanced_differs_from_accuracy(self):
        """Macro F1 should differ from accuracy for imbalanced predictions."""
        # 8 "yes" and 2 "no" in oracle. Model always predicts "yes".
        oracle = ["yes"] * 8 + ["no"] * 2
        pred = ["yes"] * 10
        acc = semantic_accuracy(oracle, pred)
        f1 = macro_f1(oracle, pred)
        assert acc == 0.8  # 8/10 correct
        # Macro F1: yes-F1 = 2*1.0*0.8/(1.0+0.8)=0.889, no-F1 = 0 → avg ≈ 0.444
        assert f1 < acc
        assert abs(f1 - 0.4444) < 0.01

    def test_three_class(self):
        oracle = ["a", "b", "c", "a", "b", "c"]
        pred = ["a", "b", "c", "a", "b", "c"]
        assert macro_f1(oracle, pred) == 1.0

    def test_empty(self):
        assert macro_f1([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            macro_f1(["a", "b"], ["a"])

    def test_class_never_predicted(self):
        """A class in oracle that is never predicted should get F1=0."""
        oracle = ["a", "a", "b"]
        pred = ["a", "a", "a"]
        f1 = macro_f1(oracle, pred)
        # a: P=2/3, R=1.0 → F1=0.8;  b: P=0, R=0 → F1=0.0
        # macro = (0.8 + 0.0) / 2 = 0.4
        assert abs(f1 - 0.4) < 0.01


# ── confusion_matrix ──────────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_binary_perfect(self):
        cm = confusion_matrix(["yes", "yes", "no", "no"], ["yes", "yes", "no", "no"])
        assert cm["labels"] == ["no", "yes"]
        # rows = oracle, cols = pred → [[2,0],[0,2]]
        assert cm["matrix"] == [[2, 0], [0, 2]]

    def test_binary_all_wrong(self):
        cm = confusion_matrix(["yes", "yes", "no", "no"], ["no", "no", "yes", "yes"])
        assert cm["matrix"] == [[0, 2], [2, 0]]

    def test_three_class(self):
        oracle = ["left", "left", "right", "straight"]
        pred = ["left", "straight", "right", "straight"]
        cm = confusion_matrix(oracle, pred)
        assert cm["labels"] == ["left", "right", "straight"]
        # left oracle:    1 left, 0 right, 1 straight
        # right oracle:   0 left, 1 right, 0 straight
        # straight oracle: 0 left, 0 right, 1 straight
        assert cm["matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]

    def test_sums_to_n(self):
        oracle = ["a", "b", "c", "a", "b"]
        pred = ["a", "c", "b", "a", "b"]
        cm = confusion_matrix(oracle, pred)
        total = sum(sum(row) for row in cm["matrix"])
        assert total == len(oracle)

    def test_explicit_labels_include_unseen(self):
        """Labels not in data should appear as zero rows/columns."""
        cm = confusion_matrix(
            ["yes", "yes"], ["yes", "yes"],
            labels=["yes", "no"],
        )
        assert cm["labels"] == ["yes", "no"]
        assert cm["matrix"] == [[2, 0], [0, 0]]

    def test_empty(self):
        cm = confusion_matrix([], [])
        assert cm["labels"] == []
        assert cm["matrix"] == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            confusion_matrix(["a", "b"], ["a"])


# ── _compute_group_metrics ────────────────────────────────────────────────

class TestComputeGroupMetrics:
    def test_structure(self):
        m = _compute_group_metrics(["a", "b"], ["a", "b"])
        assert set(m.keys()) == {"accuracy", "balanced_acc", "macro_f1", "n"}
        assert m["n"] == 2

    def test_rounding(self):
        m = _compute_group_metrics(["a", "b", "c"], ["a", "b", "x"])
        assert isinstance(m["accuracy"], float)
        # Should be rounded to 4 decimal places
        assert len(str(m["accuracy"]).split(".")[-1]) <= 4


# ── evaluate (integration) ────────────────────────────────────────────────

class TestEvaluate:
    @pytest.fixture
    def question_config(self):
        """Minimal question config for testing."""
        return {
            "braking_intensity": {
                "choices": ["emergency", "moderate", "low", "none"],
                "answer_type": "multiclass",
                "category": "direct_dynamics",
                "temporal": False,
            },
            "yaw_rate_turn_direction": {
                "choices": ["left", "right", "straight"],
                "answer_type": "multiclass",
                "category": "direct_dynamics",
                "temporal": False,
            },
            "speed_trend": {
                "choices": ["accelerating", "decelerating", "steady"],
                "answer_type": "multiclass",
                "category": "direct_dynamics",
                "temporal": True,
            },
            "speed_peak_half": {
                "choices": ["first_half", "second_half", "no_peak"],
                "answer_type": "multiclass",
                "category": "comparative",
                "temporal": True,
            },
            "emergency_maneuver": {
                "choices": ["yes", "no"],
                "answer_type": "binary",
                "category": "direct_dynamics",
                "temporal": False,
            },
            "significant_heading_change": {
                "choices": ["yes", "no"],
                "answer_type": "binary",
                "category": "direct_dynamics",
                "temporal": False,
            },
        }

    def test_perfect_predictions(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
            {"clip_id": "c2", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "none",
             "model_answer": "none"},
            {"clip_id": "c3", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "left",
             "model_answer": "left"},
        ]
        result = evaluate(records, question_config)
        assert result["global"]["accuracy"] == 1.0
        assert result["global"]["macro_f1"] == 1.0
        assert result["n_total"] == 3
        assert result["n_parsed"] == 3
        assert result["parsable_coverage"] == 1.0

    def test_partial_correctness(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "none"},
            {"clip_id": "c2", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "none",
             "model_answer": "none"},
        ]
        result = evaluate(records, question_config)
        assert result["global"]["accuracy"] == 0.5

    def test_unparsable_reduces_coverage(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
            {"clip_id": "c2", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "none",
             "model_answer": "I refuse to answer this question."},
        ]
        result = evaluate(records, question_config)
        assert result["n_total"] == 2
        assert result["n_parsed"] == 1
        assert result["parsable_coverage"] == 0.5

    def test_unknown_question_id_skipped(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "nonexistent_question",
             "category": "unknown", "oracle_label": "foo",
             "model_answer": "bar"},
        ]
        result = evaluate(records, question_config)
        assert result["n_parsed"] == 0
        assert result["parsable_coverage"] == 0.0

    def test_per_category_breakdown(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
            {"clip_id": "c2", "question_id": "speed_peak_half",
             "category": "comparative", "oracle_label": "first_half",
             "model_answer": "first half"},
        ]
        result = evaluate(records, question_config)
        assert "direct_dynamics" in result["per_category"]
        assert "comparative" in result["per_category"]
        assert result["per_category"]["direct_dynamics"]["n"] == 1
        assert result["per_category"]["comparative"]["n"] == 1

    def test_per_question_breakdown(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
            {"clip_id": "c2", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "left",
             "model_answer": "right"},
        ]
        result = evaluate(records, question_config)
        assert result["per_question"]["braking_intensity"]["accuracy"] == 1.0
        assert result["per_question"]["yaw_rate_turn_direction"]["accuracy"] == 0.0

    def test_output_is_json_serializable(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_confusion_matrix_in_per_question(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "left",
             "model_answer": "left"},
            {"clip_id": "c2", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "left",
             "model_answer": "straight"},
            {"clip_id": "c3", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "right",
             "model_answer": "right"},
            {"clip_id": "c4", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "straight",
             "model_answer": "straight"},
        ]
        result = evaluate(records, question_config)
        cm = result["per_question"]["yaw_rate_turn_direction"]["confusion_matrix"]
        assert cm["labels"] == ["left", "right", "straight"]
        # left: 1 correct, 1 predicted straight
        # right: 1 correct
        # straight: 1 correct
        assert cm["matrix"] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
        assert sum(sum(row) for row in cm["matrix"]) == 4

    def test_temporal_subset(self, question_config):
        """Temporal metrics should only include temporal questions."""
        records = [
            # Non-temporal: correct
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
            # Temporal: correct
            {"clip_id": "c2", "question_id": "speed_trend",
             "category": "direct_dynamics", "oracle_label": "accelerating",
             "model_answer": "accelerating"},
            # Temporal: wrong
            {"clip_id": "c3", "question_id": "speed_peak_half",
             "category": "comparative", "oracle_label": "first_half",
             "model_answer": "second half"},
        ]
        result = evaluate(records, question_config)
        assert result["global"]["accuracy"] == round(2 / 3, 4)
        # Temporal subset: 1 correct out of 2
        assert result["temporal"]["n"] == 2
        assert result["temporal"]["accuracy"] == 0.5

    def test_temporal_empty_when_no_temporal_questions(self, question_config):
        """Temporal metrics should handle no temporal questions gracefully."""
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        assert result["temporal"]["n"] == 0
        assert result["temporal"]["accuracy"] == 0.0

    def test_verbose_model_answers(self, question_config):
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "The braking intensity is emergency level."},
            {"clip_id": "c2", "question_id": "yaw_rate_turn_direction",
             "category": "direct_dynamics", "oracle_label": "straight",
             "model_answer": "The vehicle appears to be going straight."},
        ]
        result = evaluate(records, question_config)
        assert result["global"]["accuracy"] == 1.0

    def test_consistency_in_output(self, question_config):
        """evaluate() should include consistency with all expected fields."""
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        c = result["consistency"]
        assert "rate" in c
        assert "n_clips" in c
        assert "n_evaluable" in c
        assert "consistency_coverage" in c
        assert "mean_violations" in c
        assert "per_rule" in c

    def test_answer_field_name_fallback(self, question_config):
        """Records using 'answer' (QA generator output) instead of 'oracle_label'."""
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics", "answer": "emergency",
             "model_answer": "emergency"},
            {"clip_id": "c2", "question_id": "braking_intensity",
             "category": "direct_dynamics", "answer": "none",
             "model_answer": "none"},
        ]
        result = evaluate(records, question_config)
        assert result["n_parsed"] == 2
        assert result["global"]["accuracy"] == 1.0

    def test_oracle_label_takes_precedence_over_answer(self, question_config):
        """When both fields exist, oracle_label wins."""
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics",
             "oracle_label": "emergency", "answer": "none",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        assert result["global"]["accuracy"] == 1.0

    def test_missing_both_oracle_fields_skipped(self, question_config):
        """Records with neither oracle_label nor answer should be skipped."""
        records = [
            {"clip_id": "c1", "question_id": "braking_intensity",
             "category": "direct_dynamics",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        assert result["n_parsed"] == 0

    def test_missing_clip_id_skipped(self, question_config):
        """Records missing clip_id should be skipped, not crash."""
        records = [
            {"question_id": "braking_intensity",
             "category": "direct_dynamics", "oracle_label": "emergency",
             "model_answer": "emergency"},
        ]
        result = evaluate(records, question_config)
        assert result["n_parsed"] == 0

    def test_missing_question_id_skipped(self, question_config):
        """Records missing question_id should be skipped, not crash."""
        records = [
            {"clip_id": "c1",
             "category": "direct_dynamics", "oracle_label": "yes",
             "model_answer": "yes"},
        ]
        result = evaluate(records, question_config)
        assert result["n_parsed"] == 0

    def test_numeric_oracle_normalization(self):
        """Numeric oracle '42.0' should match model answer '42'."""
        qcfg = {
            "max_speed": {
                "choices": None,
                "answer_type": "numeric",
                "category": "direct_dynamics",
                "temporal": False,
            },
        }
        records = [
            {"clip_id": "c1", "question_id": "max_speed",
             "category": "direct_dynamics", "oracle_label": "42.0",
             "model_answer": "42"},
            {"clip_id": "c2", "question_id": "max_speed",
             "category": "direct_dynamics", "oracle_label": "3.50",
             "model_answer": "The speed is about 3.5 m/s."},
        ]
        result = evaluate(records, qcfg)
        assert result["n_parsed"] == 2
        assert result["global"]["accuracy"] == 1.0


# ── _check_rule (unit) ───────────────────────────────────────────────────

class TestCheckRule:
    RULE_EQ = {
        "name": "test_eq",
        "if": ("q_a", "==", "yes"),
        "then": ("q_b", "==", "no"),
    }
    RULE_NEQ = {
        "name": "test_neq",
        "if": ("q_a", "==", "yes"),
        "then": ("q_b", "!=", "bad"),
    }

    def test_condition_not_triggered(self):
        """When condition doesn't match, rule is not applicable."""
        applicable, violated = _check_rule(
            self.RULE_EQ, {"q_a": "no", "q_b": "yes"},
        )
        assert applicable is False
        assert violated is False

    def test_missing_condition_question(self):
        applicable, violated = _check_rule(
            self.RULE_EQ, {"q_b": "no"},
        )
        assert applicable is False

    def test_missing_implication_question(self):
        applicable, violated = _check_rule(
            self.RULE_EQ, {"q_a": "yes"},
        )
        assert applicable is False

    def test_satisfied_eq(self):
        applicable, violated = _check_rule(
            self.RULE_EQ, {"q_a": "yes", "q_b": "no"},
        )
        assert applicable is True
        assert violated is False

    def test_violated_eq(self):
        applicable, violated = _check_rule(
            self.RULE_EQ, {"q_a": "yes", "q_b": "yes"},
        )
        assert applicable is True
        assert violated is True

    def test_satisfied_neq(self):
        applicable, violated = _check_rule(
            self.RULE_NEQ, {"q_a": "yes", "q_b": "good"},
        )
        assert applicable is True
        assert violated is False

    def test_violated_neq(self):
        applicable, violated = _check_rule(
            self.RULE_NEQ, {"q_a": "yes", "q_b": "bad"},
        )
        assert applicable is True
        assert violated is True


# ── compute_consistency ──────────────────────────────────────────────────

class TestComputeConsistency:
    RULES = [
        {
            "name": "straight_no_heading",
            "if": ("turn_dir", "==", "straight"),
            "then": ("heading_change", "==", "no"),
        },
        {
            "name": "brake_not_accel",
            "if": ("braking", "==", "yes"),
            "then": ("speed_trend", "!=", "accelerating"),
        },
    ]

    def test_all_consistent(self):
        clips = {
            "c1": {"turn_dir": "straight", "heading_change": "no"},
            "c2": {"turn_dir": "left", "heading_change": "yes"},
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["rate"] == 1.0
        assert result["n_consistent"] == 1  # only c1 is evaluable
        assert result["n_evaluable"] == 1
        assert result["n_clips"] == 2

    def test_one_violation(self):
        clips = {
            "c1": {"turn_dir": "straight", "heading_change": "yes"},  # violation
            "c2": {"turn_dir": "straight", "heading_change": "no"},   # ok
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["rate"] == 0.5
        assert result["n_evaluable"] == 2
        assert result["n_consistent"] == 1
        assert result["per_rule"]["straight_no_heading"]["n_applicable"] == 2
        assert result["per_rule"]["straight_no_heading"]["n_violations"] == 1
        assert result["per_rule"]["straight_no_heading"]["compliance"] == 0.5

    def test_rule_not_applicable_missing_answer(self):
        """Clips missing one of the rule's questions are not evaluable."""
        clips = {
            "c1": {"turn_dir": "straight"},  # no heading_change → not evaluable
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["n_evaluable"] == 0
        assert result["rate"] == 0.0
        assert result["consistency_coverage"] == 0.0
        assert result["per_rule"]["straight_no_heading"]["n_applicable"] == 0

    def test_condition_not_triggered(self):
        """When condition doesn't match, the rule is not applicable (clip not evaluable)."""
        clips = {
            "c1": {"turn_dir": "left", "heading_change": "yes"},
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["n_evaluable"] == 0
        assert result["per_rule"]["straight_no_heading"]["n_applicable"] == 0

    def test_multiple_violations_same_clip(self):
        """A clip violating two rules is still counted as one inconsistent clip."""
        clips = {
            "c1": {
                "turn_dir": "straight", "heading_change": "yes",       # violates rule 1
                "braking": "yes", "speed_trend": "accelerating",       # violates rule 2
            },
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["rate"] == 0.0
        assert result["n_consistent"] == 0
        assert result["n_evaluable"] == 1
        assert result["mean_violations"] == 2.0
        assert result["per_rule"]["straight_no_heading"]["n_violations"] == 1
        assert result["per_rule"]["brake_not_accel"]["n_violations"] == 1

    def test_empty_clips(self):
        result = compute_consistency({}, rules=self.RULES)
        assert result["rate"] == 0.0
        assert result["n_clips"] == 0
        assert result["n_evaluable"] == 0
        assert result["mean_violations"] == 0.0

    def test_per_rule_compliance(self):
        clips = {
            "c1": {"turn_dir": "straight", "heading_change": "no"},  # ok
            "c2": {"turn_dir": "straight", "heading_change": "yes"}, # violation
            "c3": {"turn_dir": "straight", "heading_change": "no"},  # ok
        }
        result = compute_consistency(clips, rules=self.RULES)
        rule_stats = result["per_rule"]["straight_no_heading"]
        assert rule_stats["n_applicable"] == 3
        assert rule_stats["n_violations"] == 1
        assert abs(rule_stats["compliance"] - 0.6667) < 0.001

    def test_consistency_coverage(self):
        """Coverage should reflect what fraction of clips had evaluable rules."""
        clips = {
            "c1": {"turn_dir": "straight", "heading_change": "no"},   # evaluable
            "c2": {"turn_dir": "left", "heading_change": "yes"},      # not evaluable
            "c3": {"braking": "yes", "speed_trend": "decelerating"},  # evaluable
            "c4": {"unrelated": "answer"},                            # not evaluable
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["n_clips"] == 4
        assert result["n_evaluable"] == 2
        assert result["consistency_coverage"] == 0.5

    def test_mean_violations(self):
        """Mean violations should average over evaluable clips only."""
        clips = {
            "c1": {"turn_dir": "straight", "heading_change": "yes"},  # 1 violation
            "c2": {"turn_dir": "straight", "heading_change": "no"},   # 0 violations
            "c3": {"unrelated": "answer"},                            # not evaluable
        }
        result = compute_consistency(clips, rules=self.RULES)
        assert result["n_evaluable"] == 2
        assert result["mean_violations"] == 0.5  # 1 violation / 2 evaluable

    def test_default_rules_count(self):
        """CONSISTENCY_RULES should contain all 10 validated kinematic rules."""
        assert len(CONSISTENCY_RULES) == 10
        names = {r["name"] for r in CONSISTENCY_RULES}
        expected = {
            "heading_change_implies_turning",
            "lateral_accel_implies_turning",
            "straight_implies_no_heading_change",
            "straight_implies_no_high_lateral_accel",
            "highway_implies_not_low_speed",
            "stopped_implies_low_speed",
            "stopped_not_accelerating",
            "brake_then_turn_implies_braking",
            "brake_then_turn_implies_turning",
            "stop_and_go_not_stopped",
        }
        assert names == expected
