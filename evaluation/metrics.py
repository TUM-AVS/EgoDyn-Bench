"""Evaluation metrics for EgoDyn-Bench.

Computes Semantic Accuracy, Macro F1, and Ego-Motion Consistency Rate at
global, per-category, and per-question granularity.  No sklearn dependency —
all metrics are implemented in pure Python + stdlib.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evaluation.parsers import load_question_config, normalize_numeric, parse_answer


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def semantic_accuracy(oracle: list[str], pred: list[str]) -> float:
    """Fraction of exact matches between *oracle* and *pred*."""
    if len(oracle) != len(pred):
        raise ValueError(
            f"oracle and pred must have the same length, got {len(oracle)} vs {len(pred)}"
        )
    if not oracle:
        return 0.0
    return sum(o == p for o, p in zip(oracle, pred)) / len(oracle)


def confusion_matrix(
    oracle: list[str],
    pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build a confusion matrix from *oracle* and *pred* label lists.

    Parameters
    ----------
    oracle / pred:
        Parallel lists of ground-truth and predicted labels.
    labels:
        Explicit ordered label set.  If ``None``, derived from the union
        of *oracle* and *pred* (sorted).

    Returns
    -------
    ``{"labels": [...], "matrix": [[...]]}`` where ``matrix[i][j]`` is the
    count of samples with ``oracle == labels[i]`` and ``pred == labels[j]``.
    """
    if len(oracle) != len(pred):
        raise ValueError(
            f"oracle and pred must have the same length, got {len(oracle)} vs {len(pred)}"
        )
    if labels is None:
        labels = sorted(set(oracle) | set(pred))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for o, p in zip(oracle, pred):
        i = label_to_idx.get(o)
        j = label_to_idx.get(p)
        if i is not None and j is not None:
            mat[i][j] += 1
    return {"labels": labels, "matrix": mat}


def macro_f1(
    oracle: list[str],
    pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Macro-averaged F1 across all classes.

    Parameters
    ----------
    oracle / pred:
        Parallel lists of ground-truth and predicted labels.
    labels:
        Fixed set of classes to evaluate.  If ``None``, defaults to the
        union of *oracle* and *pred*.

    Each class gets equal weight regardless of frequency.  Classes present
    in labels but never predicted receive F1 = 0.
    """
    if len(oracle) != len(pred):
        raise ValueError(
            f"oracle and pred must have the same length, got {len(oracle)} vs {len(pred)}"
        )
    if labels is None:
        classes = sorted(set(oracle) | set(pred))
    else:
        classes = labels

    if not classes:
        return 0.0

    f1s: list[float] = []
    for cls in classes:
        tp = sum(o == cls and p == cls for o, p in zip(oracle, pred))
        fp = sum(o != cls and p == cls for o, p in zip(oracle, pred))
        fn = sum(o == cls and p != cls for o, p in zip(oracle, pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)

    return sum(f1s) / len(f1s)


def balanced_accuracy(
    oracle: list[str],
    pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Balanced Accuracy (arithmetic mean of recalls per class).

    Parameters
    ----------
    oracle / pred:
        Parallel lists of ground-truth and predicted labels.
    labels:
        Fixed set of classes to evaluate.  If ``None``, defaults to the
        union of *oracle* and *pred*.
    """
    if len(oracle) != len(pred):
        raise ValueError(
            f"oracle and pred must have the same length, got {len(oracle)} vs {len(pred)}"
        )
    if labels is None:
        classes = sorted(set(oracle) | set(pred))
    else:
        classes = labels

    if not classes:
        return 0.0

    recalls: list[float] = []
    for cls in classes:
        tp = sum(o == cls and p == cls for o, p in zip(oracle, pred))
        fn = sum(o == cls and p != cls for o, p in zip(oracle, pred))

        # If a class has no ground truth support, recall is mathematically undefined.
        # However, for balanced accuracy in a fixed-label setting, we typically only
        # average over classes that actually appear in the ground truth (support > 0).
        # Including zero-support classes as 0.0 would unfairly penalize the model
        # for a class it couldn't possibly get right.
        if (tp + fn) > 0:
            recalls.append(tp / (tp + fn))

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Grouped metric helper
# ---------------------------------------------------------------------------

def _compute_group_metrics(
    oracle: list[str],
    pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Return accuracy + balanced_acc + macro_f1 + n for a single group."""
    return {
        "accuracy": round(semantic_accuracy(oracle, pred), 4),
        "balanced_acc": round(balanced_accuracy(oracle, pred, labels), 4),
        "macro_f1": round(macro_f1(oracle, pred, labels), 4),
        "n": len(oracle),
    }


# ---------------------------------------------------------------------------
# Ego-Motion Consistency
# ---------------------------------------------------------------------------

# Each rule is a hard kinematic contradiction: if the model answers X for
# one question, it must answer Y for another — otherwise the model is
# contradicting basic physics.  Only "physics-obvious" rules that are
# independent of threshold calibration are included; soft/subjective
# implications (e.g. smooth ↔ emergency, lateral accel ↔ turning) are
# deliberately excluded to avoid reviewer pushback.
#
# IMPORTANT: All rules are validated against ground-truth labels to ensure
# zero oracle violations.  Rules involving braking_intensity ↔ speed_trend
# or braking_intensity ↔ extreme_maneuver were removed because different
# aggregation methods (min vs mean) cause legitimate threshold mismatches
# (e.g. a clip can accelerate on average while having a brief hard brake).

CONSISTENCY_RULES: list[dict[str, Any]] = [
    # --- Heading / lateral dynamics (bidirectional) ---
    {
        "name": "heading_change_implies_turning",
        "description": "Significant heading change implies the vehicle is turning (not straight)",
        "if": ("significant_heading_change", "==", "yes"),
        "then": ("yaw_rate_turn_direction", "!=", "straight"),
    },
    {
        "name": "lateral_accel_implies_turning",
        "description": "High lateral acceleration implies the vehicle is turning (not straight)",
        "if": ("high_lateral_accel", "==", "yes"),
        "then": ("yaw_rate_turn_direction", "!=", "straight"),
    },
    {
        "name": "straight_implies_no_heading_change",
        "description": "Going straight implies no significant heading change",
        "if": ("yaw_rate_turn_direction", "==", "straight"),
        "then": ("significant_heading_change", "==", "no"),
    },
    {
        "name": "straight_implies_no_high_lateral_accel",
        "description": "Going straight implies no high lateral acceleration",
        "if": ("yaw_rate_turn_direction", "==", "straight"),
        "then": ("high_lateral_accel", "==", "no"),
    },
    # --- Speed regime ↔ mean speed ---
    {
        "name": "highway_implies_not_low_speed",
        "description": "Highway speed regime implies mean speed is not low",
        "if": ("speed_regime", "==", "highway"),
        "then": ("mean_speed_low", "==", "no"),
    },
    {
        "name": "stopped_implies_low_speed",
        "description": "Stopped speed regime implies mean speed is low",
        "if": ("speed_regime", "==", "stopped"),
        "then": ("mean_speed_low", "==", "yes"),
    },
    {
        "name": "stopped_not_accelerating",
        "description": "Stopped speed regime implies speed trend is not accelerating",
        "if": ("speed_regime", "==", "stopped"),
        "then": ("speed_trend", "!=", "accelerating"),
    },
    # --- Brake-then-turn compound event ---
    {
        "name": "brake_then_turn_implies_braking",
        "description": "Brake-then-turn implies braking intensity is not none",
        "if": ("brake_then_turn", "==", "yes"),
        "then": ("braking_intensity", "!=", "none"),
    },
    {
        "name": "brake_then_turn_implies_turning",
        "description": "Brake-then-turn implies vehicle is not going straight",
        "if": ("brake_then_turn", "==", "yes"),
        "then": ("yaw_rate_turn_direction", "!=", "straight"),
    },
    # --- Stop-and-go ---
    {
        "name": "stop_and_go_not_stopped",
        "description": "Stop-and-go implies speed regime is not stopped (vehicle must also move)",
        "if": ("stop_and_go", "==", "yes"),
        "then": ("speed_regime", "!=", "stopped"),
    },
]


def _check_rule(
    rule: dict[str, Any],
    clip_answers: dict[str, str],
) -> tuple[bool, bool]:
    """Check whether a consistency rule is violated for one clip.

    Returns ``(applicable, violated)`` where *applicable* means the
    condition was triggered (both questions answered and condition matched).
    """
    cond_qid, cond_op, cond_val = rule["if"]
    impl_qid, impl_op, impl_val = rule["then"]

    cond_answer = clip_answers.get(cond_qid)
    impl_answer = clip_answers.get(impl_qid)

    # Rule not testable if either question wasn't answered for this clip
    if cond_answer is None or impl_answer is None:
        return False, False

    # Check if condition is triggered
    cond_matches = (cond_answer == cond_val) if cond_op == "==" else (cond_answer != cond_val)
    if not cond_matches:
        return False, False

    # Condition triggered — check if implication holds
    impl_holds = (impl_answer == impl_val) if impl_op == "==" else (impl_answer != impl_val)
    return True, not impl_holds


def compute_consistency(
    clip_answers: dict[str, dict[str, str]],
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute ego-motion consistency rate across clips.

    Parameters
    ----------
    clip_answers:
        ``{clip_id: {question_id: parsed_pred_label}}`` for all
        successfully parsed model predictions.
    rules:
        Consistency rules to check.  Defaults to :data:`CONSISTENCY_RULES`.

    Returns
    -------
    Dict with:

    * ``rate`` — fraction of clips with zero violations (EMCR).
    * ``n_clips`` — total clips with at least one parsed answer.
    * ``n_evaluable`` — clips where at least one rule was evaluable.
    * ``n_consistent`` — evaluable clips with zero violations.
    * ``consistency_coverage`` — ``n_evaluable / n_clips``.
    * ``mean_violations`` — average violation count per evaluable clip.
    * ``per_rule`` — per-rule breakdown with ``n_applicable``,
      ``n_violations``, and ``compliance``.
    """
    if rules is None:
        rules = CONSISTENCY_RULES

    per_rule: dict[str, dict[str, Any]] = {
        r["name"]: {"n_applicable": 0, "n_violations": 0}
        for r in rules
    }
    n_clips = len(clip_answers)
    n_consistent = 0
    n_evaluable = 0
    total_violations = 0
    clip_compliance_scores: list[float] = []  # per-clip fraction of satisfied rules

    for answers in clip_answers.values():
        clip_violations = 0
        clip_n_applicable = 0
        for rule in rules:
            applicable, violated = _check_rule(rule, answers)
            if applicable:
                clip_n_applicable += 1
                per_rule[rule["name"]]["n_applicable"] += 1
                if violated:
                    per_rule[rule["name"]]["n_violations"] += 1
                    clip_violations += 1
        if clip_n_applicable > 0:
            n_evaluable += 1
            total_violations += clip_violations
            clip_compliance_scores.append(
                (clip_n_applicable - clip_violations) / clip_n_applicable
            )
            if clip_violations == 0:
                n_consistent += 1

    # Compute per-rule compliance rates
    for stats in per_rule.values():
        n_app = stats["n_applicable"]
        stats["compliance"] = round(1 - stats["n_violations"] / n_app, 4) if n_app > 0 else 1.0

    emcr = round(n_consistent / n_evaluable, 4) if n_evaluable > 0 else 0.0
    # Rule coverage: fraction of rules that triggered at least once
    n_rules_triggered = sum(
        1 for stats in per_rule.values() if stats["n_applicable"] > 0
    )
    rule_coverage = n_rules_triggered / len(rules) if rules else 0.0
    # Weighted EMCR: penalises models where few rules trigger (degenerate answers)
    wemcr = round(emcr * rule_coverage, 4)

    # Graded consistency: mean per-clip fraction of satisfied rules
    mean_compliance = (
        round(sum(clip_compliance_scores) / len(clip_compliance_scores), 4)
        if clip_compliance_scores
        else 0.0
    )

    return {
        "rate": emcr,
        "wemcr": wemcr,
        "mean_compliance": mean_compliance,
        "rule_coverage": round(rule_coverage, 4),
        "n_rules_triggered": n_rules_triggered,
        "n_rules_total": len(rules),
        "n_clips": n_clips,
        "n_evaluable": n_evaluable,
        "n_consistent": n_consistent,
        "consistency_coverage": round(n_evaluable / n_clips, 4) if n_clips > 0 else 0.0,
        "mean_violations": round(total_violations / n_evaluable, 4) if n_evaluable > 0 else 0.0,
        "per_rule": per_rule,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    records: list[dict[str, Any]],
    question_config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run full evaluation over a list of prediction records.

    Parameters
    ----------
    records:
        Each dict must have keys ``question_id``, ``category``,
        ``model_answer``, and either ``oracle_label`` or ``answer``
        (the QA generator uses ``answer``; ``oracle_label`` is also
        accepted for convenience).
    question_config:
        Output of :func:`evaluation.parsers.load_question_config`.

    Returns
    -------
    Nested dict with ``global``, ``per_category``, ``per_question`` metrics
    plus parsability stats.
    """
    n_total = len(records)

    # Parse all answers
    parsed_oracle: list[str] = []
    parsed_pred: list[str] = []
    n_unparsed = 0

    # Group buckets
    cat_oracle: dict[str, list[str]] = defaultdict(list)
    cat_pred: dict[str, list[str]] = defaultdict(list)
    q_oracle: dict[str, list[str]] = defaultdict(list)
    q_pred: dict[str, list[str]] = defaultdict(list)
    temporal_oracle: list[str] = []
    temporal_pred: list[str] = []
    clip_answers: dict[str, dict[str, str]] = defaultdict(dict)
    # Per-source (nuScenes vs CARLA) buckets
    src_oracle: dict[str, list[str]] = defaultdict(list)
    src_pred: dict[str, list[str]] = defaultdict(list)
    src_clip_answers: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for rec in records:
        qid = rec.get("question_id")
        clip_id = rec.get("clip_id")
        if qid is None or clip_id is None:
            n_unparsed += 1
            continue
        category = rec.get("category", "unknown")

        # Accept both "oracle_label" (evaluation convention) and "answer"
        # (QA generator output) so the same evaluate() works on either format.
        oracle_raw = rec.get("oracle_label", rec.get("answer"))
        if oracle_raw is None:
            n_unparsed += 1
            continue
        oracle_label = str(oracle_raw).lower().strip()

        model_answer = rec.get("model_answer", "")

        qcfg = question_config.get(qid)
        if qcfg is None:
            n_unparsed += 1
            continue

        # Normalize numeric oracle labels so "42.0" and "42" compare equal
        if qcfg["answer_type"] == "numeric":
            oracle_label = normalize_numeric(oracle_label)

        pred_label = parse_answer(
            model_answer,
            qcfg["choices"],
            qcfg["answer_type"],
        )

        if pred_label is None:
            n_unparsed += 1
            continue

        parsed_oracle.append(oracle_label)
        parsed_pred.append(pred_label)

        cat_oracle[category].append(oracle_label)
        cat_pred[category].append(pred_label)
        q_oracle[qid].append(oracle_label)
        q_pred[qid].append(pred_label)

        clip_answers[clip_id][qid] = pred_label

        source = "nuscenes" if clip_id.startswith("clip_") else "carla"
        src_oracle[source].append(oracle_label)
        src_pred[source].append(pred_label)
        src_clip_answers[source][clip_id][qid] = pred_label

        if qcfg.get("temporal", False):
            temporal_oracle.append(oracle_label)
            temporal_pred.append(pred_label)

    n_parsed = len(parsed_oracle)

    result: dict[str, Any] = {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parsable_coverage": round(n_parsed / n_total, 4) if n_total else 0.0,
        "global": _compute_group_metrics(parsed_oracle, parsed_pred),
        "temporal": _compute_group_metrics(temporal_oracle, temporal_pred),
        "per_category": {
            cat: _compute_group_metrics(cat_oracle[cat], cat_pred[cat])
            for cat in sorted(cat_oracle)
        },
        "per_question": {
            qid: {
                **_compute_group_metrics(
                    q_oracle[qid],
                    q_pred[qid],
                    labels=question_config[qid].get("choices"),
                ),
                "confusion_matrix": confusion_matrix(
                    q_oracle[qid],
                    q_pred[qid],
                    labels=question_config[qid].get("choices"),
                ),
            }
            for qid in sorted(q_oracle)
        },
        "consistency": compute_consistency(dict(clip_answers)),
        "per_source": {
            src: {
                **_compute_group_metrics(src_oracle[src], src_pred[src]),
                "consistency": compute_consistency(dict(src_clip_answers[src])),
            }
            for src in sorted(src_oracle)
        },
    }

    return result
