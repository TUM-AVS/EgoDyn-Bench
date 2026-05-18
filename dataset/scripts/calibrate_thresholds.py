#!/usr/bin/env python3
"""
Threshold calibration for all questions defined in questions_template.yaml.

For each question, analyses the underlying feature distribution and shows
the resulting class balance with current thresholds.  Suggests
percentile-based alternatives for better balance.

Usage:
    python dataset/scripts/calibrate_thresholds.py \
        --clips-index output/nuscenes_clips/clips_index.jsonl \
                       output/carla_clips/clips_index.jsonl \
        --questions-config dataset/configs/questions_template.yaml \
        --output-report threshold_calibration_report.md
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Add project root so labeling_rules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset.generation.labeling_rules import apply_rule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clips(index_paths: List[Path]) -> List[dict]:
    """Load clips from one or more clips_index.jsonl files."""
    clips = []
    for path in index_paths:
        clips_dir = str(path.parent)
        with open(path) as f:
            for line in f:
                clip = json.loads(line)
                clip["_clips_dir"] = clips_dir
                clips.append(clip)
    logger.info("Loaded %d clips from %d index file(s)", len(clips), len(index_paths))
    return clips


def load_arrays(clip: dict) -> Dict[str, np.ndarray]:
    """Load time-series arrays from a clip's NPZ file."""
    npz_path = Path(clip["_clips_dir"]) / clip["array_ref"]
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def get_feature_values(clips: List[dict], feature_name: str) -> np.ndarray:
    """Extract a scalar feature from all clips that have it."""
    vals = [
        float(c["features"][feature_name])
        for c in clips
        if feature_name in c.get("features", {})
    ]
    return np.array(vals)


def resolve_feature_name(feature: str, aggregation: str) -> str:
    """
    Resolve a template feature name to its clip-features key.

    The question template uses array names (e.g. ``"accel"``) with an
    aggregation (e.g. ``"min"``).  The clip features dict stores these
    as ``"min_accel"``, ``"max_accel"`` etc.  This function tries, in
    order:

    1. The feature name as-is (e.g. ``"mean_abs_jerk"``).
    2. ``"{aggregation}_{feature}"`` (e.g. ``"min_accel"``).
    3. ``"{aggregation}_abs_{feature}"`` (e.g. ``"max_abs_jerk"``).
    """
    # Build ordered list of candidate names
    candidates = [feature]
    if aggregation:
        candidates.append(f"{aggregation}_{feature}")
        candidates.append(f"{aggregation}_abs_{feature}")
    return candidates


def get_resolved_feature_values(
    clips: List[dict],
    feature: str,
    aggregation: str,
) -> tuple[np.ndarray, str]:
    """Try multiple candidate feature names, return (values, used_name)."""
    for name in resolve_feature_name(feature, aggregation):
        vals = get_feature_values(clips, name)
        if len(vals) > 0:
            return vals, name
    return np.array([]), feature


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def fmt_distribution(values: np.ndarray, unit: str = "") -> str:
    """Format distribution statistics as a Markdown table."""
    u = f" {unit}" if unit else ""
    pcts = [5, 10, 25, 50, 75, 90, 95]
    lines = [
        "| Statistic | Value |",
        "|-----------|------:|",
        f"| Count | {len(values)} |",
        f"| Mean | {np.mean(values):.4f}{u} |",
        f"| Std | {np.std(values):.4f}{u} |",
        f"| Min | {np.min(values):.4f}{u} |",
    ]
    for p in pcts:
        lines.append(f"| p{p} | {np.percentile(values, p):.4f}{u} |")
    lines.append(f"| Max | {np.max(values):.4f}{u} |")
    return "\n".join(lines)


def fmt_class_balance(labels: list) -> str:
    """Format class distribution as a Markdown table."""
    counts = Counter(labels)
    total = len(labels)
    lines = [
        "| Class | Count | Fraction |",
        "|-------|------:|:--------:|",
    ]
    for cls in sorted(counts, key=str):
        cnt = counts[cls]
        lines.append(f"| {cls} | {cnt} | {cnt / total:.1%} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def suggest_equal_freq(
    values: np.ndarray,
    n_classes: int,
    labels: List[str],
) -> str:
    """Suggest equal-frequency thresholds and show resulting balance."""
    pcts = [100 * (i + 1) / n_classes for i in range(n_classes - 1)]
    thresholds = [float(np.percentile(values, p)) for p in pcts]
    result = _apply_multi_threshold(values, thresholds, labels)
    counts = Counter(result)
    total = len(result)
    lines = [
        f"**Equal-frequency thresholds** (targeting ~{100 / n_classes:.0f}% per class):",
        "",
        f"Suggested thresholds: `{[round(t, 4) for t in thresholds]}`",
        "",
        "| Class | Count | Fraction |",
        "|-------|------:|:--------:|",
    ]
    for cls in labels:
        cnt = counts.get(cls, 0)
        lines.append(f"| {cls} | {cnt} | {cnt / total:.1%} |")
    return "\n".join(lines)


def _apply_multi_threshold(
    values: np.ndarray,
    thresholds: list,
    labels: list,
) -> list:
    """Classify values using ordered ascending thresholds."""
    result = []
    for v in values:
        assigned = labels[-1]
        for i, t in enumerate(thresholds):
            if v < t:
                assigned = labels[i]
                break
        result.append(assigned)
    return result


def threshold_sensitivity_table(
    values: np.ndarray,
    below_label: str,
    above_label: str,
    current: float,
) -> str:
    """Show how different thresholds change binary class balance."""
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    lines = [
        f"| Threshold | Percentile | {below_label} | {above_label} |",
        "|----------:|:----------:|------:|------:|",
    ]
    total = len(values)
    for p in pcts:
        t = float(np.percentile(values, p))
        n_below = int(np.sum(values < t))
        n_above = total - n_below
        marker = " **<--**" if abs(t - current) / max(abs(current), 1e-9) < 0.05 else ""
        lines.append(
            f"| {t:.4f} | p{p} | {n_below} ({n_below / total:.1%}) "
            f"| {n_above} ({n_above / total:.1%}) |{marker}"
        )
    return "\n".join(lines)


def deadzone_sensitivity_table(
    values: np.ndarray,
    pos_label: str,
    neg_label: str,
    neutral_label: str,
    current_pos: float,
) -> str:
    """Show how different deadzone widths change a 3-class balance."""
    abs_vals = np.abs(values)
    pcts = [10, 20, 30, 40, 50, 60, 70]
    candidates = sorted(set(
        [float(np.percentile(abs_vals, p)) for p in pcts] + [current_pos]
    ))
    lines = [
        f"| Deadzone | {pos_label} | {neutral_label} | {neg_label} |",
        "|----------:|----------:|----------:|----------:|",
    ]
    total = len(values)
    for dz in candidates:
        n_pos = int(np.sum(values > dz))
        n_neg = int(np.sum(values < -dz))
        n_mid = total - n_pos - n_neg
        marker = " **<--**" if abs(dz - current_pos) < 1e-9 else ""
        lines.append(
            f"| {dz:.4f}{marker} | {n_pos} ({n_pos / total:.1%}) "
            f"| {n_mid} ({n_mid / total:.1%}) | {n_neg} ({n_neg / total:.1%}) |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-rule calibration functions
# ---------------------------------------------------------------------------

def calibrate_multi_threshold(
    q: dict, clips: List[dict], f
) -> None:
    """multi_threshold_classification: single feature, N thresholds."""
    params = q["rule"]["params"]
    feature = params["feature"]
    thresholds = params["thresholds"]
    labels = params["labels"]
    agg = params.get("aggregation", "mean")

    values, resolved = get_resolved_feature_values(clips, feature, agg)
    if len(values) == 0:
        f.write(f"> **Warning:** feature `{feature}` not found in any clip.\n\n")
        return

    unit = q.get("units") or ""
    f.write(f"**Feature:** `{resolved}` &ensp; **Aggregation:** `{agg}`\n\n")
    f.write(f"**Current thresholds:** `{thresholds}`  \n")
    f.write(f"**Labels:** `{labels}`\n\n")

    f.write("#### Feature Distribution\n\n")
    f.write(fmt_distribution(values, unit) + "\n\n")

    f.write("#### Current Class Balance\n\n")
    result = _apply_multi_threshold(values, thresholds, labels)
    f.write(fmt_class_balance(result) + "\n\n")

    f.write("#### Suggested Equal-Frequency Alternative\n\n")
    f.write(suggest_equal_freq(values, len(labels), labels) + "\n\n")


def calibrate_threshold(
    q: dict, clips: List[dict], f
) -> None:
    """threshold_classification: single feature, binary threshold."""
    params = q["rule"]["params"]
    feature = params["feature"]
    threshold = params["threshold"]
    below = params["below_label"]
    above = params["above_label"]
    agg = params.get("aggregation", "mean")

    values, resolved = get_resolved_feature_values(clips, feature, agg)
    if len(values) == 0:
        f.write(f"> **Warning:** feature `{feature}` not found in any clip.\n\n")
        return

    f.write(f"**Feature:** `{resolved}` &ensp; **Threshold:** `{threshold}`\n\n")

    f.write("#### Feature Distribution\n\n")
    f.write(fmt_distribution(values) + "\n\n")

    n_below = int(np.sum(values < threshold))
    n_above = len(values) - n_below
    total = len(values)
    f.write("#### Current Class Balance\n\n")
    f.write("| Class | Count | Fraction |\n")
    f.write("|-------|------:|:--------:|\n")
    f.write(f"| {below} | {n_below} | {n_below / total:.1%} |\n")
    f.write(f"| {above} | {n_above} | {n_above / total:.1%} |\n\n")

    f.write("#### Threshold Sensitivity\n\n")
    f.write(threshold_sensitivity_table(values, below, above, threshold) + "\n\n")


def calibrate_deadzone(
    q: dict, clips: List[dict], f
) -> None:
    """yaw_rate_sign_with_deadzone or trend_classification."""
    params = q["rule"]["params"]
    rule_name = q["rule"]["name"]

    if rule_name == "yaw_rate_sign_with_deadzone":
        values = get_feature_values(clips, "signed_max_yaw_rate")
        pos_label = params.get("positive_label", "left")
        neg_label = params.get("negative_label", "right")
        neutral_label = "straight"
        unit = "rad/s"
    else:  # trend_classification
        values = get_feature_values(clips, "mean_accel")
        pos_label = "accelerating"
        neg_label = "decelerating"
        neutral_label = "steady"
        unit = "m/s\u00b2"

    threshold_pos = params["threshold_positive"]
    threshold_neg = params["threshold_negative"]

    if len(values) == 0:
        f.write("> **Warning:** decision variable not found in any clip.\n\n")
        return

    f.write(f"**Deadzone:** `[{threshold_neg}, {threshold_pos}]` {unit}\n\n")

    f.write("#### Feature Distribution\n\n")
    f.write(fmt_distribution(values, unit) + "\n\n")

    result = []
    for v in values:
        if v > threshold_pos:
            result.append(pos_label)
        elif v < threshold_neg:
            result.append(neg_label)
        else:
            result.append(neutral_label)

    f.write("#### Current Class Balance\n\n")
    f.write(fmt_class_balance(result) + "\n\n")

    f.write("#### Deadzone Sensitivity\n\n")
    f.write(
        deadzone_sensitivity_table(
            values, pos_label, neg_label, neutral_label, threshold_pos
        )
        + "\n\n"
    )


def calibrate_or_threshold(
    q: dict, clips: List[dict], f
) -> None:
    """or_threshold_event: OR of multiple conditions."""
    params = q["rule"]["params"]
    conditions = params["conditions"]

    FEATURE_MAP = {
        ("max_abs_jerk", "max"): "max_abs_jerk",
        ("accel", "min"): "min_accel",
        ("accel", "max"): "max_accel",
    }

    condition_masks: List[Optional[np.ndarray]] = []

    for i, cond in enumerate(conditions):
        feat = cond["feature"]
        threshold = cond["threshold"]
        operator = cond["operator"]
        agg = cond.get("aggregation", "max")

        clip_feat = FEATURE_MAP.get((feat, agg), feat)
        values = get_feature_values(clips, clip_feat)

        f.write(f"#### Condition {i + 1}: `{clip_feat}` {operator} `{threshold}`\n\n")

        if len(values) == 0:
            f.write(f"> No values found for `{clip_feat}`.\n\n")
            condition_masks.append(None)
            continue

        f.write(fmt_distribution(values) + "\n\n")

        if operator == "greater_than":
            mask = values > threshold
        else:
            mask = values < threshold
        n_yes = int(np.sum(mask))
        f.write(f"Triggered: {n_yes}/{len(values)} ({n_yes / len(values):.1%})\n\n")

        # Sensitivity
        f.write("| Threshold | Triggered |\n")
        f.write("|----------:|----------:|\n")
        if operator == "greater_than":
            for p in [70, 80, 85, 90, 95]:
                t = float(np.percentile(values, p))
                n = int(np.sum(values > t))
                f.write(f"| {t:.3f} (p{p}) | {n} ({n / len(values):.1%}) |\n")
        else:
            for p in [5, 10, 15, 20, 30]:
                t = float(np.percentile(values, p))
                n = int(np.sum(values < t))
                f.write(f"| {t:.3f} (p{p}) | {n} ({n / len(values):.1%}) |\n")
        f.write("\n")

        condition_masks.append(mask)

    valid = [m for m in condition_masks if m is not None]
    if valid:
        min_len = min(len(m) for m in valid)
        combined = valid[0][:min_len]
        for m in valid[1:]:
            combined = combined | m[:min_len]
        n_yes = int(np.sum(combined))
        total = len(combined)
        f.write("#### Combined (OR)\n\n")
        f.write("| Class | Count | Fraction |\n")
        f.write("|-------|------:|:--------:|\n")
        f.write(f"| yes | {n_yes} | {n_yes / total:.1%} |\n")
        f.write(f"| no | {total - n_yes} | {(total - n_yes) / total:.1%} |\n\n")


def calibrate_lateral_accel(
    q: dict, clips: List[dict], f
) -> None:
    """lateral_accel_threshold: max lateral accel from clip features."""
    params = q["rule"]["params"]
    threshold = params["threshold"]

    values = get_feature_values(clips, "max_lateral_accel")
    if len(values) == 0:
        f.write("> **Warning:** `max_lateral_accel` not found.\n\n")
        return

    f.write(f"**Feature:** `max_lateral_accel` &ensp; **Threshold:** `{threshold}` m/s\u00b2\n\n")
    f.write("#### Feature Distribution\n\n")
    f.write(fmt_distribution(values, "m/s\u00b2") + "\n\n")

    n_yes = int(np.sum(values > threshold))
    total = len(values)
    f.write("#### Current Class Balance\n\n")
    f.write("| Class | Count | Fraction |\n")
    f.write("|-------|------:|:--------:|\n")
    f.write(f"| yes | {n_yes} | {n_yes / total:.1%} |\n")
    f.write(f"| no | {total - n_yes} | {(total - n_yes) / total:.1%} |\n\n")

    f.write("#### Threshold Sensitivity\n\n")
    f.write(threshold_sensitivity_table(values, "no", "yes", threshold) + "\n\n")


def calibrate_feature_conversion(
    q: dict, clips: List[dict], f
) -> None:
    """feature_conversion: numeric question, no classification threshold."""
    params = q["rule"]["params"]
    feature = params["feature"]
    factor = params.get("conversion_factor", 1.0)
    unit = q.get("units") or ""

    values = get_feature_values(clips, feature)
    if len(values) == 0:
        f.write(f"> **Warning:** `{feature}` not found.\n\n")
        return

    converted = values * factor
    f.write(f"**Feature:** `{feature}` \u00d7 {factor}\n\n")
    f.write("#### Distribution (converted)\n\n")
    f.write(fmt_distribution(converted, unit) + "\n\n")
    f.write("*Numeric question \u2014 no thresholds to calibrate.*\n\n")


def calibrate_array_rule(
    q: dict, clips: List[dict], f, *, max_clips: int = 5000
) -> None:
    """
    Generic calibration for array-dependent rules.

    Runs the actual labeling rule on a sample of clips, then shows
    the resulting class balance and threshold sensitivities.
    """
    rule_name = q["rule"]["name"]
    params = q["rule"]["params"]

    sample = clips if len(clips) <= max_clips else _sample(clips, max_clips)
    f.write(f"*Evaluated on {len(sample)} clips (arrays loaded).*\n\n")

    results = []
    evidence_list = []
    skipped = 0
    for clip in sample:
        try:
            arrays = load_arrays(clip)
            answer, evidence = apply_rule(
                rule_name, clip.get("features", {}), arrays, params
            )
            results.append(str(answer))
            evidence_list.append(evidence)
        except Exception:
            skipped += 1

    if not results:
        f.write("> Could not evaluate any clips for this rule.\n\n")
        return

    if skipped:
        f.write(f"*Skipped {skipped} clips due to missing data.*\n\n")

    f.write("#### Current Class Balance\n\n")
    f.write(fmt_class_balance(results) + "\n\n")

    # Rule-specific threshold analysis
    if rule_name == "sequential_event":
        _sequential_sensitivity(q, evidence_list, f)
    elif rule_name == "stop_and_go_detection":
        _stop_and_go_sensitivity(q, sample, f)
    elif rule_name == "half_comparison":
        _half_comparison_sensitivity(q, evidence_list, f)
    elif rule_name == "dominant_axis_comparison":
        _dominant_axis_sensitivity(q, evidence_list, f)
    elif rule_name == "peak_half_detection":
        f.write("*No adjustable thresholds for peak-half detection.*\n\n")


# ---------------------------------------------------------------------------
# Array-rule sensitivity helpers
# ---------------------------------------------------------------------------

def _sequential_sensitivity(q: dict, evidence_list: list, f) -> None:
    """Show first/second event trigger rates for sequential_event."""
    params = q["rule"]["params"]
    first_counts = [e.get("first_event_count", 0) for e in evidence_list]
    second_counts = [e.get("second_event_count", 0) for e in evidence_list]
    total = len(evidence_list)

    first_cfg = params["first_event"]
    second_cfg = params["second_event"]

    n_first = sum(1 for c in first_counts if c > 0)
    n_second = sum(1 for c in second_counts if c > 0)

    f.write("#### Event Trigger Rates\n\n")
    f.write("| Event | Threshold | Clips with event |\n")
    f.write("|-------|----------:|:----------------:|\n")
    f.write(
        f"| 1st: `{first_cfg['feature']}` {first_cfg['operator']} "
        f"| {first_cfg['threshold']} | {n_first}/{total} ({n_first / total:.1%}) |\n"
    )
    f.write(
        f"| 2nd: `{second_cfg['feature']}` {second_cfg['operator']} "
        f"| {second_cfg['threshold']} | {n_second}/{total} ({n_second / total:.1%}) |\n"
    )
    f.write("\n")

    # Gap distribution for clips where sequence was found
    gaps = [e["gap_seconds"] for e in evidence_list if e.get("sequence_found")]
    if gaps:
        f.write("#### Gap Distribution (clips with sequence)\n\n")
        f.write(fmt_distribution(np.array(gaps), "s") + "\n\n")


def _stop_and_go_sensitivity(q: dict, clips: list, f) -> None:
    """Show how stop/go thresholds affect stop-and-go detection."""
    params = q["rule"]["params"]
    go_thr = params["go_speed_threshold"]
    min_stops = params.get("min_stops", 1)

    f.write("#### Stop Threshold Sensitivity\n\n")
    f.write("| Stop (m/s) | Go (m/s) | Yes | No |\n")
    f.write("|----------:|--------:|----:|---:|\n")

    for stop_thr in [0.3, 0.5, 0.8, 1.0, 1.5]:
        n_yes = 0
        n_total = 0
        for clip in clips:
            try:
                speed = load_arrays(clip)["speed"]
                cycles = 0
                was_stopped = bool(speed[0] < stop_thr)
                for s in speed[1:]:
                    if not was_stopped and s < stop_thr:
                        was_stopped = True
                    elif was_stopped and s > go_thr:
                        cycles += 1
                        was_stopped = False
                n_total += 1
                if cycles >= min_stops:
                    n_yes += 1
            except Exception:
                n_total += 1
        if n_total:
            marker = " **<--**" if abs(stop_thr - params["stop_speed_threshold"]) < 1e-9 else ""
            f.write(
                f"| {stop_thr}{marker} | {go_thr} "
                f"| {n_yes} ({n_yes / n_total:.1%}) "
                f"| {n_total - n_yes} ({(n_total - n_yes) / n_total:.1%}) |\n"
            )
    f.write("\n")


def _half_comparison_sensitivity(q: dict, evidence_list: list, f) -> None:
    """Show how similarity_threshold affects half_comparison."""
    params = q["rule"]["params"]
    agg = params.get("aggregation", "rms")

    first_key = f"{agg}_first_half"
    second_key = f"{agg}_second_half"

    ratios = []
    bigger = []
    for e in evidence_list:
        v1 = abs(e.get(first_key, 0))
        v2 = abs(e.get(second_key, 0))
        denom = max(v1, v2, 1e-9)
        ratios.append(abs(v1 - v2) / denom)
        bigger.append("first_half" if v1 > v2 else "second_half")

    if not ratios:
        return

    ratio_arr = np.array(ratios)
    f.write("#### Relative Difference Distribution\n\n")
    f.write(fmt_distribution(ratio_arr) + "\n\n")

    f.write("#### Similarity Threshold Sensitivity\n\n")
    f.write("| Threshold | Similar | First Half | Second Half |\n")
    f.write("|----------:|--------:|-----------:|------------:|\n")
    total = len(ratios)
    current = params.get("similarity_threshold", 0.2)
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        n_sim = int(np.sum(ratio_arr < t))
        n_first = sum(1 for r, b in zip(ratios, bigger) if r >= t and b == "first_half")
        n_second = total - n_sim - n_first
        marker = " **<--**" if abs(t - current) < 1e-9 else ""
        f.write(
            f"| {t}{marker} | {n_sim} ({n_sim / total:.1%}) "
            f"| {n_first} ({n_first / total:.1%}) "
            f"| {n_second} ({n_second / total:.1%}) |\n"
        )
    f.write("\n")


def _dominant_axis_sensitivity(q: dict, evidence_list: list, f) -> None:
    """Show how ratio_threshold affects dominant_axis_comparison."""
    params = q["rule"]["params"]
    agg = params.get("aggregation", "rms")

    long_key = f"{agg}_longitudinal_accel"
    lat_key = f"{agg}_lateral_accel"

    ratios = []
    for e in evidence_list:
        long_v = e.get(long_key, 0)
        lat_v = e.get(lat_key, 0)
        if long_v > 1e-9:
            ratios.append(lat_v / long_v)

    if not ratios:
        return

    ratio_arr = np.array(ratios)
    f.write("#### Lateral / Longitudinal Ratio Distribution\n\n")
    f.write(fmt_distribution(ratio_arr) + "\n\n")

    f.write("#### Ratio Threshold Sensitivity\n\n")
    f.write("| Threshold | Longitudinal | Lateral |\n")
    f.write("|----------:|-------------:|--------:|\n")
    total = len(ratios)
    current = params.get("ratio_threshold", 1.0)
    for t in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        n_lat = int(np.sum(ratio_arr > t))
        n_long = total - n_lat
        marker = " **<--**" if abs(t - current) < 1e-9 else ""
        f.write(
            f"| {t}{marker} | {n_long} ({n_long / total:.1%}) "
            f"| {n_lat} ({n_lat / total:.1%}) |\n"
        )
    f.write("\n")


def _sample(clips: list, n: int) -> list:
    """Deterministically sample n clips."""
    rng = np.random.RandomState(42)
    indices = rng.choice(len(clips), size=n, replace=False)
    return [clips[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Rule dispatcher
# ---------------------------------------------------------------------------

CALIBRATORS = {
    "multi_threshold_classification": calibrate_multi_threshold,
    "threshold_classification": calibrate_threshold,
    "yaw_rate_sign_with_deadzone": "deadzone",
    "trend_classification": "deadzone",
    "or_threshold_event": calibrate_or_threshold,
    "lateral_accel_threshold": calibrate_lateral_accel,
    "feature_conversion": calibrate_feature_conversion,
    "sequential_event": "array",
    "peak_half_detection": "array",
    "stop_and_go_detection": "array",
    "half_comparison": "array",
    "dominant_axis_comparison": "array",
}


def calibrate_question(
    q: dict,
    clips: List[dict],
    f,
    *,
    max_array_clips: int,
) -> None:
    """Dispatch calibration for a single question."""
    rule_name = q["rule"]["name"]
    calibrator = CALIBRATORS.get(rule_name)

    if calibrator is None:
        f.write(f"*No calibrator implemented for rule `{rule_name}`.*\n\n")
    elif calibrator == "deadzone":
        calibrate_deadzone(q, clips, f)
    elif calibrator == "array":
        calibrate_array_rule(q, clips, f, max_clips=max_array_clips)
    else:
        calibrator(q, clips, f)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    clips: List[dict],
    questions: List[dict],
    output_path: Path,
    max_array_clips: int,
) -> None:
    """Generate a comprehensive threshold calibration report."""
    with open(output_path, "w") as f:
        f.write("# Threshold Calibration Report\n\n")
        f.write(f"**Clips analysed:** {len(clips)}\n\n")
        f.write("---\n\n")

        # Table of contents
        f.write("## Contents\n\n")
        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            rule = q["rule"]["name"]
            f.write(f"{i}. [{qid}](#{qid}) \u2014 `{rule}`\n")
        f.write("\n---\n\n")

        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            rule_name = q["rule"]["name"]
            category = q.get("category", "")
            desc = q.get("metadata", {}).get("description", "")

            f.write(f"## {i}. {qid}\n\n")
            f.write(f"**Question:** {q['question_text']}  \n")
            f.write(f"**Category:** {category} &ensp; ")
            f.write(f"**Answer type:** {q['answer_type']} &ensp; ")
            f.write(f"**Rule:** `{rule_name}`  \n")
            if desc:
                f.write(f"*{desc}*\n")
            f.write("\n")

            logger.info(
                "Calibrating %d/%d: %s (%s)...",
                i, len(questions), qid, rule_name,
            )
            calibrate_question(
                q, clips, f, max_array_clips=max_array_clips
            )

            f.write("---\n\n")

        f.write("*Report generated by `calibrate_thresholds.py`*\n")

    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate thresholds for all questions in the template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips-index",
        nargs="+",
        required=True,
        help="Path(s) to clips_index.jsonl (one per source)",
    )
    parser.add_argument(
        "--questions-config",
        default="dataset/configs/questions_template.yaml",
        help="Path to questions_template.yaml",
    )
    parser.add_argument(
        "--output-report",
        default="threshold_calibration_report.md",
        help="Output path for calibration report",
    )
    parser.add_argument(
        "--max-array-clips",
        type=int,
        default=5000,
        help="Max clips to load arrays for (array-dependent rules)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.questions_config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1
    with open(config_path) as f:
        config = yaml.safe_load(f)
    questions = config["questions"]
    logger.info("Loaded %d questions from %s", len(questions), config_path)

    # Load clips
    index_paths = [Path(p) for p in args.clips_index]
    for p in index_paths:
        if not p.exists():
            logger.error("Clips index not found: %s", p)
            return 1
    clips = load_clips(index_paths)

    # Generate report
    generate_report(
        clips,
        questions,
        Path(args.output_report),
        max_array_clips=args.max_array_clips,
    )

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
