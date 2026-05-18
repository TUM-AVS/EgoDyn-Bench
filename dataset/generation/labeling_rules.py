"""
Labeling rules engine for automatic QA generation.

Implements configurable rule functions that take clip features/arrays
and return ground truth labels with evidence.
"""

import logging
from typing import Any, Callable, Dict, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class LabelingRuleRegistry:
    """Registry for labeling rule functions."""

    _rules: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a rule function."""
        def decorator(func: Callable):
            if name in cls._rules:
                logger.warning(
                    "Overwriting existing rule '%s' (was %s, now %s)",
                    name, cls._rules[name].__name__, func.__name__,
                )
            cls._rules[name] = func
            return func
        return decorator

    @classmethod
    def get_rule(cls, name: str) -> Callable:
        """Get a registered rule function by name."""
        if name not in cls._rules:
            raise ValueError(
                f"Unknown rule '{name}'. Available rules: {list(cls._rules.keys())}"
            )
        return cls._rules[name]

    @classmethod
    def list_rules(cls) -> list[str]:
        """List all registered rule names."""
        return list(cls._rules.keys())


def apply_rule(
    rule_name: str,
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply a labeling rule to generate an answer.

    Args:
        rule_name: Name of the rule function
        clip_features: Pre-computed clip features
        clip_arrays: Clip time series arrays
        params: Rule parameters from config

    Returns:
        Tuple of (answer, evidence_dict)
    """
    rule_func = LabelingRuleRegistry.get_rule(rule_name)
    return rule_func(clip_features, clip_arrays, params)


def aggregate_array(
    arr: np.ndarray,
    aggregation: str
) -> float:
    """
    Aggregate array using specified method.

    Uses NaN-safe operations so that isolated NaN values do not
    silently corrupt the result.

    Args:
        arr: Input array (must be non-empty)
        aggregation: Aggregation method (mean, min, max, rms, p05, p25, p50, p75, p95)

    Returns:
        Aggregated value

    Raises:
        ValueError: If *arr* is empty or *aggregation* is unknown.
    """
    if len(arr) == 0:
        raise ValueError(
            f"Cannot aggregate empty array with method '{aggregation}'"
        )

    if aggregation == "mean":
        return float(np.nanmean(arr))
    elif aggregation == "min":
        return float(np.nanmin(arr))
    elif aggregation == "max":
        return float(np.nanmax(arr))
    elif aggregation == "rms":
        return float(np.sqrt(np.nanmean(arr**2)))
    elif aggregation == "abs_max":
        # Returns the value with the largest absolute magnitude, preserving its sign
        idx = np.nanargmax(np.abs(arr))
        return float(arr[idx])
    elif aggregation == "max_abs":
        # Returns the raw maximum absolute magnitude
        return float(np.nanmax(np.abs(arr)))
    elif aggregation.startswith("p"):
        # Percentile: p05, p50, p95, etc.
        percentile = int(aggregation[1:])
        return float(np.nanpercentile(arr, percentile))
    else:
        raise ValueError(f"Unknown aggregation method: '{aggregation}'")


# ============================================================================
# Rule Implementations
# ============================================================================

@LabelingRuleRegistry.register("yaw_rate_sign_with_deadzone")
def yaw_rate_sign_with_deadzone(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify turn direction based on yaw rate sign with deadzone.

    Params:
        threshold_positive: Positive threshold (rad/s)
        threshold_negative: Negative threshold (rad/s)
        aggregation: How to aggregate yaw_rate (mean, max, min, p50, etc.)
        positive_label: Label when yaw rate > threshold_positive (default: "left")
        negative_label: Label when yaw rate < threshold_negative (default: "right")

    Returns:
        (positive_label | negative_label | "straight", evidence)
    """
    threshold_pos = params["threshold_positive"]
    threshold_neg = params["threshold_negative"]
    aggregation = params.get("aggregation", "mean")
    positive_label = params.get("positive_label", "left")
    negative_label = params.get("negative_label", "right")

    if "yaw_rate" not in clip_arrays:
        raise ValueError("Required array 'yaw_rate' not found in clip arrays")

    # Aggregate yaw rate
    yaw_rate = clip_arrays["yaw_rate"]
    agg_yaw_rate = aggregate_array(yaw_rate, aggregation)

    # Classify
    if agg_yaw_rate > threshold_pos:
        answer = positive_label
    elif agg_yaw_rate < threshold_neg:
        answer = negative_label
    else:
        answer = "straight"

    evidence = {
        f"{aggregation}_yaw_rate": agg_yaw_rate,
        "threshold_positive": threshold_pos,
        "threshold_negative": threshold_neg,
        "positive_label": positive_label,
        "negative_label": negative_label,
    }

    return answer, evidence


@LabelingRuleRegistry.register("threshold_event")
def threshold_event(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect if a threshold event occurs.

    Params:
        feature: Feature name (from arrays or features)
        threshold: Threshold value
        operator: "less_than" or "greater_than"
        aggregation: How to aggregate (min, max, mean, etc.)

    Returns:
        ("yes" | "no", evidence)
    """
    feature_name = params["feature"]
    threshold = params["threshold"]
    operator = params["operator"]
    aggregation = params.get("aggregation", "min")

    # Get feature value
    if feature_name in clip_arrays:
        # Aggregate from time series
        feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
    elif feature_name in clip_features:
        # Use pre-computed feature
        feature_value = clip_features[feature_name]
    else:
        raise ValueError(f"Feature '{feature_name}' not found in arrays or features")

    # Apply threshold
    if operator == "less_than":
        event_occurred = feature_value < threshold
    elif operator == "greater_than":
        event_occurred = feature_value > threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")

    answer = "yes" if event_occurred else "no"

    evidence = {
        f"{aggregation}_{feature_name}": feature_value,
        "threshold": threshold,
        "operator": operator
    }

    return answer, evidence


@LabelingRuleRegistry.register("or_threshold_event")
def or_threshold_event(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect if ANY of several threshold conditions are met (logical OR).

    Params:
        conditions: List of dicts, each with:
            feature: Feature name (from arrays or features)
            threshold: Threshold value
            operator: "less_than" or "greater_than"
            aggregation: How to aggregate (min, max, mean, etc.)
        above_label: Label when at least one condition is met (default: "yes")
        below_label: Label when no condition is met (default: "no")

    Returns:
        (above_label | below_label, evidence)
    """
    conditions = params["conditions"]
    above_label = params.get("above_label", "yes")
    below_label = params.get("below_label", "no")

    any_met = False
    evidence = {"conditions": []}

    for cond in conditions:
        feature_name = cond["feature"]
        threshold = cond["threshold"]
        operator = cond["operator"]
        aggregation = cond.get("aggregation", "max")

        # Get feature value
        if feature_name in clip_arrays:
            feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
        elif feature_name in clip_features:
            feature_value = clip_features[feature_name]
        else:
            raise ValueError(f"Feature '{feature_name}' not found in arrays or features")

        # Apply threshold
        if operator == "less_than":
            met = feature_value < threshold
        elif operator == "greater_than":
            met = feature_value > threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")

        if met:
            any_met = True

        evidence["conditions"].append({
            "feature": feature_name,
            f"{aggregation}_{feature_name}": feature_value,
            "threshold": threshold,
            "operator": operator,
            "met": met,
        })

    answer = above_label if any_met else below_label
    return answer, evidence


@LabelingRuleRegistry.register("feature_conversion")
def feature_conversion(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Convert and return a feature value (possibly with unit conversion).

    Params:
        feature: Feature name (from clip_features)
        conversion_factor: Multiplicative factor (default: 1.0)
        rounding: Number of decimal places (default: None)

    Returns:
        (converted_value, evidence)
    """
    feature_name = params["feature"]
    conversion_factor = params.get("conversion_factor", 1.0)
    rounding = params.get("rounding", None)

    if feature_name not in clip_features:
        raise ValueError(f"Feature '{feature_name}' not found in clip_features")

    # Get and convert value
    raw_value = clip_features[feature_name]
    converted_value = raw_value * conversion_factor

    # Round if specified
    if rounding is not None:
        converted_value = round(converted_value, rounding)

    evidence = {
        "raw_value": raw_value,
        "conversion_factor": conversion_factor,
        "converted_value": converted_value
    }

    return converted_value, evidence


@LabelingRuleRegistry.register("threshold_classification")
def threshold_classification(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Binary classification based on threshold.

    Params:
        feature: Feature name (from features or arrays)
        threshold: Threshold value
        below_label: Label when value < threshold
        above_label: Label when value >= threshold
        aggregation: Optional, if using arrays (default: mean)

    Returns:
        (label, evidence)
    """
    feature_name = params["feature"]
    threshold = params["threshold"]
    below_label = params["below_label"]
    above_label = params["above_label"]
    aggregation = params.get("aggregation", "mean")

    # Get feature value
    if feature_name in clip_features:
        feature_value = clip_features[feature_name]
    elif feature_name in clip_arrays:
        feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
    else:
        raise ValueError(f"Feature '{feature_name}' not found")

    # Classify
    answer = below_label if feature_value < threshold else above_label

    evidence = {
        f"value_{feature_name}": feature_value,
        "threshold": threshold
    }

    return answer, evidence


@LabelingRuleRegistry.register("trend_classification")
def trend_classification(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify trend (accelerating/decelerating/steady) with deadzone.

    Params:
        feature: Feature name (typically 'accel')
        threshold_positive: Threshold for increasing trend
        threshold_negative: Threshold for decreasing trend
        aggregation: How to aggregate (mean, median, etc.)

    Returns:
        ("accelerating" | "decelerating" | "steady", evidence)
    """
    feature_name = params["feature"]
    threshold_pos = params["threshold_positive"]
    threshold_neg = params["threshold_negative"]
    aggregation = params.get("aggregation", "mean")

    # Get feature value
    if feature_name in clip_arrays:
        feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
    elif feature_name in clip_features:
        feature_value = clip_features[feature_name]
    else:
        raise ValueError(f"Feature '{feature_name}' not found")

    # Classify trend
    if feature_value > threshold_pos:
        answer = "accelerating"
    elif feature_value < threshold_neg:
        answer = "decelerating"
    else:
        answer = "steady"

    evidence = {
        f"{aggregation}_{feature_name}": feature_value,
        "threshold_positive": threshold_pos,
        "threshold_negative": threshold_neg
    }

    return answer, evidence


@LabelingRuleRegistry.register("range_check")
def range_check(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Check if value falls within a specified range.

    Params:
        feature: Feature name
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        in_range_label: Label when in range (default: "yes")
        out_range_label: Label when out of range (default: "no")
        aggregation: Optional for arrays

    Returns:
        (label, evidence)
    """
    feature_name = params["feature"]
    min_value = params.get("min_value", -float('inf'))
    max_value = params.get("max_value", float('inf'))
    in_range_label = params.get("in_range_label", "yes")
    out_range_label = params.get("out_range_label", "no")
    aggregation = params.get("aggregation", "mean")

    # Get feature value
    if feature_name in clip_features:
        feature_value = clip_features[feature_name]
    elif feature_name in clip_arrays:
        feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
    else:
        raise ValueError(f"Feature '{feature_name}' not found")

    # Check range
    in_range = min_value <= feature_value <= max_value
    answer = in_range_label if in_range else out_range_label

    evidence = {
        f"value_{feature_name}": feature_value,
        "min_value": min_value,
        "max_value": max_value,
        "in_range": in_range
    }

    return answer, evidence


@LabelingRuleRegistry.register("dominant_axis_comparison")
def dominant_axis_comparison(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Compare longitudinal vs lateral acceleration dominance.

    Params:
        longitudinal_feature: Array name for longitudinal accel (default: "accel").
                              If "accel_long" is present in clip_arrays it is
                              preferred automatically.
        lateral_method: How to compute lateral accel (default: "speed_times_yaw")
        aggregation: Aggregation method (default: "rms")
        ratio_threshold: lateral/longitudinal ratio above which -> "lateral" (default: 1.0)
        deadzone_long: Minimum longitudinal magnitude to be considered
                       non-negligible (m/s^2, default: 0.2)
        deadzone_lat: Minimum lateral magnitude to be considered
                      non-negligible (m/s^2, default: 0.2)

    Returns:
        ("longitudinal" | "lateral" | "none", evidence)
    """
    long_feature = params.get("longitudinal_feature", "accel")
    lateral_method = params.get("lateral_method", "speed_times_yaw")
    aggregation = params.get("aggregation", "rms")
    ratio_threshold = params.get("ratio_threshold", 1.0)
    deadzone_long = params.get("deadzone_long", 0.2)
    deadzone_lat = params.get("deadzone_lat", 0.2)

    # Prefer accel_long if available, fall back to configured feature
    if "accel_long" in clip_arrays:
        long_feature_used = "accel_long"
    elif long_feature in clip_arrays:
        long_feature_used = long_feature
    else:
        raise ValueError(
            f"Required array '{long_feature}' (or 'accel_long') "
            f"not found in clip arrays"
        )

    long_accel = np.abs(clip_arrays[long_feature_used])

    # Compute lateral acceleration
    if lateral_method == "speed_times_yaw":
        for arr_name in ("speed", "yaw_rate"):
            if arr_name not in clip_arrays:
                raise ValueError(
                    f"Required array '{arr_name}' not found in clip arrays"
                )
        lat_accel = clip_arrays["speed"] * np.abs(clip_arrays["yaw_rate"])
    else:
        raise ValueError(f"Unknown lateral_method: '{lateral_method}'")

    long_agg = aggregate_array(long_accel, aggregation)
    lat_agg = aggregate_array(lat_accel, aggregation)

    # Deadzone: both below threshold -> motion is negligible
    if long_agg < deadzone_long and lat_agg < deadzone_lat:
        answer = "none"
    elif long_agg < 1e-9:
        answer = "lateral"
    else:
        ratio = lat_agg / long_agg
        answer = "lateral" if ratio > ratio_threshold else "longitudinal"

    evidence = {
        f"{aggregation}_longitudinal_accel": long_agg,
        f"{aggregation}_lateral_accel": lat_agg,
        "ratio_threshold": ratio_threshold,
        "longitudinal_feature_used": long_feature_used,
        "deadzone_long": deadzone_long,
        "deadzone_lat": deadzone_lat,
    }

    return answer, evidence


@LabelingRuleRegistry.register("lateral_accel_threshold")
def lateral_accel_threshold(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect high lateral acceleration (a_lat = speed * |yaw_rate|).

    Params:
        threshold: Lateral acceleration threshold (m/s^2)
        aggregation: How to aggregate lateral accel array (default: "max")

    Returns:
        ("yes" | "no", evidence)
    """
    threshold = params["threshold"]
    aggregation = params.get("aggregation", "max")

    for arr_name in ("speed", "yaw_rate"):
        if arr_name not in clip_arrays:
            raise ValueError(f"Required array '{arr_name}' not found in clip arrays")

    lat_accel = clip_arrays["speed"] * np.abs(clip_arrays["yaw_rate"])
    lat_value = aggregate_array(lat_accel, aggregation)

    answer = "yes" if lat_value > threshold else "no"

    evidence = {
        f"{aggregation}_lateral_accel": lat_value,
        "threshold": threshold,
    }

    return answer, evidence


@LabelingRuleRegistry.register("sequential_event")
def sequential_event(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect a temporal sequence of two events (e.g. brake then turn).

    Deterministically selects the earliest first-event and the earliest
    valid second-event that follows it within the time window.

    Params:
        first_event: {feature, operator, threshold} for the first event
        second_event: {feature, operator, threshold} for the second event
        min_gap_seconds: Minimum time gap between events (default: 0.0)
        max_gap_seconds: Maximum time gap between events (default: inf)

    Operators: "less_than", "greater_than", "abs_greater_than"

    Returns:
        ("yes" | "no", evidence)
    """
    first_cfg = params["first_event"]
    second_cfg = params["second_event"]
    min_gap = params.get("min_gap_seconds", 0.0)
    max_gap = params.get("max_gap_seconds", float('inf'))

    if "timestamps" not in clip_arrays:
        raise ValueError("Required array 'timestamps' not found in clip arrays")
    timestamps = clip_arrays["timestamps"]

    def _build_mask(cfg):
        feat = cfg["feature"]
        if feat not in clip_arrays:
            raise ValueError(f"Required array '{feat}' not found in clip arrays")
        arr = clip_arrays[feat]
        op = cfg["operator"]
        thr = cfg["threshold"]
        if op == "less_than":
            return arr < thr
        elif op == "greater_than":
            return arr > thr
        elif op == "abs_greater_than":
            return np.abs(arr) > thr
        else:
            raise ValueError(f"Unknown operator: '{op}'")

    first_mask = _build_mask(first_cfg)
    second_mask = _build_mask(second_cfg)

    first_indices = np.where(first_mask)[0]
    second_indices = np.where(second_mask)[0]

    found = False
    first_event_idx = None
    second_event_idx = None
    first_time = None
    second_time = None

    # Use searchsorted for deterministic earliest-match without nested loops
    if len(first_indices) > 0 and len(second_indices) > 0:
        second_times = timestamps[second_indices]
        for fi in first_indices:
            t_first = timestamps[fi]
            # Ensure strict 'then': second event must start at least one dt after first,
            # or respect the explicitly provided min_gap.
            effective_min_gap = max(min_gap, 1e-6) 
            t_lo = t_first + effective_min_gap
            t_hi = t_first + max_gap
            # Binary search for earliest second event in [t_lo, t_hi]
            lo_pos = np.searchsorted(second_times, t_lo, side="left")
            if lo_pos < len(second_times) and second_times[lo_pos] <= t_hi:
                found = True
                first_event_idx = int(fi)
                second_event_idx = int(second_indices[lo_pos])
                first_time = float(t_first)
                second_time = float(second_times[lo_pos])
                break

    answer = "yes" if found else "no"

    evidence = {
        "first_event_count": int(np.sum(first_mask)),
        "second_event_count": int(np.sum(second_mask)),
        "sequence_found": found,
    }
    if found:
        evidence["first_event_index"] = first_event_idx
        evidence["second_event_index"] = second_event_idx
        evidence["first_event_time"] = first_time
        evidence["second_event_time"] = second_time
        evidence["gap_seconds"] = second_time - first_time

    return answer, evidence


@LabelingRuleRegistry.register("peak_half_detection")
def peak_half_detection(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Determine whether the peak of a feature occurs in the first or second half.

    Params:
        feature: Array name to analyse (e.g. "speed")
        min_peak_value: If set, return "no_peak" when max(arr) falls
                        below this value (default: None)
        prominence_ratio: Peak must be at least this much higher than the mean
                         to be considered a valid peak (default: 0.05)

    Returns:
        ("first_half" | "second_half" | "no_peak", evidence)
    """
    feature_name = params["feature"]
    min_peak_value = params.get("min_peak_value", None)
    prominence_ratio = params.get("prominence_ratio", 0.05)

    if feature_name not in clip_arrays:
        raise ValueError(
            f"Required array '{feature_name}' not found in clip arrays"
        )

    arr = clip_arrays[feature_name]
    if len(arr) == 0:
        return "no_peak", {"error": "empty_array"}

    midpoint = len(arr) // 2
    peak_idx = int(np.argmax(arr))
    peak_value = float(arr[peak_idx])
    mean_value = float(np.mean(arr))

    # Avoid labeling flat/noisy signals as having a peak
    is_prominent = True
    if abs(mean_value) > 1e-6:
        relative_increase = (peak_value - mean_value) / abs(mean_value)
        if relative_increase < prominence_ratio:
            is_prominent = False
    elif peak_value < prominence_ratio: # For signals near zero
        is_prominent = False

    if (min_peak_value is not None and peak_value < min_peak_value) or not is_prominent:
        answer = "no_peak"
    else:
        answer = "first_half" if peak_idx < midpoint else "second_half"

    evidence = {
        "peak_index": peak_idx,
        "midpoint_index": midpoint,
        "peak_value": peak_value,
        "mean_value": mean_value,
        "is_prominent": is_prominent,
        "feature_name": feature_name,
    }
    if min_peak_value is not None:
        evidence["min_peak_value"] = min_peak_value

    return answer, evidence


@LabelingRuleRegistry.register("stop_and_go_detection")
def stop_and_go_detection(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect stop-and-go driving patterns via a state machine on the speed array.

    Params:
        stop_speed_threshold: Speed below which the vehicle is considered stopped (m/s)
        go_speed_threshold: Speed above which the vehicle is considered moving (m/s)
        min_stops: Minimum number of stop->go cycles to qualify (default: 1)

    Returns:
        ("yes" | "no", evidence)
    """
    stop_thr = params["stop_speed_threshold"]
    go_thr = params["go_speed_threshold"]
    min_stops = params.get("min_stops", 1)

    if "speed" not in clip_arrays:
        raise ValueError("Required array 'speed' not found in clip arrays")

    speed = clip_arrays["speed"]
    cycles = 0
    was_stopped = bool(speed[0] < stop_thr)

    for s in speed[1:]:
        if not was_stopped and s < stop_thr:
            was_stopped = True
        elif was_stopped and s > go_thr:
            cycles += 1
            was_stopped = False

    answer = "yes" if cycles >= min_stops else "no"

    evidence = {
        "stop_go_cycles": cycles,
        "min_stops": min_stops,
        "stop_speed_threshold": stop_thr,
        "go_speed_threshold": go_thr,
    }

    return answer, evidence


@LabelingRuleRegistry.register("half_comparison")
def half_comparison(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Compare a feature's intensity between the first and second halves of the clip.

    Params:
        feature: Array name to compare (e.g. "accel")
        aggregation: Aggregation method (default: "rms")
        similarity_threshold: Relative-difference threshold below which halves
                              are considered "similar" (default: 0.2)

    Returns:
        ("first_half" | "second_half" | "similar", evidence)
    """
    feature_name = params["feature"]
    aggregation = params.get("aggregation", "rms")
    similarity_thr = params.get("similarity_threshold", 0.2)

    if feature_name not in clip_arrays:
        raise ValueError(f"Required array '{feature_name}' not found in clip arrays")

    arr = clip_arrays[feature_name]
    midpoint = len(arr) // 2

    first_agg = aggregate_array(arr[:midpoint], aggregation)
    second_agg = aggregate_array(arr[midpoint:], aggregation)

    # Use absolute values for intensity comparison
    first_mag = abs(first_agg)
    second_mag = abs(second_agg)

    max_mag = max(first_mag, second_mag)
    if max_mag < 1e-9:
        answer = "similar"
    else:
        relative_diff = abs(first_mag - second_mag) / max_mag
        if relative_diff < similarity_thr:
            answer = "similar"
        elif first_mag > second_mag:
            answer = "first_half"
        else:
            answer = "second_half"

    evidence = {
        f"{aggregation}_first_half": first_agg,
        f"{aggregation}_second_half": second_agg,
        "similarity_threshold": similarity_thr,
    }

    return answer, evidence


@LabelingRuleRegistry.register("multi_threshold_classification")
def multi_threshold_classification(
    clip_features: Dict[str, float],
    clip_arrays: Dict[str, np.ndarray],
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Multi-level classification based on ordered thresholds.

    Given ascending thresholds [t0, t1, ..., tn] and labels [l0, l1, ..., l(n+1)]:
        value < t0       -> labels[0]
        t0 <= value < t1 -> labels[1]
        ...
        value >= tn      -> labels[n+1]

    Params:
        feature: Feature name (from features or arrays)
        thresholds: List of threshold values in ascending order
        labels: List of labels (len must equal len(thresholds) + 1)
        aggregation: How to aggregate if using arrays (default: "mean")

    Returns:
        (label, evidence)
    """
    feature_name = params["feature"]
    thresholds = params["thresholds"]
    labels = params["labels"]
    aggregation = params.get("aggregation", "mean")

    if len(labels) != len(thresholds) + 1:
        raise ValueError(
            f"Expected {len(thresholds) + 1} labels for {len(thresholds)} thresholds, "
            f"got {len(labels)}"
        )

    # Get feature value
    if feature_name in clip_features:
        feature_value = clip_features[feature_name]
    elif feature_name in clip_arrays:
        feature_value = aggregate_array(clip_arrays[feature_name], aggregation)
    else:
        raise ValueError(f"Feature '{feature_name}' not found")

    # Classify: find the first threshold the value is below
    answer = labels[-1]
    for i, thr in enumerate(thresholds):
        if feature_value < thr:
            answer = labels[i]
            break

    evidence = {
        f"{aggregation}_{feature_name}": feature_value,
        "thresholds": thresholds,
        "labels": labels,
    }

    return answer, evidence
