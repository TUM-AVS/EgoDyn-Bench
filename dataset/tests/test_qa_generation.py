"""
Unit tests for QA generation pipeline.

Tests labeling rules, config loading, and QA generation without requiring
a full nuScenes dataset.
"""

import numpy as np
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.labeling_rules import (
    yaw_rate_sign_with_deadzone,
    threshold_event,
    feature_conversion,
    threshold_classification,
    trend_classification,
    aggregate_array,
    dominant_axis_comparison,
    lateral_accel_threshold,
    sequential_event,
    peak_half_detection,
    stop_and_go_detection,
    half_comparison,
    multi_threshold_classification,
    or_threshold_event,
)
from dataset.generation.config_loader import QuestionConfigLoader, QuestionConfigValidator
from dataset.generation.qa_generator import QAGenerator


def test_aggregate_array():
    """Test array aggregation methods."""
    print("\n" + "="*60)
    print("TEST: Array Aggregation")
    print("="*60)

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert aggregate_array(arr, "mean") == 3.0
    assert aggregate_array(arr, "min") == 1.0
    assert aggregate_array(arr, "max") == 5.0
    assert aggregate_array(arr, "p50") == 3.0
    assert aggregate_array(arr, "p75") == 4.0

    # NaN-safe: NaN values are ignored, not propagated
    arr_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    assert aggregate_array(arr_nan, "mean") == 3.0  # nanmean of [1,3,5]
    assert aggregate_array(arr_nan, "min") == 1.0
    assert aggregate_array(arr_nan, "max") == 5.0

    # Empty array raises ValueError
    try:
        aggregate_array(np.array([]), "mean")
        assert False, "Should have raised ValueError for empty array"
    except ValueError:
        pass

    print("✓ All aggregation methods work correctly")
    print("✓ NaN-safe aggregation works correctly")
    print("✓ Empty array raises ValueError")
    print("PASSED")


def test_yaw_rate_rule():
    """Test yaw rate sign classification."""
    print("\n" + "="*60)
    print("TEST: Yaw Rate Sign Classification")
    print("="*60)

    clip_features = {}
    clip_arrays = {
        "yaw_rate": np.array([0.15, 0.18, 0.20, 0.17])  # All positive > 0.05
    }
    params = {
        "threshold_positive": 0.05,
        "threshold_negative": -0.05,
        "aggregation": "mean"
    }

    answer, evidence = yaw_rate_sign_with_deadzone(clip_features, clip_arrays, params)

    assert answer == "left", f"Expected 'left', got '{answer}'"
    assert "mean_yaw_rate" in evidence
    assert evidence["mean_yaw_rate"] > 0.05

    # Test straight
    clip_arrays["yaw_rate"] = np.array([0.01, -0.01, 0.02, 0.00])
    answer, _ = yaw_rate_sign_with_deadzone(clip_features, clip_arrays, params)
    assert answer == "straight"

    # Test right
    clip_arrays["yaw_rate"] = np.array([-0.15, -0.18, -0.20])
    answer, _ = yaw_rate_sign_with_deadzone(clip_features, clip_arrays, params)
    assert answer == "right"

    print("✓ Left turn detected correctly")
    print("✓ Straight detected correctly")
    print("✓ Right turn detected correctly")
    print("PASSED")


def test_threshold_event():
    """Test threshold event detection."""
    print("\n" + "="*60)
    print("TEST: Threshold Event Detection")
    print("="*60)

    clip_features = {}
    clip_arrays = {
        "accel": np.array([0.5, -1.0, -3.5, -2.0])  # Min is -3.5
    }
    params = {
        "feature": "accel",
        "threshold": -3.0,
        "operator": "less_than",
        "aggregation": "min"
    }

    answer, evidence = threshold_event(clip_features, clip_arrays, params)

    assert answer == "yes", f"Expected 'yes', got '{answer}'"
    assert evidence["min_accel"] == -3.5

    # Test no event
    clip_arrays["accel"] = np.array([0.5, -1.0, -2.5, -2.0])
    answer, _ = threshold_event(clip_features, clip_arrays, params)
    assert answer == "no"

    print("✓ Hard braking event detected")
    print("✓ No event detected correctly")
    print("PASSED")


def test_feature_conversion():
    """Test feature conversion with units."""
    print("\n" + "="*60)
    print("TEST: Feature Conversion")
    print("="*60)

    clip_features = {
        "max_speed": 10.0  # m/s
    }
    clip_arrays = {}
    params = {
        "feature": "max_speed",
        "conversion_factor": 3.6,  # m/s to km/h
        "rounding": 1
    }

    answer, evidence = feature_conversion(clip_features, clip_arrays, params)

    expected = 36.0  # 10 * 3.6
    assert answer == expected, f"Expected {expected}, got {answer}"
    assert evidence["raw_value"] == 10.0
    assert evidence["converted_value"] == 36.0

    print(f"✓ Converted 10.0 m/s to {answer} km/h")
    print("PASSED")


def test_threshold_classification():
    """Test binary threshold classification."""
    print("\n" + "="*60)
    print("TEST: Threshold Classification")
    print("="*60)

    clip_features = {
        "max_abs_jerk": 4.5  # m/s³
    }
    clip_arrays = {}
    params = {
        "feature": "max_abs_jerk",
        "threshold": 3.0,
        "below_label": "smooth",
        "above_label": "aggressive"
    }

    answer, evidence = threshold_classification(clip_features, clip_arrays, params)

    assert answer == "aggressive", f"Expected 'aggressive', got '{answer}'"
    assert evidence["value_max_abs_jerk"] == 4.5

    # Test smooth
    clip_features["max_abs_jerk"] = 2.0
    answer, _ = threshold_classification(clip_features, clip_arrays, params)
    assert answer == "smooth"

    print("✓ Aggressive driving detected")
    print("✓ Smooth driving detected")
    print("PASSED")


def test_trend_classification():
    """Test trend classification."""
    print("\n" + "="*60)
    print("TEST: Trend Classification")
    print("="*60)

    clip_features = {}
    clip_arrays = {
        "accel": np.array([1.0, 1.2, 1.5, 1.3])  # Mean > 0.5
    }
    params = {
        "feature": "accel",
        "threshold_positive": 0.5,
        "threshold_negative": -0.5,
        "aggregation": "mean"
    }

    answer, evidence = trend_classification(clip_features, clip_arrays, params)

    assert answer == "accelerating", f"Expected 'accelerating', got '{answer}'"

    # Test decelerating
    clip_arrays["accel"] = np.array([-1.0, -1.2, -1.5])
    answer, _ = trend_classification(clip_features, clip_arrays, params)
    assert answer == "decelerating"

    # Test steady
    clip_arrays["accel"] = np.array([0.1, -0.1, 0.2, 0.0])
    answer, _ = trend_classification(clip_features, clip_arrays, params)
    assert answer == "steady"

    print("✓ Accelerating detected")
    print("✓ Decelerating detected")
    print("✓ Steady detected")
    print("PASSED")


def test_multi_threshold_classification():
    """Test multi-level threshold classification."""
    print("\n" + "="*60)
    print("TEST: Multi-Threshold Classification")
    print("="*60)

    clip_features = {}

    # --- Braking intensity (thresholds ascending, min aggregation) ---
    braking_params = {
        "feature": "accel",
        "aggregation": "min",
        "thresholds": [-3.924, -1.962, -0.981],
        "labels": ["emergency", "moderate", "low", "none"],
    }

    # Emergency: min(accel) < -3.924
    clip_arrays = {"accel": np.array([0.5, -1.0, -5.0, -2.0])}
    answer, evidence = multi_threshold_classification(clip_features, clip_arrays, braking_params)
    assert answer == "emergency", f"Expected 'emergency', got '{answer}'"

    # Moderate: -3.924 <= min(accel) < -1.962
    clip_arrays["accel"] = np.array([0.5, -1.0, -3.0, -2.0])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, braking_params)
    assert answer == "moderate", f"Expected 'moderate', got '{answer}'"

    # Low: -1.962 <= min(accel) < -0.981
    clip_arrays["accel"] = np.array([0.5, -1.0, -1.5, -0.5])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, braking_params)
    assert answer == "low", f"Expected 'low', got '{answer}'"

    # None: min(accel) >= -0.981
    clip_arrays["accel"] = np.array([0.5, -0.5, -0.3, 0.0])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, braking_params)
    assert answer == "none", f"Expected 'none', got '{answer}'"

    print("  Braking: emergency/moderate/low/none")

    # --- Acceleration intensity (thresholds ascending, max aggregation) ---
    accel_params = {
        "feature": "accel",
        "aggregation": "max",
        "thresholds": [0.981, 1.962, 3.924],
        "labels": ["none", "low", "moderate", "high"],
    }

    # High: max(accel) >= 3.924
    clip_arrays["accel"] = np.array([0.5, 1.0, 5.0, 2.0])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, accel_params)
    assert answer == "high", f"Expected 'high', got '{answer}'"

    # Moderate: 1.962 <= max(accel) < 3.924
    clip_arrays["accel"] = np.array([0.5, 1.0, 3.0, 2.0])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, accel_params)
    assert answer == "moderate", f"Expected 'moderate', got '{answer}'"

    # Low: 0.981 <= max(accel) < 1.962
    clip_arrays["accel"] = np.array([0.5, 1.0, 1.5, 0.5])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, accel_params)
    assert answer == "low", f"Expected 'low', got '{answer}'"

    # None: max(accel) < 0.981
    clip_arrays["accel"] = np.array([0.5, 0.3, 0.0, -0.5])
    answer, _ = multi_threshold_classification(clip_features, clip_arrays, accel_params)
    assert answer == "none", f"Expected 'none', got '{answer}'"

    print("  Acceleration: none/low/moderate/high")

    # --- Test with clip_features (scalar) ---
    answer, _ = multi_threshold_classification(
        {"my_val": -2.5}, {},
        {"feature": "my_val", "thresholds": [-3.924, -1.962, -0.981], "labels": ["emergency", "moderate", "low", "none"]},
    )
    assert answer == "moderate"

    print("  Scalar feature lookup")
    print("PASSED")


def test_or_threshold_event():
    """Test OR-combined threshold event detection (emergency maneuver)."""
    print("\n" + "="*60)
    print("TEST: OR Threshold Event")
    print("="*60)

    params = {
        "conditions": [
            {"feature": "max_abs_jerk", "threshold": 20.0, "operator": "greater_than", "aggregation": "max"},
            {"feature": "accel", "threshold": -3.924, "operator": "less_than", "aggregation": "min"},
        ],
    }

    # Both conditions met
    clip_arrays = {"max_abs_jerk": np.array([5.0, 25.0, 10.0]), "accel": np.array([1.0, -5.0, 0.0])}
    answer, evidence = or_threshold_event({}, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes' (both met), got '{answer}'"
    assert evidence["conditions"][0]["met"] and evidence["conditions"][1]["met"]
    print("  Both conditions met → yes")

    # Only jerk condition met
    clip_arrays = {"max_abs_jerk": np.array([5.0, 25.0, 10.0]), "accel": np.array([1.0, -1.0, 0.0])}
    answer, _ = or_threshold_event({}, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes' (jerk only), got '{answer}'"
    print("  Jerk only → yes")

    # Only braking condition met
    clip_arrays = {"max_abs_jerk": np.array([5.0, 10.0, 3.0]), "accel": np.array([1.0, -5.0, 0.0])}
    answer, _ = or_threshold_event({}, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes' (braking only), got '{answer}'"
    print("  Braking only → yes")

    # Neither condition met
    clip_arrays = {"max_abs_jerk": np.array([5.0, 10.0, 3.0]), "accel": np.array([1.0, -1.0, 0.0])}
    answer, _ = or_threshold_event({}, clip_arrays, params)
    assert answer == "no", f"Expected 'no' (neither met), got '{answer}'"
    print("  Neither met → no")

    # Test with clip_features fallback
    answer, _ = or_threshold_event({"max_abs_jerk": 25.0, "min_accel": -1.0}, {}, {
        "conditions": [
            {"feature": "max_abs_jerk", "threshold": 20.0, "operator": "greater_than"},
            {"feature": "min_accel", "threshold": -3.924, "operator": "less_than"},
        ],
    })
    assert answer == "yes", f"Expected 'yes' (feature lookup), got '{answer}'"
    print("  Feature dict lookup → yes")

    # Test custom labels
    answer, _ = or_threshold_event({}, {"max_abs_jerk": np.array([25.0]), "accel": np.array([0.0])}, {
        "conditions": [
            {"feature": "max_abs_jerk", "threshold": 20.0, "operator": "greater_than", "aggregation": "max"},
        ],
        "above_label": "detected",
        "below_label": "not_detected",
    })
    assert answer == "detected", f"Expected 'detected', got '{answer}'"
    print("  Custom labels → detected")

    print("PASSED")


def test_dominant_axis_comparison():
    """Test longitudinal vs lateral dominance classification."""
    print("\n" + "="*60)
    print("TEST: Dominant Axis Comparison")
    print("="*60)

    clip_features = {}
    params = {
        "longitudinal_feature": "accel",
        "lateral_method": "speed_times_yaw",
        "aggregation": "rms",
        "ratio_threshold": 1.0,
    }

    # Longitudinal dominant: high accel, negligible turning
    clip_arrays = {
        "accel": np.array([2.0, 2.0, 2.0, 2.0]),
        "speed": np.array([5.0, 5.0, 5.0, 5.0]),
        "yaw_rate": np.array([0.01, 0.01, 0.01, 0.01]),
    }
    answer, evidence = dominant_axis_comparison(clip_features, clip_arrays, params)
    assert answer == "longitudinal", f"Expected 'longitudinal', got '{answer}'"
    print("✓ Longitudinal dominant detected")

    # Lateral dominant: low accel, sharp turn at speed
    clip_arrays = {
        "accel": np.array([0.1, 0.1, 0.1, 0.1]),
        "speed": np.array([10.0, 10.0, 10.0, 10.0]),
        "yaw_rate": np.array([0.5, 0.5, 0.5, 0.5]),
    }
    answer, evidence = dominant_axis_comparison(clip_features, clip_arrays, params)
    assert answer == "lateral", f"Expected 'lateral', got '{answer}'"
    print("✓ Lateral dominant detected")

    # Deadzone: both negligible → "none"
    clip_arrays = {
        "accel": np.array([0.05, 0.05, 0.05, 0.05]),
        "speed": np.array([1.0, 1.0, 1.0, 1.0]),
        "yaw_rate": np.array([0.01, 0.01, 0.01, 0.01]),
    }
    answer, evidence = dominant_axis_comparison(clip_features, clip_arrays, params)
    assert answer == "none", f"Expected 'none', got '{answer}'"
    assert evidence["deadzone_long"] == 0.2
    assert evidence["deadzone_lat"] == 0.2
    print("✓ Deadzone 'none' detected for negligible motion")

    # accel_long preference: when present, it is chosen over "accel"
    clip_arrays = {
        "accel_long": np.array([2.0, 2.0, 2.0, 2.0]),
        "accel": np.array([999.0, 999.0, 999.0, 999.0]),  # should be ignored
        "speed": np.array([5.0, 5.0, 5.0, 5.0]),
        "yaw_rate": np.array([0.01, 0.01, 0.01, 0.01]),
    }
    answer, evidence = dominant_axis_comparison(clip_features, clip_arrays, params)
    assert evidence["longitudinal_feature_used"] == "accel_long"
    assert answer == "longitudinal"
    print("✓ accel_long preferred over accel")

    # Missing feature raises ValueError
    try:
        dominant_axis_comparison({}, {"accel": np.array([1.0])}, params)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ Missing feature raises ValueError")

    print("PASSED")


def test_lateral_accel_threshold():
    """Test high lateral acceleration detection."""
    print("\n" + "="*60)
    print("TEST: Lateral Acceleration Threshold")
    print("="*60)

    clip_features = {}
    params = {
        "threshold": 2.0,
        "aggregation": "max",
    }

    # High lateral accel: 15.0 * 0.2 = 3.0 > 2.0
    clip_arrays = {
        "speed": np.array([15.0, 15.0, 15.0]),
        "yaw_rate": np.array([0.2, 0.15, 0.1]),
    }
    answer, evidence = lateral_accel_threshold(clip_features, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes', got '{answer}'"
    assert evidence["max_lateral_accel"] == 3.0
    print("✓ High lateral accel detected")

    # Low lateral accel: 5.0 * 0.1 = 0.5 < 2.0
    clip_arrays = {
        "speed": np.array([5.0, 5.0, 5.0]),
        "yaw_rate": np.array([0.1, 0.05, 0.08]),
    }
    answer, _ = lateral_accel_threshold(clip_features, clip_arrays, params)
    assert answer == "no", f"Expected 'no', got '{answer}'"
    print("✓ Low lateral accel correctly classified")

    print("PASSED")


def test_sequential_event():
    """Test sequential event detection (brake-then-turn)."""
    print("\n" + "="*60)
    print("TEST: Sequential Event Detection")
    print("="*60)

    clip_features = {}
    params = {
        "first_event": {
            "feature": "accel",
            "operator": "less_than",
            "threshold": -1.5,
        },
        "second_event": {
            "feature": "yaw_rate",
            "operator": "abs_greater_than",
            "threshold": 0.1,
        },
        "min_gap_seconds": 0.0,
        "max_gap_seconds": 2.0,
    }

    # Brake at t=0.5-1.0, then turn at t=2.0-2.5
    clip_arrays = {
        "timestamps": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "accel": np.array([0.0, -2.0, -2.5, -0.5, 0.0, 0.0, 0.0]),
        "yaw_rate": np.array([0.0, 0.0, 0.0, 0.05, 0.2, 0.3, 0.15]),
    }
    answer, evidence = sequential_event(clip_features, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes', got '{answer}'"
    assert evidence["sequence_found"] is True
    assert evidence["gap_seconds"] <= 2.0
    assert "first_event_index" in evidence
    assert "second_event_index" in evidence
    assert evidence["first_event_index"] == 1  # earliest braking at t=0.5
    assert evidence["second_event_index"] == 4  # earliest turning at t=2.0
    print("✓ Brake-then-turn sequence detected (with deterministic indices)")

    # Wrong order: turn first, then brake → no valid sequence
    clip_arrays = {
        "timestamps": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "accel": np.array([0.0, 0.0, 0.0, -0.5, -2.0, -2.5, -0.5]),
        "yaw_rate": np.array([0.0, 0.2, 0.3, 0.05, 0.0, 0.0, 0.0]),
    }
    answer, evidence = sequential_event(clip_features, clip_arrays, params)
    assert answer == "no", f"Expected 'no', got '{answer}'"
    print("✓ Wrong-order sequence correctly rejected")

    # No braking at all
    clip_arrays = {
        "timestamps": np.array([0.0, 0.5, 1.0]),
        "accel": np.array([0.5, 0.3, 0.1]),
        "yaw_rate": np.array([0.2, 0.3, 0.1]),
    }
    answer, _ = sequential_event(clip_features, clip_arrays, params)
    assert answer == "no"
    print("✓ No first-event correctly returns 'no'")

    print("PASSED")


def test_peak_half_detection():
    """Test peak temporal localization."""
    print("\n" + "="*60)
    print("TEST: Peak Half Detection")
    print("="*60)

    clip_features = {}
    params = {"feature": "speed"}

    # Peak in first half (index 2, midpoint = 3)
    clip_arrays = {
        "speed": np.array([1.0, 3.0, 5.0, 4.0, 2.0, 1.0])
    }
    answer, evidence = peak_half_detection(clip_features, clip_arrays, params)
    assert answer == "first_half", f"Expected 'first_half', got '{answer}'"
    assert evidence["peak_index"] == 2
    assert evidence["peak_value"] == 5.0
    print("✓ Peak in first half detected")

    # Peak in second half (index 4, midpoint = 3)
    clip_arrays = {
        "speed": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0])
    }
    answer, evidence = peak_half_detection(clip_features, clip_arrays, params)
    assert answer == "second_half", f"Expected 'second_half', got '{answer}'"
    assert evidence["peak_index"] == 4
    print("✓ Peak in second half detected")

    # no_peak: peak below min_peak_value
    params_min = {"feature": "speed", "min_peak_value": 10.0}
    clip_arrays = {
        "speed": np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5])
    }
    answer, evidence = peak_half_detection(clip_features, clip_arrays, params_min)
    assert answer == "no_peak", f"Expected 'no_peak', got '{answer}'"
    assert evidence["peak_value"] == 3.0
    assert evidence["min_peak_value"] == 10.0
    print("✓ no_peak returned when peak below threshold")

    print("PASSED")


def test_stop_and_go_detection():
    """Test stop-and-go pattern detection."""
    print("\n" + "="*60)
    print("TEST: Stop-and-Go Detection")
    print("="*60)

    clip_features = {}
    params = {
        "stop_speed_threshold": 0.5,
        "go_speed_threshold": 2.0,
        "min_stops": 1,
    }

    # Two stop-go cycles
    clip_arrays = {
        "speed": np.array([5.0, 3.0, 0.3, 0.2, 3.0, 5.0, 0.4, 0.1, 2.5, 4.0])
    }
    answer, evidence = stop_and_go_detection(clip_features, clip_arrays, params)
    assert answer == "yes", f"Expected 'yes', got '{answer}'"
    assert evidence["stop_go_cycles"] == 2
    print("✓ Two stop-go cycles detected")

    # No stops at all
    clip_arrays = {
        "speed": np.array([5.0, 4.0, 3.0, 4.0, 5.0])
    }
    answer, evidence = stop_and_go_detection(clip_features, clip_arrays, params)
    assert answer == "no", f"Expected 'no', got '{answer}'"
    assert evidence["stop_go_cycles"] == 0
    print("✓ No stops correctly classified")

    # Stops but never resumes above go_threshold
    clip_arrays = {
        "speed": np.array([3.0, 0.3, 0.2, 0.4, 1.0])
    }
    answer, evidence = stop_and_go_detection(clip_features, clip_arrays, params)
    assert answer == "no", f"Expected 'no', got '{answer}'"
    assert evidence["stop_go_cycles"] == 0
    print("✓ Stop without go correctly classified")

    print("PASSED")


def test_half_comparison():
    """Test first-half vs second-half dynamics comparison."""
    print("\n" + "="*60)
    print("TEST: Half Comparison")
    print("="*60)

    clip_features = {}
    params = {
        "feature": "accel",
        "aggregation": "rms",
        "similarity_threshold": 0.2,
    }

    # First half more dynamic
    clip_arrays = {
        "accel": np.array([3.0, -3.0, 4.0, -4.0, 0.1, -0.1, 0.2, -0.2])
    }
    answer, evidence = half_comparison(clip_features, clip_arrays, params)
    assert answer == "first_half", f"Expected 'first_half', got '{answer}'"
    assert evidence["rms_first_half"] > evidence["rms_second_half"]
    print("✓ First half more dynamic detected")

    # Second half more dynamic
    clip_arrays = {
        "accel": np.array([0.1, -0.1, 0.2, -0.2, 3.0, -3.0, 4.0, -4.0])
    }
    answer, _ = half_comparison(clip_features, clip_arrays, params)
    assert answer == "second_half", f"Expected 'second_half', got '{answer}'"
    print("✓ Second half more dynamic detected")

    # Similar halves
    clip_arrays = {
        "accel": np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    }
    answer, _ = half_comparison(clip_features, clip_arrays, params)
    assert answer == "similar", f"Expected 'similar', got '{answer}'"
    print("✓ Similar halves detected")

    print("PASSED")


def test_config_validation():
    """Test configuration validation."""
    print("\n" + "="*60)
    print("TEST: Config Validation")
    print("="*60)

    # Valid config (uses a real registered rule with correct param types)
    valid_config = {
        "questions": [
            {
                "question_id": "test_q1",
                "category": "direct_dynamics",
                "question_text": "Test question?",
                "answer_type": "binary",
                "choices": ["yes", "no"],
                "rule": {
                    "name": "threshold_event",
                    "params": {
                        "feature": "accel",
                        "threshold": -3.0,
                        "operator": "less_than",
                        "aggregation": "min",
                    }
                }
            }
        ]
    }

    is_valid, errors = QuestionConfigValidator.validate_config(valid_config)
    assert is_valid, f"Valid config failed: {errors}"
    print("✓ Valid config passes validation")

    # Missing required field
    invalid_config = {
        "questions": [
            {
                "question_id": "test_q1",
                # Missing required fields
            }
        ]
    }

    is_valid, errors = QuestionConfigValidator.validate_config(invalid_config)
    assert not is_valid, "Invalid config should fail"
    print("✓ Invalid config detected")

    # Duplicate question_id
    dup_config = {
        "questions": [
            {
                "question_id": "test_q1",
                "category": "test",
                "question_text": "Test?",
                "answer_type": "binary",
                "choices": ["yes", "no"],
                "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": 1.0, "operator": "greater_than"}}
            },
            {
                "question_id": "test_q1",  # Duplicate
                "category": "test",
                "question_text": "Test?",
                "answer_type": "binary",
                "choices": ["yes", "no"],
                "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": 1.0, "operator": "greater_than"}}
            }
        ]
    }

    is_valid, errors = QuestionConfigValidator.validate_config(dup_config)
    assert not is_valid, "Duplicate question_id should fail"
    print("✓ Duplicate question_id detected")

    # --- New validation checks ---

    # Unknown rule name (typo)
    unknown_rule_config = {
        "questions": [{
            "question_id": "test_q1",
            "category": "test",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["yes", "no"],
            "rule": {"name": "threshhold_event", "params": {"threshold": 1.0}}
        }]
    }
    is_valid, errors = QuestionConfigValidator.validate_config(unknown_rule_config)
    assert not is_valid, "Unknown rule name should fail"
    assert any("Unknown rule name" in e for e in errors)
    print("✓ Unknown rule name detected")

    # Non-numeric threshold
    bad_threshold_config = {
        "questions": [{
            "question_id": "test_q1",
            "category": "test",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["yes", "no"],
            "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": "five", "operator": "greater_than"}}
        }]
    }
    is_valid, errors = QuestionConfigValidator.validate_config(bad_threshold_config)
    assert not is_valid, "Non-numeric threshold should fail"
    assert any("numeric" in e.lower() for e in errors)
    print("✓ Non-numeric threshold detected")

    # Invalid aggregation method
    bad_agg_config = {
        "questions": [{
            "question_id": "test_q1",
            "category": "test",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["yes", "no"],
            "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": 1.0, "operator": "greater_than", "aggregation": "mens"}}
        }]
    }
    is_valid, errors = QuestionConfigValidator.validate_config(bad_agg_config)
    assert not is_valid, "Invalid aggregation should fail"
    assert any("aggregation" in e.lower() for e in errors)
    print("✓ Invalid aggregation method detected")

    # Invalid operator
    bad_op_config = {
        "questions": [{
            "question_id": "test_q1",
            "category": "test",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["yes", "no"],
            "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": 1.0, "operator": "less_then"}}
        }]
    }
    is_valid, errors = QuestionConfigValidator.validate_config(bad_op_config)
    assert not is_valid, "Invalid operator should fail"
    assert any("operator" in e.lower() for e in errors)
    print("✓ Invalid operator detected")

    # Valid percentile aggregation (p75) should pass
    percentile_config = {
        "questions": [{
            "question_id": "test_q1",
            "category": "test",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["yes", "no"],
            "rule": {"name": "threshold_event", "params": {"feature": "accel", "threshold": 1.0, "operator": "greater_than", "aggregation": "p75"}}
        }]
    }
    is_valid, errors = QuestionConfigValidator.validate_config(percentile_config)
    assert is_valid, f"Percentile aggregation should pass: {errors}"
    print("✓ Valid percentile aggregation (p75) accepted")

    print("PASSED")


def test_qa_generator():
    """Test QA generator with mock data."""
    print("\n" + "="*60)
    print("TEST: QA Generator")
    print("="*60)

    # Mock clip data
    clip_record = {
        "clip_id": "test_clip_001",
        "features": {
            "max_speed": 10.0,
            "mean_speed": 8.0,
            "min_accel": -2.0,
            "max_abs_yaw_rate": 0.15,
            "max_abs_jerk": 4.0,
            "total_heading_change": 0.3,
        },
        "split": "test",
        "t_end": 1234567890.0,
    }

    clip_arrays = {
        "timestamps": np.linspace(0, 3.0, 31),
        "speed": np.ones(31) * 8.0,
        "accel": np.ones(31) * 0.5,
        "yaw_rate": np.ones(31) * 0.1,
    }

    questions = [
        {
            "question_id": "test_speed",
            "category": "test_category",
            "question_text": "What is max speed?",
            "answer_type": "numeric",
            "units": "km/h",
            "rule": {
                "name": "feature_conversion",
                "params": {
                    "feature": "max_speed",
                    "conversion_factor": 3.6,
                    "rounding": 1
                }
            },
            "metadata": {"difficulty": "easy"}
        }
    ]

    defaults = {
        "label_source": "sensor_rule",
        "include_evidence": True
    }

    generator = QAGenerator(include_evidence=True)

    qa_items = generator.generate_qa_for_clip(
        clip_record, clip_arrays, questions, defaults
    )

    assert len(qa_items) == 1
    qa = qa_items[0]

    # Check schema
    required_fields = [
        "qa_id", "clip_id", "question_id", "category", "question",
        "answer", "answer_type", "label_source", "rule_name", "split", "t_end"
    ]
    for field in required_fields:
        assert field in qa, f"Missing field: {field}"

    assert qa["clip_id"] == "test_clip_001"
    assert qa["answer"] == 36.0  # 10.0 * 3.6
    assert qa["split"] == "test"
    assert qa["t_end"] == 1234567890.0
    assert qa["evidence"] is not None

    print(f"✓ Generated QA item: {qa['qa_id']}")
    print(f"✓ Answer: {qa['answer']} {qa['units']}")
    print(f"✓ All required fields present")

    # Answer-choice mismatch: rule returns "yes" but choices are ["left", "right"]
    bad_choice_questions = [
        {
            "question_id": "test_bad_choices",
            "category": "test_category",
            "question_text": "Test?",
            "answer_type": "binary",
            "choices": ["left", "right"],
            "rule": {
                "name": "threshold_event",
                "params": {
                    "feature": "max_speed",
                    "threshold": 5.0,
                    "operator": "greater_than",
                    "aggregation": "max",
                }
            },
        }
    ]

    generator2 = QAGenerator(include_evidence=True)
    bad_qa_items = generator2.generate_qa_for_clip(
        clip_record, clip_arrays, bad_choice_questions, defaults
    )
    assert len(bad_qa_items) == 0, (
        f"Expected 0 QA items (answer 'yes' not in choices ['left', 'right']), "
        f"got {len(bad_qa_items)}"
    )
    print("✓ Answer-choice mismatch correctly rejected")

    print("PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING QA GENERATION TESTS")
    print("="*60)

    tests = [
        test_aggregate_array,
        test_yaw_rate_rule,
        test_threshold_event,
        test_feature_conversion,
        test_threshold_classification,
        test_trend_classification,
        test_multi_threshold_classification,
        test_or_threshold_event,
        test_dominant_axis_comparison,
        test_lateral_accel_threshold,
        test_sequential_event,
        test_peak_half_detection,
        test_stop_and_go_detection,
        test_half_comparison,
        test_config_validation,
        test_qa_generator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
        except AssertionError as e:
            print(f"\n✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
