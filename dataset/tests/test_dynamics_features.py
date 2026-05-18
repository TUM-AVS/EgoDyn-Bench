"""
Unit tests for dynamics features computation.

These tests validate the core dynamics computation logic without requiring
a full nuScenes dataset.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.dynamics_features import (
    DynamicsProcessor,
    validate_dynamics_arrays,
)


def test_dynamics_processor_basic():
    """Test basic dynamics processing with synthetic data."""
    print("\n" + "="*60)
    print("TEST: Basic Dynamics Processing")
    print("="*60)

    # Create synthetic trajectory: constant velocity
    processor = DynamicsProcessor(sampling_hz=10.0)

    # 3 seconds at ~2 Hz (irregular sampling)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Moving at constant 10 m/s in x direction
    positions = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [15.0, 0.0],
        [20.0, 0.0],
        [25.0, 0.0],
        [30.0, 0.0],
    ])

    # Constant heading (0 radians)
    yaws = np.zeros(7)

    # Process
    arrays = processor.process_ego_trajectory(timestamps, positions, yaws)

    # Validate
    is_valid, errors = validate_dynamics_arrays(arrays)
    assert is_valid, f"Validation failed: {errors}"

    # Check shapes
    expected_samples = int(3.0 * 10.0) + 1  # 31 samples
    assert len(arrays['timestamps']) == expected_samples, \
        f"Expected {expected_samples} samples, got {len(arrays['timestamps'])}"
    assert arrays['position'].shape == (expected_samples, 2)
    assert arrays['speed'].shape == (expected_samples,)

    # Check speed is approximately 10 m/s
    mean_speed = np.mean(arrays['speed'])
    assert 9.5 < mean_speed < 10.5, f"Mean speed {mean_speed} not close to 10 m/s"

    # Check acceleration is close to zero (constant velocity)
    max_accel = np.max(np.abs(arrays['accel']))
    assert max_accel < 0.5, f"Max acceleration {max_accel} too high for constant velocity"

    # Check yaw rate is close to zero (straight line)
    max_yaw_rate = np.max(np.abs(arrays['yaw_rate']))
    assert max_yaw_rate < 0.01, f"Max yaw rate {max_yaw_rate} too high for straight line"

    print("✓ Shapes correct")
    print(f"✓ Mean speed: {mean_speed:.2f} m/s (expected ~10 m/s)")
    print(f"✓ Max accel: {max_accel:.3f} m/s² (expected ~0)")
    print(f"✓ Max yaw rate: {max_yaw_rate:.4f} rad/s (expected ~0)")
    print("PASSED")


def test_dynamics_processor_turn():
    """Test dynamics processing with a turning trajectory."""
    print("\n" + "="*60)
    print("TEST: Turning Trajectory")
    print("="*60)

    processor = DynamicsProcessor(sampling_hz=10.0)

    # Create circular arc trajectory
    t = np.linspace(0, 3.0, 31)  # Already at 10 Hz
    radius = 20.0  # 20 meter radius
    angular_vel = 0.2  # 0.2 rad/s

    # Circular motion
    angles = angular_vel * t
    positions = np.column_stack([
        radius * np.sin(angles),
        radius * (1 - np.cos(angles))
    ])
    yaws = angles

    # Process
    arrays = processor.process_ego_trajectory(t, positions, yaws)

    # Validate
    is_valid, errors = validate_dynamics_arrays(arrays)
    assert is_valid, f"Validation failed: {errors}"

    # Check yaw rate is approximately constant
    mean_yaw_rate = np.mean(arrays['yaw_rate'])
    assert 0.15 < mean_yaw_rate < 0.25, \
        f"Mean yaw rate {mean_yaw_rate} not close to {angular_vel} rad/s"

    # Check speed is approximately constant (v = r * omega)
    expected_speed = radius * angular_vel  # 4 m/s
    mean_speed = np.mean(arrays['speed'])
    assert 3.0 < mean_speed < 5.0, \
        f"Mean speed {mean_speed} not close to {expected_speed} m/s"

    print(f"✓ Mean yaw rate: {mean_yaw_rate:.3f} rad/s (expected ~{angular_vel} rad/s)")
    print(f"✓ Mean speed: {mean_speed:.2f} m/s (expected ~{expected_speed} m/s)")
    print("PASSED")


def test_jerk_computation():
    """Test jerk computation."""
    print("\n" + "="*60)
    print("TEST: Jerk Computation")
    print("="*60)

    processor = DynamicsProcessor(sampling_hz=10.0)

    # Create acceleration profile with a step change
    timestamps = np.linspace(0, 3.0, 31)
    accel = np.ones(31) * 2.0  # Constant 2 m/s²
    accel[15:] = -1.0  # Step to -1 m/s² at t=1.5s

    # Compute jerk
    jerk = processor.compute_jerk(timestamps, accel)

    # Check jerk has large spike at transition
    max_jerk = np.max(np.abs(jerk))
    assert max_jerk > 5.0, f"Max jerk {max_jerk} too low for step change"

    # Check jerk is near zero elsewhere
    jerk_no_transition = np.concatenate([jerk[:13], jerk[17:]])
    max_jerk_elsewhere = np.max(np.abs(jerk_no_transition))
    assert max_jerk_elsewhere < 1.0, \
        f"Jerk {max_jerk_elsewhere} too high away from transition"

    print(f"✓ Max jerk at transition: {max_jerk:.2f} m/s³")
    print(f"✓ Max jerk elsewhere: {max_jerk_elsewhere:.2f} m/s³")
    print("PASSED")


def test_summary_features():
    """Test summary feature computation."""
    print("\n" + "="*60)
    print("TEST: Summary Features")
    print("="*60)

    processor = DynamicsProcessor(sampling_hz=10.0)

    # Create synthetic data with known properties
    speed = np.array([5.0, 10.0, 15.0, 10.0, 5.0])
    accel = np.array([2.0, 1.0, -2.0, -1.0, 0.0])
    yaw_rate = np.array([0.1, 0.2, 0.15, -0.1, 0.05])
    jerk = np.array([1.0, -2.0, 0.5, 1.5, -1.0])
    yaws = np.array([0.0, 0.1, 0.3, 0.2, 0.25])

    features = processor.compute_summary_features(speed, accel, yaw_rate, jerk, yaws)

    # Verify features
    assert features['max_speed'] == 15.0
    assert features['mean_speed'] == 9.0
    assert features['min_accel'] == -2.0
    assert features['max_accel'] == 2.0
    assert features['mean_accel'] == 0.0  # mean([2, 1, -2, -1, 0]) = 0
    assert features['max_abs_yaw_rate'] == 0.2
    # signed_max_yaw_rate should be 0.2 (at index 1, which has max |yaw_rate|)
    assert features['signed_max_yaw_rate'] == 0.2
    assert features['max_abs_jerk'] == 2.0

    # Check lateral acceleration: max(speed * |yaw_rate|)
    expected_lateral = max(
        5.0 * 0.1, 10.0 * 0.2, 15.0 * 0.15, 10.0 * 0.1, 5.0 * 0.05
    )  # = max(0.5, 2.0, 2.25, 1.0, 0.25) = 2.25
    assert abs(features['max_lateral_accel'] - expected_lateral) < 0.01, \
        f"max_lateral_accel {features['max_lateral_accel']} != {expected_lateral}"

    # Check heading change
    expected_heading_change = abs(0.1 - 0.0) + abs(0.3 - 0.1) + abs(0.2 - 0.3) + abs(0.25 - 0.2)
    assert abs(features['total_heading_change'] - expected_heading_change) < 0.01

    print(f"✓ max_speed: {features['max_speed']}")
    print(f"✓ mean_speed: {features['mean_speed']}")
    print(f"✓ min_accel: {features['min_accel']}")
    print(f"✓ max_abs_yaw_rate: {features['max_abs_yaw_rate']}")
    print(f"✓ max_abs_jerk: {features['max_abs_jerk']}")
    print(f"✓ max_lateral_accel: {features['max_lateral_accel']}")
    print(f"✓ total_heading_change: {features['total_heading_change']:.3f}")
    print("PASSED")


def test_smoothing_reduces_noise():
    """Test that Savitzky-Golay smoothing reduces derivative noise."""
    print("\n" + "="*60)
    print("TEST: Smoothing Reduces Derivative Noise")
    print("="*60)

    # Create trajectory with measurement noise on a constant-velocity path
    np.random.seed(42)
    t = np.linspace(0, 3.0, 31)  # 10 Hz
    noise_std = 0.05  # 5 cm position noise
    positions = np.column_stack([
        10.0 * t + np.random.randn(31) * noise_std,
        np.random.randn(31) * noise_std,
    ])
    yaws = np.zeros(31)

    # Process WITHOUT smoothing
    proc_raw = DynamicsProcessor(sampling_hz=10.0, smooth_window=1, smooth_polyorder=0)
    # smooth_window=1 effectively disables smoothing (falls through to return signal)
    arrays_raw = proc_raw.process_ego_trajectory(t, positions, yaws)

    # Process WITH smoothing (default window=5, polyorder=2)
    proc_smooth = DynamicsProcessor(sampling_hz=10.0)
    arrays_smooth = proc_smooth.process_ego_trajectory(t, positions, yaws)

    # Smoothed jerk should have lower variance than raw jerk
    jerk_std_raw = np.std(arrays_raw['jerk'])
    jerk_std_smooth = np.std(arrays_smooth['jerk'])
    assert jerk_std_smooth < jerk_std_raw, \
        f"Smoothed jerk std ({jerk_std_smooth:.2f}) should be < raw ({jerk_std_raw:.2f})"

    # Smoothed accel should have lower variance than raw accel
    accel_std_raw = np.std(arrays_raw['accel'])
    accel_std_smooth = np.std(arrays_smooth['accel'])
    assert accel_std_smooth < accel_std_raw, \
        f"Smoothed accel std ({accel_std_smooth:.2f}) should be < raw ({accel_std_raw:.2f})"

    # Mean speed should still be close to 10 m/s (smoothing preserves signal)
    mean_speed = np.mean(arrays_smooth['speed'])
    assert 9.0 < mean_speed < 11.0, \
        f"Smoothed mean speed {mean_speed:.2f} too far from 10 m/s"

    print(f"✓ Jerk std: raw={jerk_std_raw:.2f}, smoothed={jerk_std_smooth:.2f}")
    print(f"✓ Accel std: raw={accel_std_raw:.2f}, smoothed={accel_std_smooth:.2f}")
    print(f"✓ Mean speed preserved: {mean_speed:.2f} m/s")
    print("PASSED")


def test_validation():
    """Test array validation."""
    print("\n" + "="*60)
    print("TEST: Array Validation")
    print("="*60)

    # Valid arrays
    valid_arrays = {
        'timestamps': np.linspace(0, 3, 31),
        'position': np.random.randn(31, 2),
        'yaw': np.random.randn(31),
        'speed': np.abs(np.random.randn(31)),
        'accel': np.random.randn(31),
        'yaw_rate': np.random.randn(31),
        'jerk': np.random.randn(31),
    }

    is_valid, errors = validate_dynamics_arrays(valid_arrays)
    assert is_valid, f"Valid arrays failed: {errors}"
    print("✓ Valid arrays pass")

    # Missing key
    invalid_arrays = valid_arrays.copy()
    del invalid_arrays['speed']
    is_valid, errors = validate_dynamics_arrays(invalid_arrays)
    assert not is_valid, "Should fail with missing key"
    print("✓ Missing key detected")

    # NaN values
    invalid_arrays = valid_arrays.copy()
    invalid_arrays['speed'] = invalid_arrays['speed'].copy()
    invalid_arrays['speed'][5] = np.nan
    is_valid, errors = validate_dynamics_arrays(invalid_arrays)
    assert not is_valid, "Should fail with NaN"
    print("✓ NaN values detected")

    # Wrong shape
    invalid_arrays = valid_arrays.copy()
    invalid_arrays['position'] = np.random.randn(31, 3)  # Should be (31, 2)
    is_valid, errors = validate_dynamics_arrays(invalid_arrays)
    assert not is_valid, "Should fail with wrong shape"
    print("✓ Wrong shape detected")

    print("PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING DYNAMICS FEATURES TESTS")
    print("="*60)

    tests = [
        test_dynamics_processor_basic,
        test_dynamics_processor_turn,
        test_jerk_computation,
        test_summary_features,
        test_smoothing_reduces_noise,
        test_validation,
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
        except Exception as e:
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
