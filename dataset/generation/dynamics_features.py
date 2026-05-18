"""
Vehicle dynamics feature computation and time series processing.

Computes derived quantities (speed, acceleration, yaw-rate, jerk) from ego poses,
resamples to uniform grid, and extracts summary features for clip characterization.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


logger = logging.getLogger(__name__)


class DynamicsProcessor:
    """Processes ego pose time series to compute dynamics features."""

    def __init__(
        self,
        sampling_hz: float = 10.0,
        smooth_window: int = 5,
        smooth_polyorder: int = 2,
    ):
        """
        Initialize dynamics processor.

        Args:
            sampling_hz: Target sampling frequency for resampling (default: 10 Hz)
            smooth_window: Savitzky-Golay filter window length (must be odd,
                          default: 5 samples = 0.5 s at 10 Hz)
            smooth_polyorder: Polynomial order for Savitzky-Golay filter (default: 2)
        """
        self.sampling_hz = sampling_hz
        self.dt = 1.0 / sampling_hz
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to a 1D signal.

        Falls back to returning the signal unchanged if it is too short
        for the configured window and polynomial order.
        """
        min_length = self.smooth_polyorder + 2
        if len(signal) < min_length:
            return signal

        window = min(self.smooth_window, len(signal))
        # Window must be odd
        if window % 2 == 0:
            window -= 1
        if window < self.smooth_polyorder + 2:
            return signal

        return savgol_filter(signal, window, self.smooth_polyorder, mode='mirror')

    def process_ego_trajectory(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        yaws: np.ndarray,
        target_duration: float = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process raw ego pose time series into uniform dynamics arrays.

        Args:
            timestamps: Array of timestamps (seconds), shape [N]
            positions: Array of (x, y) positions, shape [N, 2]
            yaws: Array of yaw angles (radians), shape [N]
            target_duration: Optional target duration (seconds) for resampling.
                           If provided, resamples to exactly this duration.
                           If None, uses actual timestamp span.

        Returns:
            Dictionary containing:
                - timestamps: Resampled uniform timestamps, shape [M]
                - position: Resampled positions, shape [M, 2]
                - yaw: Resampled yaws, shape [M]
                - speed: Computed speeds (m/s), shape [M]
                - accel: Computed accelerations (m/s²), shape [M]
                - yaw_rate: Computed yaw rates (rad/s), shape [M]
                - jerk: Computed jerk (m/s³), shape [M]
        """
        # Validate inputs
        assert len(timestamps) == len(positions) == len(yaws), \
            "Timestamp, position, and yaw arrays must have same length"
        assert positions.shape[1] == 2, "Positions must be (x, y) pairs"

        # Create uniform time grid
        t_start = timestamps[0]

        if target_duration is not None:
            # Use target duration for resampling
            duration = target_duration
            t_end = t_start + duration
        else:
            # Use actual timestamp span
            t_end = timestamps[-1]
            duration = t_end - t_start

        n_samples = int(np.ceil(duration * self.sampling_hz)) + 1
        timestamps_uniform = np.linspace(t_start, t_end, n_samples)

        # Resample positions using linear interpolation
        interp_x = interp1d(timestamps, positions[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(timestamps, positions[:, 1], kind='linear', fill_value='extrapolate')
        positions_uniform = np.column_stack([
            interp_x(timestamps_uniform),
            interp_y(timestamps_uniform)
        ])

        # Resample yaw using unwrapped angles
        yaws_unwrapped = np.unwrap(yaws)
        interp_yaw = interp1d(timestamps, yaws_unwrapped, kind='linear', fill_value='extrapolate')
        yaws_uniform = interp_yaw(timestamps_uniform)

        # Compute speed from position derivatives, then smooth before
        # feeding into the next differentiation stage to prevent noise
        # amplification through the derivative chain.
        speed = self._compute_speed(timestamps_uniform, positions_uniform)
        speed = self._smooth(speed)

        # Compute longitudinal acceleration from smoothed speed
        accel = self._compute_acceleration(timestamps_uniform, speed)
        accel = self._smooth(accel)

        # Compute yaw rate from yaw derivatives
        yaw_rate = self._compute_yaw_rate(timestamps_uniform, yaws_uniform)
        yaw_rate = self._smooth(yaw_rate)

        # Compute jerk from smoothed acceleration
        jerk = self.compute_jerk(timestamps_uniform, accel)
        jerk = self._smooth(jerk)

        return {
            'timestamps': timestamps_uniform - t_start,  # Relative to clip start
            'position': positions_uniform,
            'yaw': yaws_uniform,
            'speed': speed,
            'accel': accel,
            'yaw_rate': yaw_rate,
            'jerk': jerk,
        }

    def _compute_speed(self, timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Compute speed from position time series.

        Args:
            timestamps: Uniform timestamps, shape [N]
            positions: Positions (x, y), shape [N, 2]

        Returns:
            Speed array (m/s), shape [N]
        """
        # Use central differences for interior points, forward/backward for edges
        speed = np.zeros(len(timestamps))

        if len(timestamps) < 2:
            return speed

        # Central differences for interior
        for i in range(1, len(timestamps) - 1):
            dx = positions[i + 1, 0] - positions[i - 1, 0]
            dy = positions[i + 1, 1] - positions[i - 1, 1]
            dt = timestamps[i + 1] - timestamps[i - 1]
            speed[i] = np.sqrt(dx**2 + dy**2) / dt

        # Forward difference for first point
        dx = positions[1, 0] - positions[0, 0]
        dy = positions[1, 1] - positions[0, 1]
        dt = timestamps[1] - timestamps[0]
        speed[0] = np.sqrt(dx**2 + dy**2) / dt

        # Backward difference for last point
        dx = positions[-1, 0] - positions[-2, 0]
        dy = positions[-1, 1] - positions[-2, 1]
        dt = timestamps[-1] - timestamps[-2]
        speed[-1] = np.sqrt(dx**2 + dy**2) / dt

        return speed

    def _compute_acceleration(self, timestamps: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """
        Compute acceleration from speed time series.

        Args:
            timestamps: Uniform timestamps, shape [N]
            speed: Speed array, shape [N]

        Returns:
            Acceleration array (m/s²), shape [N]
        """
        accel = np.zeros(len(timestamps))

        if len(timestamps) < 2:
            return accel

        # Central differences for interior
        for i in range(1, len(timestamps) - 1):
            dv = speed[i + 1] - speed[i - 1]
            dt = timestamps[i + 1] - timestamps[i - 1]
            accel[i] = dv / dt

        # Forward difference for first point
        dv = speed[1] - speed[0]
        dt = timestamps[1] - timestamps[0]
        accel[0] = dv / dt

        # Backward difference for last point
        dv = speed[-1] - speed[-2]
        dt = timestamps[-1] - timestamps[-2]
        accel[-1] = dv / dt

        return accel

    def _compute_yaw_rate(self, timestamps: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """
        Compute yaw rate from yaw angle time series.

        Args:
            timestamps: Uniform timestamps, shape [N]
            yaws: Yaw angles (already unwrapped), shape [N]

        Returns:
            Yaw rate array (rad/s), shape [N]
        """
        yaw_rate = np.zeros(len(timestamps))

        if len(timestamps) < 2:
            return yaw_rate

        # Central differences for interior
        for i in range(1, len(timestamps) - 1):
            dyaw = yaws[i + 1] - yaws[i - 1]
            dt = timestamps[i + 1] - timestamps[i - 1]
            yaw_rate[i] = dyaw / dt

        # Forward difference for first point
        dyaw = yaws[1] - yaws[0]
        dt = timestamps[1] - timestamps[0]
        yaw_rate[0] = dyaw / dt

        # Backward difference for last point
        dyaw = yaws[-1] - yaws[-2]
        dt = timestamps[-1] - timestamps[-2]
        yaw_rate[-1] = dyaw / dt

        return yaw_rate

    def compute_jerk(self, timestamps: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """
        Compute jerk (rate of change of acceleration).

        Args:
            timestamps: Uniform timestamps, shape [N]
            accel: Acceleration array, shape [N]

        Returns:
            Jerk array (m/s³), shape [N]
        """
        jerk = np.zeros(len(timestamps))

        if len(timestamps) < 2:
            return jerk

        # Central differences for interior
        for i in range(1, len(timestamps) - 1):
            da = accel[i + 1] - accel[i - 1]
            dt = timestamps[i + 1] - timestamps[i - 1]
            jerk[i] = da / dt

        # Forward difference for first point
        da = accel[1] - accel[0]
        dt = timestamps[1] - timestamps[0]
        jerk[0] = da / dt

        # Backward difference for last point
        da = accel[-1] - accel[-2]
        dt = timestamps[-1] - timestamps[-2]
        jerk[-1] = da / dt

        return jerk

    def compute_summary_features(
        self,
        speed: np.ndarray,
        accel: np.ndarray,
        yaw_rate: np.ndarray,
        jerk: np.ndarray,
        yaws: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute summary statistics for a clip.

        Args:
            speed: Speed time series
            accel: Acceleration time series
            yaw_rate: Yaw rate time series
            jerk: Jerk time series
            yaws: Yaw angle time series (unwrapped)

        Returns:
            Dictionary of summary features:
                - max_speed: Maximum speed (m/s)
                - mean_speed: Mean speed (m/s)
                - min_accel: Minimum acceleration (m/s², most negative = max braking)
                - max_accel: Maximum acceleration (m/s², most positive = max acceleration)
                - mean_accel: Mean acceleration (m/s², for speed trend classification)
                - max_abs_yaw_rate: Maximum absolute yaw rate (rad/s)
                - signed_max_yaw_rate: Yaw rate at peak |yaw_rate| with sign (rad/s, + = left)
                - max_abs_jerk: Maximum absolute jerk (m/s³)
                - max_lateral_accel: Maximum lateral (centripetal) acceleration (m/s²)
                - total_heading_change: Total heading change (radians)
        """
        # Handle empty arrays
        if len(speed) == 0:
            return {
                'max_speed': 0.0,
                'mean_speed': 0.0,
                'min_accel': 0.0,
                'max_accel': 0.0,
                'mean_accel': 0.0,
                'max_abs_yaw_rate': 0.0,
                'signed_max_yaw_rate': 0.0,
                'max_abs_jerk': 0.0,
                'max_lateral_accel': 0.0,
                'total_heading_change': 0.0,
            }

        # Compute total heading change (sum of absolute yaw differences)
        total_heading_change = 0.0
        if len(yaws) > 1:
            yaw_diffs = np.diff(yaws)
            total_heading_change = np.sum(np.abs(yaw_diffs))

        # Lateral (centripetal) acceleration: a_lat = speed * |yaw_rate|
        lateral_accel = speed * np.abs(yaw_rate)

        # Signed yaw rate at the index of max absolute yaw rate
        # (preserves left/right turn direction for classification)
        abs_yr = np.abs(yaw_rate)
        signed_max_yr = float(yaw_rate[np.argmax(abs_yr)])

        return {
            'max_speed': float(np.max(speed)),
            'mean_speed': float(np.mean(speed)),
            'min_accel': float(np.min(accel)),
            'max_accel': float(np.max(accel)),
            'mean_accel': float(np.mean(accel)),
            'max_abs_yaw_rate': float(np.max(abs_yr)),
            'signed_max_yaw_rate': signed_max_yr,
            'max_abs_jerk': float(np.max(np.abs(jerk))),
            'p95_abs_jerk': float(np.percentile(np.abs(jerk), 95)),
            'mean_abs_jerk': float(np.mean(np.abs(jerk))),
            'max_lateral_accel': float(np.max(lateral_accel)),
            'total_heading_change': float(total_heading_change),
        }


def validate_dynamics_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    """
    Validate dynamics arrays for NaN/Inf and shape consistency.

    Args:
        arrays: Dictionary of dynamics arrays

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    required_keys = ['timestamps', 'position', 'yaw', 'speed', 'accel', 'yaw_rate', 'jerk']
    for key in required_keys:
        if key not in arrays:
            errors.append(f"Missing required array: {key}")
            continue

        arr = arrays[key]

        # Check for NaN/Inf
        if np.any(~np.isfinite(arr)):
            errors.append(f"Array '{key}' contains NaN or Inf values")

        # Check shapes
        if key == 'position':
            if arr.ndim != 2 or arr.shape[1] != 2:
                errors.append(f"Array 'position' has invalid shape: {arr.shape}, expected (N, 2)")
        else:
            if arr.ndim != 1:
                errors.append(f"Array '{key}' has invalid shape: {arr.shape}, expected (N,)")

    # Check length consistency
    if not errors:
        n_samples = len(arrays['timestamps'])
        for key in ['yaw', 'speed', 'accel', 'yaw_rate', 'jerk']:
            if len(arrays[key]) != n_samples:
                errors.append(
                    f"Array '{key}' length {len(arrays[key])} "
                    f"doesn't match timestamps length {n_samples}"
                )

        if arrays['position'].shape[0] != n_samples:
            errors.append(
                f"Position array length {arrays['position'].shape[0]} "
                f"doesn't match timestamps length {n_samples}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
