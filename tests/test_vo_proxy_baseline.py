"""Tests for the VO-proxy baseline.

Uses synthetic affine transforms to generate deterministic frame sequences
so that KLT tracking + essential matrix decomposition can be validated
without real video data.
"""

import numpy as np
import cv2
import pytest

from baselines.vo_proxy_baseline import (
    VOProxyBaseline,
    compute_vo_signals,
    SUPPORTED_QUESTIONS,
    _default_intrinsics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 240, 320  # small resolution for speed


def _make_textured_image(seed: int = 42) -> np.ndarray:
    """Create a repeatable textured image with strong gradients for tracking."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    grid = np.zeros((H, W, 3), dtype=np.uint8)
    grid[::8, :, :] = 255
    grid[:, ::8, :] = 255
    blended = cv2.addWeighted(noise, 0.5, grid, 0.5, 0)
    return cv2.GaussianBlur(blended, (5, 5), 1.0)


def _apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image about its centre by *angle_deg* degrees."""
    cy, cx = image.shape[0] / 2.0, image.shape[1] / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def _apply_translation(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Translate image by (tx, ty) pixels."""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def _make_rotated_sequence(
    angle_per_frame: float, n_frames: int = 8,
) -> list[np.ndarray]:
    """Generate frames rotated incrementally from a textured base."""
    base = _make_textured_image()
    return [_apply_rotation(base, angle_per_frame * i) for i in range(n_frames)]


def _make_identical_sequence(n_frames: int = 8) -> list[np.ndarray]:
    """Generate identical frames (no motion)."""
    base = _make_textured_image()
    return [base.copy() for _ in range(n_frames)]


def _make_translated_sequence(
    tx_per_frame: float,
    ty_per_frame: float = 0.0,
    n_frames: int = 8,
) -> list[np.ndarray]:
    """Generate frames translated incrementally."""
    base = _make_textured_image()
    return [_apply_translation(base, tx_per_frame * i, ty_per_frame * i)
            for i in range(n_frames)]


def _make_accelerating_sequence(n_frames: int = 10) -> list[np.ndarray]:
    """Translation that increases over time (accelerating)."""
    base = _make_textured_image()
    frames = [base.copy()]
    cumul = 0.0
    for i in range(1, n_frames):
        cumul += i * 1.5  # increasing displacement
        frames.append(_apply_translation(base, cumul, 0))
    return frames


def _make_decelerating_sequence(n_frames: int = 10) -> list[np.ndarray]:
    """Translation that decreases over time (decelerating)."""
    base = _make_textured_image()
    frames = [base.copy()]
    cumul = 0.0
    for i in range(1, n_frames):
        cumul += max((n_frames - i) * 1.5, 0)  # decreasing displacement
        frames.append(_apply_translation(base, cumul, 0))
    return frames


# ---------------------------------------------------------------------------
# Tests: compute_vo_signals
# ---------------------------------------------------------------------------

class TestComputeVoSignals:
    """Test the low-level signal computation."""

    def test_identical_frames_zero_signals(self):
        """Identical frames should produce near-zero yaw and displacement."""
        frames = _make_identical_sequence(5)
        signals = compute_vo_signals(frames)

        assert len(signals["yaw_deg"]) == 4
        assert len(signals["disp_mag"]) == 4

        # All signals near zero
        assert np.all(np.abs(signals["yaw_deg"]) < 1.0), (
            f"Expected near-zero yaw, got {signals['yaw_deg']}"
        )
        assert np.all(signals["disp_mag"] < 0.5), (
            f"Expected near-zero disp, got {signals['disp_mag']}"
        )

    def test_rotation_produces_yaw(self):
        """In-plane rotation should produce non-zero yaw_deg."""
        frames = _make_rotated_sequence(angle_per_frame=3.0, n_frames=6)
        signals = compute_vo_signals(frames)
        cumul_yaw = np.sum(signals["yaw_deg"])
        assert abs(cumul_yaw) > 0.5, (
            f"Expected non-zero cumulative yaw, got {cumul_yaw}"
        )

    def test_opposite_rotations_differ_in_sign(self):
        """CW and CCW rotations should yield opposite yaw signs."""
        frames_ccw = _make_rotated_sequence(angle_per_frame=4.0, n_frames=8)
        frames_cw = _make_rotated_sequence(angle_per_frame=-4.0, n_frames=8)
        sig_ccw = compute_vo_signals(frames_ccw)
        sig_cw = compute_vo_signals(frames_cw)

        yaw_ccw = np.sum(sig_ccw["yaw_deg"])
        yaw_cw = np.sum(sig_cw["yaw_deg"])

        # They should have opposite signs
        assert yaw_ccw * yaw_cw < 0, (
            f"Expected opposite signs: CCW={yaw_ccw:.2f}, CW={yaw_cw:.2f}"
        )

    def test_translation_produces_displacement(self):
        """Horizontal translation should produce non-zero displacement."""
        frames = _make_translated_sequence(tx_per_frame=8.0, n_frames=6)
        signals = compute_vo_signals(frames)
        mean_disp = np.mean(signals["disp_mag"])
        assert mean_disp > 2.0, (
            f"Expected significant displacement, got {mean_disp}"
        )

    def test_single_frame_returns_empty(self):
        """Single frame should return empty arrays."""
        frames = [_make_textured_image()]
        signals = compute_vo_signals(frames)
        assert len(signals["yaw_deg"]) == 0
        assert len(signals["disp_mag"]) == 0

    def test_default_intrinsics_shape(self):
        """Default intrinsics should be a 3x3 matrix."""
        K = _default_intrinsics(W, H)
        assert K.shape == (3, 3)
        assert K[0, 0] == pytest.approx(0.9 * W)
        assert K[1, 1] == pytest.approx(0.9 * W)
        assert K[0, 2] == pytest.approx(W / 2.0)
        assert K[1, 2] == pytest.approx(H / 2.0)


# ---------------------------------------------------------------------------
# Tests: VOProxyBaseline.predict
# ---------------------------------------------------------------------------

class TestVOProxyBaseline:
    """Test the full predict method on synthetic data."""

    @pytest.fixture
    def baseline(self):
        return VOProxyBaseline()

    def test_identical_frames_straight_steady(self, baseline):
        """Identical frames -> straight, steady, no stop_and_go."""
        frames = _make_identical_sequence(8)
        preds = baseline.predict(frames)

        assert preds["yaw_rate_turn_direction"] == "straight"
        assert preds["speed_trend"] == "steady"
        assert preds["stop_and_go"] == "no"

    def test_rotation_detected_as_turn(self, baseline):
        """In-plane rotation should NOT be 'straight'."""
        frames = _make_rotated_sequence(angle_per_frame=4.0, n_frames=8)
        preds = baseline.predict(frames)
        assert preds["yaw_rate_turn_direction"] in ("left", "right"), (
            f"Expected a turn, got {preds['yaw_rate_turn_direction']}"
        )

    def test_opposite_rotations_yield_opposite_turns(self, baseline):
        """CW and CCW rotations should produce opposite turn labels."""
        frames_ccw = _make_rotated_sequence(angle_per_frame=4.0, n_frames=8)
        frames_cw = _make_rotated_sequence(angle_per_frame=-4.0, n_frames=8)
        preds_ccw = baseline.predict(frames_ccw)
        preds_cw = baseline.predict(frames_cw)

        assert preds_ccw["yaw_rate_turn_direction"] != preds_cw["yaw_rate_turn_direction"], (
            f"CCW={preds_ccw['yaw_rate_turn_direction']}, "
            f"CW={preds_cw['yaw_rate_turn_direction']} — expected different"
        )

    def test_yaw_sign_flip(self):
        """yaw_sign_flip should invert the turn direction."""
        frames = _make_rotated_sequence(angle_per_frame=5.0, n_frames=8)
        normal = VOProxyBaseline({"yaw_sign_flip": False})
        flipped = VOProxyBaseline({"yaw_sign_flip": True})
        preds_n = normal.predict(frames)
        preds_f = flipped.predict(frames)

        # If normal detects a turn, flipped should detect the opposite
        if preds_n["yaw_rate_turn_direction"] in ("left", "right"):
            assert preds_n["yaw_rate_turn_direction"] != preds_f["yaw_rate_turn_direction"]

    def test_accelerating_detected(self, baseline):
        """Increasing displacement should be classified as accelerating."""
        frames = _make_accelerating_sequence(n_frames=10)
        preds = baseline.predict(frames)
        assert preds["speed_trend"] == "accelerating", (
            f"Expected accelerating, got {preds['speed_trend']}"
        )

    def test_decelerating_detected(self, baseline):
        """Decreasing displacement should be classified as decelerating."""
        frames = _make_decelerating_sequence(n_frames=10)
        preds = baseline.predict(frames)
        assert preds["speed_trend"] == "decelerating", (
            f"Expected decelerating, got {preds['speed_trend']}"
        )

    def test_all_supported_questions_present(self, baseline):
        """predict() should return an answer for every supported question."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)

        for qid in SUPPORTED_QUESTIONS:
            assert qid in preds, f"Missing question_id: {qid}"
            assert preds[qid] in SUPPORTED_QUESTIONS[qid], (
                f"Invalid answer '{preds[qid]}' for {qid}. "
                f"Expected one of {SUPPORTED_QUESTIONS[qid]}"
            )

    def test_six_questions_returned(self, baseline):
        """predict() should return exactly 6 answers."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)
        assert len(preds) == 6, f"Expected 6 answers, got {len(preds)}: {list(preds.keys())}"

    def test_unsupported_questions_not_present(self, baseline):
        """Removed questions should not appear in output."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)
        removed = [
            "speed_regime", "mean_speed_low", "speed_peak_half",
            "driving_smoothness",
        ]
        for qid in removed:
            assert qid not in preds, (
                f"Unsupported question '{qid}' should not be in output"
            )

    def test_identical_frames_no_heading_change(self, baseline):
        """Identical frames should not have significant heading change."""
        frames = _make_identical_sequence(8)
        preds = baseline.predict(frames)
        assert preds["significant_heading_change"] == "no"
        assert preds["high_lateral_accel"] == "no"
        assert preds["brake_then_turn"] == "no"

    def test_custom_intrinsics(self, baseline):
        """Passing custom intrinsics should not crash."""
        frames = _make_translated_sequence(tx_per_frame=5.0, n_frames=6)
        K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]], dtype=np.float64)
        preds = baseline.predict(frames, intrinsics=K)
        assert "yaw_rate_turn_direction" in preds

    def test_config_dict_applied(self):
        """Config dict should override defaults."""
        config = {
            "yaw_deadzone_deg": 5.0,
            "disp_stopped_thr": 1.0,
            "max_corners": 500,
        }
        bl = VOProxyBaseline(config)
        assert bl.yaw_deadzone_deg == 5.0
        assert bl.disp_stopped_thr == 1.0
        assert bl.max_corners == 500

    def test_median_peak_turn_logic(self):
        """Turn requires both median and peak above thresholds."""
        # With very high peak threshold, even rotation should be "straight"
        config = {"yaw_peak_thr_deg": 999.0}
        bl = VOProxyBaseline(config)
        frames = _make_rotated_sequence(angle_per_frame=4.0, n_frames=8)
        preds = bl.predict(frames)
        assert preds["yaw_rate_turn_direction"] == "straight"
