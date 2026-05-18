"""Tests for the optical-flow heuristic baseline.

Uses synthetic affine transforms to generate deterministic frame sequences
so that flow signals can be validated without real video data.
"""

import numpy as np
import cv2
import pytest

from baselines.flow_heuristic import (
    FlowHeuristicBaseline,
    compute_flow_signals,
    SUPPORTED_QUESTIONS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 240, 320  # small resolution for speed


def _make_textured_image(seed: int = 42) -> np.ndarray:
    """Create a repeatable textured image with sufficient gradient for flow."""
    rng = np.random.RandomState(seed)
    # Blend random noise with a grid pattern for strong gradients
    noise = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    grid = np.zeros((H, W, 3), dtype=np.uint8)
    grid[::8, :, :] = 255
    grid[:, ::8, :] = 255
    blended = cv2.addWeighted(noise, 0.5, grid, 0.5, 0)
    # Blur slightly so Farneback can track
    return cv2.GaussianBlur(blended, (5, 5), 1.0)


def _apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image about its centre by *angle_deg* degrees."""
    cy, cx = image.shape[0] / 2.0, image.shape[1] / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def _apply_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """Zoom in (*scale* > 1) or out (*scale* < 1) about image centre."""
    cy, cx = image.shape[0] / 2.0, image.shape[1] / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def _apply_translation(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Translate image by (tx, ty) pixels."""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def _make_rotated_sequence(
    angle_per_frame: float, n_frames: int = 8
) -> list[np.ndarray]:
    """Generate a sequence of frames rotated incrementally."""
    base = _make_textured_image()
    return [_apply_rotation(base, angle_per_frame * i) for i in range(n_frames)]


def _make_zoom_sequence(
    scale_step: float, n_frames: int = 8
) -> list[np.ndarray]:
    """Generate a sequence of frames zoomed incrementally."""
    base = _make_textured_image()
    return [_apply_zoom(base, 1.0 + scale_step * i) for i in range(n_frames)]


def _make_identical_sequence(n_frames: int = 8) -> list[np.ndarray]:
    """Generate identical frames (no motion)."""
    base = _make_textured_image()
    return [base.copy() for _ in range(n_frames)]


# ---------------------------------------------------------------------------
# Tests: compute_flow_signals
# ---------------------------------------------------------------------------

class TestComputeFlowSignals:
    """Test the low-level signal computation."""

    def test_identical_frames_zero_flow(self):
        """Identical frames should produce near-zero signals."""
        frames = _make_identical_sequence(5)
        signals = compute_flow_signals(frames)

        assert len(signals["turn_score"]) == 4
        assert len(signals["exp_score"]) == 4
        assert len(signals["motion_mag"]) == 4

        assert np.all(np.abs(signals["turn_score"]) < 0.1)
        assert np.all(np.abs(signals["exp_score"]) < 0.1)
        assert np.all(signals["motion_mag"] < 0.1)

    def test_rotation_produces_turn_score(self):
        """Rotation should produce non-zero turn_score."""
        frames = _make_rotated_sequence(angle_per_frame=3.0, n_frames=6)
        signals = compute_flow_signals(frames)
        mean_turn = np.mean(signals["turn_score"])
        assert abs(mean_turn) > 0.05, f"Expected non-zero turn, got {mean_turn}"

    def test_zoom_produces_exp_score(self):
        """Zoom-in should produce positive radial expansion proxy."""
        frames = _make_zoom_sequence(scale_step=0.03, n_frames=6)
        signals = compute_flow_signals(frames)
        mean_exp = np.mean(signals["exp_score"])
        assert mean_exp > 0.05, f"Expected positive expansion, got {mean_exp}"

    def test_zoom_out_negative_exp(self):
        """Zoom-out should produce negative radial expansion proxy."""
        frames = _make_zoom_sequence(scale_step=-0.03, n_frames=6)
        signals = compute_flow_signals(frames)
        mean_exp = np.mean(signals["exp_score"])
        assert mean_exp < -0.05, f"Expected negative expansion, got {mean_exp}"

    def test_single_frame_returns_empty(self):
        """Single frame should return empty arrays."""
        frames = [_make_textured_image()]
        signals = compute_flow_signals(frames)
        assert len(signals["turn_score"]) == 0
        assert len(signals["exp_score"]) == 0
        assert len(signals["motion_mag"]) == 0

    def test_motion_mag_nonzero_with_translation(self):
        """Translation should produce non-zero motion magnitude."""
        base = _make_textured_image()
        frames = [_apply_translation(base, 5 * i, 0) for i in range(6)]
        signals = compute_flow_signals(frames)
        assert np.mean(signals["motion_mag"]) > 0.5


# ---------------------------------------------------------------------------
# Tests: FlowHeuristicBaseline.predict
# ---------------------------------------------------------------------------

class TestFlowHeuristicBaseline:
    """Test the full predict method on synthetic data."""

    @pytest.fixture
    def baseline(self):
        return FlowHeuristicBaseline()

    def test_identical_frames_straight_steady(self, baseline):
        """Identical frames → straight, steady, no events."""
        frames = _make_identical_sequence(8)
        preds = baseline.predict(frames)

        assert preds["yaw_rate_turn_direction"] == "straight"
        assert preds["speed_trend"] == "steady"
        assert preds["significant_heading_change"] == "no"
        assert preds["high_lateral_accel"] == "no"
        assert preds["stop_and_go"] == "no"
        assert preds["brake_then_turn"] == "no"

    def test_rotation_left_turn(self, baseline):
        """Counter-clockwise rotation should be detected as a turn."""
        frames = _make_rotated_sequence(angle_per_frame=5.0, n_frames=8)
        preds = baseline.predict(frames)
        assert preds["yaw_rate_turn_direction"] in ("left", "right"), (
            f"Expected a turn direction, got {preds['yaw_rate_turn_direction']}"
        )

    def test_rotation_opposite_directions_differ(self, baseline):
        """CW and CCW rotations should yield opposite turn directions."""
        frames_ccw = _make_rotated_sequence(angle_per_frame=5.0, n_frames=8)
        frames_cw = _make_rotated_sequence(angle_per_frame=-5.0, n_frames=8)
        preds_ccw = baseline.predict(frames_ccw)
        preds_cw = baseline.predict(frames_cw)
        assert preds_ccw["yaw_rate_turn_direction"] != preds_cw["yaw_rate_turn_direction"], (
            f"CCW={preds_ccw['yaw_rate_turn_direction']}, "
            f"CW={preds_cw['yaw_rate_turn_direction']} — expected different"
        )

    def test_zoom_in_accelerating(self, baseline):
        """Zoom-in (radial expansion) should map to accelerating proxy."""
        frames = _make_zoom_sequence(scale_step=0.04, n_frames=8)
        preds = baseline.predict(frames)
        assert preds["speed_trend"] == "accelerating", (
            f"Expected accelerating, got {preds['speed_trend']}"
        )

    def test_zoom_out_decelerating(self, baseline):
        """Zoom-out (radial contraction) should map to decelerating proxy."""
        frames = _make_zoom_sequence(scale_step=-0.04, n_frames=8)
        preds = baseline.predict(frames)
        assert preds["speed_trend"] == "decelerating", (
            f"Expected decelerating, got {preds['speed_trend']}"
        )

    def test_all_supported_questions_covered(self, baseline):
        """predict() should return an answer for every supported question."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)

        for qid in SUPPORTED_QUESTIONS:
            assert qid in preds, f"Missing question_id: {qid}"
            assert preds[qid] in SUPPORTED_QUESTIONS[qid], (
                f"Invalid answer '{preds[qid]}' for {qid}. "
                f"Expected one of {SUPPORTED_QUESTIONS[qid]}"
            )

    def test_unsupported_questions_not_present(self, baseline):
        """Questions removed for scientific defensibility should not appear."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)
        removed = [
            "speed_regime", "mean_speed_low", "braking_intensity",
            "driving_smoothness", "dominant_motion_axis", "extreme_maneuver",
            "speed_peak_half", "contrastive_sequence",
        ]
        for qid in removed:
            assert qid not in preds, (
                f"Unsupported question '{qid}' should not be in output"
            )

    def test_only_six_questions_returned(self, baseline):
        """predict() should return exactly 6 answers."""
        frames = _make_identical_sequence(6)
        preds = baseline.predict(frames)
        assert len(preds) == 6, f"Expected 6 answers, got {len(preds)}: {list(preds.keys())}"
