"""Unit tests for evaluation.parsers."""

import pytest

from evaluation.parsers import normalize_text, normalize_numeric, parse_answer, _parse_numeric


# ── normalize_text ────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Hello World  ") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("a   b\t\nc") == "a b c"

    def test_strips_trailing_punctuation(self):
        assert normalize_text("yes.") == "yes"
        assert normalize_text("No!") == "no"
        assert normalize_text("smooth;") == "smooth"

    def test_empty(self):
        assert normalize_text("") == ""


# ── parse_answer: binary ──────────────────────────────────────────────────

class TestParseBinary:
    choices = ["yes", "no"]

    def test_exact_match(self):
        assert parse_answer("yes", self.choices, "binary") == "yes"
        assert parse_answer("no", self.choices, "binary") == "no"

    def test_case_insensitive(self):
        assert parse_answer("YES", self.choices, "binary") == "yes"
        assert parse_answer("No", self.choices, "binary") == "no"

    def test_with_trailing_punctuation(self):
        assert parse_answer("Yes.", self.choices, "binary") == "yes"

    def test_verbose_answer(self):
        assert parse_answer(
            "Yes, the vehicle brakes hard during the clip.",
            self.choices, "binary",
        ) == "yes"

    def test_verbose_no(self):
        assert parse_answer(
            "No, there is no hard braking event visible.",
            self.choices, "binary",
        ) == "no"

    def test_empty_returns_none(self):
        assert parse_answer("", self.choices, "binary") is None
        assert parse_answer("   ", self.choices, "binary") is None

    def test_refusal_returns_none(self):
        assert parse_answer(
            "I cannot determine this from the video.",
            self.choices, "binary",
        ) is None


# ── parse_answer: binary with non-yes/no labels ──────────────────────────

class TestParseBinaryCustomLabels:
    choices = ["smooth", "aggressive"]

    def test_exact(self):
        assert parse_answer("smooth", self.choices, "binary") == "smooth"
        assert parse_answer("aggressive", self.choices, "binary") == "aggressive"

    def test_verbose(self):
        assert parse_answer(
            "The driving appears to be smooth based on the trajectory.",
            self.choices, "binary",
        ) == "smooth"

    def test_verbose_aggressive(self):
        assert parse_answer(
            "It looks quite aggressive with sudden movements.",
            self.choices, "binary",
        ) == "aggressive"


# ── parse_answer: multiclass ─────────────────────────────────────────────

class TestParseMulticlass:
    def test_turn_direction(self):
        choices = ["left", "right", "straight"]
        assert parse_answer("left", choices, "multiclass") == "left"
        assert parse_answer("The vehicle is turning left.", choices, "multiclass") == "left"
        assert parse_answer("It goes straight ahead.", choices, "multiclass") == "straight"

    def test_speed_trend(self):
        choices = ["accelerating", "decelerating", "steady"]
        assert parse_answer("accelerating", choices, "multiclass") == "accelerating"
        assert parse_answer("The car is decelerating.", choices, "multiclass") == "decelerating"
        assert parse_answer("Maintaining a steady speed.", choices, "multiclass") == "steady"

    def test_underscore_space_equivalence(self):
        choices = ["first_half", "second_half", "no_peak"]
        assert parse_answer("first half", choices, "multiclass") == "first_half"
        assert parse_answer("second half", choices, "multiclass") == "second_half"
        assert parse_answer("first_half", choices, "multiclass") == "first_half"

    def test_no_peak_before_no(self):
        """Ensure 'no_peak' is matched before 'no' via longest-first ordering."""
        choices = ["first_half", "second_half", "no_peak"]
        assert parse_answer(
            "There is no peak, the speed is constant.",
            choices, "multiclass",
        ) == "no_peak"

    def test_dominant_axis(self):
        choices = ["longitudinal", "lateral", "none"]
        assert parse_answer("longitudinal", choices, "multiclass") == "longitudinal"
        assert parse_answer(
            "The motion is primarily lateral.",
            choices, "multiclass",
        ) == "lateral"

    def test_contrastive(self):
        choices = ["first_half", "second_half", "similar"]
        assert parse_answer("similar", choices, "multiclass") == "similar"
        assert parse_answer(
            "The first half has more dynamic driving.",
            choices, "multiclass",
        ) == "first_half"


# ── parse_answer: numeric ────────────────────────────────────────────────

class TestParseNumeric:
    def test_plain_number(self):
        assert parse_answer("42.5", None, "numeric") == "42.5"

    def test_with_units(self):
        assert parse_answer("The maximum speed is 65.3 km/h.", None, "numeric") == "65.3"

    def test_integer(self):
        assert parse_answer("30", None, "numeric") == "30"

    def test_negative(self):
        assert parse_answer("-3.5 m/s²", None, "numeric") == "-3.5"

    def test_no_number(self):
        assert parse_answer("I cannot determine the speed.", None, "numeric") is None

    def test_number_in_sentence(self):
        assert parse_answer(
            "Based on the video, the vehicle reaches approximately 85 km/h.",
            None, "numeric",
        ) == "85"

    def test_trailing_zero_stripped(self):
        """'42.0' from model should normalize to '42' to match integer oracle."""
        assert parse_answer("42.0", None, "numeric") == "42"

    def test_trailing_zeros_decimal(self):
        assert parse_answer("3.50 m/s", None, "numeric") == "3.5"


# ── normalize_numeric ──────────────────────────────────────────────────

class TestNormalizeNumeric:
    def test_integer_string(self):
        assert normalize_numeric("42") == "42"

    def test_trailing_zero(self):
        assert normalize_numeric("42.0") == "42"

    def test_trailing_zeros(self):
        assert normalize_numeric("3.500") == "3.5"

    def test_preserves_decimal(self):
        assert normalize_numeric("3.14") == "3.14"

    def test_negative(self):
        assert normalize_numeric("-5.0") == "-5"

    def test_non_numeric_passthrough(self):
        assert normalize_numeric("hello") == "hello"

    def test_zero(self):
        assert normalize_numeric("0.0") == "0"
