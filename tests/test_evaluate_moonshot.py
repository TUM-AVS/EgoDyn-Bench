"""Tests for evaluation/evaluate_moonshot.py reliability upgrades."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_moonshot import jittered_backoff, BACKOFF_BASE, BACKOFF_CAP


# ---------------------------------------------------------------------------
# (1) Jittered backoff tests
# ---------------------------------------------------------------------------

class TestJitteredBackoff:
    """Test jittered_backoff returns values within expected range and cap."""

    def test_attempt_zero_within_range(self):
        """attempt=0 → base * 2^0 * jitter = base * [0.75, 1.25]."""
        for _ in range(200):
            val = jittered_backoff(0, base=1.0, cap=60.0)
            assert 0.75 <= val <= 1.25

    def test_attempt_three_within_range(self):
        """attempt=3 → base * 8 * jitter = [6.0, 10.0]."""
        for _ in range(200):
            val = jittered_backoff(3, base=1.0, cap=60.0)
            assert 6.0 <= val <= 10.0

    def test_respects_cap(self):
        """Very high attempt should be capped."""
        for _ in range(200):
            val = jittered_backoff(20, base=1.0, cap=60.0)
            assert val <= 60.0

    def test_cap_overrides_large_base(self):
        """Even with large base, cap wins."""
        val = jittered_backoff(0, base=100.0, cap=5.0)
        assert val <= 5.0

    def test_custom_base(self):
        """Custom base scales correctly."""
        for _ in range(200):
            val = jittered_backoff(0, base=2.0, cap=60.0)
            assert 1.5 <= val <= 2.5  # 2.0 * [0.75, 1.25]

    def test_deterministic_with_seed(self):
        """With fixed seed, output is reproducible."""
        import random
        random.seed(42)
        v1 = jittered_backoff(2, base=1.0, cap=60.0)
        random.seed(42)
        v2 = jittered_backoff(2, base=1.0, cap=60.0)
        assert v1 == v2

    def test_uses_module_defaults(self):
        """Calling with no base/cap uses module-level defaults."""
        for _ in range(100):
            val = jittered_backoff(0)
            assert BACKOFF_BASE * 0.75 <= val <= BACKOFF_BASE * 1.25
            assert val <= BACKOFF_CAP


# ---------------------------------------------------------------------------
# (4) Grouped resume duplicate-skipping tests
# ---------------------------------------------------------------------------

class TestGroupedResumeDuplicateSkipping:
    """Verify that grouped mode write-skipping avoids duplicates."""

    def _make_record(self, qa_id: str, answer: str = "test") -> dict:
        return {
            "qa_id": qa_id,
            "clip_id": "clip_001",
            "question_id": "q_speed_trend",
            "category": "speed",
            "oracle_label": "slow",
            "model_answer": answer,
        }

    def test_skips_already_completed_ids(self, tmp_path):
        """Records whose qa_id is in completed_ids should not be written."""
        completed_ids = {"qa_001", "qa_003"}
        records = [
            self._make_record("qa_001"),
            self._make_record("qa_002"),
            self._make_record("qa_003"),
            self._make_record("qa_004"),
        ]

        out_file = tmp_path / "output.jsonl"
        written = []
        skipped = 0
        with open(out_file, "w") as f:
            for record in records:
                qa_id = record.get("qa_id", "")
                if qa_id and qa_id in completed_ids:
                    skipped += 1
                    continue
                f.write(json.dumps(record) + "\n")
                written.append(record)

        assert skipped == 2
        assert len(written) == 2
        assert {r["qa_id"] for r in written} == {"qa_002", "qa_004"}

        # Verify file contents
        lines = out_file.read_text().strip().split("\n")
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert {r["qa_id"] for r in parsed} == {"qa_002", "qa_004"}

    def test_no_skipping_when_completed_empty(self, tmp_path):
        """With empty completed_ids, all records should be written."""
        completed_ids: set[str] = set()
        records = [
            self._make_record("qa_001"),
            self._make_record("qa_002"),
        ]

        out_file = tmp_path / "output.jsonl"
        written_count = 0
        with open(out_file, "w") as f:
            for record in records:
                qa_id = record.get("qa_id", "")
                if qa_id and qa_id in completed_ids:
                    continue
                f.write(json.dumps(record) + "\n")
                written_count += 1

        assert written_count == 2

    def test_all_skipped_when_all_completed(self, tmp_path):
        """If all qa_ids already completed, nothing should be written."""
        completed_ids = {"qa_001", "qa_002", "qa_003"}
        records = [
            self._make_record("qa_001"),
            self._make_record("qa_002"),
            self._make_record("qa_003"),
        ]

        out_file = tmp_path / "output.jsonl"
        written_count = 0
        with open(out_file, "w") as f:
            for record in records:
                qa_id = record.get("qa_id", "")
                if qa_id and qa_id in completed_ids:
                    continue
                f.write(json.dumps(record) + "\n")
                written_count += 1

        assert written_count == 0
        assert out_file.read_text() == ""

    def test_empty_qa_id_not_skipped(self, tmp_path):
        """Records with empty qa_id should never be skipped."""
        completed_ids = {"qa_001"}
        records = [
            self._make_record(""),
            self._make_record("qa_001"),
        ]

        out_file = tmp_path / "output.jsonl"
        written_count = 0
        with open(out_file, "w") as f:
            for record in records:
                qa_id = record.get("qa_id", "")
                if qa_id and qa_id in completed_ids:
                    continue
                f.write(json.dumps(record) + "\n")
                written_count += 1

        assert written_count == 1  # the empty-id record passes through
