"""Answer parsing for EgoDyn-Bench evaluation.

Maps free-text model answers to canonical semantic labels defined in the
question template YAML.  Parsing is deterministic and driven by the
``choices`` list of each question — no per-question_id special-casing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_question_config(
    yaml_path: str | Path = "dataset/configs/questions_template.yaml",
) -> dict[str, dict[str, Any]]:
    """Load question definitions keyed by *question_id*.

    Returns a dict like::

        {
            "braking_intensity": {
                "choices": ["emergency", "moderate", "low", "none"],
                "answer_type": "multiclass",
                "category": "direct_dynamics",
            },
            ...
        }
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    out: dict[str, dict[str, Any]] = {}
    for q in cfg["questions"]:
        meta = q.get("metadata", {})
        out[q["question_id"]] = {
            "choices": [str(c).lower() for c in q["choices"]] if q.get("choices") else None,
            "answer_type": q["answer_type"],
            "category": q.get("category", "unknown"),
            "temporal": meta.get("temporal", False),
        }
    return out


def normalize_text(raw: str) -> str:
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""
    text = raw.lower().strip()
    # Strip markdown bold/italic markers and backticks
    text = re.sub(r"[*`]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:!?")
    return text


def parse_answer(
    model_answer: str,
    choices: list[str] | None,
    answer_type: str,
) -> str | None:
    """Map a model's free-text answer to a canonical label.

    Parameters
    ----------
    model_answer:
        Raw text produced by the model.
    choices:
        Canonical label set (e.g. ``["yes", "no"]``).  ``None`` for numeric.
    answer_type:
        One of ``"binary"``, ``"multiclass"``, ``"numeric"``.

    Returns
    -------
    The matched canonical label as a string, or ``None`` if unparsable.
    """
    if not model_answer or not model_answer.strip():
        return None

    if answer_type == "numeric":
        return _parse_numeric(model_answer)

    if choices is None:
        return None

    norm = normalize_text(model_answer)

    # 1) Exact match on full text
    if norm in choices:
        return norm

    # 2) Handle underscore/space equivalence  (first_half ↔ first half)
    for choice in choices:
        if "_" in choice and choice.replace("_", " ") == norm:
            return choice
        if " " in choice and choice.replace(" ", "_") == norm:
            return choice

    # 3) Last-line matching — models that ignore "answer only" instructions
    #    typically put their final answer on the last line.  Check there
    #    before scanning the full (potentially noisy) reasoning text.
    last_line = _extract_last_line(model_answer)
    if last_line:
        last_norm = normalize_text(last_line)
        # Exact match on last line
        if last_norm in choices:
            return last_norm
        for choice in choices:
            if "_" in choice and choice.replace("_", " ") == last_norm:
                return choice
            if " " in choice and choice.replace(" ", "_") == last_norm:
                return choice
        # Substring match on last line
        match = _substring_match(last_norm, choices)
        if match is not None:
            return match

    # 4) Full-text substring match with word boundaries — longest choices
    #    first to avoid partial collisions (e.g. "no_peak" before "no")
    #    and to prevent "no" matching inside "cannot".
    return _substring_match(norm, choices)


def _extract_last_line(text: str) -> str | None:
    """Return the last non-empty line from *text*, or ``None``."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _substring_match(norm: str, choices: list[str]) -> str | None:
    """Find the longest choice that appears as a whole word in *norm*."""
    for choice in sorted(choices, key=len, reverse=True):
        variants = [choice]
        if "_" in choice:
            variants.append(choice.replace("_", " "))
        for variant in variants:
            if re.search(r"\b" + re.escape(variant) + r"\b", norm):
                return choice
    return None


def normalize_numeric(value: str) -> str:
    """Normalize a numeric string to its canonical form.

    Strips trailing zeros so that ``"42.0"`` and ``"42"`` compare equal,
    and ``"3.50"`` becomes ``"3.5"``.  Non-numeric strings pass through
    unchanged.
    """
    try:
        return f"{float(value):g}"
    except (ValueError, OverflowError):
        return value


def _parse_numeric(model_answer: str) -> str | None:
    """Extract the first number from a model answer string."""
    # Match integers and decimals, possibly negative
    m = re.search(r"-?\d+(?:\.\d+)?", model_answer)
    if m is None:
        return None
    return normalize_numeric(m.group(0))
