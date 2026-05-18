"""Evaluate Google Gemini models on EgoDyn-Bench.

Supports Gemini 2.5 and 3 vision-capable models via the ``--model`` flag.
Gemini 3 models are detected automatically and use ``thinking_level``
instead of ``thinking_budget``.

Usage:
    # Gemini 2.5 Flash (default)
    python evaluation/evaluate_gemini.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gemini_flash_run.jsonl \
        --max_samples 5 --run_eval

    # Gemini 2.5 Pro
    python evaluation/evaluate_gemini.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gemini_pro_run.jsonl \
        --model gemini-2.5-pro

    # Gemini 3 Flash
    python evaluation/evaluate_gemini.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gemini3_flash_run.jsonl \
        --model gemini-3-flash-preview --thinking_level low

    # Gemini 3 Pro
    python evaluation/evaluate_gemini.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gemini3_pro_run.jsonl \
        --model gemini-3-pro-preview

    # Vision-only ablation
    python evaluation/evaluate_gemini.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gemini_vision_only.jsonl \
        --no_trajectory
"""

import base64
import logging
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not installed. Run: pip install google-genai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator_common import build_common_parser, run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


def _is_gemini3(model: str) -> bool:
    """Check whether *model* belongs to the Gemini 3 family."""
    return "gemini-3" in model


def _make_caller(
    client: genai.Client,
    model: str,
    temperature: float,
    thinking_level: str | None,
):
    """Return a closure ``(prompt, image_data) -> model_answer``.

    For Gemini 3 models the config uses ``thinking_level`` (not
    ``thinking_budget``) and defaults temperature to 1.0 as recommended
    by Google.
    """
    is_g3 = _is_gemini3(model)

    # Build the GenerateContentConfig once — reused across calls.
    config_kwargs: dict = {"max_output_tokens": 256}

    if is_g3:
        # Gemini 3: Google recommends temperature=1.0; lower values can
        # cause looping.  Only override when the user explicitly set it.
        config_kwargs["temperature"] = temperature
        level = thinking_level or "low"
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=level,
        )
        logger.info(
            f"Gemini 3 detected — using thinking_level={level!r}, "
            f"temperature={temperature}"
        )
    else:
        config_kwargs["temperature"] = temperature

    gen_config = types.GenerateContentConfig(**config_kwargs)

    def call_api(prompt: str, image_data: list[tuple[str, str]]) -> str:
        # Build content parts: images first, then text (Gemini recommendation)
        parts: list[types.Part] = []
        for b64, mime in image_data:
            raw_bytes = base64.b64decode(b64)
            parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
        parts.append(types.Part.from_text(text=prompt))

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=gen_config,
                )
                text = response.text
                return text.strip() if text else ""

            except Exception as exc:
                exc_str = str(exc).lower()
                # Permanent quota error
                if "quota" in exc_str or "resource_exhausted" in exc_str:
                    if "per" not in exc_str:
                        # Likely a billing quota, not a per-minute rate limit
                        logger.error(
                            f"Gemini quota exhausted: {exc}. "
                            "Check your Google Cloud billing."
                        )
                        raise SystemExit(1) from exc
                # Rate limit — back off
                if "429" in exc_str or "resource_exhausted" in exc_str:
                    wait = 2 ** (attempt + 2)
                    logger.warning(f"Rate limit (attempt {attempt + 1}), waiting {wait}s")
                    time.sleep(wait)
                    continue
                # Other errors
                wait = 2 ** attempt
                logger.error(f"Gemini error (attempt {attempt + 1}): {exc}")
                time.sleep(wait)

        logger.error(f"Failed after {MAX_RETRIES} retries")
        return ""

    return call_api


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_common_parser(
        description="Evaluate Google Gemini on EgoDyn-Bench",
        default_model="gemini-2.5-flash",
        api_key_env_var="GOOGLE_API_KEY",
    )
    parser.add_argument(
        "--thinking_level",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Thinking level for Gemini 3 models (default: 'low'). "
             "Ignored for Gemini 2.5 models. "
             "'minimal'/'low' = fast, 'medium'/'high' = deeper reasoning.",
    )
    args = parser.parse_args()

    # Default temperature to 1.0 for Gemini 3 when user didn't explicitly set it
    if _is_gemini3(args.model) and args.temperature == 0.0:
        logger.info(
            "Gemini 3 detected with default temperature=0.0 — "
            "overriding to 1.0 (recommended by Google to avoid looping)."
        )
        args.temperature = 1.0

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "Google API key not found. "
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) or pass --api_key."
        )
        return 1

    client = genai.Client(api_key=api_key)
    call_api = _make_caller(client, args.model, args.temperature, args.thinking_level)

    return run_evaluation(args, call_api)


if __name__ == "__main__":
    sys.exit(main())
