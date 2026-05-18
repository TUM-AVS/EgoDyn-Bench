"""Evaluate Anthropic Claude models on EgoDyn-Bench.

Supports any Claude vision-capable model (Sonnet, Opus, Haiku) via the
``--model`` flag.

Usage:
    # Sonnet 4.5 (default)
    python evaluation/evaluate_claude.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/claude_sonnet_run.jsonl \
        --max_samples 5 --run_eval

    # Opus 4.6
    python evaluation/evaluate_claude.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/claude_opus_run.jsonl \
        --model claude-opus-4-6

    # Haiku 4.5 (budget)
    python evaluation/evaluate_claude.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/claude_haiku_run.jsonl \
        --model claude-haiku-4-5-20251001

    # Vision-only ablation
    python evaluation/evaluate_claude.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/claude_vision_only.jsonl \
        --no_trajectory
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from anthropic import (
        Anthropic,
        RateLimitError,
        APIError,
        APIConnectionError,
    )
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
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


def _make_caller(
    client: Anthropic,
    model: str,
    temperature: float,
):
    """Return a closure ``(prompt, image_data) -> model_answer``."""

    def call_api(prompt: str, image_data: list[tuple[str, str]]) -> str:
        # Claude expects images before text in the content array
        content: list[dict[str, Any]] = []
        for b64, mime in image_data:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=256,
                    temperature=temperature,
                    messages=[{"role": "user", "content": content}],
                )
                return message.content[0].text.strip()

            except RateLimitError as exc:
                wait = 2 ** (attempt + 2)
                logger.warning(f"Rate limit (attempt {attempt + 1}), waiting {wait}s: {exc}")
                time.sleep(wait)
            except (APIError, APIConnectionError) as exc:
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt + 1}), waiting {wait}s: {exc}")
                time.sleep(wait)
            except Exception as exc:
                wait = 2 ** attempt
                logger.error(f"Unexpected error (attempt {attempt + 1}): {exc}")
                time.sleep(wait)

        logger.error(f"Failed after {MAX_RETRIES} retries")
        return ""

    return call_api


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_common_parser(
        description="Evaluate Anthropic Claude on EgoDyn-Bench",
        default_model="claude-sonnet-4-5-20250929",
        api_key_env_var="ANTHROPIC_API_KEY",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass --api_key."
        )
        return 1

    client = Anthropic(api_key=api_key, max_retries=0)
    call_api = _make_caller(client, args.model, args.temperature)

    return run_evaluation(args, call_api)


if __name__ == "__main__":
    sys.exit(main())
