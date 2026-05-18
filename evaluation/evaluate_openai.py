"""Evaluate OpenAI models on EgoDyn-Bench.

Supports both GPT-4 family (GPT-4o, GPT-4.1) and GPT-5 family (GPT-5,
GPT-5.1, GPT-5.2) models.  The script auto-detects the model family and
uses the correct API parameters (token limit field, temperature handling).

Usage:
    # GPT-4o (default)
    python evaluation/evaluate_openai.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gpt4o_run.jsonl \
        --max_samples 5 --run_eval

    # GPT-4.1
    python evaluation/evaluate_openai.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gpt41_run.jsonl \
        --model gpt-4.1

    # GPT-5.2
    python evaluation/evaluate_openai.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gpt52_run.jsonl \
        --model gpt-5.2

    # GPT-5 mini (budget)
    python evaluation/evaluate_openai.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gpt5mini_run.jsonl \
        --model gpt-5-mini

    # GPT-5.2 with reasoning effort
    python evaluation/evaluate_openai.py \
        --qa_jsonl ./output/pipeline_all/splits/val_qa.jsonl \
        --output ./results/gpt52_high_run.jsonl \
        --model gpt-5.2 --reasoning_effort high
"""

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
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
# Model family detection
# ---------------------------------------------------------------------------

def _is_gpt5_family(model: str) -> bool:
    """Return True if the model uses GPT-5 API conventions.

    GPT-5 family uses ``max_completion_tokens`` instead of ``max_tokens``
    on the Chat Completions endpoint.
    """
    return bool(re.match(r"gpt-5", model))


def _is_reasoning_model(model: str) -> bool:
    """Return True if the model is a reasoning model that doesn't support temperature.

    Reasoning models (o-series, GPT-5-mini) only accept the default
    temperature (1) and reject explicit temperature=0.0.
    """
    return bool(re.match(r"(o1|o3|o4|gpt-5-mini)", model))


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

MAX_RETRIES = 10


def _make_caller(
    client: OpenAI,
    model: str,
    temperature: float,
    frame_detail: str,
    reasoning_effort: str | None = None,
):
    """Return a closure ``(prompt, image_data) -> model_answer``."""
    is_gpt5 = _is_gpt5_family(model)
    is_reasoning = _is_reasoning_model(model)

    if is_gpt5:
        logger.info(
            f"GPT-5 family detected ({model}): using max_completion_tokens"
        )
    if is_reasoning:
        logger.info(
            f"Reasoning model detected ({model}): omitting temperature parameter"
        )

    def call_api(prompt: str, image_data: list[tuple[str, str]]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64, mime in image_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": frame_detail,
                },
            })

        # Build request kwargs — GPT-5 uses different parameter names,
        # and reasoning models don't support custom temperature.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if is_gpt5:
            # Reasoning models use tokens for internal reasoning too,
            # so give them a larger budget to avoid empty outputs.
            kwargs["max_completion_tokens"] = 2048 if is_reasoning else 256
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
        else:
            kwargs["max_tokens"] = 256

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content
                return text.strip() if text else ""

            except RateLimitError as exc:
                if "insufficient_quota" in str(exc):
                    logger.error(
                        "Insufficient OpenAI quota. Add credits at "
                        "https://platform.openai.com/account/billing"
                    )
                    raise SystemExit(1) from exc
                wait = 2 ** (attempt + 2)
                logger.warning(f"Rate limit (attempt {attempt + 1}), waiting {wait}s")
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
        description="Evaluate OpenAI models on EgoDyn-Bench",
        default_model="gpt-4o",
        api_key_env_var="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--reasoning_effort", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for GPT-5 models (ignored for GPT-4 family)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY or pass --api_key.")
        return 1

    client = OpenAI(api_key=api_key, max_retries=0)
    call_api = _make_caller(
        client, args.model, args.temperature,
        args.frame_detail, args.reasoning_effort,
    )

    return run_evaluation(args, call_api)


if __name__ == "__main__":
    sys.exit(main())
