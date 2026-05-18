"""Evaluate HuggingFace-hosted VLMs on EgoDyn-Bench via OpenAI-compatible API.

Supports any model on HuggingFace's inference router that exposes an
OpenAI-compatible chat completions endpoint (Kimi K2.5, Qwen3-VL, etc.).

Features:
- Jittered exponential backoff with configurable cap (default 60s)
- Per-request timeout via ``--timeout_s`` (default 60)
- Safe future handling: individual failures don't crash the run
- Duplicate-safe grouped resume (skips already-completed QA IDs at write time)
- Batched flush every N records (default 50) for reduced I/O overhead

Usage:
    # Sequential (default)
    python evaluation/evaluate_moonshot.py \
        --selected_clips selected_clips.json \
        --model moonshotai/Kimi-K2.5:novita \
        --no_trajectory --resume \
        --output generated/kimi_k25_answers.jsonl \
        --run_eval --metrics_output results/kimi_k25.json

    # Parallel with custom timeout
    python evaluation/evaluate_moonshot.py \
        --selected_clips selected_clips.json \
        --model moonshotai/Kimi-K2.5:novita \
        --workers 4 --timeout_s 90 --no_trajectory --resume \
        --output generated/kimi_k25_answers.jsonl
"""

import concurrent.futures
import json
import logging
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator_common import (  # noqa: E402
    build_common_parser,
    load_clips_index,
    load_qa_items,
    load_completed_qa_ids,
    select_frames,
    encode_image,
    extract_video_frames,
    build_prompt,
    build_grouped_prompt,
    group_qa_by_clip,
    parse_grouped_response,
    make_prediction_record,
    print_eval_results,
    resolve_carla_video_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
MAX_RETRIES = 10
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
FLUSH_EVERY = 50


# ---------------------------------------------------------------------------
# Jittered backoff helper
# ---------------------------------------------------------------------------

def jittered_backoff(attempt: int, base: float = BACKOFF_BASE,
                     cap: float = BACKOFF_CAP) -> float:
    """Compute a jittered exponential backoff delay.

    Returns ``min(cap, base * 2^attempt * jitter)`` where jitter is drawn
    uniformly from [0.75, 1.25].
    """
    raw = base * (2 ** attempt)
    jitter = random.uniform(0.75, 1.25)
    return min(cap, raw * jitter)


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _make_caller(
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int = 2048,
    timeout_s: float = 60.0,
):
    """Return a closure ``(prompt, image_data) -> model_answer``."""

    def call_api(prompt: str, image_data: list[tuple[str, str]]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64, mime in image_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                },
            })

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                # Strip <think>...</think> CoT blocks from reasoning models
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                return text.strip()

            except RateLimitError as exc:
                wait = jittered_backoff(attempt + 2)
                logger.warning(
                    f"Rate limit (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
            except (APIError, APIConnectionError) as exc:
                wait = jittered_backoff(attempt)
                logger.warning(
                    f"API error (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
            except TimeoutError as exc:
                wait = jittered_backoff(attempt)
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
            except Exception as exc:
                wait = jittered_backoff(attempt)
                logger.error(
                    f"Unexpected error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}"
                )
                time.sleep(wait)

        logger.error(f"Failed after {MAX_RETRIES} retries")
        return ""

    return call_api


# ---------------------------------------------------------------------------
# Parallel evaluation loop
# ---------------------------------------------------------------------------

def _run_parallel_evaluation(args, call_api) -> int:
    """Run inference with optional parallelism via ThreadPoolExecutor."""
    if args.metrics_output:
        args.run_eval = True

    resolve_carla_video_dir(args)

    nuscenes_root = Path(args.nuscenes_root)
    if not args.no_images and not nuscenes_root.exists():
        logger.error(f"nuScenes root not found: {nuscenes_root}")
        return 1

    # --- load data ---------------------------------------------------------
    clips: dict[str, dict] = {}

    ns_idx_path = Path(args.nuscenes_index)
    if ns_idx_path.exists():
        clips.update(load_clips_index(ns_idx_path))
    ca_idx_path = Path(args.carla_index)
    if ca_idx_path.exists():
        clips.update(load_clips_index(ca_idx_path))

    if not clips:
        logger.error("No clips loaded.")
        return 1

    with open(args.selected_clips) as f:
        selected_data = json.load(f)
        if selected_data and isinstance(selected_data[0], dict):
            selected_ids = set(c["id"] for c in selected_data)
        else:
            selected_ids = set(selected_data)
    logger.info(f"Loaded {len(selected_ids)} selected clips")

    all_qa: list[dict] = []
    for qa_path in [args.nuscenes_qa, args.carla_qa]:
        p = Path(qa_path)
        if p.exists():
            all_qa.extend(load_qa_items(p))
    qa_items = [qa for qa in all_qa if qa["clip_id"] in selected_ids]
    logger.info(f"Filtered to {len(qa_items)} QA items")

    if args.max_samples is not None:
        qa_items = qa_items[: args.max_samples]
        logger.info(f"Subsampled to {len(qa_items)} items")

    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_qa_ids(args.output)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} already completed")

    trajectory_mode = getattr(args, "trajectory_mode", "summary") or "summary"
    if getattr(args, "no_trajectory", False) and trajectory_mode == "summary":
        trajectory_mode = "none"
    include_trajectory = trajectory_mode != "none"
    include_images = not args.no_images
    carla_video_dir = Path(args.carla_video_dir)
    grouped = getattr(args, "group_by_clip", False)
    workers = getattr(args, "workers", 1)

    logger.info(
        f"Starting inference: model={args.model}, "
        f"workers={workers}, "
        f"trajectory={'ON' if include_trajectory else 'OFF'}, "
        f"images={'ON' if include_images else 'OFF'}, "
        f"grouped={'ON' if grouped else 'OFF'}"
    )

    def _encode_clip_images(clip_id: str, clip: dict) -> list[tuple[str, str]]:
        image_data: list[tuple[str, str]] = []
        if not include_images:
            return image_data
        frame_paths = clip.get("frame_paths", [])
        if frame_paths:
            selected = select_frames(frame_paths, args.num_frames)
            for rel_path in selected:
                abs_path = nuscenes_root / rel_path
                if abs_path.exists():
                    image_data.append(encode_image(str(abs_path)))
        else:
            video_path = carla_video_dir / f"{clip_id}.mp4"
            if video_path.exists():
                image_data = extract_video_frames(str(video_path), args.num_frames)
        return image_data

    # --- build work items --------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    unflushed = 0

    from tqdm import tqdm

    processed = 0
    skipped = 0
    failed = 0

    if grouped:
        # --- Grouped mode: one call per clip ---
        clip_groups = group_qa_by_clip(qa_items)
        work_items = []
        for clip_id, group in clip_groups.items():
            group_ids = [qa.get("qa_id", "") for qa in group]
            if completed_ids and all(
                qid and qid in completed_ids for qid in group_ids
            ):
                skipped += len(group)
                continue
            clip = clips.get(clip_id)
            if clip is None:
                continue
            work_items.append((clip_id, group, clip))

        logger.info(
            f"{len(work_items)} clip groups to process "
            f"({skipped} questions skipped via resume)"
        )

        def process_group(item):
            clip_id, group, clip = item
            image_data = _encode_clip_images(clip_id, clip)
            prompt = build_grouped_prompt(
                group, clip,
                include_trajectory=include_trajectory,
                num_frames_sent=len(image_data),
            )
            raw_response = call_api(prompt, image_data)
            answers = parse_grouped_response(raw_response, len(group))
            records = []
            for qa_item, answer in zip(group, answers):
                records.append(make_prediction_record(qa_item, answer))
            return records

        mode = "a" if args.resume else "w"
        with open(output_path, mode) as out_f:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = {
                    executor.submit(process_group, item): item
                    for item in work_items
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Evaluating clips",
                ):
                    item = futures[future]
                    try:
                        records = future.result()
                    except Exception as exc:
                        # (3) Don't crash — emit failure records
                        _, group, _ = item
                        logger.error(
                            f"Future failed for clip group "
                            f"({len(group)} questions): {exc}"
                        )
                        records = [
                            make_prediction_record(qa, "")
                            for qa in group
                        ]
                        failed += len(group)

                    with write_lock:
                        for record in records:
                            # (4) Skip duplicates from partial resume
                            qa_id = record.get("qa_id", "")
                            if qa_id and qa_id in completed_ids:
                                skipped += 1
                                continue
                            out_f.write(json.dumps(record) + "\n")
                            processed += 1
                            unflushed += 1
                        # (5) Flush every N records
                        if unflushed >= FLUSH_EVERY:
                            out_f.flush()
                            unflushed = 0
            # Final flush
            out_f.flush()

    else:
        # --- Per-question mode ---
        work_items = []
        for qa_item in qa_items:
            qa_id = qa_item.get("qa_id", "")
            if qa_id and qa_id in completed_ids:
                skipped += 1
                continue
            clip_id = qa_item["clip_id"]
            clip = clips.get(clip_id)
            if clip is None:
                continue
            work_items.append((qa_item, clip))

        logger.info(
            f"{len(work_items)} questions to process "
            f"({skipped} skipped via resume)"
        )

        def process_question(item):
            qa_item, clip = item
            clip_id = qa_item["clip_id"]
            image_data = _encode_clip_images(clip_id, clip)
            prompt = build_prompt(
                qa_item, clip,
                include_trajectory=include_trajectory,
                num_frames_sent=len(image_data),
            )
            model_answer = call_api(prompt, image_data)
            return make_prediction_record(qa_item, model_answer)

        mode = "a" if args.resume else "w"
        with open(output_path, mode) as out_f:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = {
                    executor.submit(process_question, item): item
                    for item in work_items
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Evaluating",
                ):
                    item = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        # (3) Don't crash — emit failure record
                        qa_item, _ = item
                        logger.error(
                            f"Future failed for {qa_item.get('qa_id', '?')}: "
                            f"{exc}"
                        )
                        record = make_prediction_record(qa_item, "")
                        failed += 1

                    with write_lock:
                        out_f.write(json.dumps(record) + "\n")
                        processed += 1
                        unflushed += 1
                        # (5) Flush every N records
                        if unflushed >= FLUSH_EVERY:
                            out_f.flush()
                            unflushed = 0
            # Final flush
            out_f.flush()

    logger.info(
        f"Done: {processed} predictions written to {output_path} "
        f"({skipped} skipped via resume, {failed} failed)"
    )

    # --- optional evaluation -----------------------------------------------
    if args.run_eval:
        from evaluation.parsers import load_question_config
        from evaluation.metrics import evaluate

        question_config = load_question_config(args.config)

        records: list[dict] = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            logger.error("No predictions to evaluate.")
            return 1

        result = evaluate(records, question_config)
        print_eval_results(result, args.model, include_trajectory, include_images)

        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            logger.info(f"Metrics written to {metrics_path}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_common_parser(
        description="Evaluate HuggingFace-hosted VLMs on EgoDyn-Bench",
        default_model="moonshotai/Kimi-K2.5:fastest",
        api_key_env_var="HF_TOKEN",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel API workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--base_url", type=str, default=HF_ROUTER_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--timeout_s", type=float, default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("HF_TOKEN")
    if not api_key:
        logger.error(
            "HuggingFace token not found. "
            "Set HF_TOKEN in .env or pass --api_key."
        )
        return 1

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        max_retries=0,  # we handle retries ourselves
    )

    call_api = _make_caller(
        client, args.model, args.temperature,
        timeout_s=args.timeout_s,
    )

    return _run_parallel_evaluation(args, call_api)


if __name__ == "__main__":
    sys.exit(main())
