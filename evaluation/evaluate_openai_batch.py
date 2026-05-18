"""OpenAI Batch API evaluator for EgoDyn-Bench.

50% cheaper than real-time API, no rate-limit pressure, results within 24h.

Workflow (4 steps):

    # 1. Prepare batch request files (resizes images, splits into chunks)
    python evaluation/evaluate_openai_batch.py prepare \
        --selected_clips selected_clips.json \
        --nuscenes_qa output/nuscenes_clips/qa_pairs.jsonl \
        --carla_qa output/carla_clips/qa_pairs.jsonl \
        --model gpt-4o-mini --trajectory_mode none \
        --batch_dir generated/batch_gpt4o_mini

    # 2. Submit all batch files to OpenAI
    python evaluation/evaluate_openai_batch.py submit \
        --batch_dir generated/batch_gpt4o_mini

    # 3. Check status (re-run until all complete)
    python evaluation/evaluate_openai_batch.py status \
        --batch_dir generated/batch_gpt4o_mini

    # 4. Collect results and optionally run evaluation
    python evaluation/evaluate_openai_batch.py collect \
        --batch_dir generated/batch_gpt4o_mini \
        --output generated/gpt_4o_mini_answers.jsonl \
        --run_eval --metrics_output results/gpt_4o_mini.json
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator_common import (
    load_dotenv,
    load_clips_index,
    load_qa_items,
    select_frames,
    encode_image,
    extract_video_frames,
    build_prompt,
    build_grouped_prompt,
    group_qa_by_clip,
    parse_grouped_response,
    print_eval_results,
    resolve_carla_video_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# gpt-4o-mini limit is 200 MB; larger models allow up to 512 MB.
# Use 190 MB as a safe default that works for all models.
MAX_BATCH_FILE_BYTES = 190 * 1024 * 1024
# Max requests per batch (OpenAI limit is 50,000)
MAX_REQUESTS_PER_BATCH = 50_000
# Resize images before base64 encoding to reduce file size
DEFAULT_MAX_IMAGE_DIM = 512


# ---------------------------------------------------------------------------
# Image resizing (uses cv2 which is already a dependency)
# ---------------------------------------------------------------------------

def _resize_b64_image(
    b64_data: str,
    mime_type: str,
    max_dim: int = DEFAULT_MAX_IMAGE_DIM,
    jpeg_quality: int = 75,
) -> tuple[str, str]:
    """Resize a base64-encoded image to fit within max_dim, return new b64."""
    if not _HAS_CV2:
        return b64_data, mime_type

    import numpy as np

    raw = base64.b64decode(b64_data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return b64_data, mime_type

    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        # Already small enough — just re-encode as JPEG to save space
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("utf-8"), "image/jpeg"
        return b64_data, mime_type

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ok:
        return base64.b64encode(buf.tobytes()).decode("utf-8"), "image/jpeg"
    return b64_data, mime_type


# ---------------------------------------------------------------------------
# Batch request builder
# ---------------------------------------------------------------------------

def _build_batch_line(
    custom_id: str,
    model: str,
    prompt: str,
    image_data: list[tuple[str, str]],
    frame_detail: str,
    temperature: float,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Build one Batch API request object."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64, mime in image_data:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": frame_detail,
            },
        })

    # GPT-5+ models require "max_completion_tokens" instead of "max_tokens"
    if model.startswith("gpt-5") or model.startswith("o"):
        token_key = "max_completion_tokens"
    else:
        token_key = "max_tokens"

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            token_key: max_tokens,
            "temperature": temperature,
        },
    }


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    """Build batch request JSONL files from QA items + clips."""
    batch_dir = Path(args.batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    nuscenes_root = Path(args.nuscenes_root)
    resolve_carla_video_dir(args)
    carla_video_dir = Path(args.carla_video_dir)
    trajectory_mode = getattr(args, "trajectory_mode", "summary") or "summary"
    if getattr(args, "no_trajectory", False) and trajectory_mode == "summary":
        trajectory_mode = "none"
    include_trajectory = trajectory_mode != "none"
    include_images = not args.no_images
    max_image_dim = args.max_image_dim
    max_per_file = args.max_requests_per_file or MAX_REQUESTS_PER_BATCH

    # --- load data (same as evaluator_common) ---
    clips: dict[str, dict] = {}
    ns_idx = Path(args.nuscenes_index)
    if ns_idx.exists():
        clips.update(load_clips_index(ns_idx))
    ca_idx = Path(args.carla_index)
    if ca_idx.exists():
        clips.update(load_clips_index(ca_idx))

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

    grouped = args.group_by_clip

    # --- helper: encode + resize frames for a clip ---
    def _encode_clip(clip_id: str, clip: dict) -> list[tuple[str, str]]:
        image_data: list[tuple[str, str]] = []
        if not include_images:
            return image_data
        frame_paths = clip.get("frame_paths", [])
        if frame_paths:
            selected = select_frames(frame_paths, args.num_frames)
            for rel_path in selected:
                abs_path = nuscenes_root / rel_path
                if not abs_path.exists():
                    continue
                b64, mime = encode_image(str(abs_path))
                b64, mime = _resize_b64_image(b64, mime, max_image_dim)
                image_data.append((b64, mime))
        else:
            video_path = carla_video_dir / f"{clip_id}.mp4"
            if video_path.exists():
                for b64, mime in extract_video_frames(str(video_path), args.num_frames):
                    b64, mime = _resize_b64_image(b64, mime, max_image_dim)
                    image_data.append((b64, mime))
        return image_data

    # --- build requests and split into files ---
    file_idx = 0
    current_size = 0
    current_count = 0
    current_path = batch_dir / f"batch_{file_idx:04d}.jsonl"
    current_file = open(current_path, "w")
    batch_files: list[str] = [str(current_path)]
    id_map: dict[str, Any] = {}

    from tqdm import tqdm

    def _write_line(line: str) -> None:
        nonlocal file_idx, current_size, current_count, current_path, current_file
        line_bytes = len(line.encode("utf-8"))
        if (current_size + line_bytes > MAX_BATCH_FILE_BYTES
                or current_count >= max_per_file):
            current_file.close()
            file_idx += 1
            current_path = batch_dir / f"batch_{file_idx:04d}.jsonl"
            current_file = open(current_path, "w")
            batch_files.append(str(current_path))
            current_size = 0
            current_count = 0
        current_file.write(line)
        current_size += line_bytes
        current_count += 1

    if grouped:
        # ---- Grouped: one request per clip, all questions combined ----
        clip_groups = group_qa_by_clip(qa_items)
        for clip_id in tqdm(clip_groups, desc="Preparing clips"):
            group = clip_groups[clip_id]
            clip = clips.get(clip_id)
            if clip is None:
                continue

            image_data = _encode_clip(clip_id, clip)
            prompt = build_grouped_prompt(
                group, clip,
                include_trajectory=include_trajectory,
                num_frames_sent=len(image_data),
            )
            req = _build_batch_line(
                custom_id=clip_id,
                model=args.model,
                prompt=prompt,
                image_data=image_data,
                frame_detail=args.frame_detail,
                temperature=args.temperature,
            )
            _write_line(json.dumps(req) + "\n")

            # Store ordered list of qa metadata for this clip
            id_map[clip_id] = [
                {
                    "qa_id": qa.get("qa_id", ""),
                    "clip_id": clip_id,
                    "question_id": qa["question_id"],
                    "category": qa.get("category", "unknown"),
                    "oracle_label": qa.get("answer", qa.get("oracle_label", "")),
                }
                for qa in group
            ]
    else:
        # ---- Original: one request per question ----
        for i, qa_item in enumerate(tqdm(qa_items, desc="Preparing")):
            qa_id = qa_item.get("qa_id", f"item_{i}")
            clip_id = qa_item["clip_id"]
            clip = clips.get(clip_id)
            if clip is None:
                continue

            image_data = _encode_clip(clip_id, clip)
            prompt = build_prompt(
                qa_item, clip,
                include_trajectory=include_trajectory,
                num_frames_sent=len(image_data),
            )
            req = _build_batch_line(
                custom_id=qa_id,
                model=args.model,
                prompt=prompt,
                image_data=image_data,
                frame_detail=args.frame_detail,
                temperature=args.temperature,
            )
            _write_line(json.dumps(req) + "\n")

            id_map[qa_id] = {
                "clip_id": clip_id,
                "question_id": qa_item["question_id"],
                "category": qa_item.get("category", "unknown"),
                "oracle_label": qa_item.get("answer", qa_item.get("oracle_label", "")),
            }

    current_file.close()

    if current_count == 0 and file_idx > 0:
        os.unlink(current_path)
        batch_files.pop()

    n_total = sum(len(v) if isinstance(v, list) else 1 for v in id_map.values())
    manifest = {
        "model": args.model,
        "grouped": grouped,
        "trajectory": trajectory_mode != "none",
        "images": not args.no_images,
        "num_frames": args.num_frames,
        "frame_detail": args.frame_detail,
        "max_image_dim": max_image_dim,
        "total_requests": current_count + sum(
            1 for _ in range(file_idx)  # approximate
        ),
        "total_questions": n_total,
        "batch_files": batch_files,
        "batches": [],
    }
    manifest_path = batch_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    id_map_path = batch_dir / "id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)

    mode_str = f"grouped ({len(id_map)} clips)" if grouped else f"{len(id_map)} requests"
    logger.info(
        f"Prepared {n_total} questions as {mode_str} "
        f"in {len(batch_files)} file(s) → {batch_dir}"
    )
    for bf in batch_files:
        size_mb = Path(bf).stat().st_size / (1024 * 1024)
        logger.info(f"  {Path(bf).name}: {size_mb:.1f} MB")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: submit
# ---------------------------------------------------------------------------

def _count_in_progress(client, batches_info: list[dict]) -> int:
    """Return the number of batches still in progress."""
    n = 0
    for info in batches_info:
        status = info.get("status", "")
        if status in ("completed", "failed", "expired", "cancelled"):
            continue
        # Refresh from API
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            n += 1
    return n


def _wait_for_batches(client, batches_info: list[dict], poll_interval: int = 60) -> None:
    """Block until all in-progress batches have finished."""
    import time

    while True:
        n = _count_in_progress(client, batches_info)
        if n == 0:
            return
        logger.info(f"  {n} batch(es) still running, polling in {poll_interval}s...")
        time.sleep(poll_interval)


def cmd_submit(args: argparse.Namespace) -> int:
    """Upload batch files and create OpenAI batch jobs."""
    batch_dir = Path(args.batch_dir)
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}. Run 'prepare' first.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY or pass --api_key")
        return 1
    client = OpenAI(api_key=api_key)

    batches_info: list[dict] = manifest.get("batches", [])

    # Remove failed batches so their files can be re-submitted
    n_failed = sum(1 for b in batches_info if b.get("status") in ("failed", "expired"))
    if n_failed:
        logger.info(f"Removing {n_failed} failed/expired batch(es) from manifest for retry")
        batches_info = [b for b in batches_info if b.get("status") not in ("failed", "expired")]
        manifest["batches"] = batches_info

    submitted_files = {b["input_file"] for b in batches_info}
    limit = args.limit
    wait = args.wait
    n_submitted_this_run = 0

    pending_files = [
        bf for bf in manifest["batch_files"] if bf not in submitted_files
    ]
    if not pending_files:
        logger.info("All batch files already submitted.")
        return 0

    # Wait for any existing in-progress batches before submitting new ones
    n_in_progress = _count_in_progress(client, batches_info)
    if n_in_progress > 0:
        logger.info(f"Waiting for {n_in_progress} existing in-progress batch(es) to complete...")
        _wait_for_batches(client, batches_info)
        manifest["batches"] = batches_info
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info(
        f"{len(pending_files)} file(s) to submit"
        + (f", limit {limit} at a time" if limit else "")
        + (" (will wait between groups)" if wait and limit else "")
    )

    for bf_path in pending_files:
        # If limit is set and we've hit it, either wait or stop
        if limit and n_submitted_this_run >= limit:
            if wait:
                logger.info(f"Reached limit of {limit}, waiting for completion...")
                _wait_for_batches(client, batches_info)
                # Save progress
                manifest["batches"] = batches_info
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                n_submitted_this_run = 0
            else:
                logger.info(
                    f"Reached limit of {limit}. "
                    f"Re-run 'submit' after current batches complete."
                )
                break

        logger.info(f"Uploading {bf_path}...")
        with open(bf_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        logger.info(f"  File ID: {file_obj.id}")

        logger.info("Creating batch...")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "egodyn-bench", "model": manifest["model"]},
        )
        logger.info(f"  Batch ID: {batch.id}, status: {batch.status}")

        batches_info.append({
            "input_file": bf_path,
            "file_id": file_obj.id,
            "batch_id": batch.id,
            "status": batch.status,
        })
        n_submitted_this_run += 1

    manifest["batches"] = batches_info
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_remaining = len([
        bf for bf in manifest["batch_files"]
        if bf not in {b["input_file"] for b in batches_info}
    ])
    if n_remaining:
        logger.info(
            f"Submitted {n_submitted_this_run} batch(es), "
            f"{n_remaining} remaining. Re-run 'submit' later."
        )
    else:
        logger.info(
            f"All {len(batches_info)} batch(es) submitted. "
            f"Run 'status' to check progress."
        )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Check and display batch status."""
    batch_dir = Path(args.batch_dir)
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"No manifest at {manifest_path}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    batches_info = manifest.get("batches", [])
    if not batches_info:
        logger.info("No batches submitted yet.")
        return 0

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY or pass --api_key")
        return 1
    client = OpenAI(api_key=api_key)

    all_complete = True
    for info in batches_info:
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        info["output_file_id"] = getattr(batch, "output_file_id", None)
        info["error_file_id"] = getattr(batch, "error_file_id", None)
        counts = batch.request_counts
        completed = counts.completed if counts else "?"
        failed = counts.failed if counts else "?"
        total = counts.total if counts else "?"
        print(
            f"  Batch {info['batch_id']}: {batch.status} "
            f"({completed}/{total} done, {failed} failed)"
        )
        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_complete = False

    # Update manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if all_complete:
        print("\nAll batches finished. Run 'collect' to download results.")
    else:
        print("\nSome batches still running. Check again later.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> int:
    """Download batch results and create evaluation-compatible JSONL."""
    batch_dir = Path(args.batch_dir)
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"No manifest at {manifest_path}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    id_map_path = batch_dir / "id_map.json"
    with open(id_map_path) as f:
        id_map = json.load(f)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY or pass --api_key")
        return 1
    client = OpenAI(api_key=api_key)

    # Download all result files
    all_results: list[dict] = []
    for info in manifest.get("batches", []):
        batch = client.batches.retrieve(info["batch_id"])
        if batch.status != "completed":
            logger.warning(
                f"Batch {info['batch_id']} not completed (status={batch.status}), skipping"
            )
            continue

        output_file_id = batch.output_file_id
        if not output_file_id:
            logger.warning(f"Batch {info['batch_id']} has no output file")
            continue

        # Download result file
        result_path = batch_dir / f"result_{info['batch_id']}.jsonl"
        if not result_path.exists():
            logger.info(f"Downloading results for batch {info['batch_id']}...")
            content = client.files.content(output_file_id)
            result_path.write_bytes(content.read())

        # Parse results
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))

        # Also download error file if present
        error_file_id = batch.error_file_id
        if error_file_id:
            error_path = batch_dir / f"errors_{info['batch_id']}.jsonl"
            if not error_path.exists():
                logger.info(f"Downloading errors for batch {info['batch_id']}...")
                content = client.files.content(error_file_id)
                error_path.write_bytes(content.read())

    logger.info(f"Collected {len(all_results)} results from batch API")

    # Convert to evaluation format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = manifest.get("grouped", False)

    n_ok = 0
    n_err = 0
    with open(output_path, "w") as out_f:
        for result in all_results:
            custom_id = result["custom_id"]
            meta = id_map.get(custom_id)
            if meta is None:
                logger.warning(f"Unknown custom_id: {custom_id}")
                n_err += 1
                continue

            response = result.get("response", {})
            body = response.get("body", {})

            # Extract model answer from response
            model_answer = ""
            if response.get("status_code") == 200:
                choices = body.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content", "")
                    model_answer = text.strip() if text else ""
                    n_ok += 1
                else:
                    n_err += 1
            else:
                error = body.get("error", {})
                logger.warning(
                    f"Request {custom_id} failed: "
                    f"{error.get('message', 'unknown error')}"
                )
                n_err += 1

            if grouped:
                # meta is a list of per-question metadata
                answers = parse_grouped_response(model_answer, len(meta))
                for qa_meta, answer in zip(meta, answers):
                    record = {
                        "qa_id": qa_meta.get("qa_id", ""),
                        "clip_id": qa_meta["clip_id"],
                        "question_id": qa_meta["question_id"],
                        "category": qa_meta["category"],
                        "oracle_label": qa_meta["oracle_label"],
                        "model_answer": answer,
                    }
                    out_f.write(json.dumps(record) + "\n")
            else:
                record = {
                    "qa_id": custom_id,
                    "clip_id": meta["clip_id"],
                    "question_id": meta["question_id"],
                    "category": meta["category"],
                    "oracle_label": meta["oracle_label"],
                    "model_answer": model_answer,
                }
                out_f.write(json.dumps(record) + "\n")

    logger.info(
        f"Wrote {n_ok + n_err} predictions to {output_path} "
        f"({n_ok} OK, {n_err} errors)"
    )

    # --- optional evaluation ---
    if args.run_eval or args.metrics_output:
        from evaluation.parsers import load_question_config
        from evaluation.metrics import evaluate

        config_path = args.config or str(
            PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"
        )
        question_config = load_question_config(config_path)

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
        print_eval_results(
            result, manifest["model"],
            manifest.get("trajectory", False),
            manifest.get("images", True),
        )

        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            logger.info(f"Metrics written to {metrics_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="OpenAI Batch API evaluator for EgoDyn-Bench",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenAI API key (default: $OPENAI_API_KEY)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Build batch request JSONL files")
    p_prep.add_argument("--selected_clips", type=str, required=True)
    p_prep.add_argument(
        "--nuscenes_qa", type=str,
        default="output/nuscenes_clips/qa_pairs.jsonl",
    )
    p_prep.add_argument(
        "--carla_qa", type=str,
        default="output/carla_clips/qa_pairs.jsonl",
    )
    p_prep.add_argument(
        "--nuscenes_index", type=str,
        default="output/nuscenes_clips/clips_index.jsonl",
    )
    p_prep.add_argument(
        "--carla_index", type=str,
        default="output/carla_clips/clips_index.jsonl",
    )
    p_prep.add_argument(
        "--nuscenes_root", type=str,
        default=None,
    )
    p_prep.add_argument(
        "--carla_video_source", type=str, default="transferred",
        choices=["simulation", "transferred"],
        help="Which CARLA video data to use (default: transferred)",
    )
    p_prep.add_argument(
        "--carla_video_dir", type=str, default=None,
        help="Explicit directory for CARLA MP4 chunks (overrides --carla_video_source)",
    )
    p_prep.add_argument("--batch_dir", type=str, required=True)
    p_prep.add_argument("--model", type=str, default="gpt-4o-mini")
    p_prep.add_argument("--num_frames", type=int, default=10)
    p_prep.add_argument("--frame_detail", type=str, default="low")
    p_prep.add_argument("--temperature", type=float, default=0.0)
    p_prep.add_argument("--trajectory_mode", type=str, default="summary", choices=["none", "summary", "timeseries", "coordinates", "full"], help="Trajectory embedding mode.")
    p_prep.add_argument("--no_trajectory", action="store_true", default=False, help="(Deprecated) Shorthand for --trajectory_mode none.")
    p_prep.add_argument("--no_images", action="store_true")
    p_prep.add_argument("--max_samples", type=int, default=None)
    p_prep.add_argument(
        "--max_image_dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
        help="Resize images so max dimension fits this (reduces batch file size)",
    )
    p_prep.add_argument(
        "--group_by_clip", action="store_true",
        help="Group all questions per clip into a single request (~14x cheaper)",
    )
    p_prep.add_argument(
        "--max_requests_per_file", type=int, default=None,
        help="Max requests per batch file (use to stay under enqueued token limits)",
    )

    # --- submit ---
    p_sub = sub.add_parser("submit", help="Upload and start batch jobs")
    p_sub.add_argument("--batch_dir", type=str, required=True)
    p_sub.add_argument(
        "--limit", type=int, default=None,
        help="Max batch files to submit at once (avoids enqueued token limits)",
    )
    p_sub.add_argument(
        "--wait", action="store_true",
        help="When --limit is set, poll and wait for completion then submit more",
    )

    # --- status ---
    p_stat = sub.add_parser("status", help="Check batch job status")
    p_stat.add_argument("--batch_dir", type=str, required=True)

    # --- collect ---
    p_coll = sub.add_parser("collect", help="Download results and evaluate")
    p_coll.add_argument("--batch_dir", type=str, required=True)
    p_coll.add_argument("--output", type=str, required=True)
    p_coll.add_argument("--run_eval", action="store_true")
    p_coll.add_argument("--metrics_output", type=str, default=None)
    p_coll.add_argument(
        "--config", type=str, default=None,
        help="Path to questions_template.yaml (for eval)",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        return cmd_prepare(args)
    elif args.command == "submit":
        return cmd_submit(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "collect":
        return cmd_collect(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
