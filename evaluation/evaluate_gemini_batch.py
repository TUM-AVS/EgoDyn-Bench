"""Google Gemini Batch API evaluator for EgoDyn-Bench.

50% cheaper than real-time API, higher throughput, results within 24h.

Gemini batch uses inline requests (no file upload). Requests are sent
directly in the API call body, so we split into batches by request count.

Workflow (4 steps):

    # 1. Prepare batch request files
    python evaluation/evaluate_gemini_batch.py prepare \
        --selected_clips selected_clips.json \
        --nuscenes_qa output/nuscenes_clips/qa_pairs.jsonl \
        --carla_qa output/carla_clips/qa_pairs.jsonl \
        --model gemini-2.5-flash --trajectory_mode none \
        --batch_dir generated/batch_gemini_flash

    # 2. Submit all batch files to Google
    python evaluation/evaluate_gemini_batch.py submit \
        --batch_dir generated/batch_gemini_flash

    # 3. Check status (re-run until all complete)
    python evaluation/evaluate_gemini_batch.py status \
        --batch_dir generated/batch_gemini_flash

    # 4. Collect results and optionally run evaluation
    python evaluation/evaluate_gemini_batch.py collect \
        --batch_dir generated/batch_gemini_flash \
        --output generated/gemini_flash_answers.jsonl \
        --run_eval --metrics_output results/gemini_flash.json
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator_common import (  # noqa: E402
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

# Gemini batch sends inline requests in the API body.
# Split to keep payloads manageable.
DEFAULT_MAX_REQUESTS_IMAGE = 500
DEFAULT_MAX_REQUESTS_TEXT = 5000
DEFAULT_MAX_IMAGE_DIM = 512

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------------
# Image resizing (cv2)
# ---------------------------------------------------------------------------

def _resize_b64_image(
    b64_data: str,
    mime_type: str,
    max_dim: int = DEFAULT_MAX_IMAGE_DIM,
    jpeg_quality: int = 75,
) -> tuple[str, str]:
    """Resize a base64-encoded image to fit within max_dim."""
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
# Model helpers
# ---------------------------------------------------------------------------

def _is_gemini3(model: str) -> bool:
    return "gemini-3" in model


# ---------------------------------------------------------------------------
# Batch request builder (Gemini inline format)
# ---------------------------------------------------------------------------

def _build_gemini_request(
    prompt: str,
    image_data: list[tuple[str, str]],
    temperature: float,
    max_output_tokens: int = 256,
    thinking_level: str | None = None,
    is_g3: bool = False,
) -> dict[str, Any]:
    """Build one Gemini inline batch request dict.

    Uses the ``InlinedRequest`` schema: ``contents`` + ``config``
    (a ``GenerateContentConfig``).
    """
    # Parts: images first, then text
    parts: list[dict[str, Any]] = []
    for b64, mime in image_data:
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": b64,
            }
        })
    parts.append({"text": prompt})

    config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    # Gemini 3 thinking config
    if is_g3 and thinking_level:
        config["thinking_config"] = {
            "thinking_level": thinking_level,
        }

    return {
        "contents": [{"role": "user", "parts": parts}],
        "config": config,
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
    is_g3 = _is_gemini3(args.model)

    max_per_batch = (
        args.max_per_batch
        or (DEFAULT_MAX_REQUESTS_IMAGE if include_images else DEFAULT_MAX_REQUESTS_TEXT)
    )

    # --- load data ---
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

    # Gemini 3 temperature default
    temperature = args.temperature
    if is_g3 and temperature == 0.0:
        temperature = 1.0
        logger.info("Gemini 3 detected — using temperature=1.0 (recommended)")

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
    current_count = 0
    current_path = batch_dir / f"batch_{file_idx:04d}.jsonl"
    current_file = open(current_path, "w")
    batch_files: list[str] = [str(current_path)]

    # Ordered list of ids per batch file (Gemini returns results in order)
    # In grouped mode these are clip_ids; in per-question mode they are qa_ids.
    batch_qa_ids: dict[str, list[str]] = {str(current_path): []}
    id_map: dict[str, Any] = {}

    from tqdm import tqdm

    def _write_line(line: str, entry_id: str) -> None:
        nonlocal file_idx, current_count, current_path, current_file
        if current_count >= max_per_batch:
            current_file.close()
            file_idx += 1
            current_path = batch_dir / f"batch_{file_idx:04d}.jsonl"
            current_file = open(current_path, "w")
            batch_files.append(str(current_path))
            batch_qa_ids[str(current_path)] = []
            current_count = 0
        current_file.write(line)
        current_count += 1
        batch_qa_ids[str(current_path)].append(entry_id)

    if grouped:
        # ---- Grouped: one request per clip ----
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
            req = _build_gemini_request(
                prompt=prompt,
                image_data=image_data,
                temperature=temperature,
                thinking_level=args.thinking_level if is_g3 else None,
                is_g3=is_g3,
            )
            _write_line(json.dumps(req) + "\n", clip_id)

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
            req = _build_gemini_request(
                prompt=prompt,
                image_data=image_data,
                temperature=temperature,
                thinking_level=args.thinking_level if is_g3 else None,
                is_g3=is_g3,
            )
            _write_line(json.dumps(req) + "\n", qa_id)

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
        del batch_qa_ids[str(current_path)]

    # Write manifest
    n_total = sum(len(v) if isinstance(v, list) else 1 for v in id_map.values())
    manifest = {
        "provider": "google",
        "model": args.model,
        "grouped": grouped,
        "trajectory": trajectory_mode != "none",
        "images": not args.no_images,
        "num_frames": args.num_frames,
        "max_image_dim": max_image_dim,
        "max_per_batch": max_per_batch,
        "temperature": temperature,
        "thinking_level": args.thinking_level if is_g3 else None,
        "total_requests": len(id_map),
        "total_questions": n_total,
        "batch_files": batch_files,
        "batch_qa_ids": batch_qa_ids,
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
        n = len(batch_qa_ids[bf])
        logger.info(f"  {Path(bf).name}: {n} requests, {size_mb:.1f} MB")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: submit
# ---------------------------------------------------------------------------

def _count_pending(batches_info: list[dict], client: genai.Client) -> int:
    """Count batch jobs that are still pending or running."""
    pending = 0
    for info in batches_info:
        if info["state"] in COMPLETED_STATES:
            continue
        try:
            job = client.batches.get(name=info["job_name"])
            info["state"] = job.state.name
            if job.state.name not in COMPLETED_STATES:
                pending += 1
        except Exception:
            pending += 1
    return pending


def _wait_for_batches(
    batches_info: list[dict],
    client: genai.Client,
    poll_interval: int = 60,
) -> None:
    """Block until all submitted batches reach a completed state."""
    while True:
        pending = _count_pending(batches_info, client)
        if pending == 0:
            return
        logger.info(
            f"  {pending} batch(es) still running, "
            f"polling in {poll_interval}s..."
        )
        time.sleep(poll_interval)


def cmd_submit(args: argparse.Namespace) -> int:
    """Read prepared JSONL files and create Gemini batch jobs."""
    batch_dir = Path(args.batch_dir)
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}. Run 'prepare' first.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    api_key = (
        args.api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        logger.error("Set GOOGLE_API_KEY or pass --api_key")
        return 1

    client_kwargs = {"api_key": api_key}
    if args.project:
        client_kwargs["project"] = args.project
    client = genai.Client(**client_kwargs)

    model = manifest["model"]
    batches_info: list[dict] = manifest.get("batches", [])
    submitted_files = {b["input_file"] for b in batches_info}

    limit = args.limit  # max batches to submit before waiting
    wait = args.wait  # wait for completion between waves

    def _save_manifest() -> None:
        manifest["batches"] = batches_info
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

    n_submitted = 0
    wave_count = 0
    for bf_path in manifest["batch_files"]:
        if bf_path in submitted_files:
            continue

        # If limit is set and we've hit it, wait for current wave
        if limit and wave_count >= limit:
            if wait:
                logger.info(
                    f"Submitted {wave_count} batch(es) this wave. "
                    f"Waiting for completion before next wave..."
                )
                _save_manifest()
                _wait_for_batches(batches_info, client)
                wave_count = 0
            else:
                logger.info(
                    f"Submitted {wave_count} batch(es) (--limit reached). "
                    f"Re-run 'submit' later or use --wait."
                )
                _save_manifest()
                return 0

        # Read inline requests from JSONL
        requests = []
        with open(bf_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    requests.append(json.loads(line))

        if not requests:
            logger.warning(f"Empty batch file: {bf_path}, skipping")
            continue

        logger.info(f"Submitting {len(requests)} requests from {bf_path}...")
        try:
            batch_job = client.batches.create(
                model=f"models/{model}",
                src=requests,
                config={"display_name": Path(bf_path).stem},
            )
        except Exception as e:
            logger.warning(f"Submit failed for {bf_path}: {e}")
            if wait and wave_count > 0:
                logger.info(
                    "Hit quota limit. Waiting for current batches "
                    "to complete before retrying..."
                )
                _save_manifest()
                _wait_for_batches(batches_info, client)
                wave_count = 0
                # Retry the failed submission
                try:
                    batch_job = client.batches.create(
                        model=f"models/{model}",
                        src=requests,
                        config={"display_name": Path(bf_path).stem},
                    )
                except Exception as e2:
                    logger.warning(f"Retry also failed for {bf_path}: {e2}")
                    logger.info("Saving progress and stopping.")
                    _save_manifest()
                    return 0
            else:
                logger.info("Saving progress and stopping. Re-run 'submit' later.")
                _save_manifest()
                return 0

        logger.info(f"  Job: {batch_job.name}, state: {batch_job.state.name}")

        batches_info.append({
            "input_file": bf_path,
            "job_name": batch_job.name,
            "state": batch_job.state.name,
            "n_requests": len(requests),
        })
        n_submitted += 1
        wave_count += 1
        # Save after each successful submission
        _save_manifest()

    # If --wait was given, wait for the final wave too
    if wait and _count_pending(batches_info, client) > 0:
        logger.info("Waiting for final batch wave to complete...")
        _wait_for_batches(batches_info, client)

    total = sum(b.get("n_requests", 0) for b in batches_info)
    n_remaining = len(manifest["batch_files"]) - len(batches_info)
    if n_remaining > 0:
        logger.info(
            f"Submitted {n_submitted} batch(es), {n_remaining} remaining. "
            f"Re-run 'submit' later."
        )
    else:
        logger.info(
            f"All {len(batches_info)} batch(es) submitted, "
            f"{total} total requests. Run 'status' to check progress."
        )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Check and display batch job status."""
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

    api_key = (
        args.api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        logger.error("Set GOOGLE_API_KEY or pass --api_key")
        return 1

    client_kwargs = {"api_key": api_key}
    if args.project:
        client_kwargs["project"] = args.project
    client = genai.Client(**client_kwargs)

    all_complete = True
    for info in batches_info:
        batch_job = client.batches.get(name=info["job_name"])
        info["state"] = batch_job.state.name
        print(f"  Job {info['job_name']}: {batch_job.state.name}")
        if batch_job.state.name not in COMPLETED_STATES:
            all_complete = False

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
    """Retrieve batch results and create evaluation-compatible JSONL."""
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

    batch_qa_ids = manifest.get("batch_qa_ids", {})

    api_key = (
        args.api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        logger.error("Set GOOGLE_API_KEY or pass --api_key")
        return 1

    client_kwargs = {"api_key": api_key}
    if args.project:
        client_kwargs["project"] = args.project
    client = genai.Client(**client_kwargs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = manifest.get("grouped", False)
    n_ok = 0
    n_err = 0

    with open(output_path, "w") as out_f:
        for info in manifest.get("batches", []):
            batch_job = client.batches.get(name=info["job_name"])
            if batch_job.state.name != "JOB_STATE_SUCCEEDED":
                logger.warning(
                    f"Job {info['job_name']} not succeeded "
                    f"(state={batch_job.state.name}), skipping"
                )
                continue

            # Get the ordered ids for this batch
            entry_ids = batch_qa_ids.get(info["input_file"], [])

            # Cache results to disk
            cache_path = batch_dir / f"result_{Path(info['job_name']).name}.json"
            if not cache_path.exists():
                logger.info(f"Downloading results for {info['job_name']}...")
                responses = []
                if batch_job.dest and batch_job.dest.inlined_responses:
                    for resp in batch_job.dest.inlined_responses:
                        if resp.response:
                            responses.append({
                                "text": resp.response.text or "",
                                "error": None,
                            })
                        elif resp.error:
                            responses.append({
                                "text": "",
                                "error": str(resp.error),
                            })
                        else:
                            responses.append({"text": "", "error": "empty response"})
                with open(cache_path, "w") as cf:
                    json.dump(responses, cf)
            else:
                with open(cache_path) as cf:
                    responses = json.load(cf)

            # Match responses to ids by order
            if len(responses) != len(entry_ids):
                logger.warning(
                    f"Response count ({len(responses)}) != request count "
                    f"({len(entry_ids)}) for {info['job_name']}"
                )

            for idx, entry_id in enumerate(entry_ids):
                meta = id_map.get(entry_id)
                if meta is None:
                    n_err += 1
                    continue

                model_answer = ""
                if idx < len(responses):
                    resp = responses[idx]
                    if resp.get("error"):
                        logger.warning(f"Request {entry_id}: {resp['error']}")
                        n_err += 1
                    else:
                        model_answer = resp.get("text", "").strip()
                        n_ok += 1
                else:
                    logger.warning(f"No response for {entry_id}")
                    n_err += 1

                if grouped:
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
                        "qa_id": entry_id,
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
        description="Google Gemini Batch API evaluator for EgoDyn-Bench",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Google API key (default: $GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Google Cloud project ID (optional)",
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
    p_prep.add_argument("--model", type=str, default="gemini-2.5-flash")
    p_prep.add_argument("--num_frames", type=int, default=10)
    p_prep.add_argument("--temperature", type=float, default=0.0)
    p_prep.add_argument("--trajectory_mode", type=str, default="summary", choices=["none", "summary", "timeseries", "coordinates", "full"], help="Trajectory embedding mode.")
    p_prep.add_argument("--no_trajectory", action="store_true", default=False, help="(Deprecated) Shorthand for --trajectory_mode none.")
    p_prep.add_argument("--no_images", action="store_true")
    p_prep.add_argument("--max_samples", type=int, default=None)
    p_prep.add_argument(
        "--max_image_dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
        help="Resize images so max dimension fits this",
    )
    p_prep.add_argument(
        "--max_per_batch", type=int, default=None,
        help=f"Max requests per batch (default: {DEFAULT_MAX_REQUESTS_IMAGE} "
             f"with images, {DEFAULT_MAX_REQUESTS_TEXT} text-only)",
    )
    p_prep.add_argument(
        "--thinking_level", type=str, default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Thinking level for Gemini 3 models (ignored for Gemini 2.x)",
    )
    p_prep.add_argument(
        "--group_by_clip", action="store_true",
        help="Group all questions per clip into a single request (~14x cheaper)",
    )

    # --- submit ---
    p_sub = sub.add_parser("submit", help="Upload and start batch jobs")
    p_sub.add_argument("--batch_dir", type=str, required=True)
    p_sub.add_argument(
        "--limit", type=int, default=None,
        help="Max batch files to submit per wave (for quota management)",
    )
    p_sub.add_argument(
        "--wait", action="store_true",
        help="Wait for batch completion between waves (use with --limit)",
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
