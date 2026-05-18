"""Anthropic Claude Batch API evaluator for EgoDyn-Bench.

50% cheaper than real-time API, no rate-limit pressure, results within 24h.

Workflow (4 steps):

    # 1. Prepare batch request files (resizes images, splits into chunks)
    python evaluation/evaluate_claude_batch.py prepare \
        --selected_clips selected_clips.json \
        --nuscenes_qa output/nuscenes_clips/qa_pairs.jsonl \
        --carla_qa output/carla_clips/qa_pairs.jsonl \
        --model claude-sonnet-4-5-20250929 --trajectory_mode none \
        --batch_dir generated/batch_claude_sonnet

    # 2. Submit all batch files to Anthropic
    python evaluation/evaluate_claude_batch.py submit \
        --batch_dir generated/batch_claude_sonnet

    # 3. Check status (re-run until all complete)
    python evaluation/evaluate_claude_batch.py status \
        --batch_dir generated/batch_claude_sonnet

    # 4. Collect results and optionally run evaluation
    python evaluation/evaluate_claude_batch.py collect \
        --batch_dir generated/batch_claude_sonnet \
        --output generated/claude_sonnet_answers.jsonl \
        --run_eval --metrics_output results/claude_sonnet.json
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
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
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

# Anthropic batch API sends requests in the HTTP body (no file upload),
# so we split batches to keep each API call at a manageable size.
# Default: 500 requests per batch for image runs, 5000 for text-only.
DEFAULT_MAX_REQUESTS_IMAGE = 500
DEFAULT_MAX_REQUESTS_TEXT = 5000
DEFAULT_MAX_IMAGE_DIM = 512


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
# Batch request builder (Anthropic format)
# ---------------------------------------------------------------------------

def _build_batch_request(
    custom_id: str,
    model: str,
    prompt: str,
    image_data: list[tuple[str, str]],
    temperature: float,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Build one Anthropic Batch API request object."""
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

    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
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
    id_map: dict[str, Any] = {}

    from tqdm import tqdm

    def _write_line(line: str) -> None:
        nonlocal file_idx, current_count, current_path, current_file
        if current_count >= max_per_batch:
            current_file.close()
            file_idx += 1
            current_path = batch_dir / f"batch_{file_idx:04d}.jsonl"
            current_file = open(current_path, "w")
            batch_files.append(str(current_path))
            current_count = 0
        current_file.write(line)
        current_count += 1

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
            req = _build_batch_request(
                custom_id=clip_id,
                model=args.model,
                prompt=prompt,
                image_data=image_data,
                temperature=args.temperature,
            )
            _write_line(json.dumps(req) + "\n")

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
            req = _build_batch_request(
                custom_id=qa_id,
                model=args.model,
                prompt=prompt,
                image_data=image_data,
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

    # Remove empty last file
    if current_count == 0 and file_idx > 0:
        os.unlink(current_path)
        batch_files.pop()

    # Write manifest
    n_total = sum(len(v) if isinstance(v, list) else 1 for v in id_map.values())
    manifest = {
        "provider": "anthropic",
        "model": args.model,
        "grouped": grouped,
        "trajectory": trajectory_mode != "none",
        "images": not args.no_images,
        "num_frames": args.num_frames,
        "max_image_dim": max_image_dim,
        "max_per_batch": max_per_batch,
        "total_requests": len(id_map),
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

def cmd_submit(args: argparse.Namespace) -> int:
    """Read prepared JSONL files and create Anthropic batch jobs."""
    batch_dir = Path(args.batch_dir)
    manifest_path = batch_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}. Run 'prepare' first.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY or pass --api_key")
        return 1
    client = Anthropic(api_key=api_key)

    batches_info: list[dict] = manifest.get("batches", [])
    submitted_files = {b["input_file"] for b in batches_info}

    for bf_path in manifest["batch_files"]:
        if bf_path in submitted_files:
            logger.info(f"Already submitted: {bf_path}, skipping")
            continue

        # Read requests from JSONL
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
        batch = client.messages.batches.create(requests=requests)
        logger.info(f"  Batch ID: {batch.id}, status: {batch.processing_status}")

        batches_info.append({
            "input_file": bf_path,
            "batch_id": batch.id,
            "status": batch.processing_status,
            "n_requests": len(requests),
        })

    manifest["batches"] = batches_info
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(b.get("n_requests", 0) for b in batches_info)
    logger.info(
        f"Submitted {len(batches_info)} batch(es), {total} total requests. "
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

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY or pass --api_key")
        return 1
    client = Anthropic(api_key=api_key)

    all_complete = True
    for info in batches_info:
        batch = client.messages.batches.retrieve(info["batch_id"])
        info["status"] = batch.processing_status
        counts = batch.request_counts
        print(
            f"  Batch {info['batch_id']}: {batch.processing_status} "
            f"(succeeded={counts.succeeded}, errored={counts.errored}, "
            f"processing={counts.processing}, canceled={counts.canceled})"
        )
        if batch.processing_status not in ("ended",):
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
    """Stream batch results and create evaluation-compatible JSONL."""
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

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY or pass --api_key")
        return 1
    client = Anthropic(api_key=api_key)

    # Collect results from all batches
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = manifest.get("grouped", False)
    n_ok = 0
    n_err = 0

    with open(output_path, "w") as out_f:
        for info in manifest.get("batches", []):
            batch = client.messages.batches.retrieve(info["batch_id"])
            if batch.processing_status != "ended":
                logger.warning(
                    f"Batch {info['batch_id']} not ended "
                    f"(status={batch.processing_status}), skipping"
                )
                continue

            # Cache results to disk for re-runs
            cache_path = batch_dir / f"result_{info['batch_id']}.jsonl"
            if not cache_path.exists():
                logger.info(f"Downloading results for batch {info['batch_id']}...")
                with open(cache_path, "w") as cache_f:
                    for entry in client.messages.batches.results(info["batch_id"]):
                        cache_f.write(json.dumps(entry.model_dump()) + "\n")

            # Parse cached results
            with open(cache_path) as cache_f:
                for line in cache_f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)

                    custom_id = entry["custom_id"]
                    meta = id_map.get(custom_id)
                    if meta is None:
                        logger.warning(f"Unknown custom_id: {custom_id}")
                        n_err += 1
                        continue

                    model_answer = ""
                    result = entry.get("result", {})
                    if result.get("type") == "succeeded":
                        message = result.get("message", {})
                        content_blocks = message.get("content", [])
                        if content_blocks:
                            model_answer = content_blocks[0].get("text", "").strip()
                        n_ok += 1
                    else:
                        error = result.get("error", {})
                        logger.warning(
                            f"Request {custom_id} failed: "
                            f"{error.get('type', 'unknown')}: "
                            f"{error.get('message', '')}"
                        )
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
        description="Anthropic Claude Batch API evaluator for EgoDyn-Bench",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)",
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
    p_prep.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    p_prep.add_argument("--num_frames", type=int, default=10)
    p_prep.add_argument("--temperature", type=float, default=0.0)
    p_prep.add_argument("--trajectory_mode", type=str, default="summary", choices=["none", "summary", "timeseries", "coordinates", "full"], help="Trajectory embedding mode.")
    p_prep.add_argument("--no_trajectory", action="store_true", default=False, help="(Deprecated) Shorthand for --trajectory_mode none.")
    p_prep.add_argument("--no_images", action="store_true")
    p_prep.add_argument("--max_samples", type=int, default=None)
    p_prep.add_argument(
        "--max_image_dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
        help="Resize images so max dimension fits this (reduces batch size)",
    )
    p_prep.add_argument(
        "--max_per_batch", type=int, default=None,
        help=f"Max requests per batch (default: {DEFAULT_MAX_REQUESTS_IMAGE} "
             f"with images, {DEFAULT_MAX_REQUESTS_TEXT} text-only)",
    )
    p_prep.add_argument(
        "--group_by_clip", action="store_true",
        help="Group all questions per clip into a single request (~14x cheaper)",
    )

    # --- submit ---
    p_sub = sub.add_parser("submit", help="Upload and start batch jobs")
    p_sub.add_argument("--batch_dir", type=str, required=True)

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
