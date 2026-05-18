"""Shared evaluation logic for EgoDyn-Bench VLM evaluators.

All model-specific evaluators (GPT-4o, Gemini, Claude) import from this
module to avoid duplicating data loading, frame handling, prompt building,
and output formatting.

The main entry point is :func:`run_evaluation`, which runs the full
inference loop given a callable that handles the API call.
"""

import argparse
import base64
import json
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directories for CARLA video sources
CARLA_VIDEO_DIRS = {
    "simulation": PROJECT_ROOT / "output" / "carla_chunks",
    # Override with EGODYN_CARLA_TRANSFERRED_DIR or pass --carla_video_dir explicitly.
    "transferred": Path(
        os.environ.get("EGODYN_CARLA_TRANSFERRED_DIR", "./data/carla/benchmark_transferred")
    ),
}

logger = logging.getLogger(__name__)


def load_dotenv(path: str | Path | None = None) -> None:
    """Load key=value pairs from a .env file into ``os.environ``.

    Skips blank lines and comments (``#``).  Does not override variables
    that are already set in the environment.  No external dependency.
    """
    if path is None:
        path = PROJECT_ROOT / ".env"
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
    logger.debug(f"Loaded .env from {path}")

# Type alias: (prompt, image_data) -> model_answer
ApiCallFn = Callable[[str, list[tuple[str, str]]], str]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clips_index(path: str | Path) -> dict[str, dict]:
    """Load clips_index.jsonl and return a dict keyed by clip_id."""
    clips: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            clips[rec["clip_id"]] = rec
    logger.info(f"Loaded {len(clips)} clips from {path}")
    return clips


def load_qa_items(
    path: str | Path,
    max_samples: int | None = None,
) -> list[dict]:
    """Load QA items from JSONL."""
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_samples is not None and len(items) >= max_samples:
                break
    logger.info(f"Loaded {len(items)} QA items from {path}")
    return items


def load_completed_qa_ids(path: str | Path) -> set[str]:
    """Load qa_ids already present in an output file (for --resume)."""
    ids: set[str] = set()
    if not Path(path).exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qa_id = rec.get("qa_id")
                if qa_id:
                    ids.add(qa_id)
            except json.JSONDecodeError:
                continue
    logger.info(f"Resuming: {len(ids)} predictions already in {path}")
    return ids


# ---------------------------------------------------------------------------
# Frame selection & encoding
# ---------------------------------------------------------------------------

def select_frames(frame_paths: list[str], num_frames: int) -> list[str]:
    """Pick *num_frames* evenly spaced paths from the full list."""
    n = len(frame_paths)
    if n == 0:
        return []
    if num_frames >= n:
        return list(frame_paths)
    if num_frames == 1:
        return [frame_paths[n // 2]]  # middle frame
    indices = [round(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frame_paths[i] for i in indices]


def encode_image(image_path: str) -> tuple[str, str]:
    """Base64-encode an image and return (data, mime_type)."""
    ext = Path(image_path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def extract_video_frames(
    video_path: str,
    num_frames: int,
) -> list[tuple[str, str]]:
    """Extract evenly spaced frames from a video file as base64 JPEG.

    Requires ``opencv-python``. Returns a list of (base64_data, mime_type).
    """
    if not _HAS_CV2:
        logger.warning(
            "opencv-python not installed — cannot extract video frames. "
            "Run: pip install opencv-python"
        )
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    if num_frames >= total:
        indices = list(range(total))
    elif num_frames == 1:
        indices = [total // 2]
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames: list[tuple[str, str]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            data = base64.b64encode(buf.tobytes()).decode("utf-8")
            frames.append((data, "image/jpeg"))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Optical flow overlay
# ---------------------------------------------------------------------------

def _compute_flow_overlay(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay dense optical flow (Farneback) as HSV colour on *curr_bgr*.

    Hue encodes flow direction, saturation encodes magnitude.
    Returns a BGR uint8 image of the same size as *curr_bgr*.
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*curr_bgr.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blended = cv2.addWeighted(curr_bgr, 1 - alpha, flow_bgr, alpha, 0)
    return blended


def overlay_flow_on_encoded_frames(
    image_data: list[tuple[str, str]],
    alpha: float = 0.5,
) -> list[tuple[str, str]]:
    """Replace each frame (except the first) with a flow-overlaid version.

    Decodes base64 JPEGs, computes pairwise Farneback flow, blends, and
    re-encodes.  The first frame is returned unmodified (no previous frame).
    """
    if not _HAS_CV2 or len(image_data) < 2:
        return image_data

    # Decode all frames
    bgr_frames: list[np.ndarray] = []
    for b64, _mime in image_data:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            bgr_frames.append(img)

    if len(bgr_frames) < 2:
        return image_data

    result: list[tuple[str, str]] = [image_data[0]]  # first frame unchanged
    for i in range(1, len(bgr_frames)):
        blended = _compute_flow_overlay(bgr_frames[i - 1], bgr_frames[i], alpha)
        ok, buf = cv2.imencode(".jpg", blended)
        if ok:
            result.append((base64.b64encode(buf.tobytes()).decode("utf-8"), "image/jpeg"))
        else:
            result.append(image_data[i])
    return result


# ---------------------------------------------------------------------------
# Trajectory array loading & formatting
# ---------------------------------------------------------------------------

# Directories containing pre-computed NPZ arrays
_ARRAYS_DIRS = {
    "nuscenes": PROJECT_ROOT / "output" / "nuscenes_clips" / "arrays",
    "carla": PROJECT_ROOT / "output" / "carla_clips" / "arrays",
}


def _select_indices(n_total: int, n_select: int) -> list[int]:
    """Pick *n_select* evenly-spaced indices from [0, n_total)."""
    if n_total == 0:
        return []
    if n_select >= n_total:
        return list(range(n_total))
    if n_select == 1:
        return [n_total // 2]
    return [round(i * (n_total - 1) / (n_select - 1)) for i in range(n_select)]


def load_trajectory_arrays(
    clip_id: str,
    clip: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    """Load the NPZ array for a clip.

    Looks up the ``array_ref`` field first, then falls back to scanning
    the known arrays directories.  Returns a dict of numpy arrays or
    *None* if the file cannot be found.
    """
    # Try array_ref from clip metadata
    array_ref = clip.get("array_ref")
    if array_ref:
        source = clip.get("source", "")
        if source == "carla":
            npz_path = PROJECT_ROOT / "output" / "carla_clips" / array_ref
        else:
            npz_path = PROJECT_ROOT / "output" / "nuscenes_clips" / array_ref
        if npz_path.exists():
            return dict(np.load(npz_path))

    # Fallback: try both directories
    for d in _ARRAYS_DIRS.values():
        npz_path = d / f"{clip_id}.npz"
        if npz_path.exists():
            return dict(np.load(npz_path))

    logger.warning(f"No NPZ array found for clip {clip_id}")
    return None


def format_timeseries_embedding(
    arrays: dict[str, np.ndarray],
    num_points: int,
) -> str:
    """Format time-series values at *num_points* evenly-spaced instants.

    Returns a compact text block like::

        Vehicle dynamics (10 time-steps over 3.0s):
        t(s):   0.00, 0.33, 0.67, ...
        speed:  5.2, 5.4, 5.6, ...
        accel:  0.12, -0.05, ...
        ...
    """
    n_samples = len(arrays.get("timestamps", []))
    if n_samples == 0:
        return ""

    idx = _select_indices(n_samples, num_points)
    ts = arrays["timestamps"][idx]
    duration = arrays["timestamps"][-1] - arrays["timestamps"][0]

    lines: list[str] = []
    lines.append(f"Vehicle dynamics ({len(idx)} time-steps over {duration:.1f}s):")
    lines.append("t(s):    " + ", ".join(f"{t:.2f}" for t in ts))

    # Core channels
    channels = [
        ("speed(m/s)", "speed", ".1f"),
        ("accel(m/s²)", "accel", ".2f"),
        ("yaw_rate(rad/s)", "yaw_rate", ".3f"),
        ("jerk(m/s³)", "jerk", ".2f"),
    ]
    for label, key, fmt in channels:
        if key in arrays:
            vals = arrays[key][idx]
            lines.append(f"{label}: " + ", ".join(f"{v:{fmt}}" for v in vals))

    return "\n".join(lines)


def format_coordinates_embedding(
    arrays: dict[str, np.ndarray],
    num_points: int,
) -> str:
    """Format (x, y) trajectory coordinates at *num_points* instants.

    Coordinates are zero-centred (relative to the first position) so the
    model sees displacement rather than global map coordinates.

    Returns a compact text block like::

        Vehicle trajectory (10 waypoints over 3.0s, metres):
        t(s):  0.00, 0.33, ...
        x(m):  0.0, 1.2, ...
        y(m):  0.0, 0.5, ...
        heading(rad): 1.57, 1.58, ...
    """
    n_samples = len(arrays.get("timestamps", []))
    if n_samples == 0 or "position" not in arrays:
        return ""

    idx = _select_indices(n_samples, num_points)
    ts = arrays["timestamps"][idx]
    pos = arrays["position"][idx]  # (N, 2)
    duration = arrays["timestamps"][-1] - arrays["timestamps"][0]

    # Zero-centre relative to first point
    origin = arrays["position"][0]
    pos_rel = pos - origin

    lines: list[str] = []
    lines.append(f"Vehicle trajectory ({len(idx)} waypoints over {duration:.1f}s, metres):")
    lines.append("t(s): " + ", ".join(f"{t:.2f}" for t in ts))
    lines.append("x(m): " + ", ".join(f"{v:.1f}" for v in pos_rel[:, 0]))
    lines.append("y(m): " + ", ".join(f"{v:.1f}" for v in pos_rel[:, 1]))

    if "yaw" in arrays:
        yaw = arrays["yaw"][idx]
        lines.append("heading(rad): " + ", ".join(f"{v:.3f}" for v in yaw))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _format_summary_embedding(clip: dict[str, Any]) -> str:
    """Format the legacy 8-scalar summary embedding."""
    features = clip.get("features", {})
    ms = features.get("max_speed", 0.0)
    return (
        "Vehicle dynamics:"
        f" max_speed={ms:.1f}m/s ({ms * 3.6:.0f}km/h),"
        f" mean_speed={features.get('mean_speed', 0.0):.1f}m/s,"
        f" min_accel={features.get('min_accel', 0.0):.2f}m/s²,"
        f" max_yaw_rate={features.get('max_abs_yaw_rate', 0.0):.3f}rad/s,"
        f" max_jerk={features.get('max_abs_jerk', 0.0):.2f}m/s³,"
        f" mean_jerk={features.get('mean_abs_jerk', 0.0):.2f}m/s³,"
        f" max_lat_accel={features.get('max_lateral_accel', 0.0):.2f}m/s²,"
        f" heading_change={features.get('total_heading_change', 0.0):.3f}rad"
    )


def _build_trajectory_text(
    trajectory_mode: str,
    clip: dict[str, Any],
    arrays: dict[str, np.ndarray] | None,
    num_points: int,
) -> str | None:
    """Return the trajectory text block for the given mode, or None."""
    if trajectory_mode == "none":
        return None
    if trajectory_mode == "summary":
        return _format_summary_embedding(clip)
    if trajectory_mode == "timeseries":
        if arrays is None:
            return _format_summary_embedding(clip)  # graceful fallback
        return format_timeseries_embedding(arrays, num_points)
    if trajectory_mode == "coordinates":
        if arrays is None:
            return _format_summary_embedding(clip)
        return format_coordinates_embedding(arrays, num_points)
    if trajectory_mode == "full":
        # Both time-series and coordinates
        parts: list[str] = []
        if arrays is not None:
            ts = format_timeseries_embedding(arrays, num_points)
            if ts:
                parts.append(ts)
            co = format_coordinates_embedding(arrays, num_points)
            if co:
                parts.append(co)
        if not parts:
            return _format_summary_embedding(clip)
        return "\n".join(parts)
    # Unknown mode: fall back to summary
    return _format_summary_embedding(clip)


def build_prompt(
    qa_item: dict[str, Any],
    clip: dict[str, Any],
    include_trajectory: bool,
    num_frames_sent: int,
    *,
    trajectory_mode: str = "summary",
    trajectory_arrays: dict[str, np.ndarray] | None = None,
    num_trajectory_points: int = 10,
) -> str:
    """Assemble a compact text prompt for a single QA item."""
    parts: list[str] = []

    if num_frames_sent > 0:
        parts.append(
            f"The {'image shows' if num_frames_sent == 1 else f'{num_frames_sent} images show'} the forward camera view "
            f"{'from' if num_frames_sent == 1 else 'at evenly spaced moments across'} a 3-second driving clip."
        )

    if include_trajectory:
        traj_text = _build_trajectory_text(
            trajectory_mode, clip, trajectory_arrays, num_trajectory_points,
        )
        if traj_text:
            parts.append(traj_text)

    question_text = qa_item.get("question", qa_item.get("question_text", ""))
    choices = qa_item.get("choices")
    answer_type = qa_item.get("answer_type", "")

    if choices:
        opts = " / ".join(str(c) for c in choices)
        parts.append(f"{question_text} [{opts}]")
        parts.append("Answer with ONLY the chosen option.")
    elif answer_type == "numeric":
        units = qa_item.get("units") or ""
        parts.append(f"{question_text} [number{', ' + units if units else ''}]")
        parts.append("Answer with ONLY the number.")
    else:
        parts.append(question_text)
        parts.append("Answer concisely.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Grouped prompt building (all questions per clip in one request)
# ---------------------------------------------------------------------------

def group_qa_by_clip(qa_items: list[dict]) -> dict[str, list[dict]]:
    """Group QA items by clip_id, preserving order."""
    groups: dict[str, list[dict]] = {}
    for qa in qa_items:
        groups.setdefault(qa["clip_id"], []).append(qa)
    return groups


def build_grouped_prompt(
    qa_items: list[dict],
    clip: dict[str, Any],
    include_trajectory: bool,
    num_frames_sent: int,
    *,
    trajectory_mode: str = "summary",
    trajectory_arrays: dict[str, np.ndarray] | None = None,
    num_trajectory_points: int = 10,
) -> str:
    """Build a single prompt containing all questions for one clip.

    Produces a compact prompt (~200-400 text tokens) with numbered questions.
    The model is asked to reply with numbered answers in the same order.
    """
    parts: list[str] = []

    # Minimal context
    if num_frames_sent > 0:
        parts.append(
            f"The {'image shows' if num_frames_sent == 1 else f'{num_frames_sent} images show'} the forward camera view "
            f"{'from' if num_frames_sent == 1 else 'at evenly spaced moments across'} a 3-second driving clip."
        )

    if include_trajectory:
        traj_text = _build_trajectory_text(
            trajectory_mode, clip, trajectory_arrays, num_trajectory_points,
        )
        if traj_text:
            parts.append(traj_text)

    # Questions
    parts.append("\nAnswer each question. Reply ONLY with the numbered answers.")
    for i, qa in enumerate(qa_items, 1):
        q_text = qa.get("question", qa.get("question_text", ""))
        choices = qa.get("choices")
        answer_type = qa.get("answer_type", "")
        if choices:
            opts = " / ".join(str(c) for c in choices)
            parts.append(f"Q{i}. {q_text} [{opts}]")
        elif answer_type == "numeric":
            units = qa.get("units") or ""
            parts.append(f"Q{i}. {q_text} [number{', ' + units if units else ''}]")
        else:
            parts.append(f"Q{i}. {q_text}")

    parts.append(f"\nFormat:\nA1. answer\nA2. answer\n...\nA{len(qa_items)}. answer")
    return "\n".join(parts)


def parse_grouped_response(
    response_text: str,
    n_questions: int,
) -> list[str]:
    """Parse a numbered response like 'A1. left\\nA2. none\\n...'

    Returns a list of *n_questions* answers (empty string for unparsed).
    """
    import re

    answers = [""] * n_questions
    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: "A1. left", "1. left", "A1: left", "1: left", "1) left"
        m = re.match(r"[Aa]?(\d+)[.):]\s*(.+)", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_questions:
                answers[idx] = m.group(2).strip()
    return answers


# ---------------------------------------------------------------------------
# Prediction record
# ---------------------------------------------------------------------------

def make_prediction_record(qa_item: dict, model_answer: str) -> dict:
    """Build the output dict compatible with scripts/evaluate.py."""
    return {
        "qa_id": qa_item.get("qa_id", ""),
        "clip_id": qa_item["clip_id"],
        "question_id": qa_item["question_id"],
        "category": qa_item.get("category", "unknown"),
        "oracle_label": qa_item.get("answer", qa_item.get("oracle_label", "")),
        "model_answer": model_answer,
    }


# ---------------------------------------------------------------------------
# Common CLI parser
# ---------------------------------------------------------------------------

def build_common_parser(
    description: str,
    default_model: str,
    api_key_env_var: str,
) -> argparse.ArgumentParser:
    """Build an argparse parser with all shared evaluation arguments.

    Each model-specific evaluator calls this, then optionally adds its own
    arguments before parsing.  Automatically loads ``.env`` from the project
    root so API keys are available via ``os.getenv()``.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa_jsonl", type=str, default=None,
        help="Path to QA file (e.g. val_qa.jsonl). Required if --selected_clips is not used.",
    )
    parser.add_argument(
        "--selected_clips", type=str, default=None,
        help="Path to selected_clips.json (overrides --qa_jsonl)",
    )
    parser.add_argument(
        "--nuscenes_qa", type=str,
        default="output/nuscenes_clips/qa.jsonl",
        help="Path to source nuScenes QA (used with --selected_clips)",
    )
    parser.add_argument(
        "--carla_qa", type=str,
        default="output/carla_clips/qa.jsonl",
        help="Path to source CARLA QA (used with --selected_clips)",
    )
    parser.add_argument(
        "--nuscenes_index", type=str,
        default="output/nuscenes_clips/clips_index.jsonl",
        help="Path to nuScenes clips index (used with --selected_clips)",
    )
    parser.add_argument(
        "--carla_index", type=str,
        default="output/carla_clips/clips_index.jsonl",
        help="Path to CARLA clips index (used with --selected_clips)",
    )
    parser.add_argument(
        "--clips_index", type=str, default=None,
        help="Path to combined clips_index.jsonl (legacy/fallback)",
    )
    parser.add_argument(
        "--nuscenes_root", type=str,
        default=None,
        help="Path to nuScenes dataset root (required for nuScenes clips)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output predictions JSONL",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"),
        help="Path to questions_template.yaml (for --run_eval)",
    )
    parser.add_argument(
        "--model", type=str, default=default_model,
        help="Model name",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit QA items")
    parser.add_argument(
        "--num_frames", type=int, default=10,
        help="Number of frames per clip (evenly spaced). More frames = better temporal context.",
    )
    parser.add_argument(
        "--frame_detail", type=str, default="low", choices=["low", "high"],
        help="Image detail level (OpenAI-specific): 'low' = 85 tokens/image, "
             "'high' = up to ~765+ tokens/image. Ignored by Gemini/Claude.",
    )
    parser.add_argument(
        "--carla_video_source", type=str, default="transferred",
        choices=["simulation", "transferred"],
        help="Which CARLA video data to use: 'simulation' = original CARLA renders, "
             "'transferred' = Cosmos-transferred realistic frames (default: transferred)",
    )
    parser.add_argument(
        "--carla_video_dir", type=str, default=None,
        help="Explicit directory for CARLA MP4 video chunks. "
             "Overrides --carla_video_source if set.",
    )
    parser.add_argument(
        "--trajectory_mode", type=str, default="summary",
        choices=["none", "summary", "timeseries", "coordinates", "full"],
        help="Trajectory embedding mode: 'none' = no dynamics (vision-only), "
             "'summary' = 8 scalar features (default), "
             "'timeseries' = per-frame speed/accel/yaw_rate/jerk, "
             "'coordinates' = (x,y) waypoints + heading, "
             "'full' = both timeseries and coordinates.",
    )
    parser.add_argument(
        "--no_trajectory", action="store_true", default=False,
        help="(Deprecated) Shorthand for --trajectory_mode none.",
    )
    parser.add_argument("--no_images", action="store_true", help="Text-only ablation")
    parser.add_argument(
        "--group_by_clip", action="store_true",
        help="Group all questions per clip into a single API call. "
             "Reduces cost ~14x by sending images only once per clip.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5, help="Seconds between calls")
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation after inference")
    parser.add_argument("--metrics_output", type=str, default=None, help="Write metrics JSON here (implies --run_eval)")
    parser.add_argument("--resume", action="store_true", help="Skip items already in output file")
    parser.add_argument(
        "--shuffle_frames", action="store_true",
        help="Randomly shuffle frame order before sending to the model. "
             "Ablation to test whether temporal ordering of frames matters.",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, default=42,
        help="Random seed for --shuffle_frames (default: 42 for reproducibility)",
    )
    parser.add_argument(
        "--overlay_flow", action="store_true",
        help="Overlay dense optical flow (Farneback) on each frame. "
             "Ablation to test whether explicit visual motion cues help.",
    )
    parser.add_argument(
        "--flow_alpha", type=float, default=0.5,
        help="Blending alpha for --overlay_flow (0=original, 1=flow only, default: 0.5)",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help=f"API key (default: ${api_key_env_var})",
    )
    # Store the env var name so run_evaluation() can look it up
    parser.set_defaults(_api_key_env_var=api_key_env_var)
    return parser


def resolve_carla_video_dir(args: argparse.Namespace) -> None:
    """Resolve ``args.carla_video_dir`` from ``--carla_video_source`` if not set explicitly."""
    if args.carla_video_dir is None:
        source = getattr(args, "carla_video_source", "transferred")
        args.carla_video_dir = str(CARLA_VIDEO_DIRS.get(source, CARLA_VIDEO_DIRS["transferred"]))
    logger.info(f"CARLA video dir: {args.carla_video_dir}")


# ---------------------------------------------------------------------------
# Evaluation result printing
# ---------------------------------------------------------------------------

def print_eval_results(
    result: dict[str, Any],
    model_name: str,
    trajectory_on: bool,
    images_on: bool,
) -> None:
    """Print a formatted summary of evaluation results."""
    g = result["global"]
    c = result["consistency"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:              {model_name}")
    print(f"Trajectory:         {'ON' if trajectory_on else 'OFF'}")
    print(f"Images:             {'ON' if images_on else 'OFF'}")
    print(f"Total records:      {result['n_total']}")
    print(f"Parsed:             {result['n_parsed']}")
    print(f"Parsable coverage:  {result['parsable_coverage']:.1%}")
    print(f"Semantic accuracy:  {g['accuracy']:.4f}")
    print(f"Macro F1:           {g['macro_f1']:.4f}")
    t = result["temporal"]
    if t["n"] > 0:
        print(f"Temporal accuracy:  {t['accuracy']:.4f}  (n={t['n']})")
        print(f"Temporal F1:        {t['macro_f1']:.4f}")
    else:
        print("Temporal accuracy:  N/A (no temporal questions in sample)")
    if c["n_evaluable"] > 0:
        print(f"Consistency (EMCR): {c['rate']:.4f}  (n={c['n_evaluable']} clips)")
    else:
        print("Consistency (EMCR): N/A (no evaluable clip pairs in sample)")
    print()
    print("Per-category:")
    for cat, m in sorted(result["per_category"].items()):
        print(f"  {cat:30s}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  n={m['n']}")
    if "per_source" in result and result["per_source"]:
        print()
        print("Per-source:")
        for src, m in sorted(result["per_source"].items()):
            emcr = m.get("consistency", {}).get("rate", 0.0)
            print(f"  {src:30s}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  emcr={emcr:.4f}  n={m['n']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace, call_api: ApiCallFn) -> int:
    """Run the full inference + optional evaluation loop.

    Parameters
    ----------
    args:
        Parsed CLI arguments (from :func:`build_common_parser`).
    call_api:
        A callable ``(prompt, image_data) -> model_answer`` that handles
        the model-specific API call including retries.  Should return an
        empty string on failure.
    """
    if args.metrics_output:
        args.run_eval = True

    # --- resolve CARLA video directory ---------------------------------
    resolve_carla_video_dir(args)

    # --- validate paths ------------------------------------------------
    nuscenes_root = Path(args.nuscenes_root)
    if not args.no_images and not nuscenes_root.exists():
        logger.error(f"nuScenes root not found: {nuscenes_root}")
        return 1

    # --- load data -----------------------------------------------------
    clips = {}
    
    if args.selected_clips:
        # Load indices from separate source files
        ns_idx_path = Path(args.nuscenes_index)
        if ns_idx_path.exists():
            clips.update(load_clips_index(ns_idx_path))
        else:
            logger.warning(f"NuScenes clips index not found: {ns_idx_path}")

        ca_idx_path = Path(args.carla_index)
        if ca_idx_path.exists():
            clips.update(load_clips_index(ca_idx_path))
        else:
            logger.warning(f"CARLA clips index not found: {ca_idx_path}")

        if not clips:
             logger.error("No clips loaded from source indices.")
             return 1
             
        # Load selected clips list
        sel_path = Path(args.selected_clips)
        if not sel_path.exists():
            logger.error(f"Selected clips file not found: {sel_path}")
            return 1
        
        with open(sel_path) as f:
            selected_data = json.load(f)
            if selected_data and isinstance(selected_data[0], dict):
                selected_ids = set(c["id"] for c in selected_data)
            else:
                selected_ids = set(selected_data)
        
        logger.info(f"Loaded {len(selected_ids)} selected clips from {sel_path}")

        # Load all source QAs and filter
        all_qa_items = []
        
        ns_qa_path = Path(args.nuscenes_qa)
        if ns_qa_path.exists():
            all_qa_items.extend(load_qa_items(ns_qa_path))
        else:
            logger.warning(f"NuScenes QA path not found: {ns_qa_path}")

        ca_qa_path = Path(args.carla_qa)
        if ca_qa_path.exists():
            all_qa_items.extend(load_qa_items(ca_qa_path))
        else:
            logger.warning(f"CARLA QA path not found: {ca_qa_path}")

        # Filter
        qa_items = [qa for qa in all_qa_items if qa["clip_id"] in selected_ids]
        logger.info(f"Filtered to {len(qa_items)} QA items for the selected clips")

        if args.max_samples is not None:
             qa_items = qa_items[:args.max_samples]
             logger.info(f"Subsampled to {len(qa_items)} items via max_samples")

    else:
        # Legacy mode
        if not args.clips_index:
            logger.error("Legacy mode: --clips_index is required when not using --selected_clips")
            return 1
            
        clips_index_path = Path(args.clips_index)
        if not clips_index_path.exists():
            logger.error(f"Clips index not found: {clips_index_path}")
            return 1
        clips = load_clips_index(clips_index_path)

        if not args.qa_jsonl:
            logger.error("Legacy mode: --qa_jsonl is required when not using --selected_clips")
            return 1
            
        qa_path = Path(args.qa_jsonl)
        if not qa_path.exists():
            logger.error(f"QA file not found: {qa_path}")
            return 1
        qa_items = load_qa_items(qa_path, max_samples=args.max_samples)

    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_qa_ids(args.output)

    # --- inference loop ------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    skipped_no_clip = 0
    skipped_resume = 0
    processed = 0

    trajectory_mode = getattr(args, "trajectory_mode", "summary") or "summary"
    # --no_trajectory is a deprecated alias for --trajectory_mode none.
    # Only apply it when trajectory_mode was not explicitly set to something else.
    if getattr(args, "no_trajectory", False) and trajectory_mode == "summary":
        trajectory_mode = "none"
    include_trajectory = trajectory_mode != "none"
    include_images = not args.no_images
    needs_arrays = trajectory_mode in ("timeseries", "coordinates", "full")
    carla_video_dir = Path(args.carla_video_dir)
    grouped = getattr(args, "group_by_clip", False)

    logger.info(
        f"Starting inference: model={args.model}, "
        f"trajectory={'ON' if include_trajectory else 'OFF'}"
        f"{f' ({trajectory_mode})' if include_trajectory else ''}, "
        f"images={'ON' if include_images else 'OFF'}, "
        f"num_frames={args.num_frames}, "
        f"grouped={'ON' if grouped else 'OFF'}"
    )

    shuffle_frames = getattr(args, "shuffle_frames", False)
    shuffle_rng = random.Random(getattr(args, "shuffle_seed", 42))
    do_overlay_flow = getattr(args, "overlay_flow", False)
    flow_alpha = getattr(args, "flow_alpha", 0.5)

    def _encode_clip_images(clip_id: str, clip: dict) -> list[tuple[str, str]]:
        """Encode frames for a clip (shared between both modes)."""
        image_data: list[tuple[str, str]] = []
        if not include_images:
            return image_data
        frame_paths = clip.get("frame_paths", [])
        if frame_paths:
            selected = select_frames(frame_paths, args.num_frames)
            for rel_path in selected:
                abs_path = nuscenes_root / rel_path
                if not abs_path.exists():
                    logger.warning(f"Image not found: {abs_path}")
                    continue
                image_data.append(encode_image(str(abs_path)))
        else:
            video_path = carla_video_dir / f"{clip_id}.mp4"
            if video_path.exists():
                image_data = extract_video_frames(
                    str(video_path), args.num_frames,
                )
            else:
                logger.warning(f"No frame_paths and no video for {clip_id}")
        if do_overlay_flow and len(image_data) >= 2:
            image_data = overlay_flow_on_encoded_frames(image_data, flow_alpha)
        if shuffle_frames and len(image_data) > 1:
            shuffle_rng.shuffle(image_data)
        return image_data

    with open(output_path, mode) as out_f:

        if grouped:
            # ---- Grouped mode: one API call per clip ----
            clip_groups = group_qa_by_clip(qa_items)
            for clip_id in tqdm(clip_groups, desc="Evaluating clips"):
                group = clip_groups[clip_id]

                # Resume: skip if ALL qa_ids for this clip are done
                group_ids = [qa.get("qa_id", "") for qa in group]
                if completed_ids and all(
                    qid and qid in completed_ids for qid in group_ids
                ):
                    skipped_resume += len(group)
                    continue

                clip = clips.get(clip_id)
                if clip is None:
                    skipped_no_clip += len(group)
                    continue

                image_data = _encode_clip_images(clip_id, clip)

                traj_arrays = None
                if needs_arrays:
                    traj_arrays = load_trajectory_arrays(clip_id, clip)

                prompt = build_grouped_prompt(
                    group, clip,
                    include_trajectory=include_trajectory,
                    num_frames_sent=len(image_data),
                    trajectory_mode=trajectory_mode,
                    trajectory_arrays=traj_arrays,
                    num_trajectory_points=args.num_frames,
                )

                raw_response = call_api(prompt, image_data)
                time.sleep(args.rate_limit_delay)

                answers = parse_grouped_response(raw_response, len(group))
                for qa_item, answer in zip(group, answers):
                    record = make_prediction_record(qa_item, answer)
                    out_f.write(json.dumps(record) + "\n")
                    processed += 1
                out_f.flush()
        else:
            # ---- Original mode: one API call per question ----
            for qa_item in tqdm(qa_items, desc="Evaluating"):
                qa_id = qa_item.get("qa_id", "")

                if qa_id and qa_id in completed_ids:
                    skipped_resume += 1
                    continue

                clip_id = qa_item["clip_id"]
                clip = clips.get(clip_id)
                if clip is None:
                    skipped_no_clip += 1
                    logger.warning(f"Clip {clip_id} not found, skipping {qa_id}")
                    continue

                image_data = _encode_clip_images(clip_id, clip)

                traj_arrays = None
                if needs_arrays:
                    traj_arrays = load_trajectory_arrays(clip_id, clip)

                prompt = build_prompt(
                    qa_item, clip,
                    include_trajectory=include_trajectory,
                    num_frames_sent=len(image_data),
                    trajectory_mode=trajectory_mode,
                    trajectory_arrays=traj_arrays,
                    num_trajectory_points=args.num_frames,
                )

                model_answer = call_api(prompt, image_data)
                time.sleep(args.rate_limit_delay)

                record = make_prediction_record(qa_item, model_answer)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                processed += 1

    logger.info(
        f"Done: {processed} predictions written to {output_path}"
        f" (skipped: {skipped_no_clip} no-clip, {skipped_resume} resumed)"
    )

    # --- optional evaluation -------------------------------------------
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