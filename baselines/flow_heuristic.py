"""Optical-flow heuristic baseline for EgoDyn-Bench.

Uses Farneback dense optical flow (OpenCV only) to derive three per-frame-pair
proxy signals and maps them to benchmark answers via temporal aggregation +
deadzones.

**All signals are unitless pixel-domain proxies, not metric quantities.**
They capture qualitative motion patterns (rotation direction, expansion rate,
overall displacement) but do not recover physical speed or acceleration.

Signals
-------
- **turn_score**: tangential flow component (proxy for rotational motion;
  positive = counter-clockwise / left turn in ego frame)
- **exp_score**: radial flow proxy (expansion / contraction about image
  centre; positive = radial expansion, negative = contraction)
- **motion_mag**: median flow magnitude in a central ROI (proxy for overall
  scene displacement between frames)

Supported questions (6 of 14)
-----------------------------
Only questions whose answers have a defensible physical mapping from
monocular optical flow are supported.  Unsupported questions return an
empty string and will be marked unparsable by the evaluator.

Usage::

    python -m baselines.flow_heuristic \
        --selected_clips selected_clips.json \
        --output preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum width for internal processing.  Frames wider than this are
# downsampled for stability and speed.  Does not affect output labels.
_MAX_PROCESS_WIDTH = 320

# ---------------------------------------------------------------------------
# Question IDs that this baseline can answer
# ---------------------------------------------------------------------------
# Only questions with a defensible physical mapping from monocular flow.
SUPPORTED_QUESTIONS: dict[str, list[str]] = {
    "yaw_rate_turn_direction": ["left", "right", "straight"],
    "speed_trend": ["accelerating", "decelerating", "steady"],
    "stop_and_go": ["yes", "no"],
    "brake_then_turn": ["yes", "no"],
    "significant_heading_change": ["yes", "no"],
    "high_lateral_accel": ["yes", "no"],
}


# ---------------------------------------------------------------------------
# Core flow computation
# ---------------------------------------------------------------------------

def _preprocess_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Convert to grayscale, optionally downsample, and apply light blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w > max_width:
        scale = max_width / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    return gray


def compute_flow_signals(
    frames: list[np.ndarray],
    max_width: int = _MAX_PROCESS_WIDTH,
) -> dict[str, np.ndarray]:
    """Compute per-frame-pair optical flow proxy signals.

    All returned values are **unitless pixel-domain proxies**.  They capture
    qualitative motion patterns but do not recover metric speed or
    acceleration.

    Parameters
    ----------
    frames:
        List of BGR images (uint8), all the same shape.
    max_width:
        Frames wider than this are downsampled before flow computation
        for stability and speed.

    Returns
    -------
    Dictionary with keys ``turn_score``, ``exp_score``, ``motion_mag``,
    each a 1-D float array of length ``len(frames) - 1``.
    """
    if len(frames) < 2:
        empty = np.zeros(0, dtype=np.float64)
        return {"turn_score": empty, "exp_score": empty, "motion_mag": empty}

    # Preprocess first frame to get working resolution
    prev_gray = _preprocess_frame(frames[0], max_width)
    h, w = prev_gray.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    # Vertical crop bounds: ignore top 20% (sky) and bottom 15% (hood).
    # All three signals are computed only within this vertical band.
    y_top = int(h * 0.20)
    y_bot = int(h * 0.85)

    # Pre-compute pixel grid for radial/tangential decomposition
    # (only within the vertical crop)
    ys, xs = np.mgrid[y_top:y_bot, 0:w].astype(np.float32)
    dx_grid = xs - cx
    dy_grid = ys - cy
    radial_norm = np.sqrt(dx_grid ** 2 + dy_grid ** 2)
    radial_norm[radial_norm < 1e-6] = 1.0

    turn_scores: list[float] = []
    exp_scores: list[float] = []
    motion_mags: list[float] = []

    for frame in frames[1:]:
        curr_gray = _preprocess_frame(frame, max_width)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Crop to valid vertical band for all signals
        flow_crop = flow[y_top:y_bot, :, :]
        flow_x = flow_crop[:, :, 0]
        flow_y = flow_crop[:, :, 1]

        # --- turn_score: tangential (rotational) flow component ---
        # Tangential = cross(flow, radial_unit) = (fx*dy - fy*dx) / |r|
        # Positive = CCW rotation = left turn in ego frame.
        # Proxy signal — not a metric yaw rate.
        tangential_flow = (flow_x * dy_grid - flow_y * dx_grid) / radial_norm
        turn_scores.append(float(np.median(tangential_flow)))

        # --- exp_score: radial flow proxy (expansion / contraction) ---
        # Radial = dot(flow, radial_unit) = (fx*dx + fy*dy) / |r|
        # Positive = expansion (scene moving outward from centre).
        # Proxy signal — not a metric speed or acceleration.
        radial_flow = (flow_x * dx_grid + flow_y * dy_grid) / radial_norm
        exp_scores.append(float(np.median(radial_flow)))

        # --- motion_mag: median flow magnitude in cropped region ---
        # Proxy for overall scene displacement — not metric speed.
        mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
        motion_mags.append(float(np.median(mag)))

        prev_gray = curr_gray

    return {
        "turn_score": np.array(turn_scores),
        "exp_score": np.array(exp_scores),
        "motion_mag": np.array(motion_mags),
    }


# ---------------------------------------------------------------------------
# FlowHeuristicBaseline
# ---------------------------------------------------------------------------

class FlowHeuristicBaseline:
    """Map optical-flow proxy signals to EgoDyn-Bench answers.

    Only answers question types that have a defensible physical mapping
    from monocular optical flow (see :data:`SUPPORTED_QUESTIONS`).
    Unsupported question types are omitted from the output.

    Parameters
    ----------
    turn_deadzone:
        ``|turn_score|`` below this → "straight".
    exp_deadzone:
        ``|exp_score|`` below this → "steady" (for speed_trend proxy).
    motion_stopped_thr:
        ``motion_mag`` below this → treated as stopped (for stop_and_go).
    motion_moving_thr:
        ``motion_mag`` above this → treated as moving (for stop_and_go).
    heading_change_thr:
        Cumulative ``|turn_score|`` above this → significant heading change.
    lateral_thr:
        Peak ``|turn_score|`` above this → high lateral acceleration proxy.
    """

    def __init__(
        self,
        turn_deadzone: float = 0.05,
        exp_deadzone: float = 0.2,
        motion_stopped_thr: float = 0.3,
        motion_moving_thr: float = 1.5,
        heading_change_thr: float = 3.0,
        lateral_thr: float = 1.5,
    ):
        self.turn_deadzone = turn_deadzone
        self.exp_deadzone = exp_deadzone
        self.motion_stopped_thr = motion_stopped_thr
        self.motion_moving_thr = motion_moving_thr
        self.heading_change_thr = heading_change_thr
        self.lateral_thr = lateral_thr

    def predict(self, frames: list[np.ndarray]) -> dict[str, str]:
        """Predict answers for supported question types only.

        Parameters
        ----------
        frames:
            List of BGR uint8 images for one clip.

        Returns
        -------
        ``{question_id: answer}`` for every question in
        :data:`SUPPORTED_QUESTIONS`.  Unsupported question types are
        not included — the evaluator will treat their absence as
        unparsable.
        """
        signals = compute_flow_signals(frames)
        turn = signals["turn_score"]
        exp = signals["exp_score"]
        mag = signals["motion_mag"]

        answers: dict[str, str] = {}

        # Aggregates
        mean_turn = float(np.mean(turn)) if len(turn) > 0 else 0.0
        mean_exp = float(np.mean(exp)) if len(exp) > 0 else 0.0
        peak_abs_turn = float(np.max(np.abs(turn))) if len(turn) > 0 else 0.0
        cumul_turn = float(np.sum(np.abs(turn))) if len(turn) > 0 else 0.0

        # --- yaw_rate_turn_direction ---
        # Tangential flow sign directly indicates rotational direction.
        if mean_turn > self.turn_deadzone:
            answers["yaw_rate_turn_direction"] = "left"
        elif mean_turn < -self.turn_deadzone:
            answers["yaw_rate_turn_direction"] = "right"
        else:
            answers["yaw_rate_turn_direction"] = "straight"

        # --- speed_trend ---
        # Radial expansion/contraction proxy: expansion correlates with
        # increasing ego-speed (objects recede faster), contraction with
        # decreasing ego-speed.  This is a proxy, not metric acceleration.
        if mean_exp > self.exp_deadzone:
            answers["speed_trend"] = "accelerating"
        elif mean_exp < -self.exp_deadzone:
            answers["speed_trend"] = "decelerating"
        else:
            answers["speed_trend"] = "steady"

        # --- significant_heading_change ---
        # Cumulative absolute rotational flow as a proxy for total heading
        # change over the clip.
        answers["significant_heading_change"] = (
            "yes" if cumul_turn > self.heading_change_thr else "no"
        )

        # --- high_lateral_accel ---
        # Peak tangential flow magnitude as a proxy for lateral dynamics.
        answers["high_lateral_accel"] = (
            "yes" if peak_abs_turn > self.lateral_thr else "no"
        )

        # --- stop_and_go ---
        # Detect transitions between low and high displacement magnitude.
        if len(mag) >= 3:
            stopped_mask = mag < self.motion_stopped_thr
            moving_mask = mag > self.motion_moving_thr
            cycles = 0
            was_stopped = bool(stopped_mask[0])
            for i in range(1, len(mag)):
                if not was_stopped and stopped_mask[i]:
                    was_stopped = True
                elif was_stopped and moving_mask[i]:
                    cycles += 1
                    was_stopped = False
            answers["stop_and_go"] = "yes" if cycles >= 1 else "no"
        else:
            answers["stop_and_go"] = "no"

        # --- brake_then_turn ---
        # Temporal sequence: radial contraction (braking proxy) followed
        # by rotational flow (turning proxy).
        if len(exp) >= 3 and len(turn) >= 3:
            brake_mask = exp < -self.exp_deadzone
            turn_mask = np.abs(turn) > self.turn_deadzone
            found = False
            brake_indices = np.where(brake_mask)[0]
            turn_indices = np.where(turn_mask)[0]
            if len(brake_indices) > 0 and len(turn_indices) > 0:
                for bi in brake_indices:
                    later_turns = turn_indices[turn_indices > bi]
                    if len(later_turns) > 0:
                        found = True
                        break
            answers["brake_then_turn"] = "yes" if found else "no"
        else:
            answers["brake_then_turn"] = "no"

        return answers


# ---------------------------------------------------------------------------
# Frame loading helpers
# ---------------------------------------------------------------------------

def load_frames_from_images(
    frame_paths: list[str],
    root: Path,
    num_frames: int = 10,
) -> list[np.ndarray]:
    """Load evenly-spaced frames from a list of image paths."""
    n = len(frame_paths)
    if n == 0:
        return []
    if num_frames >= n:
        indices = list(range(n))
    else:
        indices = [round(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        path = root / frame_paths[idx]
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(img)
    return frames


def load_frames_from_video(
    video_path: str | Path,
    num_frames: int = 10,
) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    if num_frames >= total:
        indices = list(range(total))
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optical-flow heuristic baseline on EgoDyn-Bench clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected_clips", type=str, default="selected_clips.json",
        help="Path to selected_clips.json",
    )
    parser.add_argument(
        "--nuscenes_qa", type=str, default="output/nuscenes_clips/qa.jsonl",
        help="Path to nuScenes QA JSONL",
    )
    parser.add_argument(
        "--carla_qa", type=str, default="output/carla_clips/qa.jsonl",
        help="Path to CARLA QA JSONL",
    )
    parser.add_argument(
        "--nuscenes_index", type=str,
        default="output/nuscenes_clips/clips_index.jsonl",
        help="Path to nuScenes clips index",
    )
    parser.add_argument(
        "--carla_index", type=str,
        default="output/carla_clips/clips_index.jsonl",
        help="Path to CARLA clips index",
    )
    parser.add_argument(
        "--nuscenes_root", type=str,
        default=None,
        help="Path to nuScenes dataset root (required for image/video extraction)",
    )
    parser.add_argument(
        "--carla_video_source", type=str, default="transferred",
        choices=["simulation", "transferred"],
        help="Which CARLA video data to use (default: transferred)",
    )
    parser.add_argument(
        "--carla_video_dir", type=str, default=None,
        help="Explicit directory for CARLA MP4 chunks (overrides --carla_video_source)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output predictions JSONL",
    )
    parser.add_argument(
        "--num_frames", type=int, default=10,
        help="Number of frames to extract per clip",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit QA items",
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="Run evaluation after inference",
    )
    parser.add_argument(
        "--config", type=str,
        default="dataset/configs/questions_template.yaml",
        help="Path to questions_template.yaml (for --run_eval)",
    )
    parser.add_argument(
        "--metrics_output", type=str, default=None,
        help="Write metrics JSON here (implies --run_eval)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _build_parser().parse_args(argv)
    if args.metrics_output:
        args.run_eval = True

    # --- Load data --------------------------------------------------------
    from evaluation.evaluator_common import (
        load_clips_index,
        load_qa_items,
        make_prediction_record,
        resolve_carla_video_dir,
    )

    resolve_carla_video_dir(args)

    clips: dict[str, dict] = {}
    for idx_path in (args.nuscenes_index, args.carla_index):
        p = Path(idx_path)
        if p.exists():
            clips.update(load_clips_index(p))

    sel_path = Path(args.selected_clips)
    if not sel_path.exists():
        logger.error("Selected clips file not found: %s", sel_path)
        return 1
    with open(sel_path) as f:
        selected_data = json.load(f)
        if selected_data and isinstance(selected_data[0], dict):
            selected_ids = set(c["id"] for c in selected_data)
        else:
            selected_ids = set(selected_data)

    all_qa: list[dict] = []
    for qa_path in (args.nuscenes_qa, args.carla_qa):
        p = Path(qa_path)
        if p.exists():
            all_qa.extend(load_qa_items(p))

    qa_items = [qa for qa in all_qa if qa["clip_id"] in selected_ids]
    if args.max_samples is not None:
        qa_items = qa_items[: args.max_samples]

    logger.info(
        "Loaded %d QA items for %d selected clips",
        len(qa_items), len(selected_ids),
    )

    # --- Run baseline per clip -------------------------------------------
    from evaluation.evaluator_common import group_qa_by_clip

    baseline = FlowHeuristicBaseline()
    nuscenes_root = Path(args.nuscenes_root)
    carla_video_dir = Path(args.carla_video_dir)
    clip_groups = group_qa_by_clip(qa_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    processed = 0
    skipped = 0

    with open(output_path, "w") as out_f:
        for clip_id in tqdm(clip_groups, desc="Flow baseline"):
            clip = clips.get(clip_id)
            if clip is None:
                skipped += len(clip_groups[clip_id])
                continue

            # Load frames
            frame_paths = clip.get("frame_paths", [])
            if frame_paths:
                frames = load_frames_from_images(
                    frame_paths, nuscenes_root, args.num_frames,
                )
            else:
                video_path = carla_video_dir / f"{clip_id}.mp4"
                frames = load_frames_from_video(video_path, args.num_frames)

            if len(frames) < 2:
                skipped += len(clip_groups[clip_id])
                continue

            predictions = baseline.predict(frames)

            for qa_item in clip_groups[clip_id]:
                qid = qa_item["question_id"]
                if qid not in SUPPORTED_QUESTIONS:
                    continue
                answer = predictions.get(qid, "")
                record = make_prediction_record(qa_item, answer)
                out_f.write(json.dumps(record) + "\n")
                processed += 1

    logger.info(
        "Done: %d predictions written to %s (skipped %d)",
        processed, output_path, skipped,
    )

    # --- Optional evaluation ---------------------------------------------
    if args.run_eval:
        from evaluation.parsers import load_question_config
        from evaluation.metrics import evaluate
        from evaluation.evaluator_common import print_eval_results

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
        print_eval_results(result, "flow_heuristic", False, True)

        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            logger.info("Metrics written to %s", metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
