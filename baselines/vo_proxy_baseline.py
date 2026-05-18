"""Visual-odometry proxy baseline for EgoDyn-Bench.

Uses sparse KLT feature tracking + essential matrix decomposition to estimate
per-frame-pair ego-rotation (yaw) and translational magnitude.  This is a
*light VO proxy* — it does NOT recover absolute scale or full 6-DoF pose.

Dependencies: numpy, opencv-python only.

Supported question_ids
----------------------
- ``yaw_rate_turn_direction``      — left / right / straight via mean + peak
                                      yaw from essential matrix decomposition
- ``speed_trend``                  — accelerating / decelerating / steady via
                                      linear slope of pixel-domain displacement
                                      magnitude (proxy — no absolute speed)
- ``stop_and_go``                  — yes / no via hysteresis-based stop→go cycle
                                      counting on median track displacement
- ``significant_heading_change``   — yes / no via cumulative absolute yaw
- ``high_lateral_accel``           — yes / no via peak absolute yaw rate
- ``brake_then_turn``              — yes / no via temporal deceleration→yaw
                                      sequence detection

Limitations
-----------
- No absolute speed — all displacement-based outputs are pixel-domain motion
  proxies and depend on image resolution, FoV, and scene depth.
- Essential matrix decomposition can be noisy for small baselines (near-stopped
  vehicles) or pure-rotation scenes.  A horizontal-flow fallback is used when
  ``findEssentialMat`` returns too few inliers.
- Yaw sign convention: positive yaw = left turn (CCW when viewed from above).
  Set ``yaw_sign_flip=True`` in config if your coordinate frame differs.

Usage::

    python -m baselines.vo_proxy_baseline \
        --selected_clips selected_clips.json \
        --output preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Ensure project root is on sys.path for `evaluation.*` imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question IDs that this baseline can answer
# ---------------------------------------------------------------------------
SUPPORTED_QUESTIONS: dict[str, list[str]] = {
    "yaw_rate_turn_direction": ["left", "right", "straight"],
    "speed_trend": ["accelerating", "decelerating", "steady"],
    "stop_and_go": ["yes", "no"],
    "brake_then_turn": ["yes", "no"],
    "significant_heading_change": ["yes", "no"],
    "high_lateral_accel": ["yes", "no"],
}


# ---------------------------------------------------------------------------
# Core VO proxy computation
# ---------------------------------------------------------------------------

def _default_intrinsics(w: int, h: int) -> np.ndarray:
    """Assume pinhole camera: fx = fy = 0.9 * W, principal point at centre."""
    f = 0.9 * w
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def _roi_mask(w: int, h: int, fraction: float = 0.6) -> np.ndarray:
    """Binary mask for the central *fraction* of the image (avoids sky/hood)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    rw = int(w * fraction)
    rh = int(h * fraction)
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    mask[y0:y0 + rh, x0:x0 + rw] = 255
    return mask


def _yaw_from_rotation(R: np.ndarray) -> float:
    """Extract yaw angle (rotation about Y-axis) from a 3x3 rotation matrix.

    Convention: camera looks along +Z, Y points down, X points right.
    Yaw = atan2(R[0,2], R[2,2]).  Positive yaw = camera turned left.
    """
    return math.atan2(R[0, 2], R[2, 2])


def compute_vo_signals(
    frames: list[np.ndarray],
    intrinsics: Optional[np.ndarray] = None,
    max_corners: int = 800,
    roi_fraction: float = 0.6,
    max_displacement_px: float = 50.0,
    ransac_threshold: float = 1.0,
    ransac_confidence: float = 0.999,
    min_inliers: int = 15,
    yaw_sign_flip: bool = False,
) -> dict[str, np.ndarray]:
    """Compute per-frame-pair VO signals via KLT + essential matrix.

    Parameters
    ----------
    frames:
        List of BGR uint8 images, all the same shape.
    intrinsics:
        3x3 camera matrix.  If None, a default pinhole is assumed.
    max_corners:
        Maximum corners for goodFeaturesToTrack.
    roi_fraction:
        Central fraction of image used for feature detection.
    max_displacement_px:
        Tracks with displacement larger than this are discarded.
    ransac_threshold:
        Pixel threshold for RANSAC in findEssentialMat.
    ransac_confidence:
        Confidence for RANSAC.
    min_inliers:
        Minimum inliers required to trust the essential matrix.
        Falls back to horizontal-flow heuristic otherwise.
    yaw_sign_flip:
        If True, negate the yaw sign.
    """
    n = len(frames)
    if n < 2:
        empty = np.zeros(0, dtype=np.float64)
        return {"yaw_deg": empty, "disp_mag": empty}

    h, w = frames[0].shape[:2]
    K = intrinsics if intrinsics is not None else _default_intrinsics(w, h)
    mask = _roi_mask(w, h, roi_fraction)

    # GoodFeaturesToTrack params
    gftt_params = dict(
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=mask,
    )

    # KLT params
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    yaw_degs: list[float] = []
    disp_mags: list[float] = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, n):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Detect corners in previous frame
        corners = cv2.goodFeaturesToTrack(prev_gray, **gftt_params)

        if corners is None or len(corners) < 8:
            # Not enough features — report zero motion
            yaw_degs.append(0.0)
            disp_mags.append(0.0)
            prev_gray = curr_gray
            continue

        # Track with KLT
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None, **lk_params,
        )

        # Filter: keep only valid tracks
        status = status.ravel().astype(bool)
        pts1 = corners.reshape(-1, 2)[status]
        pts2 = pts_next.reshape(-1, 2)[status]

        # Remove outlier tracks (excessive displacement)
        disp = pts2 - pts1
        disp_norms = np.linalg.norm(disp, axis=1)
        valid = disp_norms < max_displacement_px
        pts1 = pts1[valid]
        pts2 = pts2[valid]
        disp_norms = disp_norms[valid]

        # Median displacement magnitude (proxy for translational speed)
        med_disp = float(np.median(disp_norms)) if len(disp_norms) > 0 else 0.0
        disp_mags.append(med_disp)

        # Gate: if displacement is near zero, E-mat decomposition is
        # degenerate (recoverPose returns garbage R).  Report zero yaw.
        if med_disp < 0.3:
            yaw_degs.append(0.0)
            prev_gray = curr_gray
            continue

        if len(pts1) < 8:
            # Too few tracks — fallback to horizontal flow heuristic
            yaw_deg = _horizontal_flow_fallback(pts1, pts2, w)
            if yaw_sign_flip:
                yaw_deg = -yaw_deg
            yaw_degs.append(yaw_deg)
            prev_gray = curr_gray
            continue

        # Essential matrix via RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=ransac_confidence,
            threshold=ransac_threshold,
        )

        if E is None or inlier_mask is None:
            yaw_deg = _horizontal_flow_fallback(pts1, pts2, w)
            if yaw_sign_flip:
                yaw_deg = -yaw_deg
            yaw_degs.append(yaw_deg)
            prev_gray = curr_gray
            continue

        n_inliers = int(np.sum(inlier_mask))

        if n_inliers < min_inliers:
            # Not enough inliers — fallback
            yaw_deg = _horizontal_flow_fallback(pts1, pts2, w)
            if yaw_sign_flip:
                yaw_deg = -yaw_deg
            yaw_degs.append(yaw_deg)
            prev_gray = curr_gray
            continue

        # Decompose essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
        yaw_rad = _yaw_from_rotation(R)
        yaw_deg = math.degrees(yaw_rad)

        if yaw_sign_flip:
            yaw_deg = -yaw_deg

        yaw_degs.append(yaw_deg)
        prev_gray = curr_gray

    return {
        "yaw_deg": np.array(yaw_degs, dtype=np.float64),
        "disp_mag": np.array(disp_mags, dtype=np.float64),
    }


def _horizontal_flow_fallback(
    pts1: np.ndarray,
    pts2: np.ndarray,
    img_width: int,
) -> float:
    """Fallback yaw estimate from median horizontal displacement.

    When essential matrix decomposition fails, use the overall horizontal
    flow direction as a coarse turn indicator.  Returns degrees (positive =
    left turn).
    """
    if len(pts1) < 2:
        return 0.0
    dx = pts2[:, 0] - pts1[:, 0]
    # Scene moving right in image → ego turning left → positive yaw
    median_dx = float(np.median(dx))
    # Convert to rough degrees: assume ~1 px ≈ 0.06° for a typical driving camera
    return median_dx * 0.06


# ---------------------------------------------------------------------------
# VOProxyBaseline
# ---------------------------------------------------------------------------

class VOProxyBaseline:
    """Map VO-proxy signals to EgoDyn-Bench answers.

    Only predicts questions with a defensible physical mapping from
    monocular KLT tracking + essential matrix decomposition.

    Parameters
    ----------
    yaw_deadzone_deg:
        Mean |yaw| below this → "straight" (degrees per frame pair).
    yaw_peak_thr_deg:
        Peak |yaw| must exceed this to confirm a turn (degrees).
    disp_stopped_thr:
        Median displacement below this → considered "stopped" (pixels).
    disp_moving_thr:
        Median displacement above this → considered "moving" (pixels).
    trend_deadzone:
        Absolute slope of disp_mag below this → "steady".
    heading_change_thr_deg:
        Cumulative |yaw| above this → "significant heading change" (degrees).
    lateral_thr_deg:
        Peak |yaw| above this → "high lateral acceleration" (degrees).
    brake_disp_drop:
        Fractional drop in displacement to detect braking (0–1).
    yaw_sign_flip:
        Negate yaw sign if coordinate convention differs.
    max_corners:
        Maximum corners for goodFeaturesToTrack.
    roi_fraction:
        Central fraction of image for feature detection.
    max_displacement_px:
        Maximum per-frame displacement before track is discarded.
    ransac_threshold:
        Pixel threshold for essential matrix RANSAC.
    min_inliers:
        Minimum inliers to trust the essential matrix.
    """

    def __init__(self, config: dict | None = None):
        c = config or {}
        self.yaw_deadzone_deg = c.get("yaw_deadzone_deg", 0.03)
        self.yaw_peak_thr_deg = c.get("yaw_peak_thr_deg", 0.15)
        self.disp_stopped_thr = c.get("disp_stopped_thr", 0.5)
        self.disp_moving_thr = c.get("disp_moving_thr", 2.0)
        self.trend_deadzone = c.get("trend_deadzone", 0.3)
        self.heading_change_thr_deg = c.get("heading_change_thr_deg", 1.5)
        self.lateral_thr_deg = c.get("lateral_thr_deg", 0.8)
        self.brake_disp_drop = c.get("brake_disp_drop", 0.4)
        self.yaw_sign_flip = c.get("yaw_sign_flip", False)
        self.max_corners = c.get("max_corners", 800)
        self.roi_fraction = c.get("roi_fraction", 0.6)
        self.max_displacement_px = c.get("max_displacement_px", 50.0)
        self.ransac_threshold = c.get("ransac_threshold", 1.0)
        self.min_inliers = c.get("min_inliers", 15)

    def predict(
        self,
        frames: list[np.ndarray],
        intrinsics: Optional[np.ndarray] = None,
    ) -> dict[str, str]:
        """Predict answers for supported question types.

        Parameters
        ----------
        frames:
            List of BGR uint8 images for one clip.
        intrinsics:
            Optional 3x3 camera intrinsic matrix.  If None, a default
            pinhole model is assumed (fx = fy = 0.9 * image_width).

        Returns
        -------
        ``{question_id: answer}`` for every question in
        :data:`SUPPORTED_QUESTIONS`.
        """
        signals = compute_vo_signals(
            frames,
            intrinsics=intrinsics,
            max_corners=self.max_corners,
            roi_fraction=self.roi_fraction,
            max_displacement_px=self.max_displacement_px,
            ransac_threshold=self.ransac_threshold,
            min_inliers=self.min_inliers,
            yaw_sign_flip=self.yaw_sign_flip,
        )

        yaw = signals["yaw_deg"]
        disp = signals["disp_mag"]
        answers: dict[str, str] = {}

        # --- yaw_rate_turn_direction (mean direction + peak gating) ---
        # Mean yaw determines direction (robust to per-pair noise in
        # E-mat decomposition); peak confirms at least one frame pair
        # showed a convincing rotation signal.
        if len(yaw) > 0:
            yaw_mean = float(np.mean(yaw))
            yaw_peak = float(np.max(np.abs(yaw)))
        else:
            yaw_mean = 0.0
            yaw_peak = 0.0

        if (abs(yaw_mean) > self.yaw_deadzone_deg
                and yaw_peak > self.yaw_peak_thr_deg):
            answers["yaw_rate_turn_direction"] = (
                "left" if yaw_mean > 0 else "right"
            )
        else:
            answers["yaw_rate_turn_direction"] = "straight"

        # --- speed_trend (linear slope of pixel-domain displacement proxy) ---
        if len(disp) >= 3:
            t = np.arange(len(disp), dtype=np.float64)
            t_mean = np.mean(t)
            d_mean = np.mean(disp)
            slope = float(
                np.sum((t - t_mean) * (disp - d_mean))
                / max(np.sum((t - t_mean) ** 2), 1e-12)
            )
            if slope > self.trend_deadzone:
                answers["speed_trend"] = "accelerating"
            elif slope < -self.trend_deadzone:
                answers["speed_trend"] = "decelerating"
            else:
                answers["speed_trend"] = "steady"
        else:
            answers["speed_trend"] = "steady"

        # --- stop_and_go (hysteresis-based cycle counting) ---
        if len(disp) >= 3:
            stopped_mask = disp < self.disp_stopped_thr
            moving_mask = disp > self.disp_moving_thr
            cycles = 0
            was_stopped = bool(stopped_mask[0])
            for k in range(1, len(disp)):
                if not was_stopped and stopped_mask[k]:
                    was_stopped = True
                elif was_stopped and moving_mask[k]:
                    cycles += 1
                    was_stopped = False
            answers["stop_and_go"] = "yes" if cycles >= 1 else "no"
        else:
            answers["stop_and_go"] = "no"

        # --- significant_heading_change (cumulative absolute yaw) ---
        cumul_abs_yaw = float(np.sum(np.abs(yaw))) if len(yaw) > 0 else 0.0
        answers["significant_heading_change"] = (
            "yes" if cumul_abs_yaw > self.heading_change_thr_deg else "no"
        )

        # --- high_lateral_accel (peak absolute yaw rate proxy) ---
        answers["high_lateral_accel"] = (
            "yes" if yaw_peak > self.lateral_thr_deg else "no"
        )

        # --- brake_then_turn (deceleration followed by yaw) ---
        if len(disp) >= 3 and len(yaw) >= 3:
            # Detect braking: displacement drops by > brake_disp_drop fraction
            mean_disp = float(np.mean(disp)) if len(disp) > 0 else 0.0
            brake_thr = mean_disp * self.brake_disp_drop if mean_disp > 0.5 else 0.0
            brake_mask = np.zeros(len(disp), dtype=bool)
            for k in range(1, len(disp)):
                if disp[k] < disp[k - 1] - brake_thr and brake_thr > 0:
                    brake_mask[k] = True
            # Detect turning: |yaw| exceeds deadzone
            turn_mask = np.abs(yaw) > self.yaw_deadzone_deg
            found = False
            brake_indices = np.where(brake_mask)[0]
            turn_indices = np.where(turn_mask)[0]
            if len(brake_indices) > 0 and len(turn_indices) > 0:
                for bi in brake_indices:
                    if np.any(turn_indices > bi):
                        found = True
                        break
            answers["brake_then_turn"] = "yes" if found else "no"
        else:
            answers["brake_then_turn"] = "no"

        return answers


# ---------------------------------------------------------------------------
# Frame loading helpers (reuse from flow_heuristic)
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
        indices = [round(i * (n - 1) / (num_frames - 1))
                   for i in range(num_frames)]
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
        indices = [round(i * (total - 1) / (num_frames - 1))
                   for i in range(num_frames)]
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
        description="Run VO-proxy baseline on EgoDyn-Bench clips.",
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

    baseline = VOProxyBaseline()
    nuscenes_root = Path(args.nuscenes_root)
    carla_video_dir = Path(args.carla_video_dir)
    clip_groups = group_qa_by_clip(qa_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    processed = 0
    skipped = 0

    with open(output_path, "w") as out_f:
        for clip_id in tqdm(clip_groups, desc="VO-proxy baseline"):
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
        print_eval_results(result, "vo_proxy", False, True)

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
