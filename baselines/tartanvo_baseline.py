"""TartanVO learned visual odometry baseline for EgoDyn-Bench.

Replaces the classical KLT + essential-matrix VO backend in
:mod:`baselines.vo_proxy_baseline` with **TartanVO** (Wang et al., CoRL 2021),
a learning-based monocular VO trained on TartanAir synthetic data.

The heuristic mapping from per-frame-pair signals (yaw, displacement) to
the 6 supported questions is **identical** to the classical VO proxy —
only the ego-motion estimation backend is swapped.

**Narrative:** Even with a state-of-the-art *learned* monocular VO
(TartanVO, trained on diverse synthetic scenes and shown to generalise to
KITTI/EuRoC without fine-tuning), the heuristic mapping to semantic
ego-motion labels still falls short of the reasoning required by
EgoDyn-Bench.

Supported questions (6 of 14)
-----------------------------
Same as :mod:`baselines.vo_proxy_baseline`.

Dependencies: ``torch``, ``cupy`` (for PWC-Net correlation layer).

Pretrained weights
------------------
Download ``tartanvo_1914.pkl`` to ``models/``::

    wget https://cmu.box.com/shared/static/t1a5u4x6dxohl89104dyrsiz42mvq2sz.pkl \\
         -O models/tartanvo_1914.pkl

Usage::

    python -m baselines.tartanvo_baseline \\
        --selected_clips selected_clips.json \\
        --output generated/tartanvo_answers.jsonl \\
        --run_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.tartanvo_lib import VONet, make_intrinsics_layer
from baselines.vo_proxy_baseline import (
    SUPPORTED_QUESTIONS,
    load_frames_from_images,
    load_frames_from_video,
)

logger = logging.getLogger(__name__)

# TartanVO constants (from original repo)
_POSE_STD = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)
_FLOW_NORM = 20.0

# TartanAir default intrinsics (for 640x480 images)
_TARTANAIR_FX = 320.0
_TARTANAIR_FY = 320.0
_TARTANAIR_CX = 320.0
_TARTANAIR_CY = 240.0

# Network input size (TartanVO was trained on 640x448 center-cropped images)
_NET_W = 640
_NET_H = 448


# ---------------------------------------------------------------------------
# TartanVO signal computation
# ---------------------------------------------------------------------------

def compute_tartanvo_signals(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute per-frame-pair yaw and displacement using TartanVO.

    Parameters
    ----------
    frames:
        List of BGR uint8 images for one clip.
    model:
        Pre-loaded VONet in eval mode.
    device:
        Torch device (must be cuda — TartanVO requires GPU).

    Returns
    -------
    ``{"yaw_deg": np.ndarray, "disp_mag": np.ndarray}`` — same format
    as :func:`baselines.vo_proxy_baseline.compute_vo_signals`.
    """
    if len(frames) < 2:
        empty = np.zeros(0, dtype=np.float64)
        return {"yaw_deg": empty, "disp_mag": empty}

    # --- Preprocess frames: BGR → RGB, center-crop to 640x448, normalize ---
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → RGB float32 (H, W, 3), center-cropped to _NET_W x _NET_H."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Resize so that min dimension accommodates the crop
        scale = max(_NET_H / h, _NET_W / w)
        if scale > 1.0:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        elif max(h, w) > max(_NET_H, _NET_W) * 1.5:
            # Downsample large frames first for efficiency
            scale = max(_NET_H / h, _NET_W / w)
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Center crop
        h, w = rgb.shape[:2]
        y1 = (h - _NET_H) // 2
        x1 = (w - _NET_W) // 2
        rgb = rgb[y1:y1+_NET_H, x1:x1+_NET_W, :]
        return rgb

    processed = [_preprocess(f) for f in frames]

    # Build intrinsic layer (same for all frame pairs — TartanAir defaults)
    intrinsic_np = make_intrinsics_layer(
        _NET_W, _NET_H,
        _TARTANAIR_FX, _TARTANAIR_FY,
        _TARTANAIR_CX, _TARTANAIR_CY,
    )

    yaw_degs: list[float] = []
    disp_mags: list[float] = []

    with torch.no_grad():
        for i in range(len(processed) - 1):
            # Convert to tensors: (1, 3, H, W) float32 [0, 1]
            img1_np = processed[i].astype(np.float32).transpose(2, 0, 1) / 255.0
            img2_np = processed[i + 1].astype(np.float32).transpose(2, 0, 1) / 255.0

            img1 = torch.from_numpy(img1_np).unsqueeze(0).to(device)
            img2 = torch.from_numpy(img2_np).unsqueeze(0).to(device)

            # Intrinsic layer: downsample by 4x to match flow output resolution
            # (PWC-Net outputs flow at H/4 x W/4)
            intr_down = cv2.resize(
                intrinsic_np,
                (_NET_W // 4, _NET_H // 4),
                interpolation=cv2.INTER_LINEAR,
            )
            intr_np = intr_down.transpose(2, 0, 1).astype(np.float32)
            intrinsic = torch.from_numpy(intr_np).unsqueeze(0).to(device)

            # Run TartanVO
            _flow, pose = model([img1, img2, intrinsic])

            # Denormalize pose
            pose_np = pose.cpu().numpy()[0] * _POSE_STD  # (6,)
            # pose = [tx, ty, tz, rx, ry, rz]
            tx, ty, tz = pose_np[0], pose_np[1], pose_np[2]
            rx, ry, rz = pose_np[3], pose_np[4], pose_np[5]

            # Yaw: rz component (rotation around vertical axis)
            # Convert from radians to degrees
            yaw_deg = float(rz * 180.0 / math.pi)
            yaw_degs.append(yaw_deg)

            # Displacement magnitude (up-to-scale)
            disp = float(math.sqrt(tx**2 + ty**2 + tz**2))
            disp_mags.append(disp)

    return {
        "yaw_deg": np.array(yaw_degs, dtype=np.float64),
        "disp_mag": np.array(disp_mags, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# TartanVOBaseline
# ---------------------------------------------------------------------------

class TartanVOBaseline:
    """TartanVO-based learned VO baseline — same heuristic mapping as VO proxy.

    Uses the same thresholds and question mapping as
    :class:`baselines.vo_proxy_baseline.VOProxyBaseline`.
    """

    def __init__(
        self,
        weights_path: str = "models/tartanvo_1914.pkl",
        device: str | torch.device | None = None,
        config: dict[str, Any] | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TartanVO requires CUDA (CuPy correlation layer is GPU-only)."
            )

        logger.info("Loading TartanVO model from %s on %s ...", weights_path, self.device)
        self.model = VONet()
        self._load_weights(self.model, weights_path)
        self.model = self.model.to(self.device).eval()
        logger.info("TartanVO model loaded.")

        # Heuristic thresholds — recalibrated for TartanVO's output scale.
        # TartanVO's pose output (denormalized by pose_std) produces yaw
        # values in the range [-5, +5] degrees and displacement magnitudes
        # in the range [0.02, 3.5].  The VO proxy thresholds were set for
        # KLT pixel-domain signals and must be adjusted.
        #
        # Calibrated on 387 frame-pair signals from 43 clips:
        #   yaw_deg:  p25=-0.34, p50=0.00, p75=0.72, std=1.59
        #   disp_mag: p25=0.52,  p50=0.76, p75=1.14, std=0.64
        c = config or {}
        self.yaw_deadzone_deg = c.get("yaw_deadzone_deg", 0.5)
        self.yaw_peak_thr_deg = c.get("yaw_peak_thr_deg", 1.0)
        self.disp_stopped_thr = c.get("disp_stopped_thr", 0.15)
        self.disp_moving_thr = c.get("disp_moving_thr", 0.5)
        self.trend_deadzone = c.get("trend_deadzone", 0.05)
        self.heading_change_thr_deg = c.get("heading_change_thr_deg", 5.0)
        self.lateral_thr_deg = c.get("lateral_thr_deg", 2.0)
        self.brake_disp_drop = c.get("brake_disp_drop", 0.3)

    @staticmethod
    def _load_weights(model: torch.nn.Module, path: str) -> None:
        """Load TartanVO weights, handling DataParallel prefix."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_dict = model.state_dict()
        # Try direct match first
        matched = {k: v for k, v in state.items() if k in model_dict}
        if len(matched) == 0:
            # Try stripping "module." prefix (DataParallel)
            matched = {}
            for k, v in state.items():
                kk = k[7:] if k.startswith("module.") else k
                if kk in model_dict:
                    matched[kk] = v
        if len(matched) == 0:
            raise RuntimeError(f"Could not load any weights from {path}")
        model_dict.update(matched)
        model.load_state_dict(model_dict)

    def predict(self, frames: list[np.ndarray]) -> dict[str, str]:
        """Predict answers using TartanVO + heuristic mapping.

        The signal extraction uses TartanVO; the mapping to answers
        replicates :meth:`baselines.vo_proxy_baseline.VOProxyBaseline.predict`.
        """
        signals = compute_tartanvo_signals(frames, self.model, self.device)
        yaw = signals["yaw_deg"]
        disp = signals["disp_mag"]
        answers: dict[str, str] = {}

        # --- yaw_rate_turn_direction ---
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

        # --- speed_trend ---
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

        # --- stop_and_go ---
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

        # --- significant_heading_change ---
        cumul_abs_yaw = float(np.sum(np.abs(yaw))) if len(yaw) > 0 else 0.0
        answers["significant_heading_change"] = (
            "yes" if cumul_abs_yaw > self.heading_change_thr_deg else "no"
        )

        # --- high_lateral_accel ---
        answers["high_lateral_accel"] = (
            "yes" if yaw_peak > self.lateral_thr_deg else "no"
        )

        # --- brake_then_turn ---
        if len(disp) >= 3 and len(yaw) >= 3:
            mean_disp = float(np.mean(disp)) if len(disp) > 0 else 0.0
            brake_thr = mean_disp * self.brake_disp_drop if mean_disp > 0.5 else 0.0
            brake_mask = np.zeros(len(disp), dtype=bool)
            for k in range(1, len(disp)):
                if disp[k] < disp[k - 1] - brake_thr and brake_thr > 0:
                    brake_mask[k] = True
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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TartanVO learned-VO baseline on EgoDyn-Bench clips.",
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
        help="Which CARLA video data to use",
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
        "--weights", type=str, default="models/tartanvo_1914.pkl",
        help="Path to TartanVO pretrained weights",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect cuda)",
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
    parser.add_argument(
        "--dump_signals", action="store_true",
        help="Print per-clip signal statistics (for threshold calibration)",
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
        group_qa_by_clip,
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

    # --- Initialise model -------------------------------------------------
    baseline = TartanVOBaseline(
        weights_path=args.weights,
        device=args.device,
    )
    nuscenes_root = Path(args.nuscenes_root)
    carla_video_dir = Path(args.carla_video_dir)
    clip_groups = group_qa_by_clip(qa_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    processed = 0
    skipped = 0

    # Signal stats for calibration
    all_yaw_degs: list[float] = []
    all_disp_mags: list[float] = []

    with open(output_path, "w") as out_f:
        for clip_id in tqdm(clip_groups, desc="TartanVO baseline"):
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

            if args.dump_signals:
                signals = compute_tartanvo_signals(
                    frames, baseline.model, baseline.device,
                )
                all_yaw_degs.extend(signals["yaw_deg"].tolist())
                all_disp_mags.extend(signals["disp_mag"].tolist())

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

    if args.dump_signals and all_yaw_degs:
        yaw_arr = np.array(all_yaw_degs)
        disp_arr = np.array(all_disp_mags)
        logger.info(
            "Signal stats (N=%d):\n"
            "  yaw_deg:  mean=%.4f, std=%.4f, min=%.4f, p25=%.4f, p50=%.4f, p75=%.4f, max=%.4f\n"
            "  disp_mag: mean=%.4f, std=%.4f, min=%.4f, p25=%.4f, p50=%.4f, p75=%.4f, max=%.4f",
            len(yaw_arr),
            yaw_arr.mean(), yaw_arr.std(), yaw_arr.min(),
            np.percentile(yaw_arr, 25), np.percentile(yaw_arr, 50),
            np.percentile(yaw_arr, 75), yaw_arr.max(),
            disp_arr.mean(), disp_arr.std(), disp_arr.min(),
            np.percentile(disp_arr, 25), np.percentile(disp_arr, 50),
            np.percentile(disp_arr, 75), disp_arr.max(),
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
        print_eval_results(result, "tartanvo_baseline", False, True)

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
