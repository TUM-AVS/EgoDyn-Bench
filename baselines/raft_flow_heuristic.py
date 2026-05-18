"""RAFT-based learned flow heuristic baseline for EgoDyn-Bench.

Drop-in replacement of the Farneback flow backend in :mod:`baselines.flow_heuristic`
with a pre-trained **RAFT** model (``torchvision.models.optical_flow.raft_large``).
The heuristic signal mapping (tangential/radial decomposition, deadzones, temporal
pattern matching) is **identical** to the classical flow heuristic — only the
optical flow extraction is swapped.

**Narrative:** Even with state-of-the-art *learned* optical flow (RAFT, trained on
FlyingChairs + FlyingThings3D + Sintel + KITTI + HD1K), the heuristic mapping to
semantic ego-motion labels fails to capture the complex reasoning required by
EgoDyn-Bench.

Supported questions (6 of 14)
-----------------------------
Same as :mod:`baselines.flow_heuristic` — only questions with a defensible
physical mapping from monocular optical flow.

Usage::

    python -m baselines.raft_flow_heuristic \
        --selected_clips selected_clips.json \
        --output generated/raft_flow_heuristic_answers.jsonl \
        --run_eval

Requires: ``torch``, ``torchvision >= 0.14`` (RAFT weights).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor

from baselines.flow_heuristic import (
    FlowHeuristicBaseline,
    SUPPORTED_QUESTIONS,
    load_frames_from_images,
    load_frames_from_video,
    _MAX_PROCESS_WIDTH,
)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAFT flow computation
# ---------------------------------------------------------------------------

def _pad_to_multiple(img: torch.Tensor, divisor: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad image tensor so H and W are multiples of *divisor*.

    Returns the padded tensor and (pad_h, pad_w) used.
    """
    _, h, w = img.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return img, (pad_h, pad_w)


def compute_raft_flow_signals(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    max_width: int = _MAX_PROCESS_WIDTH,
) -> dict[str, np.ndarray]:
    """Compute per-frame-pair flow signals using RAFT.

    Applies the **same** radial/tangential decomposition and sky/hood
    cropping as the Farneback baseline in :func:`baselines.flow_heuristic.compute_flow_signals`.
    The only difference is the optical flow backend.

    Parameters
    ----------
    frames:
        List of BGR images (uint8).
    model:
        Pre-loaded RAFT model in eval mode.
    device:
        Torch device (cuda or cpu).
    max_width:
        Frames wider than this are downsampled before flow computation.

    Returns
    -------
    ``{turn_score, exp_score, motion_mag}`` — same format as the Farneback version.
    """
    if len(frames) < 2:
        empty = np.zeros(0, dtype=np.float64)
        return {"turn_score": empty, "exp_score": empty, "motion_mag": empty}

    # --- Preprocessing: resize + convert to tensors ---
    def _preprocess(bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → (3, H, W) float32 tensor in [0, 1], optionally resized."""
        h, w = bgr.shape[:2]
        if w > max_width:
            scale = max_width / w
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(rgb)  # (3, H, W) float32 [0, 1]

    tensors = [_preprocess(f) for f in frames]

    # Pad to multiple of 8 (RAFT requirement)
    padded = []
    pad_info = None
    for t in tensors:
        p, pad_info = _pad_to_multiple(t, 8)
        padded.append(p)

    _, h_pad, w_pad = padded[0].shape
    # Working resolution for signal decomposition (before padding)
    _, h_work, w_work = tensors[0].shape

    # Vertical crop bounds (same as Farneback: 20% top, 85% bottom)
    y_top = int(h_work * 0.20)
    y_bot = int(h_work * 0.85)
    cy, cx = h_work / 2.0, w_work / 2.0

    # Pre-compute pixel grid for radial/tangential decomposition
    ys, xs = np.mgrid[y_top:y_bot, 0:w_work].astype(np.float32)
    dx_grid = xs - cx
    dy_grid = ys - cy
    radial_norm = np.sqrt(dx_grid ** 2 + dy_grid ** 2)
    radial_norm[radial_norm < 1e-6] = 1.0

    turn_scores: list[float] = []
    exp_scores: list[float] = []
    motion_mags: list[float] = []

    # --- Run RAFT on frame pairs ---
    with torch.no_grad():
        for i in range(len(padded) - 1):
            img1 = padded[i].unsqueeze(0).to(device)
            img2 = padded[i + 1].unsqueeze(0).to(device)

            # RAFT returns list of flow refinements; [-1] is final
            flow_list = model(img1, img2)
            flow = flow_list[-1].squeeze(0).cpu().numpy()  # (2, H_pad, W_pad)

            # Remove padding and crop to working resolution
            flow_x = flow[0, :h_work, :w_work]
            flow_y = flow[1, :h_work, :w_work]

            # Crop to valid vertical band
            flow_x_crop = flow_x[y_top:y_bot, :]
            flow_y_crop = flow_y[y_top:y_bot, :]

            # --- turn_score: tangential flow component ---
            tangential_flow = (flow_x_crop * dy_grid - flow_y_crop * dx_grid) / radial_norm
            turn_scores.append(float(np.median(tangential_flow)))

            # --- exp_score: radial flow component ---
            radial_flow = (flow_x_crop * dx_grid + flow_y_crop * dy_grid) / radial_norm
            exp_scores.append(float(np.median(radial_flow)))

            # --- motion_mag: flow magnitude ---
            mag = np.sqrt(flow_x_crop ** 2 + flow_y_crop ** 2)
            motion_mags.append(float(np.median(mag)))

    return {
        "turn_score": np.array(turn_scores),
        "exp_score": np.array(exp_scores),
        "motion_mag": np.array(motion_mags),
    }


# ---------------------------------------------------------------------------
# RAFTFlowHeuristicBaseline
# ---------------------------------------------------------------------------

class RAFTFlowHeuristicBaseline(FlowHeuristicBaseline):
    """RAFT-based learned flow heuristic — same mapping, learned flow backend.

    Inherits all heuristic thresholds and question mapping from
    :class:`baselines.flow_heuristic.FlowHeuristicBaseline`.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading RAFT model (weights: C_T_SKHT_V2) on %s ...", self.device)
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self.model = self.model.to(self.device).eval()
        logger.info("RAFT model loaded.")

    def predict(self, frames: list[np.ndarray]) -> dict[str, str]:
        """Predict answers using RAFT flow + heuristic mapping.

        Computes flow signals via RAFT, then delegates to the parent's
        heuristic predict logic.
        """
        signals = compute_raft_flow_signals(frames, self.model, self.device)
        # Inject signals and call the parent's mapping logic directly
        # (parent.predict() calls compute_flow_signals internally, so
        # we replicate the mapping here to avoid double computation)
        turn = signals["turn_score"]
        exp = signals["exp_score"]
        mag = signals["motion_mag"]

        answers: dict[str, str] = {}

        mean_turn = float(np.mean(turn)) if len(turn) > 0 else 0.0
        mean_exp = float(np.mean(exp)) if len(exp) > 0 else 0.0
        peak_abs_turn = float(np.max(np.abs(turn))) if len(turn) > 0 else 0.0
        cumul_turn = float(np.sum(np.abs(turn))) if len(turn) > 0 else 0.0

        # yaw_rate_turn_direction
        if mean_turn > self.turn_deadzone:
            answers["yaw_rate_turn_direction"] = "left"
        elif mean_turn < -self.turn_deadzone:
            answers["yaw_rate_turn_direction"] = "right"
        else:
            answers["yaw_rate_turn_direction"] = "straight"

        # speed_trend
        if mean_exp > self.exp_deadzone:
            answers["speed_trend"] = "accelerating"
        elif mean_exp < -self.exp_deadzone:
            answers["speed_trend"] = "decelerating"
        else:
            answers["speed_trend"] = "steady"

        # significant_heading_change
        answers["significant_heading_change"] = (
            "yes" if cumul_turn > self.heading_change_thr else "no"
        )

        # high_lateral_accel
        answers["high_lateral_accel"] = (
            "yes" if peak_abs_turn > self.lateral_thr else "no"
        )

        # stop_and_go
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

        # brake_then_turn
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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAFT learned-flow heuristic baseline on EgoDyn-Bench clips.",
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
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect cuda/cpu)",
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
    baseline = RAFTFlowHeuristicBaseline(device=args.device)
    nuscenes_root = Path(args.nuscenes_root)
    carla_video_dir = Path(args.carla_video_dir)
    clip_groups = group_qa_by_clip(qa_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    processed = 0
    skipped = 0

    with open(output_path, "w") as out_f:
        for clip_id in tqdm(clip_groups, desc="RAFT flow baseline"):
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
        print_eval_results(result, "raft_flow_heuristic", False, True)

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
