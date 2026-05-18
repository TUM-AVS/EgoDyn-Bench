"""Evaluate DriveMM on EgoDyn-Bench.

DriveMM is an all-in-one large multimodal model for autonomous driving,
built on LLaVA-NeXT (LLaMA-3 8B + SigLIP so400m-patch14-384).

Prerequisites:
    1. Clone DriveMM:  git clone https://github.com/zhijian11/DriveMM
    2. Install:        cd DriveMM && pip install -e ".[train]"
    3. Requires:       flash-attn>=2.6, torch>=2.1

Usage:
    # Basic run (downloads weights from HuggingFace automatically)
    python evaluation/evaluate_drivemm.py \\
        --selected_clips selected_clips.json \
        --model DriveMM/DriveMM \
        --no_trajectory --resume \
        --output generated/drivemm_answers.jsonl \
        --run_eval --metrics_output results/drivemm.json

    # With local checkpoint
    python evaluation/evaluate_drivemm.py \
        --selected_clips selected_clips.json \
        --model /path/to/DriveMM \
        --no_trajectory --resume \
        --output generated/drivemm_answers.jsonl

    # Small test run
    python evaluation/evaluate_drivemm.py \
        --selected_clips selected_clips.json \
        --model DriveMM/DriveMM \
        --no_trajectory --max_samples 5 \
        --output generated/drivemm_test.jsonl --run_eval
"""

import base64
import io
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images
    from llava.train.train import preprocess_llama3
except ImportError:
    print(
        "ERROR: DriveMM's llava package not installed.\n"
        "  1. git clone https://github.com/zhijian11/DriveMM\n"
        "  2. cd DriveMM && pip install -e '.[train]'"
    )
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
# Helpers
# ---------------------------------------------------------------------------

def _decode_base64_images(
    image_data: list[tuple[str, str]],
) -> list[Image.Image]:
    """Decode base64-encoded images to PIL Images."""
    images = []
    for b64, _mime in image_data:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        images.append(img)
    return images


def _build_image_prompt(text_prompt: str, n_images: int) -> str:
    """Prepend <image> tokens to the text prompt.

    DriveMM expects one ``<image>`` token per input image.  For multiple
    images we number them so the model can reference each frame.
    """
    if n_images == 0:
        return text_prompt
    if n_images == 1:
        return f"<image>\n{text_prompt}"
    tags = " ".join(f"{i}: <image>" for i in range(1, n_images + 1))
    return f"{tags}\n{text_prompt}"


# ---------------------------------------------------------------------------
# Model loading & caller
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: str, dtype: torch.dtype):
    """Load DriveMM model, tokenizer, and image processor."""
    logger.info(f"Loading DriveMM from {model_path} (dtype={dtype})")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,
        "llama",
        device_map=torch.device(device),
        torch_dtype=str(dtype).replace("torch.", ""),
        multimodal=True,
    )
    model.eval()
    logger.info(
        f"DriveMM loaded: {model.dtype}, device={next(model.parameters()).device}, "
        f"max_length={max_length}"
    )
    return tokenizer, model, image_processor


def _make_caller(tokenizer, model, image_processor, device: str):
    """Return a closure ``(prompt, image_data) -> model_answer``."""

    def call_api(prompt: str, image_data: list[tuple[str, str]]) -> str:
        # --- decode images ---
        pil_images = _decode_base64_images(image_data)
        n_images = len(pil_images)

        # --- build prompt with <image> tokens ---
        full_prompt = _build_image_prompt(prompt, n_images)

        # --- process images ---
        if pil_images:
            image_tensors = process_images(pil_images, image_processor, model.config)
            image_tensors = [
                t.to(dtype=model.dtype, device=device) for t in image_tensors
            ]
            image_sizes = [img.size for img in pil_images]
            modalities = ["image"] * n_images
        else:
            image_tensors = None
            image_sizes = None
            modalities = None

        # --- tokenize via LLaMA-3 chat template ---
        sources = [[
            {"from": "human", "value": full_prompt},
            {"from": "gpt", "value": ""},
        ]]
        input_ids = preprocess_llama3(
            sources, tokenizer, has_image=(n_images > 0),
        )["input_ids"][:, :-1].to(device)

        # --- generate ---
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
                modalities=modalities,
            )

        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    return call_api


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_common_parser(
        description="Evaluate DriveMM on EgoDyn-Bench",
        default_model="DriveMM/DriveMM",
        api_key_env_var="HF_TOKEN",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokenizer, model, image_processor = _load_model(
        args.model, args.device, dtype,
    )

    call_api = _make_caller(tokenizer, model, image_processor, args.device)

    return run_evaluation(args, call_api)


if __name__ == "__main__":
    sys.exit(main())
