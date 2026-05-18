"""Evaluate any vLLM-supported VLM locally on EgoDyn-Bench.

Works with any vision-language model that vLLM supports (vllm>=0.11.0),
including Qwen3-VL, InternVL3, LLaVA, Phi-4-multimodal, and others.
Optionally launches a vLLM server as a managed subprocess and connects
to it via the OpenAI-compatible ``/v1/chat/completions`` endpoint.

Includes per-request inference timing metrics (mean, median, P95 latency,
throughput) that are printed after evaluation and saved to --metrics_output.

Prerequisites:
    pip install vllm>=0.11.0 openai

Usage:
    # Qwen3-VL-8B on single GPU (auto-launches vLLM)
    python evaluation/evaluate_vllm_local.py \
        --selected_clips selected_clips.json \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max_model_len 16384 \
        --no_trajectory --resume \
        --output generated/qwen3vl_8b_answers.jsonl \
        --run_eval --metrics_output results/qwen3vl_8b.json

    # InternVL3-8B on single GPU
    python evaluation/evaluate_vllm_local.py \
        --selected_clips selected_clips.json \
        --model OpenGVLab/InternVL3-8B \
        --max_model_len 16384 \
        --no_trajectory --resume \
        --output generated/internvl3_8b_answers.jsonl \
        --run_eval --metrics_output results/internvl3_8b.json

    # Qwen3-VL-30B MoE on 1 GPUs with tensor parallelism
    python evaluation/evaluate_vllm_local.py \
        --selected_clips selected_clips.json \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --tensor_parallel_size 1 --max_model_len 16384 \
        --no_trajectory --resume \
        --output generated/qwen3vl_30b_answers.jsonl

    # Connect to an already-running vLLM server
    python evaluation/evaluate_vllm_local.py \
        --selected_clips selected_clips.json \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --base_url http://localhost:8000/v1 \
        --no_launch \
        --no_trajectory --resume \
        --output generated/qwen3vl_8b_answers.jsonl
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, APIError, APIConnectionError
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator_common import build_common_parser, run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 6
VLLM_STARTUP_TIMEOUT = 600  # seconds to wait for vLLM server readiness

# Models that use chain-of-thought reasoning (need higher max_tokens and
# vLLM's --reasoning-parser to separate thinking from the final answer).
# Maps model name substring -> default reasoning parser name.
_REASONING_MODEL_PARSERS: dict[str, str] = {
    "Cosmos-Reason": "qwen3",
    "Thinking": "deepseek_r1",      # e.g. Qwen3-VL-4B-Thinking
    "Kimi-K2": "deepseek_r1",
}


def _detect_reasoning_parser(model: str) -> str | None:
    """Return the appropriate vLLM reasoning parser for *model*, or None."""
    for pattern, parser in _REASONING_MODEL_PARSERS.items():
        if pattern in model:
            return parser
    return None


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = VLLM_STARTUP_TIMEOUT) -> bool:
    """Poll the vLLM server until it responds or timeout is reached."""
    import urllib.request
    import urllib.error

    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    start = time.monotonic()
    last_log = start
    while time.monotonic() - start < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            now = time.monotonic()
            if now - last_log >= 15:
                elapsed = int(now - start)
                logger.info(
                    f"Still waiting for vLLM server... ({elapsed}s elapsed, "
                    f"timeout {int(timeout)}s)"
                )
                last_log = now
            time.sleep(3)
    return False


def _launch_vllm_server(
    model: str,
    port: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "auto",
    ]

    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    # MoE models benefit from expert parallelism
    if "A3B" in model or "A22B" in model:
        cmd.append("--enable-expert-parallel")

    # Reasoning models: separate CoT from final answer via reasoning parser
    reasoning_parser = _detect_reasoning_parser(model)
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Launching vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
    )
    return proc


def _shutdown_server(proc: subprocess.Popen) -> None:
    """Gracefully shut down the vLLM server subprocess."""
    if proc.poll() is not None:
        return
    logger.info("Shutting down vLLM server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        logger.warning("Server didn't stop gracefully, sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _make_caller(
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int = 256,
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
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content
                return text.strip() if text else ""

            except (APIError, APIConnectionError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"API error (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait}s: {exc}"
                )
                time.sleep(wait)
            except Exception as exc:
                wait = 2 ** attempt
                logger.error(
                    f"Unexpected error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}"
                )
                time.sleep(wait)

        logger.error(f"Failed after {MAX_RETRIES} retries")
        return ""

    return call_api


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

class InferenceTimer:
    """Collect per-request latencies and compute summary statistics."""

    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.n_images: list[int] = []
        self.wall_start: float | None = None
        self.wall_end: float | None = None

    def wrap(self, call_api):
        """Return a wrapped ``call_api`` that records timing per call."""
        timer = self

        def timed_call(prompt: str, image_data: list[tuple[str, str]]) -> str:
            if timer.wall_start is None:
                timer.wall_start = time.monotonic()
            t0 = time.monotonic()
            result = call_api(prompt, image_data)
            elapsed = time.monotonic() - t0
            timer.latencies.append(elapsed)
            timer.n_images.append(len(image_data))
            timer.wall_end = time.monotonic()
            return result

        return timed_call

    def summary(self) -> dict[str, Any]:
        """Return a dict of timing statistics."""
        if not self.latencies:
            return {}
        n = len(self.latencies)
        total = sum(self.latencies)
        wall = (self.wall_end - self.wall_start) if self.wall_start and self.wall_end else total
        sorted_lat = sorted(self.latencies)
        return {
            "n_requests": n,
            "total_inference_s": round(total, 2),
            "wall_clock_s": round(wall, 2),
            "mean_latency_s": round(total / n, 3),
            "median_latency_s": round(sorted_lat[n // 2], 3),
            "p95_latency_s": round(sorted_lat[int(n * 0.95)], 3),
            "min_latency_s": round(sorted_lat[0], 3),
            "max_latency_s": round(sorted_lat[-1], 3),
            "throughput_req_per_s": round(n / wall, 3) if wall > 0 else 0,
            "mean_images_per_request": round(sum(self.n_images) / n, 1),
        }

    def print_report(self, model_name: str) -> None:
        """Print a formatted timing report to stdout."""
        s = self.summary()
        if not s:
            return
        print("\n" + "=" * 60)
        print("INFERENCE TIMING")
        print("=" * 60)
        print(f"Model:                {model_name}")
        print(f"Total requests:       {s['n_requests']}")
        print(f"Wall-clock time:      {s['wall_clock_s']:.1f}s")
        print(f"Sum of latencies:     {s['total_inference_s']:.1f}s")
        print(f"Mean latency:         {s['mean_latency_s']:.3f}s")
        print(f"Median latency:       {s['median_latency_s']:.3f}s")
        print(f"P95 latency:          {s['p95_latency_s']:.3f}s")
        print(f"Min / Max:            {s['min_latency_s']:.3f}s / {s['max_latency_s']:.3f}s")
        print(f"Throughput:           {s['throughput_req_per_s']:.2f} req/s")
        print(f"Avg images/request:   {s['mean_images_per_request']:.1f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_common_parser(
        description="Evaluate any vLLM-supported VLM locally on EgoDyn-Bench",
        default_model="Qwen/Qwen3-VL-8B-Instruct",
        api_key_env_var="VLLM_API_KEY",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="vLLM server base URL (default: auto-launch on free port)",
    )
    parser.add_argument(
        "--no_launch", action="store_true",
        help="Don't launch vLLM server; connect to --base_url instead",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Maximum context length (vLLM --max-model-len). "
             "Reduce if running out of GPU memory.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max output tokens per request. Defaults to 256, or 4096 "
             "for reasoning models (e.g. Cosmos-Reason2) to avoid "
             "truncating chain-of-thought.",
    )
    parser.add_argument(
        "--gpu_name", type=str, default=None,
        help="GPU identifier for timing data (e.g. 'RTX_5090', 'Jetson_Thor'). "
             "Auto-detected from nvidia-smi if not provided.",
    )
    parser.add_argument(
        "--vllm_args", type=str, nargs="*", default=None,
        help="Extra arguments passed directly to vLLM server",
    )
    args = parser.parse_args()

    # Auto-select max_tokens for reasoning models
    if args.max_tokens is None:
        args.max_tokens = 4096 if _detect_reasoning_parser(args.model) else 256

    # --- server management -------------------------------------------------
    vllm_proc = None
    port = None

    if args.no_launch:
        if not args.base_url:
            args.base_url = "http://localhost:8000/v1"
        logger.info(f"Connecting to existing vLLM server at {args.base_url}")
    else:
        port = _find_free_port()
        args.base_url = f"http://localhost:{port}/v1"

        vllm_proc = _launch_vllm_server(
            model=args.model,
            port=port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            extra_args=args.vllm_args,
        )

        logger.info(
            f"Waiting for vLLM server (PID {vllm_proc.pid}) on port {port}..."
        )
        if not _wait_for_server(args.base_url):
            logger.error(
                f"vLLM server failed to start within {VLLM_STARTUP_TIMEOUT}s. "
                "Check GPU memory and model compatibility."
            )
            _shutdown_server(vllm_proc)
            return 1
        logger.info("vLLM server is ready.")

    try:
        client = OpenAI(
            base_url=args.base_url,
            api_key="EMPTY",  # vLLM doesn't require auth by default
            max_retries=0,
            timeout=300.0,
        )

        raw_caller = _make_caller(
            client, args.model, args.temperature, max_tokens=args.max_tokens,
        )

        timer = InferenceTimer()
        call_api = timer.wrap(raw_caller)

        ret = run_evaluation(args, call_api)

        # --- timing report -------------------------------------------------
        timer.print_report(args.model)
        timing_data = timer.summary()

        if timing_data and args.metrics_output:
            # Tag timing with GPU hardware
            gpu_name = args.gpu_name
            if not gpu_name:
                try:
                    gpu_name = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name",
                         "--format=csv,noheader,nounits"],
                        text=True,
                    ).strip().split("\n")[0]
                except Exception:
                    gpu_name = "unknown"
            timing_data["gpu"] = gpu_name

            metrics_path = Path(args.metrics_output)
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            metrics["inference_timing"] = timing_data
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
                f.write("\n")
            logger.info(f"Timing data appended to {metrics_path}")

        return ret

    finally:
        if vllm_proc is not None:
            _shutdown_server(vllm_proc)


if __name__ == "__main__":
    sys.exit(main())
