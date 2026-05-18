# Evaluation

## Overview

The evaluation module scores VLM predictions against deterministic oracle labels using three metric families:

1. **Balanced Accuracy** -- mean per-class recall (robust to residual class imbalance)
2. **Macro F1** -- class-balanced F1 (each label weighted equally)
3. **Weighted Physics Consistency Rate (WPCR)** -- fraction of clips with zero kinematic contradictions, weighted by rule coverage

Metrics are computed at global, per-category, per-question, and per-source (nuScenes vs CARLA) granularity. All implementations are pure Python (no sklearn dependency).

---

## Prediction File Format

All evaluation scripts produce a JSONL file with one record per question:

```json
{
  "clip_id": "clip_00042",
  "question_id": "braking_intensity",
  "category": "direct_dynamics",
  "oracle_label": "moderate",
  "model_answer": "The braking appears to be moderate based on the deceleration."
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `clip_id` | yes | Clip identifier (`clip_*` = nuScenes, otherwise CARLA) |
| `question_id` | yes | Must match a key in `questions_template.yaml` |
| `category` | no | Used for per-category breakdown (defaults to `unknown`) |
| `oracle_label` or `answer` | yes | Ground-truth label; `oracle_label` takes precedence |
| `model_answer` | yes | Raw model output (free text) |

---

## Running Evaluation

> **Data path notes.** The CLI examples below show only the essential flags
> for clarity. In practice, add `--nuscenes_root /path/to/nuscenes/v1.0-trainval`
> to any command that evaluates on nuScenes clips. CARLA video paths are read
> from `EGODYN_CARLA_TRANSFERRED_DIR` (set as an env var) or `--carla_video_dir`.
> See [SETUP.md → Data sources](SETUP.md#data-sources) for the full path
> convention.

### CLI

```bash
# Print metrics to stdout
python scripts/evaluate.py --predictions generated/model_answers.jsonl

# Write metrics to file
python scripts/evaluate.py \
    --predictions generated/model_answers.jsonl \
    --output results/model_metrics.json

# Use a different question config
python scripts/evaluate.py \
    --predictions generated/model_answers.jsonl \
    --config path/to/questions_template.yaml
```

### Python API

```python
from evaluation.parsers import load_question_config
from evaluation.metrics import evaluate

question_config = load_question_config("dataset/configs/questions_template.yaml")

records = [
    {
        "clip_id": "clip_00042",
        "question_id": "braking_intensity",
        "category": "direct_dynamics",
        "oracle_label": "moderate",
        "model_answer": "The braking is moderate.",
    },
    # ... more records
]

result = evaluate(records, question_config)
print(result["global"]["balanced_acc"])    # 0.45
print(result["global"]["macro_f1"])        # 0.40
print(result["consistency"]["wemcr"])      # 0.85
```

### Build Leaderboard

After running evaluations for multiple models, consolidate all results:

```bash
python scripts/build_leaderboard.py
```

This reads every `*.jsonl` in `generated/`, runs the full evaluation pipeline, and writes `leaderboard/results.json`.

---

## Output Structure

The `evaluate()` function returns a nested dict:

```json
{
  "n_total": 42000,
  "n_parsed": 41500,
  "parsable_coverage": 0.9881,
  "global": {
    "accuracy": 0.54,
    "balanced_acc": 0.45,
    "macro_f1": 0.40,
    "n": 41500
  },
  "temporal": {
    "accuracy": 0.48,
    "balanced_acc": 0.43,
    "macro_f1": 0.38,
    "n": 12000
  },
  "per_category": {
    "direct_dynamics": { "accuracy": 0.55, "balanced_acc": 0.46, "macro_f1": 0.41, "n": 33000 },
    "comparative":     { "accuracy": 0.50, "balanced_acc": 0.42, "macro_f1": 0.37, "n": 8500 }
  },
  "per_question": {
    "braking_intensity": {
      "accuracy": 0.40,
      "balanced_acc": 0.38,
      "macro_f1": 0.35,
      "n": 3000,
      "confusion_matrix": {
        "labels": ["emergency", "moderate", "low", "none"],
        "matrix": [[ ... ]]
      }
    }
  },
  "consistency": {
    "rate": 0.85,
    "wemcr": 0.83,
    "mean_compliance": 0.95,
    "rule_coverage": 0.98,
    "n_clips": 3000,
    "n_evaluable": 2500,
    "n_consistent": 2125,
    "consistency_coverage": 0.833,
    "mean_violations": 0.18,
    "per_rule": {
      "heading_change_implies_turning": { "n_applicable": 800, "n_violations": 32, "compliance": 0.96 }
    }
  },
  "per_source": {
    "nuscenes": { "accuracy": 0.56, "balanced_acc": 0.47, "macro_f1": 0.42, "n": 21000, "consistency": { ... } },
    "carla":    { "accuracy": 0.52, "balanced_acc": 0.43, "macro_f1": 0.38, "n": 20500, "consistency": { ... } }
  }
}
```

---

## Answer Parsing

The parser (`evaluation/parsers.py`) maps free-text model output to canonical labels using a 4-stage cascade:

1. **Exact match** after normalization (lowercase, strip, collapse whitespace, strip markdown formatting)
2. **Underscore/space equivalence** (`first_half` matches `first half`)
3. **Last-line extraction** -- models that ignore "answer only" instructions often put their answer on the last line; this is checked before scanning potentially noisy reasoning text
4. **Substring match with word boundaries** (longest-first to avoid partial collisions; `\b` prevents "no" matching inside "cannot")

For numeric questions, the first number is extracted via regex and normalized (e.g., `42.0` becomes `42`).

Unparsable answers are excluded from metrics and counted in `n_total - n_parsed`.

---

## Consistency Rules

Ten hard kinematic rules detect physics contradictions in model predictions across questions within the same clip:

| Rule | Condition | Implication |
|------|-----------|-------------|
| `heading_change_implies_turning` | Heading change = `yes` | Turn direction != `straight` |
| `lateral_accel_implies_turning` | High lateral accel = `yes` | Turn direction != `straight` |
| `straight_implies_no_heading_change` | Turn direction = `straight` | Heading change = `no` |
| `straight_implies_no_high_lateral_accel` | Turn direction = `straight` | High lateral accel = `no` |
| `highway_implies_not_low_speed` | Speed regime = `highway` | Mean speed low = `no` |
| `stopped_implies_low_speed` | Speed regime = `stopped` | Mean speed low = `yes` |
| `stopped_not_accelerating` | Speed regime = `stopped` | Speed trend != `accelerating` |
| `brake_then_turn_implies_braking` | Brake-then-turn = `yes` | Braking intensity != `none` |
| `brake_then_turn_implies_turning` | Brake-then-turn = `yes` | Turn direction != `straight` |
| `stop_and_go_not_stopped` | Stop-and-go = `yes` | Speed regime != `stopped` |

Only clips where at least one rule is evaluable (both questions answered and condition triggered) contribute to the WPCR score. The WPCR is additionally weighted by **rule coverage** (fraction of rules that triggered at least once) to penalize degenerate answer distributions that trivially avoid violations.

---

## VLM Evaluators

### Setup

All evaluators load API keys from `.env` in the project root:

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=...
#   ANTHROPIC_API_KEY=sk-ant-...
```

### Common Arguments

All evaluators share the same CLI via `evaluation/evaluator_common.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--selected_clips` | - | Path to `selected_clips.json` (primary input) |
| `--output` | *(required)* | Output predictions JSONL |
| `--model` | varies | Model name or HuggingFace ID |
| `--num_frames` | `10` | Frames per clip (evenly spaced from available frames) |
| `--trajectory_mode` | `summary` | `none` / `summary` / `timeseries` / `coordinates` / `full` |
| `--no_trajectory` | false | Shorthand for `--trajectory_mode none` (vision-only) |
| `--no_images` | false | Text-only ablation (omit images) |
| `--carla_video_source` | `transferred` | `simulation` (raw CARLA) or `transferred` (Cosmos) |
| `--group_by_clip` | false | Group all 14 questions per clip into one API call (~14x cheaper) |
| `--temperature` | `0.0` | Sampling temperature |
| `--resume` | false | Skip QA items already in the output file |
| `--run_eval` | false | Run evaluation after inference and print metrics |
| `--metrics_output` | - | Write metrics JSON to file (implies `--run_eval`) |
| `--max_samples` | - | Limit total QA items (for testing) |
| `--frame_detail` | `low` | Image detail level (OpenAI-specific, ignored by others) |
| `--shuffle_frames` | false | Ablation: randomly shuffle frame order |
| `--overlay_flow` | false | Ablation: overlay dense optical flow on frames |
| `--flow_alpha` | `0.5` | Blending alpha for flow overlay |

### Data paths

By default, evaluators look for QA and clip data at:
- `output/nuscenes_clips/qa.jsonl` and `output/carla_clips/qa.jsonl`
- `output/nuscenes_clips/clips_index.jsonl` and `output/carla_clips/clips_index.jsonl`

These can be overridden with `--nuscenes_qa`, `--carla_qa`, `--nuscenes_index`, `--carla_index`.

---

## Batch API Evaluators

For large-scale evaluation, batch APIs offer 50% cost savings and no rate-limit pressure. All three providers follow the same 4-step workflow:

### Workflow

```
prepare  →  submit  →  status  →  collect
```

1. **prepare** -- encodes images, builds request files, splits into chunks
2. **submit** -- uploads batch files to the provider
3. **status** -- polls batch job status (re-run until all complete)
4. **collect** -- downloads results, merges into a single JSONL, optionally runs evaluation

### OpenAI Batch

```bash
pip install openai

# Prepare (splits into files of --max_requests_per_file requests each)
python evaluation/evaluate_openai_batch.py prepare \
    --selected_clips selected_clips.json \
    --model gpt-4o-mini \
    --trajectory_mode none \
    --batch_dir generated/batch_gpt4o_mini

# Submit (use --limit and --wait for quota management)
python evaluation/evaluate_openai_batch.py submit \
    --batch_dir generated/batch_gpt4o_mini \
    --limit 5 --wait

# Check status
python evaluation/evaluate_openai_batch.py status \
    --batch_dir generated/batch_gpt4o_mini

# Collect results
python evaluation/evaluate_openai_batch.py collect \
    --batch_dir generated/batch_gpt4o_mini \
    --output generated/gpt4o_mini_answers.jsonl \
    --run_eval --metrics_output results/gpt4o_mini.json
```

### Gemini Batch

```bash
pip install google-genai

python evaluation/evaluate_gemini_batch.py prepare \
    --selected_clips selected_clips.json \
    --model gemini-2.5-flash \
    --trajectory_mode none \
    --batch_dir generated/batch_gemini_flash

python evaluation/evaluate_gemini_batch.py submit \
    --batch_dir generated/batch_gemini_flash \
    --limit 10 --wait

python evaluation/evaluate_gemini_batch.py status \
    --batch_dir generated/batch_gemini_flash

python evaluation/evaluate_gemini_batch.py collect \
    --batch_dir generated/batch_gemini_flash \
    --output generated/gemini_flash_answers.jsonl \
    --run_eval --metrics_output results/gemini_flash.json
```

**Note:** Gemini batch API has per-model enqueued token quotas. Use `--limit` to control how many batch files are submitted at once to stay within quota.

### Claude Batch

```bash
pip install anthropic

python evaluation/evaluate_claude_batch.py prepare \
    --selected_clips selected_clips.json \
    --model claude-sonnet-4-5-20250929 \
    --trajectory_mode none \
    --batch_dir generated/batch_claude_sonnet

python evaluation/evaluate_claude_batch.py submit \
    --batch_dir generated/batch_claude_sonnet

python evaluation/evaluate_claude_batch.py status \
    --batch_dir generated/batch_claude_sonnet

python evaluation/evaluate_claude_batch.py collect \
    --batch_dir generated/batch_claude_sonnet \
    --output generated/claude_sonnet_answers.jsonl \
    --run_eval --metrics_output results/claude_sonnet.json
```

### HuggingFace Inference Router

For models available via the HuggingFace inference API (Kimi K2.5, etc.):

```bash
python evaluation/evaluate_moonshot.py \
    --selected_clips selected_clips.json \
    --model moonshotai/Kimi-K2.5:novita \
    --no_trajectory --resume \
    --output generated/kimi_k25_answers.jsonl \
    --run_eval
```

---

## Local Evaluation with vLLM

`evaluate_vllm_local.py` evaluates any [vLLM-supported](https://docs.vllm.ai/en/latest/models/supported_models/) vision-language model locally. It auto-launches a vLLM server as a managed subprocess, runs inference via the OpenAI-compatible API, shuts it down, and reports timing metrics.

### Prerequisites

```bash
pip install vllm>=0.11.0 openai
```

### Usage

```bash
# Single GPU (auto-launches vLLM)
python evaluation/evaluate_vllm_local.py \
    --selected_clips selected_clips.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --max_model_len 16384 \
    --no_trajectory --resume \
    --output generated/qwen3vl_8b_answers.jsonl \
    --run_eval --metrics_output results/qwen3vl_8b.json

# Multi-GPU with tensor parallelism
python evaluation/evaluate_vllm_local.py \
    --selected_clips selected_clips.json \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor_parallel_size 2 --max_model_len 16384 \
    --no_trajectory --resume \
    --output generated/qwen3vl_30b_answers.jsonl

# Connect to an already-running vLLM server
python evaluation/evaluate_vllm_local.py \
    --selected_clips selected_clips.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --base_url http://localhost:8000/v1 --no_launch \
    --no_trajectory --resume \
    --output generated/qwen3vl_8b_answers.jsonl
```

### vLLM-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_model_len` | - | Maximum KV cache length (required, caps VRAM usage) |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu_memory_utilization` | `0.9` | Fraction of GPU memory for KV cache |
| `--base_url` | auto | vLLM server URL (auto-assigned if launching) |
| `--no_launch` | false | Connect to an existing vLLM server instead of launching one |

### Tested Models

| Model | HuggingFace ID | VRAM (approx) | Single GPU (32GB)? |
|---|---|---|---|
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | ~17GB | Yes |
| Qwen3-VL-30B (MoE) | `Qwen/Qwen3-VL-30B-A3B-Instruct` | ~60GB | No (2x GPU) |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | ~17GB | Yes |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | ~17GB | Yes |
| InternVL3.5-38B | `OpenGVLab/InternVL3_5-38B` | ~76GB | No (2x GPU) |

Inference timing metrics (mean/median/P95 latency, throughput) are printed after evaluation and saved to `--metrics_output` under the `"inference_timing"` key.

---

## Visual Baselines

Geometric baselines estimate dynamics from visual motion cues without access to sensor telemetry. They answer a subset of 6 questions: `yaw_rate_turn_direction`, `speed_trend`, `stop_and_go`, `brake_then_turn`, `significant_heading_change`, `high_lateral_accel`.

```bash
# Visual odometry proxy
python -m baselines vo_proxy \
    --selected_clips selected_clips.json \
    --output generated/vo_proxy_answers.jsonl

# Optical flow heuristic
python -m baselines flow_heuristic \
    --selected_clips selected_clips.json \
    --output generated/flow_heuristic_answers.jsonl

# RAFT-based optical flow
python -m baselines raft_flow_heuristic \
    --selected_clips selected_clips.json \
    --output generated/raft_flow_answers.jsonl

# TartanVO visual odometry
python -m baselines tartanvo \
    --selected_clips selected_clips.json \
    --output generated/tartanvo_answers.jsonl
```

Baselines support `--carla_video_source simulation|transferred` to evaluate on both CARLA visual domains.

---

## Adding a New Evaluator

The fastest way is to copy [`evaluation/evaluate_example.py`](../evaluation/evaluate_example.py) -- a heavily commented template that runs as-is against the dummy "always answer yes" model. Swap the `call_my_model` body with your inference call and you have a working evaluator.

The contract is small:

1. Build the CLI parser with `build_common_parser(description, default_model, api_key_env_var)` -- this hands you every shared flag (`--selected_clips`, `--qa_jsonl`, `--num_frames`, `--trajectory_mode`, `--resume`, `--run_eval`, ...).
2. Implement one closure: `call_api(prompt: str, image_data: list[tuple[str, str]]) -> str` where each `image_data` element is `(base64_string, mime_type)`.
3. Hand both to `run_evaluation(args, call_api)`.

The shared `run_evaluation()` handles data loading, prompt construction, frame extraction, progress tracking, resume logic, and optional post-hoc evaluation. You don't need to touch any of that.

---

## Submitting Your Model to the Leaderboard

The leaderboard is regenerated by [`scripts/build_leaderboard.py`](../scripts/build_leaderboard.py), which reads every `generated/*.jsonl` and consolidates the metrics into `leaderboard/results.json` plus per-model files under `results/`. A submission is two files plus a PR.

### What you submit

| File | Produced by | Required |
|---|---|---|
| `generated/<your_model>_answers.jsonl` | your evaluator (the JSONL contract above) | yes |
| `results/<your_model>.json` | `--run_eval --metrics_output ...` on your evaluator, or `scripts/build_leaderboard.py` | yes |
| `evaluation/evaluate_<your_model>.py` | your evaluator script (optional but recommended) | preferred |

`<your_model>` becomes the model key in the leaderboard. Use lowercase snake_case (e.g. `qwen3vl_8b`, `gemini2_flash`, `internvl3_8b_w_traj`). Suffix `_w_traj` if your run includes trajectory text -- the website figures group models by this suffix and won't render correctly without it.

### End-to-end flow

```bash
# 1. (Once) Generate the QA file for the curated benchmark clips.
#    The released selected_clips.json references QAs that live in
#    output/{nuscenes,carla}_clips/qa.jsonl — produced by:
python dataset/scripts/generate_qa.py \
    --clips_index output/nuscenes_clips/clips_index.jsonl \
    --output output/nuscenes_clips/qa.jsonl
python dataset/scripts/generate_qa.py \
    --clips_index output/carla_clips/clips_index.jsonl \
    --output output/carla_clips/qa.jsonl

# 2. Run your evaluator over the full 1000 clips × 14 questions = 14000 rows.
python evaluation/evaluate_<your_model>.py \
    --selected_clips selected_clips.json \
    --output generated/<your_model>_answers.jsonl \
    --resume --run_eval \
    --metrics_output results/<your_model>.json

# 3. Rebuild the consolidated leaderboard. This step computes BAcc / F1 /
#    WPCR / per-rule compliance / per-source slices for every model,
#    keeping your new entry in sync with the others.
python scripts/build_leaderboard.py

# 4. Open a PR with the three files (evaluator script, answers JSONL, results JSON).
```

The `--resume` flag means partial runs can be restarted cheaply; the harness skips QA rows that already appear in the output JSONL. Plan for ~14 000 inference calls per model.

### PR checklist

- [ ] `generated/<your_model>_answers.jsonl` has 14 000 rows (or `n_total == 14000` in `results/<your_model>.json`).
- [ ] `results/<your_model>.json` was produced by `scripts/build_leaderboard.py` after running your evaluator (so per-rule and per-source fields are present).
- [ ] If you added an `evaluation/evaluate_<your_model>.py`, it follows the [`evaluate_example.py`](../evaluation/evaluate_example.py) pattern -- one `call_api` closure, parser from `build_common_parser`, entry through `run_evaluation`.
- [ ] Model name uses lowercase snake_case; trajectory-augmented runs end in `_w_traj`.
- [ ] PR description includes the model identifier (HuggingFace ID, API model string, or local checkpoint hash), hardware used, and an estimate of total inference time.

If you can't share the model itself (closed weights, license restrictions), the answers JSONL + results JSON are enough on their own. The leaderboard stays reproducible for everyone who has the same model access; the evaluator script is the artifact that lets others verify the prompt assembly.
