# EgoDyn-Bench

**A physics-grounded VQA benchmark for evaluating Vision-Language Models on trajectory-based dynamics reasoning in autonomous driving.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

---

## Key Idea

Existing driving VQA datasets (DriveLM, etc.) can be solved from vision alone.
EgoDyn-Bench enforces **trajectory dependency**: the same scene with a different trajectory must produce a different answer.

The benchmark covers 14 question types across direct dynamics, comparative, and temporal categories, evaluated with Balanced Accuracy, Macro F1, and Weighted Physics Consistency Rate (WPCR).

---

## What's in the Benchmark

- **1,000 curated clips** (500 nuScenes + 500 CARLA-Cosmos-transferred), 3 seconds each at 10 Hz.
- **14 question types** × 1,000 clips ≈ **14,000 QA pairs**, with deterministic oracle labels derived from sensor telemetry (no manual annotation).
- **Two visual domains**: real-world (nuScenes) and photorealistic synthetic (CARLA frames passed through NVIDIA Cosmos Transfer 2.5).
- **Released artifacts:** `selected_clips.json` (the 1k clip spec with per-clip dynamics features and oracle answers), the evaluation harness, baselines, and a consolidated leaderboard.

> **Note on raw data:** nuScenes is licensed CC BY-NC-SA 4.0, so only derived QA + dynamics arrays are shipped here. You download nuScenes separately from <https://www.nuscenes.org/> and point the harness at your local copy.

---

## Quick Start: Evaluate Your Model

```bash
# 1. Environment
conda env create -f environment.yml && conda activate dynamics-benchmark

# 2. Download the benchmark data (~3-5 GB) from Hugging Face
pip install -U "huggingface_hub[cli]"
hf download fnc1901/EgoDyn-Bench --repo-type=dataset --local-dir data/egodyn-bench

# 3. Tell the harness where the data lives (one-off; see docs/SETUP.md for all paths)
export EGODYN_CARLA_TRANSFERRED_DIR=./data/egodyn-bench/carla_videos_transferred
# nuScenes you download yourself from nuscenes.org — pass via --nuscenes_root per command

# 4. Run your model — example with a local Qwen3-VL via vLLM
python evaluation/evaluate_vllm_local.py \
    --selected_clips selected_clips.json \
    --nuscenes_root /path/to/nuscenes \
    --model Qwen/Qwen3-VL-8B-Instruct --max_model_len 16384 \
    --no_trajectory --resume \
    --output generated/qwen3vl_8b_answers.jsonl \
    --metrics_output results/qwen3vl_8b.json --run_eval

# 5. Or score an existing predictions JSONL standalone
python scripts/evaluate.py --predictions generated/qwen3vl_8b_answers.jsonl

# 6. Inspect failures interactively
jupyter lab analysis/notebooks/failure_analysis.ipynb
```

To add your own model: copy [`evaluation/evaluate_example.py`](evaluation/evaluate_example.py), swap the `call_my_model` stub for your inference call, and run. See [docs/EVALUATION.md → Submitting Your Model](docs/EVALUATION.md#submitting-your-model-to-the-leaderboard) for the full submission flow.

---

## Model Evaluation

All evaluation scripts share the same interface via `evaluation/evaluator_common.py` and expect `selected_clips.json` as input. Each writes a JSONL of predictions and (optionally) a `results/<model>.json` with all metrics.

### Cloud API models

Each cloud provider has a batch evaluator with prepare/submit/collect stages:

```bash
# OpenAI (GPT-4o, GPT-4.1, GPT-5.1)
python evaluation/evaluate_openai_batch.py prepare --selected_clips selected_clips.json ...
python evaluation/evaluate_openai_batch.py submit  --batch_dir generated/batch_gpt4o/ ...

# Google Gemini (2.0 Flash, 3 Pro)
python evaluation/evaluate_gemini_batch.py prepare --selected_clips selected_clips.json ...
python evaluation/evaluate_gemini_batch.py submit  --batch_dir generated/batch_gemini/ ...

# Anthropic Claude (Sonnet, Opus)
python evaluation/evaluate_claude_batch.py prepare --selected_clips selected_clips.json ...
python evaluation/evaluate_claude_batch.py submit  --batch_dir generated/batch_claude/ ...

# HuggingFace inference router (Kimi K2.5, etc.)
python evaluation/evaluate_moonshot.py \
    --selected_clips selected_clips.json \
    --model moonshotai/Kimi-K2.5:novita \
    --no_trajectory --resume \
    --output generated/kimi_k25_answers.jsonl
```

### Local models via vLLM

`evaluate_vllm_local.py` evaluates any [vLLM-supported](https://docs.vllm.ai/en/latest/models/supported_models/) vision-language model locally. It auto-launches a vLLM server, runs inference, shuts it down, and reports timing metrics.

**Tested models:**

| Model | HuggingFace ID | VRAM (approx) | Single GPU (32GB)? |
|---|---|---|---|
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | ~17GB | Yes |
| Qwen3-VL-30B (MoE) | `Qwen/Qwen3-VL-30B-A3B-Instruct` | ~60GB | No (2x GPU) |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | ~17GB | Yes |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | ~17GB | Yes |
| InternVL3.5-38B | `OpenGVLab/InternVL3_5-38B` | ~76GB | No (2x GPU) |

```bash
# Auto-launch vLLM and evaluate (single GPU)
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

**Key flags:** `--max_model_len 16384` (caps KV cache to fit in VRAM), `--tensor_parallel_size N` (multi-GPU), `--no_launch` (use existing server), `--gpu_memory_utilization 0.9` (default).

### Visual baselines

Geometric baselines estimate dynamics from visual motion cues (optical flow, visual odometry) without access to sensor telemetry:

```bash
# Visual odometry proxy baseline
python -m baselines vo_proxy \
    --selected_clips selected_clips.json \
    --output generated/vo_proxy_answers.jsonl

# Optical flow heuristic
python -m baselines flow_heuristic \
    --selected_clips selected_clips.json \
    --output generated/flow_heuristic_answers.jsonl
```

### Build leaderboard

After running evaluations, consolidate all results into a single leaderboard:

```bash
python scripts/build_leaderboard.py
```

This reads every `*.jsonl` in `generated/`, runs the full evaluation pipeline (parsing + metrics + consistency), and writes `leaderboard/results.json`.

---

## Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| Balanced Accuracy | Global, per-category, per-question | Mean per-class recall (robust to class imbalance) |
| Macro F1 | Global, per-category, per-question | Class-balanced F1 score |
| WPCR | Per-clip | Weighted Physics Consistency Rate -- fraction of clips with zero kinematic contradictions, weighted by rule coverage |
| Confusion Matrix | Per-question | Full label-vs-label counts |
| Parsable Coverage | Global | Fraction of model answers successfully parsed |

---

## Rebuilding the Benchmark from Scratch

The full pipeline that produces `selected_clips.json` from raw nuScenes + CARLA data. Skip this section if you only want to evaluate a model against the released benchmark.

### Step 1: Obtain source data

EgoDyn-Bench combines two data sources:

- **nuScenes** -- real-world driving scenes with CAN bus telemetry.
  Download from [nuscenes.org](https://www.nuscenes.org/).
- **CARLA** -- simulated driving scenes recorded with the [Frenetix](https://github.com/TUM-AVS/Frenetix) planner.

See [docs/SETUP.md → Data sources](docs/SETUP.md#data-sources) for directory layout conventions and the `EGODYN_*` environment variables the scripts read.

### Step 2: Extract 3-second clips

Chunk continuous driving sequences into fixed-length 3-second clips (31 samples at 10 Hz). Each clip produces a `clips_index.jsonl` with per-clip metadata and feature arrays.

```bash
# nuScenes clips
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root /path/to/nuscenes \
    --output_dir output/nuscenes_clips

# CARLA clips
python dataset/scripts/extract_carla_clips.py \
    --carla_logs /path/to/frenetix_logs \
    --output_dir output/carla_clips \
    --carla_video_dir /path/to/videos \
    --require_video
```

**Output per source:**
- `clips_index.jsonl` -- one record per clip with dynamics features
- `arrays/<clip_id>.npz` -- numeric arrays (speed, accel, yaw_rate, ...)
- `metadata.json` -- dataset-level metadata

### Step 3 (optional): Cosmos style transfer

CARLA frames can be transformed into photorealistic images using [NVIDIA Cosmos Transfer](https://github.com/NVIDIA/Cosmos). This produces a second visual domain (Cosmos-transferred) alongside the raw simulation frames, enabling domain gap analysis. See `scripts/prepare_carla_cosmos.sh` and `scripts/slurm_cosmos_transfer.sh` for the SLURM-driven prep + transfer pipeline.

### Step 4: Generate QA pairs

Apply the labeling rules defined in `dataset/configs/questions_template.yaml` to each clip's dynamics features. This produces one QA pair per question per clip, with deterministic ground-truth labels derived from sensor data.

```bash
# nuScenes QA
python dataset/scripts/generate_qa.py \
    --clips_index output/nuscenes_clips/clips_index.jsonl \
    --questions_config dataset/configs/questions_template.yaml \
    --output_qa_jsonl output/nuscenes_clips/qa.jsonl

# CARLA QA
python dataset/scripts/generate_qa.py \
    --clips_index output/carla_clips/clips_index.jsonl \
    --questions_config dataset/configs/questions_template.yaml \
    --output_qa_jsonl output/carla_clips/qa.jsonl
```

### Step 5 (optional): Calibrate thresholds

Inspect the feature distributions across the full clip pool and adjust the classification thresholds in `questions_template.yaml` for better answer-class balance.

```bash
python dataset/scripts/calibrate_thresholds.py \
    --clips-index output/nuscenes_clips/clips_index.jsonl \
                  output/carla_clips/clips_index.jsonl \
    --questions-config dataset/configs/questions_template.yaml \
    --output-report docs/threshold_calibration_report.md
```

If thresholds are changed, re-run Step 4 to regenerate QA labels.

### Step 6: Select balanced clip subset

Select a balanced subset of clips from the full QA pool using a greedy algorithm that minimizes answer-class imbalance across all 14 questions.

```bash
python scripts/select_balanced_clips.py \
    --target 1000 \
    --min-nuscenes-frac 0.5 \
    --output selected_clips.json
```

The output `selected_clips.json` is the final clip list used by all evaluation scripts.

---

## Toolkit (Beyond the Benchmark)

The same pipeline that produces the curated 1,000-clip benchmark also supports research workflows that go beyond evaluation:

- **Training-data splits.** `dataset/scripts/build_splits.py` + `dataset/generation/split_builder.py` produce stratified train/val splits from any clip pool. Useful if you want to fine-tune on EgoDyn-style QA. See [docs/TASK3_SPLITS.md](docs/TASK3_SPLITS.md).
- **Custom labelling rules.** Add a rule to `dataset/generation/labeling_rules.py` (registry-based, 12 rules to start from) and a question to `dataset/configs/questions_template.yaml` to extend the question set.
- **Interactive analysis.** `analysis/notebooks/statistics.ipynb` and `analysis/notebooks/failure_analysis.ipynb` are tutorial notebooks showing how to load and visualize the dataset and model results.
- **Clip viewer.** `scripts/clip_viewer.py` serves a local web UI for browsing any clip with its dynamics arrays + frames.

These are independent of the benchmark itself — using them is optional.

---

## Repository Structure

```
egodyn-bench/
├── dataset/
│   ├── configs/
│   │   └── questions_template.yaml      # 14 question templates with labeling rules
│   ├── generation/
│   │   ├── nuscenes_extract.py          # nuScenes clip extraction logic
│   │   ├── dynamics_features.py         # Speed, accel, yaw-rate, jerk computation
│   │   ├── config_loader.py             # YAML config loader with validation
│   │   ├── labeling_rules.py            # 14 labeling rules (registry pattern)
│   │   ├── qa_generator.py              # QA dataset generation
│   │   └── split_builder.py             # Train/val split builder (toolkit)
│   ├── scripts/
│   │   ├── extract_nuscenes_clips.py    # CLI: nuScenes clip extraction
│   │   ├── extract_carla_clips.py       # CLI: CARLA clip extraction
│   │   ├── generate_qa.py               # CLI: QA generation
│   │   ├── calibrate_thresholds.py      # CLI: threshold calibration
│   │   └── build_splits.py              # CLI: train/val split builder (toolkit)
│   └── tests/
├── evaluation/
│   ├── parsers.py                       # Free-text answer parsing (4-stage cascade)
│   ├── metrics.py                       # Balanced accuracy, F1, WPCR, confusion matrices
│   ├── evaluator_common.py              # Shared evaluation loop and CLI arguments
│   ├── evaluate_example.py              # ★ Template — copy-paste starting point
│   ├── evaluate_openai_batch.py         # OpenAI Batch API evaluator
│   ├── evaluate_gemini_batch.py         # Google Gemini Batch API evaluator
│   ├── evaluate_claude_batch.py         # Anthropic Claude Batch API evaluator
│   ├── evaluate_moonshot.py             # HuggingFace inference router evaluator
│   ├── evaluate_vllm_local.py           # Local vLLM evaluator (Qwen3-VL, InternVL3, etc.)
│   └── evaluate_drivemm.py              # DriveMM baseline evaluator
├── baselines/
│   ├── vo_proxy_baseline.py             # Visual odometry proxy baseline
│   ├── flow_heuristic.py                # Optical flow heuristic baseline
│   ├── raft_flow_heuristic.py           # RAFT-based optical flow baseline
│   └── tartanvo_baseline.py             # TartanVO visual odometry baseline
├── scripts/
│   ├── evaluate.py                      # CLI: evaluate a single predictions JSONL
│   ├── build_leaderboard.py             # Leaderboard builder
│   ├── select_balanced_clips.py         # Balanced clip subset selection
│   ├── bootstrap_confidence.py          # Bootstrap 95% confidence intervals
│   ├── threshold_sensitivity.py         # Threshold-sensitivity analysis (paper Appx)
│   ├── per_threshold_sensitivity.py     # Per-question sensitivity (paper Appx)
│   ├── visualize_website.py             # Leaderboard / website figures + page JSONs
│   ├── visualize_distributions.py       # Dataset distribution figures + page JSONs
│   ├── clip_viewer.py                   # Interactive clip browser (local web UI)
│   └── prepare_carla_cosmos.sh          # CARLA Cosmos sim-to-real prep
├── analysis/
│   └── notebooks/
│       ├── statistics.ipynb             # Tutorial: dataset & benchmark statistics
│       └── failure_analysis.ipynb       # Tutorial: model post-mortem
├── tests/                               # Unit tests for parsers, metrics, baselines
├── docs/                                # Documentation
├── leaderboard/results.json             # Consolidated leaderboard
├── selected_clips.json                  # The released 1k-clip benchmark spec
├── environment.yml                      # Conda environment spec
└── requirements.txt                     # pip requirements
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **Benchmark consumption** | |
| [docs/SETUP.md](docs/SETUP.md) | Environment setup, data sources, env vars |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics, answer parsing, how to submit a new model |
| [evaluation/evaluate_example.py](evaluation/evaluate_example.py) | Template evaluator — start here when adding a model |
| **Going deeper** | |
| [docs/DATASET_GENERATION.md](docs/DATASET_GENERATION.md) | Detailed dataset generation pipeline |
| [dataset/README.md](dataset/README.md) | Dataset format specification |
| [docs/threshold_calibration_report.md](docs/threshold_calibration_report.md) | How the labelling thresholds were calibrated |
| **Toolkit** | |
| [docs/TASK3_SPLITS.md](docs/TASK3_SPLITS.md) | Building train/val splits beyond the benchmark |
| [analysis/notebooks/](analysis/notebooks/) | Interactive tutorial notebooks |

---

## Citation

```bibtex
@inproceedings{schaefer2026egodyn,
  title={EgoDyn-Bench: Evaluating Ego-Motion Understanding in Vision-Centric Foundation Models for Autonomous Driving},
  author={Sch{\"a}fer, Finn Rasmus and Gao, Yuan and Wang, Dingrui and Stauner, Thomas and G{\"u}nnemann, Stephan and Piccinini, Mattia and Schmidt, Sebastian and Betz, Johannes},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).