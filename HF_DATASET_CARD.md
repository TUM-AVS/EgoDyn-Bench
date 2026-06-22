---
license: cc-by-nc-sa-4.0
task_categories:
  - visual-question-answering
  - video-classification
language:
  - en
tags:
  - autonomous-driving
  - dynamics-reasoning
  - vqa
  - vlm-benchmark
  - trajectory
  - carla
  - nuscenes
size_categories:
  - 1K<n<10K
pretty_name: EgoDyn-Bench
configs:
  - config_name: default
    data_files:
      - split: test
        path: selected_clips.json
---

# EgoDyn-Bench

**A physics-grounded VQA benchmark for evaluating Vision-Language Models on trajectory-based dynamics reasoning in autonomous driving.**

This repository contains the **data artifacts** for the benchmark. The evaluation harness, baselines, and reference implementations live in the companion [GitHub repository](https://github.com/TUM-AVS/EgoDyn-Bench).

> **Note on licensing.** The nuScenes-derived portion of this dataset is released under CC BY-NC-SA 4.0 to comply with nuScenes' upstream license. Raw nuScenes imagery is **not** redistributed here — users must download nuScenes themselves from <https://www.nuscenes.org/> and join via the `sample_token` references included in `nuscenes_clips/clips_index.jsonl`. The CARLA-derived portion (videos, dynamics arrays, QA) is permissively licensed.

---

## What this dataset is

EgoDyn-Bench enforces **trajectory dependency** in driving VQA: the same scene with a different trajectory must produce a different answer. Existing benchmarks (DriveLM, etc.) can largely be solved from vision alone — this one cannot.

- **1,000 curated 3-second clips** at 10 Hz (500 nuScenes + 500 CARLA-Cosmos-transferred)
- **14 question types** × 1,000 clips ≈ **14,000 QA pairs** with deterministic oracle labels derived from sensor telemetry
- **Two visual domains** per CARLA clip: raw simulation + photorealistic Cosmos-Transfer 2.5 sim-to-real
- **Per-clip dynamics arrays**: 31-sample sequences of speed, acceleration, yaw-rate, jerk, position, yaw, timestamps
- **Reference leaderboard** with 49 models evaluated end-to-end

---

## Repository layout

```
EgoDyn-Bench/
├── selected_clips.json                       # The 1000-clip benchmark spec
├── leaderboard.json                          # 49-model reference leaderboard
├── nuscenes_clips/
│   ├── clips_index.jsonl                     # Per-clip metadata + sample_tokens
│   ├── arrays/clip_*.npz                     # 31-sample dynamics arrays
│   └── qa.jsonl                              # Oracle QA pairs
├── carla_clips/
│   ├── clips_index.jsonl
│   ├── arrays/*.npz
│   └── qa.jsonl
├── carla_videos_simulation/                  # Raw CARLA Frenetix replays, 1280x720
│   └── <clip_id>.mp4                         # 500 clips
└── carla_videos_transferred/                 # Cosmos-Transfer 2.5 sim-to-real
    └── <clip_id>.mp4                         # 500 clips, paired with simulation
```

### File schemas

**`selected_clips.json`** — the canonical 1000-clip benchmark spec:
```json
{
  "id": "clip_19765",                      // or e.g. "DEU_Heilbronn-163_1_T-8__Balanced__w0"
  "source": "nuscenes",                    // or "carla"
  "features": {"mean_speed": 6.2, ...},    // per-clip dynamics summary
  "answers": {"yaw_rate_turn_direction": "left", ...}
}
```

**`{nuscenes,carla}_clips/clips_index.jsonl`** — one record per clip with metadata, timestamps, and (for nuScenes) `sample_token` joins to raw nuScenes:
```json
{
  "clip_id": "clip_19765",
  "scene_token": "...",
  "sample_tokens": ["...", "..."],
  "start_time": 0.0,
  "duration": 3.0
}
```

**`{nuscenes,carla}_clips/arrays/<clip_id>.npz`** — keys: `timestamps`, `position` (T,2), `yaw` (T,), `speed` (T,), `accel` (T,), `yaw_rate` (T,), `jerk` (T,) where T=31.

**`{nuscenes,carla}_clips/qa.jsonl`** — one row per (clip, question) pair:
```json
{
  "clip_id": "clip_19765",
  "question_id": "braking_intensity",
  "category": "direct_dynamics",
  "oracle_label": "moderate",
  "question": "How would you classify the braking intensity in this clip?",
  "choices": ["none", "low", "moderate", "emergency"]
}
```

---

## Quickstart

```bash
# 1. Download the dataset
pip install -U "huggingface_hub[cli]"
hf download fnc1901/EgoDyn-Bench --repo-type=dataset --local-dir data/egodyn-bench

# 2. Clone the evaluation harness
git clone https://github.com/TUM-AVS/EgoDyn-Bench.git
cd EgoDyn-Bench

# 3. Set up environment
conda env create -f environment.yml && conda activate dynamics-benchmark

# 4. Point the harness at the downloaded data
export EGODYN_CARLA_TRANSFERRED_DIR=$(pwd)/../data/egodyn-bench/carla_videos_transferred
cp ../data/egodyn-bench/selected_clips.json .
mkdir -p output && ln -s ../data/egodyn-bench/nuscenes_clips output/nuscenes_clips
ln -s ../data/egodyn-bench/carla_clips output/carla_clips

# 5. Download nuScenes separately (required for vision-only evaluation)
#    https://www.nuscenes.org/ — v1.0-trainval

# 6. Evaluate your model
python evaluation/evaluate_vllm_local.py \
    --selected_clips selected_clips.json \
    --nuscenes_root /path/to/nuscenes \
    --model your/hf-model-id --max_model_len 16384 \
    --output generated/your_model_answers.jsonl \
    --run_eval --metrics_output results/your_model.json
```

See [docs/EVALUATION.md](https://github.com/TUM-AVS/EgoDyn-Bench/blob/main/docs/EVALUATION.md) in the GitHub repo for the full evaluation flow, leaderboard submission, and answer-parsing details.

---

## Loading without the harness

If you only need the labels/arrays for your own research (no harness, no models), the canonical format is plain JSON/JSONL/NPZ — no `datasets` library required:

```python
import json, numpy as np
from pathlib import Path

ROOT = Path("data/egodyn-bench")
clips = json.load(open(ROOT / "selected_clips.json"))

# All QA pairs for the benchmark
qa_nu = [json.loads(l) for l in open(ROOT / "nuscenes_clips/qa.jsonl")]
qa_ca = [json.loads(l) for l in open(ROOT / "carla_clips/qa.jsonl")]

# Dynamics arrays for one clip
clip_id = clips[0]["id"]
src = clips[0]["source"]
arrays = np.load(ROOT / f"{src}_clips/arrays/{clip_id}.npz")
print(arrays["speed"].shape)  # (31,) — 3 s @ 10 Hz
```

---

## Natural Visual-Artifact Subset

80 of the 500 CARLA-transferred clips (16%) carry visible spatial artifacts inherited 
from upstream CARLA rendering — missing thin geometry, lighting glitches, melted 
textures. Because these artifacts are temporally stable within each 3-second window, 
the optical-flow signal driving the dynamics oracle is preserved while photometric 
quality is severely degraded.

**This subset functions as an unintended natural ablation for the paper's central 
"perception bottleneck" finding.** If models were genuinely vision-grounded, accuracy 
should drop noticeably on these 80 clips relative to the other 420. It does not: 
per-clip accuracy differs by at most 3 pp across six representative leaderboard 
models, with mixed direction — additional independent evidence that models do not 
meaningfully exploit photometric quality for ego-motion reasoning (Sec. 5.3 of the 
paper).

All 500 clips remain part of the benchmark for leaderboard consistency. The flagged 
subset is provided in `visual_artifact_subset.json` for downstream studies — e.g., 
fine-grained perception-quality ablations or visual robustness work.

---

## Determinism guarantees

- **Curation of `selected_clips.json` is a one-time decision** — the released file is the canonical artifact. The selection algorithm in the GitHub repo is provided for transparency, not as a bit-exact reproducer. (This matches how nuScenes, KITTI, BDD100K, DriveLM, etc. ship.)
- **Everything downstream is fully deterministic** — given `selected_clips.json` and a model's predictions JSONL, the evaluation harness reproduces `leaderboard.json` entries bit-for-bit. Verified on all 49 reference models.

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

- **Code (GitHub repo):** Apache 2.0
- **nuScenes-derived artifacts** (`nuscenes_clips/`): CC BY-NC-SA 4.0 — derivative of nuScenes (© 2019 Motional). NonCommercial only; share-alike.
- **CARLA-derived artifacts** (`carla_clips/`, `carla_videos_*`): CC BY 4.0. CARLA is MIT-licensed; Cosmos-Transfer 2.5 outputs follow NVIDIA's permissive research-output terms.
- The dataset bundle as a whole is published under CC BY-NC-SA 4.0 to satisfy the most restrictive component.