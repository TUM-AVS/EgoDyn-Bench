# Dataset Generation Pipeline

The dataset is built in four sequential stages:

1. **Clip extraction** -- pull 3-second ego-motion clips from nuScenes and CARLA
2. **QA generation** -- apply labeling rules to produce question-answer pairs
3. **Balanced subset selection** -- curate a class-balanced 1000-clip benchmark from the full candidate pool
4. **Split building** -- (optional, toolkit) create stratified train/val partitions for follow-up work

Each stage has a CLI script, a validation script, and unit tests.

---

## About the released benchmark

The artifact you actually evaluate against is **`selected_clips.json`** at the repo root: a fixed list of 1,000 clip IDs (500 nuScenes + 500 CARLA) chosen to maximise answer-class balance across the 14 question types. Every entry in `leaderboard/results.json` was computed against *that* specific 1,000-clip set.

The selection script (`scripts/select_balanced_clips.py`, or the legacy `scripts/improved_clip_selection.py` that produced the released JSON) is included for **transparency about the curation methodology**, not as a bit-exact reproducer:

- **Re-running the selection produces a statistically equivalent but not identical subset.** The greedy balancing algorithm depends on iteration order of intermediate Python sets/dicts, which is not guaranteed across runs even with a fixed `--seed`. Two consecutive runs typically overlap by ~75–80% of clips.
- **This is consistent with how every major driving benchmark ships** (nuScenes' `v1.0-trainval` tokens, KITTI's train/test split, BDD100K, DriveLM — curation is a one-time decision, the artifact is the citation).
- **Everything downstream of the selection is deterministic**: QA generation, answer parsing, metrics, and rule-based consistency scoring all reproduce bit-for-bit given a fixed `selected_clips.json` and a model's predictions JSONL. The leaderboard is fully re-derivable.

In practice: **users of the benchmark consume `selected_clips.json` as-is** via the evaluation harness. Stages 1–3 of this document are relevant only if you want to rebuild a similar-but-different curated set from raw data, e.g. to extend the benchmark or audit the methodology.

---

## Quick start (smoke test)

Run the full pipeline on a small subset (50 clips) to verify everything works before scaling.

```bash
conda activate dynamics-benchmark

# 1. Extract clips
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root /path/to/nuscenes \
    --nuscenes_version v1.0-trainval \
    --output_dir ./output/clips_test \
    --num_clips 50

# 2. Validate clips
python dataset/validation/validate_clips.py \
    --clips_dir ./output/clips_test/clips_index.jsonl

# 3. Generate QA
python dataset/scripts/generate_qa.py \
    --clips_index ./output/clips_test/clips_index.jsonl \
    --questions_config ./dataset/configs/questions_template.yaml \
    --output_qa_jsonl ./output/qa_test.jsonl

# 4. Validate QA
python dataset/validation/validate_qa.py \
    --qa_jsonl ./output/qa_test.jsonl \
    --clips_index ./output/clips_test/clips_index.jsonl

# 5. Build splits
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips_test/clips_index.jsonl \
    --qa_jsonl ./output/qa_test.jsonl \
    --output_dir ./output/splits_test \
    --num_val_clips 10 \
    --num_train_clips 30 \
    --min_per_bin 2

# 6. Validate splits
python dataset/validation/validate_splits.py \
    --splits_dir ./output/splits_test
```

---

## Stage 1: Clip extraction

**Script:** `dataset/scripts/extract_nuscenes_clips.py`

Iterates over nuScenes keyframe samples and extracts a backward-looking window of ego poses + camera frames, computes dynamics features (speed, acceleration, yaw-rate, jerk), and writes each clip as an NPZ array file plus a JSONL index record.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--nuscenes_root` | *(required)* | Path to nuScenes dataset root |
| `--nuscenes_version` | `v1.0-trainval` | Dataset version (`v1.0-mini` for testing) |
| `--output_dir` | *(required)* | Output directory |
| `--num_clips` | all | Maximum clips to extract |
| `--clip_seconds` | `3.0` | Clip duration in seconds |
| `--sampling_hz` | `10.0` | Resampling frequency (Hz) |
| `--min_frames` | `20` | Minimum camera frames per clip |
| `--camera` | `CAM_FRONT` | Camera sensor to use |
| `--seed` | `42` | Random seed (stored in metadata) |

### Output

```
output_dir/
├── clips_index.jsonl    # One record per clip (metadata + summary features)
├── arrays/
│   ├── clip_00000.npz   # timestamps, positions, yaw, speed, accel, yaw_rate, jerk
│   ├── clip_00001.npz
│   └── ...
└── metadata.json        # Dataset-level metadata (version, params, counts)
```

### Validation

```bash
python dataset/validation/validate_clips.py \
    --clips_dir ./output/clips/clips_index.jsonl
```

Checks: file existence, array shapes, timestamp monotonicity, NaN/Inf values, feature consistency.

---

## Stage 2: QA generation

**Script:** `dataset/scripts/generate_qa.py`

Loads the clips index and question templates, runs each labeling rule against each clip, and writes one QA record per (clip, question) pair.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--clips_index` | *(required)* | Path to `clips_index.jsonl` |
| `--questions_config` | *(required)* | Path to `questions_template.yaml` |
| `--output_qa_jsonl` | *(required)* | Output QA file |
| `--seed` | `42` | Random seed |
| `--max_qas_per_clip` | all | Limit QA items per clip |
| `--category_filter` | all | Only generate for one category (e.g. `direct_dynamics`) |
| `--no_evidence` | false | Exclude evidence dicts from output |

### Question template format

Question templates are defined in `dataset/configs/questions_template.yaml`. Each question specifies:

- `question_id` -- unique identifier
- `question_text` -- human-readable prompt
- `answer_type` -- `binary`, `multiclass`, or `numeric`
- `choices` -- valid answer labels
- `category` -- grouping (e.g. `direct_dynamics`, `comparative`)
- `rule.name` -- labeling rule to apply (registered in `labeling_rules.py`)
- `rule.params` -- rule-specific parameters (features, thresholds, etc.)

### Labeling rules

All labeling rules live in `dataset/generation/labeling_rules.py` and use a registry pattern. Currently implemented (12 rules):

| Rule | Description |
|------|-------------|
| `yaw_rate_sign_with_deadzone` | Turn direction from mean yaw-rate |
| `threshold_event` | Binary event detection (e.g. hard braking) |
| `or_threshold_event` | Logical OR of multiple threshold conditions |
| `threshold_classification` | Above/below threshold classification |
| `multi_threshold_classification` | Multi-level classification (e.g. smoothness: smooth/moderate/aggressive/emergency) |
| `trend_classification` | Speed trend (accelerating/decelerating/steady) |
| `feature_conversion` | Convert and round a feature value (e.g. m/s to km/h) |
| `range_check` | Check if value falls within a specified range |
| `dominant_axis_comparison` | Compare longitudinal vs lateral acceleration |
| `lateral_accel_threshold` | Classify lateral acceleration magnitude |
| `sequential_event` | Detect temporal sequence of events (e.g. brake-then-turn) |
| `peak_half_detection` | Locate peak in first or second half of clip |
| `stop_and_go_detection` | Detect stop-and-go events |
| `half_comparison` | Compare a feature between first and second half |

### Output format

Each line in `qa.jsonl`:

```json
{
  "qa_id": "clip_00000__hard_braking_event",
  "clip_id": "clip_00000",
  "question_id": "hard_braking_event",
  "question_text": "Did the ego vehicle perform a hard braking maneuver?",
  "answer": "no",
  "answer_type": "binary",
  "choices": ["yes", "no"],
  "category": "direct_dynamics",
  "evidence": { "feature_value": -1.2, "threshold": -3.0 }
}
```

### Validation

```bash
python dataset/validation/validate_qa.py \
    --qa_jsonl ./output/qa.jsonl \
    --clips_index ./output/clips/clips_index.jsonl \
    --verbose
```

Checks: schema completeness, answer validity, clip coverage, distribution balance.

---

## Stage 3: Balanced subset selection

**Scripts:** `scripts/improved_clip_selection.py` (legacy — produced the released JSON), `scripts/select_balanced_clips.py` (newer, equivalent).

Greedily picks N clips from the full QA pool to maximise answer-class balance across all categorical questions. The released `selected_clips.json` was produced with the legacy script at `--target 1000 --min-nuscenes-frac 0.5`.

```bash
# Produces a 1000-clip subset (50% nuScenes / 50% CARLA)
python scripts/improved_clip_selection.py \
    --target 1000 \
    --min-nuscenes-frac 0.5 \
    --output selected_clips.json
```

### Non-determinism caveat

As covered in [About the released benchmark](#about-the-released-benchmark), the selection is not bit-exact across runs. **For evaluating a model, do not regenerate `selected_clips.json` — use the released file as-is.** Regeneration is meant for users who want to construct a comparable curated set on different data, not to reproduce the released benchmark.

### Output format

Each entry in `selected_clips.json`:

```json
{
  "id": "clip_19765",                        // or e.g. "DEU_Heilbronn-163_1_T-8__Balanced__w0"
  "source": "nuscenes",                      // or "carla"
  "features": {"mean_speed": 6.2, ...},      // per-clip dynamics summaries
  "answers": {"yaw_rate_turn_direction": "left", ...}
}
```

---

## Stage 4: Split building (toolkit)

**Script:** `dataset/scripts/build_splits.py`

Creates stratified train/val partitions based on QA-derived dynamics tags. This stage is **independent of the benchmark** — it lets you generate training data for fine-tuning, using the same QA pipeline as the benchmark but with different size/balance targets. See [TASK3_SPLITS.md](TASK3_SPLITS.md) for the full workflow.

### Stratification

Each clip is tagged with three boolean flags from its QA answers:

| Tag | Source question | Positive condition |
|-----|----------------|--------------------|
| `has_turn` | `yaw_rate_turn_direction` | answer is `left` or `right` |
| `has_braking` | `hard_braking_event` | answer is `yes` |
| `has_aggressive` | `driving_smoothness` | answer is `aggressive` or `emergency` |

These 3 bits produce up to 8 stratification bins. Validation clips are sampled in two phases:

1. **Minimum coverage** -- take `min_per_bin` clips from each bin
2. **Proportional fill** -- distribute remaining quota proportionally to bin sizes

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--clips_index` | *(required)* | Path to `clips_index.jsonl` |
| `--qa_jsonl` | *(required)* | Path to `qa.jsonl` |
| `--output_dir` | *(required)* | Output directory |
| `--seed` | `42` | Random seed |
| `--num_val_clips` | `500` | Target validation clips |
| `--num_train_clips` | `3000` | Target training clips |
| `--min_per_bin` | `10` | Minimum clips per stratification bin |
| `--tag_turn_qid` | `yaw_rate_turn_direction` | Question ID for turn tag |
| `--tag_braking_qid` | `hard_braking_event` | Question ID for braking tag |
| `--tag_aggressive_qid` | `driving_smoothness` | Question ID for aggressive tag |
| `--dry_run` | false | Print statistics without writing files |

### Output

```
output_dir/
├── train_clips.jsonl      # Training clips (with split: "train")
├── val_clips.jsonl        # Validation clips (with split: "val")
├── train_qa.jsonl         # Training QA items
├── val_qa.jsonl           # Validation QA items
└── split_metadata.json    # Seed, strategy, bin counts, actual sizes
```

### Validation

```bash
python dataset/validation/validate_splits.py \
    --splits_dir ./output/splits
```

Checks: no clip overlap between splits, no QA overlap, split field correctness, metadata accuracy, distribution balance.

---

## Production run

Generate the full benchmark dataset from nuScenes v1.0-trainval:

```bash
conda activate dynamics-benchmark

# 1. Extract all clips (~29,000 from v1.0-trainval)
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root /path/to/nuscenes \
    --nuscenes_version v1.0-trainval \
    --output_dir ./output/clips

# 2. Validate
python dataset/validation/validate_clips.py \
    --clips_dir ./output/clips/clips_index.jsonl

# 3. Generate QA (~14 questions per clip)
python dataset/scripts/generate_qa.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --questions_config ./dataset/configs/questions_template.yaml \
    --output_qa_jsonl ./output/qa.jsonl

# 4. Validate
python dataset/validation/validate_qa.py \
    --qa_jsonl ./output/qa.jsonl \
    --clips_index ./output/clips/clips_index.jsonl \
    --verbose

# 5. Build splits
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --qa_jsonl ./output/qa.jsonl \
    --output_dir ./output/splits \
    --num_val_clips 500 \
    --num_train_clips 3000

# 6. Final validation
python dataset/validation/validate_splits.py \
    --splits_dir ./output/splits
```

---

## Threshold calibration

Question thresholds were calibrated iteratively. The `driving_smoothness` question uses a 4-level classification on `mean_abs_jerk` (mean of absolute jerk) with thresholds at 0.3, 0.9, and 2.0 m/s³. See [THRESHOLD_CALIBRATION.md](THRESHOLD_CALIBRATION.md) for methodology and values.

If you re-extract clips or use a different dataset version, re-run calibration:

```bash
python dataset/scripts/calibrate_thresholds.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --output_report ./threshold_calibration_report.md
```

## Sign conventions

Yaw-rate and acceleration sign conventions have been verified against nuScenes data.
See [sign_convention_detailed_analysis.md](sign_convention_detailed_analysis.md) for the full analysis.

**Summary:**
- Positive yaw-rate = left turn (counter-clockwise from above)
- Negative acceleration = braking (speed decrease)
