# Task 3: Dataset Splits

This document describes the implementation of stratified train/val split generation for the Dynamic Trajectory Understanding Benchmark.

## Overview

The split generation pipeline creates balanced train and validation sets using **stratified sampling** based on dynamics characteristics derived from QA answers. This ensures both splits have representative coverage of different driving scenarios (turning, braking, aggressive driving).

## Key Features

- **QA-tag-based stratification**: Clips are binned based on three boolean tags derived from QA answers
- **Two-phase sampling**: Minimum coverage per bin + proportional sampling to reach target size
- **Configurable tag questions**: Tag derivation uses configurable question IDs
- **Deterministic**: Fixed random seed ensures reproducible splits
- **Comprehensive validation**: Checks for data integrity and distribution balance

## Stratification Strategy

### Tag Derivation

Each clip is tagged with three boolean flags based on QA answers:

1. **`has_turn`**: True if clip contains left or right turn
   - Derived from question ID: `yaw_rate_turn_direction`
   - Positive if answer is `"left"` or `"right"`

2. **`has_braking`**: True if clip contains hard braking event
   - Derived from question ID: `hard_braking_event`
   - Positive if answer is `"yes"`

3. **`has_aggressive`**: True if clip exhibits aggressive driving
   - Derived from question ID: `driving_smoothness`
   - Positive if answer is `"aggressive"`

**Tag logic**: "Any positive evidence" - if ANY QA item for a clip indicates the condition, the clip is tagged.

### Stratification Bins

Clips are stratified into **8 possible bins** based on the combination of tags:

```
(has_turn, has_braking, has_aggressive)
Examples:
  (0, 0, 0): No turn, no braking, smooth driving
  (1, 0, 1): Turn present, no braking, aggressive driving
  (1, 1, 1): Turn + braking + aggressive
```

### Validation Sampling

Two-phase approach to balance coverage and proportionality:

**Phase 1 - Minimum Coverage:**
- Sample `min_per_bin` clips from each bin (default: 10)
- Ensures rare scenarios are represented in validation set
- Caps at available clips if bin is smaller

**Phase 2 - Proportional Sampling:**
- Remaining validation quota distributed proportionally to bin sizes
- Maintains overall dataset distribution
- Uses clips not yet sampled in Phase 1

**Training Set:**
- All clips NOT in validation set form the training pool
- Subsample to `num_train_clips` if pool is larger than target
- Random sampling (already stratified by validation split)

## File Descriptions

### Generation

**`dataset/generation/split_builder.py`** (452 lines)
- Core split builder class with stratified sampling logic
- Classes:
  - `ClipTags`: Container for clip-level boolean tags
  - `SplitBuilder`: Main builder with two-phase sampling
- Key methods:
  - `derive_clip_tags()`: Extract tags from QA answers
  - `stratify_clips()`: Bin clips by tag combinations
  - `sample_validation_set()`: Two-phase stratified sampling
  - `build_splits()`: Main pipeline orchestration
  - `write_splits()`: Output JSONL files with split assignments

### Scripts

**`dataset/scripts/build_splits.py`** (220 lines)
- CLI interface for split generation
- Arguments:
  - `--clips_index`: Path to clips_index.jsonl (required)
  - `--qa_jsonl`: Path to qa.jsonl (required)
  - `--output_dir`: Output directory for split files (required)
  - `--seed`: Random seed (default: 42)
  - `--num_val_clips`: Target validation clips (default: 500)
  - `--num_train_clips`: Target training clips (default: 3000)
  - `--min_per_bin`: Minimum clips per bin (default: 10)
  - `--tag_turn_qid`: Question ID for turn tag (default: yaw_rate_turn_direction)
  - `--tag_braking_qid`: Question ID for braking tag (default: hard_braking_event)
  - `--tag_aggressive_qid`: Question ID for aggressive tag (default: driving_smoothness)
  - `--dry_run`: Print statistics without writing files

### Validation

**`dataset/validation/validate_splits.py`** (561 lines)
- Comprehensive validation suite for generated splits
- Checks:
  - No clip_id overlap between train and val
  - No qa_id overlap between splits
  - All QA items assigned to exactly one split
  - Split field correctly set in all records
  - Metadata counts match actual data
- Reports:
  - Overall counts (clips, QA, QA/clip ratio)
  - QA distribution by category
  - QA distribution by question_id
  - Answer distributions (train vs val comparison)
  - Dynamics tag coverage
  - Stratification bin statistics

## Output Files

Split generation creates 5 files in the output directory:

### 1. `train_clips.jsonl`

JSONL file with one clip record per line. Each record includes:
- All original clip metadata
- `split: "train"` field added

### 2. `val_clips.jsonl`

JSONL file with validation clip records.
- Same schema as train_clips.jsonl
- `split: "val"` field added

### 3. `train_qa.jsonl`

JSONL file with training QA items.
- All QA records for clips in training set
- `split: "train"` field added

### 4. `val_qa.jsonl`

JSONL file with validation QA items.
- All QA records for clips in validation set
- `split: "val"` field added

### 5. `split_metadata.json`

JSON file with split generation metadata:

```json
{
  "seed": 42,
  "balance_strategy": "qa_tags_stratified",
  "tag_question_ids": {
    "turn": "yaw_rate_turn_direction",
    "braking": "hard_braking_event",
    "aggressive": "driving_smoothness"
  },
  "timestamp": "2026-01-05T12:34:56.789Z",
  "target_num_val_clips": 500,
  "target_num_train_clips": 3000,
  "min_per_bin": 10,
  "actual_counts": {
    "total_clips": 5432,
    "total_qa": 43456,
    "eligible_clips": 5432,
    "dropped_clips_no_qa": 0,
    "val_clips": 500,
    "train_clips": 3000,
    "val_qa": 4000,
    "train_qa": 24000
  },
  "stratification_bins": {
    "(0, 0, 0)": {
      "total": 1200,
      "sampled_min": 10,
      "sampled_proportional": 100,
      "sampled_total": 110
    },
    "(1, 0, 1)": {
      "total": 800,
      "sampled_min": 10,
      "sampled_proportional": 70,
      "sampled_total": 80
    }
    // ... other bins
  }
}
```

## Usage Examples

### Basic Usage

Generate splits with default settings:

```bash
conda activate dynamics-benchmark

python dataset/scripts/build_splits.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --qa_jsonl ./output/qa.jsonl \
    --output_dir ./output/splits \
    --seed 42
```

This creates:
- 500 validation clips
- 3000 training clips
- Minimum 10 clips per bin

### Custom Split Sizes

Generate smaller splits for testing:

```bash
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips_50/clips_index.jsonl \
    --qa_jsonl ./output/qa_50.jsonl \
    --output_dir ./output/splits_test \
    --num_val_clips 10 \
    --num_train_clips 30 \
    --min_per_bin 2 \
    --seed 42
```

### Custom Tag Question IDs

Use different questions for tag derivation:

```bash
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --qa_jsonl ./output/qa.jsonl \
    --output_dir ./output/splits \
    --tag_turn_qid custom_turn_question \
    --tag_braking_qid custom_braking_question \
    --tag_aggressive_qid custom_smoothness_question \
    --seed 42
```

### Dry Run

Preview statistics without writing files:

```bash
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --qa_jsonl ./output/qa.jsonl \
    --output_dir ./output/splits \
    --dry_run
```

### Validate Generated Splits

```bash
python dataset/validation/validate_splits.py \
    --splits_dir ./output/splits
```

Expected output:
```
✓ All validation checks passed

DISTRIBUTION SUMMARY:
Split      Clips      QA Items     QA/Clip
Train      3000       24000        8.0
Val        500        4000         8.0
Total      3500       28000

QA DISTRIBUTION BY CATEGORY:
Category                       Train        Val          Total
direct_dynamics                24000        4000         28000

... detailed distributions ...

✓ Split dataset is valid and ready for use.
```

## Design Decisions

### Why QA-tag-based stratification?

1. **Observational labels**: Tags derived from QA answers reflect actual clip content, not hypothetical scenarios
2. **Implicit coverage**: Ensures validation set covers diverse dynamics without explicit filtering
3. **Downstream alignment**: Tags match the question types models will answer during evaluation
4. **Flexibility**: Tag question IDs are configurable via CLI

### Why two-phase sampling?

1. **Rare scenario coverage**: Minimum coverage ensures underrepresented bins appear in validation
2. **Distribution preservation**: Proportional phase maintains overall dataset characteristics
3. **Tunable balance**: `min_per_bin` parameter controls coverage vs. proportionality tradeoff

### Why clip-level splitting (not QA-level)?

1. **No data leakage**: Prevents same clip appearing in both train and val
2. **Realistic evaluation**: Models see entirely new clips during validation
3. **Simpler implementation**: Avoids complex QA-level balancing

## Implementation Details

### Determinism

All random operations use a seeded `numpy.random.RandomState`:
- Validation sampling
- Training set subsampling
- All outputs sorted by clip_id/qa_id for reproducibility

### Edge Cases

**Insufficient clips for target:**
- Warning logged if validation set smaller than target
- Uses all available clips up to target

**Bins with fewer clips than `min_per_bin`:**
- Samples all available clips from that bin
- Logs actual sampled count in metadata

**Clips without QA:**
- Dropped from splits (logged in metadata as `dropped_clips_no_qa`)
- Only clips with at least one QA item are eligible

**Proportional sampling underflow:**
- If Phase 2 allocation doesn't reach target, warning logged
- May occur if many bins are smaller than `min_per_bin`

### Performance

- Loads all clips and QA into memory (suitable for 2K-5K clips)
- Tag derivation: O(Q) where Q = number of QA items
- Stratification: O(C) where C = number of clips
- Sampling: O(C log C) due to sorting

For larger datasets (>50K clips), consider chunked processing or disk-backed storage.

## Validation Checklist

After generating splits, verify:

1. **No overlap**: No clip_id appears in both train and val
2. **Complete assignment**: All eligible QA items assigned to exactly one split
3. **Split field**: All records have correct `split` field
4. **Metadata accuracy**: Counts in metadata match actual files
5. **Distribution balance**: Tag coverage similar between train and val
6. **Answer distributions**: Major answer distributions preserved in both splits

Use `validate_splits.py` to automate these checks.

## Troubleshooting

### Warning: "Could not reach target N, sampled M"

**Cause:** Not enough eligible clips to fill validation set after minimum coverage.

**Solutions:**
- Reduce `num_val_clips`
- Reduce `min_per_bin`
- Generate more clips/QA

### Error: "QA file not found"

**Cause:** QA file path is incorrect or QA generation hasn't been run.

**Solutions:**
- Verify `--qa_jsonl` path
- Run `generate_qa.py` first

### Warning: "Dropped N clips without QA"

**Cause:** Some clips in clips_index.jsonl don't have corresponding QA items.

**Solutions:**
- Ensure QA generation covered all clips
- Check for clip_id mismatches
- Review QA generation logs for failures

### Validation error: "Duplicate qa_id in train split"

**Cause:** Bug in split generation or file corruption.

**Solutions:**
- Delete output directory and regenerate splits
- Check for manual file edits
- Report issue if reproducible

## Testing

Test split generation with small dataset:

```bash
# Generate 50 clips
python dataset/scripts/extract_nuscenes_clips.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir ./output/clips_test \
    --max_clips 50

# Generate QA
python dataset/scripts/generate_qa.py \
    --clips_index ./output/clips_test/clips_index.jsonl \
    --questions_config ./dataset/configs/questions_template.yaml \
    --output_qa_jsonl ./output/qa_test.jsonl

# Build splits
python dataset/scripts/build_splits.py \
    --clips_index ./output/clips_test/clips_index.jsonl \
    --qa_jsonl ./output/qa_test.jsonl \
    --output_dir ./output/splits_test \
    --num_val_clips 10 \
    --num_train_clips 30 \
    --min_per_bin 2

# Validate
python dataset/validation/validate_splits.py \
    --splits_dir ./output/splits_test
```

Expected: All validation checks pass.

## Future Enhancements

Potential improvements for larger-scale deployment:

1. **Multi-level stratification**: Add scene type, weather conditions, time of day
2. **Balanced answer distributions**: Ensure rare answer classes appear in validation
3. **Cross-validation splits**: Generate multiple folds for robust evaluation
4. **Stratification diagnostics**: Visualize bin distributions and sampling statistics
5. **Incremental updates**: Add new clips to existing splits without regeneration

## Related Documentation

- [Task 1: Clip Extraction](./TASK1_EXTRACTION.md) (if exists)
- [Task 2: QA Generation](./TASK2_QA_GENERATION.md) (if exists)
- [Sign Convention Analysis](../sign_convention_detailed_analysis.md)
- [Setup Instructions](./SETUP.md)
