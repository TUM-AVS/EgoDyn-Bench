# Dataset Pipeline

This directory contains the dataset generation and validation pipeline for the Dynamic Trajectory Understanding Benchmark.

## Task 1: nuScenes Clip Extraction

Extract 3-second clips from nuScenes with aligned ego dynamics and camera frames.

### Quick Start

#### 1. Install Dependencies

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate dynamics-benchmark
```

Or using pip:
```bash
pip install -r requirements.txt
```

#### 2. Run Unit Tests (No nuScenes Required)

Verify the implementation without needing the full nuScenes dataset:

```bash
# Using conda environment
conda run -n dynamics-benchmark python dataset/tests/test_dynamics_features.py

# Or if environment is already activated
python dataset/tests/test_dynamics_features.py
```

Expected output: `✓ ALL TESTS PASSED`

#### 3. Extract Clips

```bash
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root /path/to/nuscenes \
    --nuscenes_version v1.0-trainval \
    --output_dir ./output/clips \
    --num_clips 100 \
    --seed 42
```

**Smoke test** (fast, 10 clips):
```bash
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root /path/to/nuscenes \
    --nuscenes_version v1.0-mini \
    --output_dir ./output/test_clips \
    --num_clips 10 \
    --seed 42
```

#### 4. Validate Clips

```bash
python dataset/validation/validate_clips.py \
    --clips_dir ./output/clips \
    --nuscenes_root /path/to/nuscenes \
    --verbose
```

### CLI Arguments

#### extract_nuscenes_clips.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--nuscenes_root` | Yes | - | Path to nuScenes dataset root |
| `--nuscenes_version` | No | `v1.0-trainval` | nuScenes version (e.g., `v1.0-mini`, `v1.0-trainval`) |
| `--output_dir` | Yes | - | Output directory for clips |
| `--num_clips` | No | `None` (all) | Maximum number of clips to extract |
| `--clip_seconds` | No | `3.0` | Clip duration in seconds |
| `--sampling_hz` | No | `10.0` | Resampling frequency for uniform time series |
| `--min_frames` | No | `20` | Minimum frames required per clip |
| `--camera` | No | `CAM_FRONT` | Camera sensor to use |
| `--seed` | No | `42` | Random seed for reproducibility |

#### validate_clips.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--clips_dir` | Yes | - | Directory containing `clips_index.jsonl` and `arrays/` |
| `--nuscenes_root` | No | `None` | Optional path to nuScenes root for frame validation |
| `--timestamp_tolerance` | No | `0.1` | Tolerance for timestamp alignment (seconds) |
| `--verbose` | No | `False` | Print all validation errors |

### Output Format

The extraction script produces:

```
{output_dir}/
├── clips_index.jsonl          # One record per clip
├── arrays/                    # Numeric data
│   ├── clip_00000.npz
│   ├── clip_00001.npz
│   └── ...
└── metadata.json              # Dataset-level metadata
```

#### clips_index.jsonl Format

Each line is a JSON record:

```json
{
  "clip_id": "clip_00000",
  "scene_token": "...",
  "sample_token": "...",
  "t_start": 1234567890.123,
  "t_end": 1234567893.123,
  "camera": "CAM_FRONT",
  "frame_tokens": ["...", "...", ...],
  "frame_paths": ["samples/CAM_FRONT/...", ...],
  "num_frames": 36,
  "array_ref": "arrays/clip_00000.npz",
  "split": "unsplit",
  "features": {
    "max_speed": 12.3,
    "mean_speed": 8.1,
    "min_accel": -2.5,
    "max_abs_yaw_rate": 0.15,
    "max_abs_jerk": 1.2,
    "p95_abs_jerk": 0.85,
    "mean_abs_jerk": 0.62,
    "total_heading_change": 0.45
  }
}
```

#### NPZ Array Format

Each `.npz` file contains:

```python
{
    'timestamps': [N,],        # Relative time (seconds from t_start)
    'position': [N, 2],        # (x, y) position in meters
    'yaw': [N,],              # Yaw angle in radians (unwrapped)
    'speed': [N,],            # Speed in m/s
    'accel': [N,],            # Longitudinal acceleration in m/s²
    'yaw_rate': [N,],         # Yaw rate in rad/s
    'jerk': [N,],             # Jerk in m/s³
}
```

Where `N = clip_seconds × sampling_hz + 1` (default: 31 samples for 3.0s at 10 Hz).

### Design Details

#### Clip Extraction Strategy

- **Anchoring**: Each clip ends at a nuScenes keyframe sample, looking backward 3.0 seconds
- **Frame selection**: All `CAM_FRONT` sample_data with timestamps in `[t_start, t_end]`
- **Minimum frames**: Clips with fewer than 20 frames are rejected
- **Deterministic**: Extraction order is deterministic given the same seed

#### Dynamics Computation

1. **Ego poses** are queried from nuScenes for all frame timestamps
2. **Derivation from ego_pose** (for consistency):
   - Position: `(x, y)` from ego translation
   - Yaw: quaternion → Euler angle conversion
   - Speed: `sqrt(dx² + dy²) / dt` from position deltas
   - Acceleration: numerical derivative of speed
   - Yaw rate: numerical derivative of yaw (unwrapped)
3. **Resampling**: Linear interpolation to uniform 10 Hz grid
4. **Jerk**: numerical derivative of acceleration
5. **Features**: Summary statistics computed from time series

#### Validation Checks

The validation suite verifies:
- ✅ Clip duration ≈ 3.0s (within tolerance)
- ✅ Timestamps monotonically increasing
- ✅ Frame references exist (if `--nuscenes_root` provided)
- ✅ Frame count ≥ minimum threshold
- ✅ Arrays contain no NaN/Inf values
- ✅ Array shapes consistent across clips
- ✅ Timestamp alignment within tolerance
- ✅ Distribution summaries for debugging (speed, accel, yaw_rate, jerk)

### Module Structure

```
dataset/
├── generation/
│   ├── nuscenes_extract.py    # Clip extraction from nuScenes
│   ├── dynamics_features.py   # Dynamics computation and features
│   ├── config_loader.py        # Question template config loader
│   ├── labeling_rules.py       # Labeling rule engine
│   └── qa_generator.py         # QA generation logic
├── scripts/
│   ├── extract_nuscenes_clips.py  # CLI for clip extraction
│   └── generate_qa.py          # CLI for QA generation
├── validation/
│   ├── validate_clips.py      # Clip validation suite
│   └── validate_qa.py          # QA validation suite
├── tests/
│   ├── test_dynamics_features.py  # Dynamics tests
│   └── test_qa_generation.py   # QA generation tests
└── configs/
    └── questions_template.yaml # Question template config
```

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'nuscenes'`
- **Fix**: Install nuScenes devkit: `pip install nuscenes-devkit`

**Issue**: Validation reports "Frame not found"
- **Fix**: Ensure `--nuscenes_root` points to the correct nuScenes dataset root

**Issue**: Very few clips extracted
- **Fix**: Check `--min_frames` threshold; lower it if your dataset has sparse frames

**Issue**: Arrays contain NaN values
- **Fix**: This indicates missing ego_pose data; check nuScenes version compatibility

## Task 2: Question Template Loading + QA Generation

Generate multiple physically grounded QA items per clip using configurable question templates.

### Quick Start

```bash
# Generate QA from clips
python dataset/scripts/generate_qa.py \
    --clips_index ./output/clips/clips_index.jsonl \
    --questions_config ./dataset/configs/questions_template.yaml \
    --output_qa_jsonl ./output/qa.jsonl \
    --seed 42

# Validate generated QA
python dataset/validation/validate_qa.py \
    --qa_jsonl ./output/qa.jsonl \
    --clips_index ./output/clips/clips_index.jsonl \
    --verbose

# Run unit tests
python dataset/tests/test_qa_generation.py
```

### Configuration

Edit [dataset/configs/questions_template.yaml](configs/questions_template.yaml) to customize:
- Question templates and text
- Labeling rule parameters (thresholds, aggregation methods)
- Answer types (binary, multiclass, numeric)
- Metadata (difficulty, required features)

**Important:** All threshold values are PLACEHOLDERS. Adjust based on:
- nuScenes data distribution analysis
- Driving comfort standards (ISO 2631)
- Desired question difficulty

### Adapting Thresholds to New Vehicle Platforms

Our thresholds fall into two categories:

1. **Physics-anchored** (platform-independent): These derive from driving standards and generalize across vehicles without recalibration:
   - Speed trend deadzone: ±0.25 m/s² (ISO 15622 ACC steady-state error)
   - High lateral acceleration: 2.0 m/s² (AASHTO comfort limit)
   - Heading change: 0.2618 rad / 15° (geometric, vehicle-independent)
   - Stop-and-go: stopped < 0.5 m/s, moving > 2.0 m/s
   - Speed regime boundaries: 0.5 / 5.0 / 13.9 m/s

2. **Percentile-calibrated** (platform-dependent): These are tuned to the data distribution and should be recalibrated for new platforms with different acceleration/yaw-rate envelopes:
   - Braking intensity thresholds (depends on vehicle braking capability)
   - Driving smoothness / jerk thresholds (depends on suspension, drivetrain)
   - Extreme maneuver thresholds (depends on vehicle dynamics limits)
   - Yaw rate deadzone (depends on steering system noise floor)

**Recalibration workflow for a new platform:**

```bash
# 1. Extract clips from your dataset
python dataset/scripts/extract_nuscenes_clips.py --nuscenes_root /path/to/data ...

# 2. Run threshold calibration to see distributions and suggested equal-frequency thresholds
python dataset/scripts/calibrate_thresholds.py \
    --clips_index output/your_clips/clips_index.jsonl \
    --config dataset/configs/questions_template.yaml \
    --output docs/your_calibration_report.md

# 3. Review the report, update thresholds in questions_template.yaml
# 4. Regenerate QA pairs with updated thresholds
```

The calibration script outputs per-question feature distributions, current class balance, and suggested equal-frequency thresholds. See [docs/threshold_calibration_report.md](../docs/threshold_calibration_report.md) for a complete example.

**Sensitivity analysis:** Our [threshold sensitivity study](../docs/threshold_calibration_report.md) shows that most thresholds are robust — moderate perturbations (±20%) shift class boundaries by only a few percentage points. The exception is the yaw rate deadzone (0.04 rad/s), where small changes significantly affect the left/right/straight balance due to the high density of near-zero yaw rates.

### Available Labeling Rules

1. **`yaw_rate_sign_with_deadzone`**: Turn direction (left/right/straight)
2. **`threshold_event`**: Binary event detection (e.g., hard braking)
3. **`or_threshold_event`**: Logical OR of multiple conditions (e.g., emergency maneuver)
4. **`feature_conversion`**: Unit conversion (e.g., m/s → km/h)
5. **`threshold_classification`**: Binary classification (e.g., high lateral accel yes/no)
6. **`multi_threshold_classification`**: Multi-level classification (e.g., driving smoothness: smooth/moderate/aggressive/emergency)
7. **`trend_classification`**: Trend detection (accelerating/decelerating/steady)
8. **`range_check`**: Range validation

See [dataset/generation/labeling_rules.py](generation/labeling_rules.py) for implementation details.

### Output Format

**qa.jsonl** - One JSON record per QA item:
```json
{
  "qa_id": "qa_00000001",
  "clip_id": "clip_00042",
  "question_id": "yaw_rate_turn_direction",
  "category": "direct_dynamics",
  "question": "Is the vehicle turning left, right, or going straight?",
  "answer": "left",
  "answer_type": "multiclass",
  "choices": ["left", "right", "straight"],
  "units": null,
  "label_source": "sensor_rule",
  "rule_name": "yaw_rate_sign_with_deadzone",
  "split": "unsplit",
  "t_end": 1531883533.550364,
  "evidence": {...},
  "metadata": {...}
}
```

### Next Steps

- **Task 3**: Add dataset split logic (train/val)
- **Task 4**: Upgrade storage from NPZ to Zarr/HDF5 for scalability
