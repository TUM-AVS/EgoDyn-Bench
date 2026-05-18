# Detailed Sign Convention Analysis Report

## Executive Summary

✅ **All sign conventions verified as CORRECT.** No fixes needed to Task 1 or Task 2 code.

---

## 1. Yaw-Rate Sign Convention

### Analysis Results

**Dataset:** 50 clips analyzed (highest max_abs_yaw_rate)

**Findings:**
- **100% consistency rate** between yaw_rate sign and net heading change direction
- 36 clips with positive yaw_rate → all showed positive heading change (LEFT turns)
- 14 clips with negative yaw_rate → all showed negative heading change (RIGHT turns)
- 0 inconsistencies detected

### Sample Data (Top 10 Clips)

| Clip ID    | Mean Yaw-Rate (rad/s) | Net Δ Heading (rad) | Consistent? |
|------------|----------------------|---------------------|-------------|
| clip_00038 | +0.2977              | +0.8939             | ✓           |
| clip_00003 | +0.1461              | +0.4327             | ✓           |
| clip_00002 | +0.2075              | +0.6247             | ✓           |
| clip_00000 | +0.3099              | +0.9418             | ✓           |
| clip_00001 | +0.2692              | +0.8167             | ✓           |
| clip_00041 | +0.2671              | +0.8102             | ✓           |
| clip_00042 | +0.2150              | +0.6495             | ✓           |
| clip_00039 | +0.3158              | +0.9540             | ✓           |
| clip_00043 | +0.1572              | +0.4683             | ✓           |
| clip_00040 | +0.3036              | +0.9205             | ✓           |

### Physical Interpretation

**nuScenes coordinate system:**
- x-axis: Forward (vehicle front)
- y-axis: Left
- z-axis: Up
- Yaw: Rotation around z-axis

**Convention:**
- **Positive yaw rotation** = Counter-clockwise when viewed from above = **LEFT turn**
- **Negative yaw rotation** = Clockwise when viewed from above = **RIGHT turn**

### Implementation Verification

**Task 1 (nuscenes_extract.py):**
```python
# Yaw extracted from ego_pose quaternion
quat = ego_pose['rotation']  # [w, x, y, z] in nuScenes
rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to scipy
yaw = rotation.as_euler('xyz', degrees=False)[2]  # Z-axis rotation
```
✅ Correctly extracts yaw as rotation around z-axis

**Task 1 (dynamics_features.py):**
```python
# Yaw-rate via numerical derivative
yaws_unwrapped = np.unwrap(yaws)
# Central difference: (yaw[i+1] - yaw[i-1]) / (2*dt)
```
✅ Correctly preserves sign of yaw derivative

**Task 2 (labeling_rules.py):**
```python
# yaw_rate_sign_with_deadzone rule
if agg_yaw_rate > threshold_pos:
    answer = "left"   # Positive → LEFT
elif agg_yaw_rate < threshold_neg:
    answer = "right"  # Negative → RIGHT
```
✅ Correctly interprets positive yaw_rate as LEFT turn

### Conclusion

**✅ NO FIX NEEDED.**

The yaw-rate sign convention is physically correct and consistent with nuScenes coordinate system. Current labeling rules correctly interpret:
- Positive yaw_rate → LEFT turn
- Negative yaw_rate → RIGHT turn

---

## 2. Acceleration Sign Convention

### Analysis Results

**Dataset:** 20 clips analyzed (most negative min_accel)

**Findings:**
- **100% consistency rate** for meaningful comparisons
- 13 clips with negative mean_accel showed decreasing speed (braking)
- 0 clips with negative mean_accel showed increasing speed
- 7 clips had brief negative min_accel but positive mean_accel (momentary deceleration during overall acceleration)

### Sample Data

| Clip ID    | Min Accel (m/s²) | Mean Accel (m/s²) | Δ Speed (m/s) | Braking? |
|------------|------------------|-------------------|---------------|----------|
| clip_00030 | -1.261           | -0.070            | -0.200        | Minor    |
| clip_00007 | -1.189           | -0.059            | -0.201        | Minor    |
| clip_00011 | -1.189           | -0.460            | -1.412        | ✓ Yes    |
| clip_00009 | -1.189           | -0.405            | -1.204        | ✓ Yes    |
| clip_00010 | -1.189           | -0.439            | -1.381        | ✓ Yes    |
| clip_00008 | -1.189           | -0.209            | -0.649        | ✓ Yes    |
| clip_00048 | -1.142           | +0.345            | +1.082        | N/A*     |
| clip_00049 | -1.142           | +0.260            | +0.767        | N/A*     |

*N/A: Clips with positive mean_accel show brief deceleration within overall acceleration

### Physical Interpretation

**Expected behavior:**
- **Negative acceleration** → Speed decreases (braking/deceleration)
- **Positive acceleration** → Speed increases (accelerating)

**Observed behavior:**
- ✅ All clips with **negative mean acceleration** show **decreasing speed**
- ✅ Clips with brief negative min_accel but positive mean_accel show overall speed increase (expected for transient deceleration)

### Implementation Verification

**Task 1 (dynamics_features.py):**
```python
# Speed from position deltas
dx = positions[i + 1, 0] - positions[i - 1, 0]
dy = positions[i + 1, 1] - positions[i - 1, 1]
speed[i] = np.sqrt(dx**2 + dy**2) / dt

# Acceleration as derivative of speed
accel[i] = (speed[i + 1] - speed[i - 1]) / (2*dt)
```
✅ Correctly computes: accel = d(speed)/dt

**Task 2 (labeling_rules.py):**
```python
# threshold_event for hard braking
if operator == "less_than":
    event_occurred = feature_value < threshold  # Negative accel < -3.0
answer = "yes" if event_occurred else "no"
```
✅ Correctly identifies hard braking as negative acceleration

### Important Note on Braking Magnitude

The dataset shows:
- **Most negative acceleration observed:** -1.26 m/s²
- **Typical braking range:** -0.4 to -1.3 m/s²

This is **normal for highway/urban driving**. The placeholder threshold of `-3.0 m/s²` for "hard braking" in the template config may need adjustment:

**Recommended threshold adjustments:**
- **Normal braking:** < -0.5 m/s²
- **Moderate braking:** < -1.0 m/s²
- **Hard braking:** < -1.5 m/s² (not -3.0 m/s² which is emergency-level)

### Conclusion

**✅ NO FIX NEEDED for sign convention.**

The acceleration sign convention is physically correct:
- Negative acceleration → Decreasing speed (braking)
- Positive acceleration → Increasing speed (accelerating)

**⚠️ RECOMMENDED CONFIG ADJUSTMENT:**

Update `hard_braking_event` threshold in `questions_template.yaml`:
```yaml
threshold: -1.5  # m/s² (was -3.0, which is too extreme)
```

---

## 3. Overall Recommendations

### For Task 1 & Task 2 Code

**✅ NO CHANGES NEEDED**

All sign conventions are correct:
1. Yaw-rate: Positive = LEFT, Negative = RIGHT
2. Acceleration: Negative = Braking, Positive = Accelerating

### For Configuration Templates

**⚠️ ADJUST THRESHOLDS** in `dataset/configs/questions_template.yaml`:

```yaml
# Current (too extreme for typical driving)
hard_braking_event:
  threshold: -3.0  # m/s²

# Recommended (based on observed data)
hard_braking_event:
  threshold: -1.5  # m/s² (captures hard braking without requiring emergency stops)
```

**Additional threshold suggestions based on data:**
- Yaw-rate deadzone: ±0.05 rad/s ✅ (appropriate)
- Speed trend deadzone: ±0.5 m/s² ✅ (appropriate)
- Jerk smoothness: 3.0 m/s³ ⚠️ (may need adjustment based on distribution)

### Verification Methodology

The diagnostic approach was:

1. **For yaw-rate:** Compare sign of mean yaw_rate with sign of net heading change (yaw_end - yaw_start)
2. **For acceleration:** Compare sign of mean accel with sign of speed change (speed_end - speed_start)
3. **Sample size:** 50 clips for yaw-rate, 20 clips for acceleration
4. **Selection:** Sorted by magnitude to analyze clips with most pronounced effects

This methodology ensures we're testing the sign conventions on clips where the effects are most visible and measurable.

---

## 4. Coordinate System Reference

### nuScenes Global Frame
- Origin: Arbitrary global reference
- x: East
- y: North
- z: Up

### nuScenes Ego Frame (Vehicle Frame)
- Origin: Vehicle center
- x: Forward (vehicle front)
- y: Left
- z: Up

### Rotation Convention
- Yaw (ψ): Rotation around z-axis
- Pitch (θ): Rotation around y-axis
- Roll (φ): Rotation around x-axis

### Sign Conventions (Right-Hand Rule)
- **Positive yaw:** Counter-clockwise rotation around z-axis (when viewed from above)
- **Positive pitch:** Nose up
- **Positive roll:** Right side down

---

## Appendix: Diagnostic Script

The verification was performed using:
- **Script:** `dataset/scripts/check_sign_conventions.py`
- **Dataset:** 50 clips extracted from nuScenes v1.0-trainval
- **Method:** Numerical analysis of derived dynamics signals
- **Validation:** 100% consistency across all analyzed clips

To reproduce this analysis:
```bash
python dataset/scripts/check_sign_conventions.py \
    --clips_index ./output/clips_50/clips_index.jsonl \
    --output_report ./sign_convention_report.md \
    --n_yaw_clips 50 \
    --n_accel_clips 20
```

---

**Report Date:** 2026-01-05
**Analysis Status:** ✅ COMPLETE
**Action Required:** Update threshold parameters only (no code changes)
