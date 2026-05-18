# Threshold Calibration Report

**Clips analysed:** 41999

---

## Contents

1. [yaw_rate_turn_direction](#yaw_rate_turn_direction) — `yaw_rate_sign_with_deadzone`
2. [braking_intensity](#braking_intensity) — `multi_threshold_classification`
3. [acceleration_intensity](#acceleration_intensity) — `multi_threshold_classification`
4. [max_speed_kmh](#max_speed_kmh) — `feature_conversion`
5. [driving_smoothness](#driving_smoothness) — `multi_threshold_classification`
6. [speed_trend](#speed_trend) — `trend_classification`
7. [mean_speed_low](#mean_speed_low) — `threshold_classification`
8. [significant_heading_change](#significant_heading_change) — `threshold_classification`
9. [emergency_maneuver](#emergency_maneuver) — `or_threshold_event`
10. [dominant_motion_axis](#dominant_motion_axis) — `dominant_axis_comparison`
11. [high_lateral_accel](#high_lateral_accel) — `lateral_accel_threshold`
12. [brake_then_turn](#brake_then_turn) — `sequential_event`
13. [speed_peak_half](#speed_peak_half) — `peak_half_detection`
14. [stop_and_go](#stop_and_go) — `stop_and_go_detection`
15. [contrastive_sequence](#contrastive_sequence) — `half_comparison`

---

## 1. yaw_rate_turn_direction

**Question:** Is the vehicle turning left, right, or going straight?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `yaw_rate_sign_with_deadzone`  
*Tests understanding of yaw rate sign convention*

**Deadzone:** `[-0.04, 0.04]` rad/s

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | -0.0013 rad/s |
| Std | 0.1627 rad/s |
| Min | -1.8920 rad/s |
| p5 | -0.2724 rad/s |
| p10 | -0.1343 rad/s |
| p25 | -0.0311 rad/s |
| p50 | -0.0000 rad/s |
| p75 | 0.0299 rad/s |
| p90 | 0.1329 rad/s |
| p95 | 0.2585 rad/s |
| Max | 2.0821 rad/s |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| left | 9191 | 21.9% |
| right | 9312 | 22.2% |
| straight | 23496 | 55.9% |

#### Deadzone Sensitivity

| Deadzone | left | straight | right |
|----------:|----------:|----------:|----------:|
| 0.0004 | 18873 (44.9%) | 4200 (10.0%) | 18926 (45.1%) |
| 0.0042 | 16652 (39.6%) | 8400 (20.0%) | 16947 (40.4%) |
| 0.0124 | 14551 (34.6%) | 12601 (30.0%) | 14847 (35.4%) |
| 0.0196 | 12477 (29.7%) | 16800 (40.0%) | 12722 (30.3%) |
| 0.0305 | 10420 (24.8%) | 21000 (50.0%) | 10579 (25.2%) |
| 0.0400 **<--** | 9191 (21.9%) | 23496 (55.9%) | 9312 (22.2%) |
| 0.0483 | 8378 (19.9%) | 25201 (60.0%) | 8420 (20.0%) |
| 0.0776 | 6359 (15.1%) | 29399 (70.0%) | 6241 (14.9%) |

---

## 2. braking_intensity

**Question:** What is the intensity level of the vehicle's braking?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `multi_threshold_classification`  
*Tests classification of braking intensity into severity levels*

**Feature:** `min_accel` &ensp; **Aggregation:** `min`

**Current thresholds:** `[-3.924, -1.962, -0.981]`  
**Labels:** `['emergency', 'moderate', 'low', 'none']`

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | -1.2356 |
| Std | 1.5194 |
| Min | -9.8025 |
| p5 | -4.5331 |
| p10 | -2.9359 |
| p25 | -1.5858 |
| p50 | -0.8881 |
| p75 | -0.1809 |
| p90 | -0.0000 |
| p95 | 0.0110 |
| Max | 2.3604 |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| emergency | 2689 | 6.4% |
| low | 11725 | 27.9% |
| moderate | 4897 | 11.7% |
| none | 22688 | 54.0% |

#### Suggested Equal-Frequency Alternative

**Equal-frequency thresholds** (targeting ~25% per class):

Suggested thresholds: `[-1.5858, -0.8881, -0.1809]`

| Class | Count | Fraction |
|-------|------:|:--------:|
| emergency | 10500 | 25.0% |
| moderate | 10499 | 25.0% |
| low | 10500 | 25.0% |
| none | 10500 | 25.0% |

---

## 3. acceleration_intensity

**Question:** What is the intensity level of the vehicle's acceleration?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `multi_threshold_classification`  
*Tests classification of acceleration intensity into severity levels*

**Feature:** `max_accel` &ensp; **Aggregation:** `max`

**Current thresholds:** `[0.981, 1.962, 3.924]`  
**Labels:** `['none', 'low', 'moderate', 'high']`

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 0.7198 |
| Std | 0.9917 |
| Min | -3.2203 |
| p5 | -0.5985 |
| p10 | -0.1585 |
| p25 | 0.0000 |
| p50 | 0.6306 |
| p75 | 1.2308 |
| p90 | 1.8596 |
| p95 | 2.3081 |
| Max | 12.0675 |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| high | 433 | 1.0% |
| low | 11139 | 26.5% |
| moderate | 3167 | 7.5% |
| none | 27260 | 64.9% |

#### Suggested Equal-Frequency Alternative

**Equal-frequency thresholds** (targeting ~25% per class):

Suggested thresholds: `[0.0, 0.6306, 1.2308]`

| Class | Count | Fraction |
|-------|------:|:--------:|
| none | 10500 | 25.0% |
| low | 10499 | 25.0% |
| moderate | 10499 | 25.0% |
| high | 10501 | 25.0% |

---

## 4. max_speed_kmh

**Question:** What is the maximum speed of the vehicle (in km/h)?  
**Category:** direct_dynamics &ensp; **Answer type:** numeric &ensp; **Rule:** `feature_conversion`  
*Tests ability to read maximum speed*

**Feature:** `max_speed` × 3.6

#### Distribution (converted)

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 24.7048 km/h |
| Std | 18.0853 km/h |
| Min | 0.0000 km/h |
| p5 | 0.0006 km/h |
| p10 | 0.0259 km/h |
| p25 | 11.8094 km/h |
| p50 | 24.0948 km/h |
| p75 | 34.2029 km/h |
| p90 | 46.3339 km/h |
| p95 | 57.0610 km/h |
| Max | 129.6023 km/h |

*Numeric question — no thresholds to calibrate.*

---

## 5. driving_smoothness

**Question:** How smooth is the driving based on jerk?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `multi_threshold_classification`  
*Tests understanding of jerk as comfort metric (4-level)*

**Feature:** `mean_abs_jerk` &ensp; **Aggregation:** `mean`

**Current thresholds:** `[0.3, 0.9, 2.0]`  
**Labels:** `['smooth', 'moderate', 'aggressive', 'emergency']`

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 1.7294 |
| Std | 1.1727 |
| Min | 0.0000 |
| p5 | 0.0001 |
| p10 | 0.0007 |
| p25 | 1.0139 |
| p50 | 1.7490 |
| p75 | 2.3470 |
| p90 | 3.0551 |
| p95 | 3.6252 |
| Max | 12.3205 |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| aggressive | 16179 | 38.5% |
| emergency | 16238 | 38.7% |
| moderate | 2927 | 7.0% |
| smooth | 6655 | 15.8% |

#### Suggested Equal-Frequency Alternative

**Equal-frequency thresholds** (targeting ~25% per class):

Suggested thresholds: `[1.0139, 1.749, 2.347]`

| Class | Count | Fraction |
|-------|------:|:--------:|
| smooth | 10500 | 25.0% |
| moderate | 10499 | 25.0% |
| aggressive | 10500 | 25.0% |
| emergency | 10500 | 25.0% |

---

## 6. speed_trend

**Question:** Is the vehicle accelerating, decelerating, or maintaining steady speed?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `trend_classification`  
*Tests understanding of acceleration trends*

**Deadzone:** `[-0.25, 0.25]` m/s²

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | -0.1990 m/s² |
| Std | 0.8519 m/s² |
| Min | -5.6807 m/s² |
| p5 | -1.9081 m/s² |
| p10 | -1.3242 m/s² |
| p25 | -0.4883 m/s² |
| p50 | -0.0027 m/s² |
| p75 | 0.1931 m/s² |
| p90 | 0.6641 m/s² |
| p95 | 0.9591 m/s² |
| Max | 5.7465 m/s² |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| accelerating | 9390 | 22.4% |
| decelerating | 14393 | 34.3% |
| steady | 18216 | 43.4% |

#### Deadzone Sensitivity

| Deadzone | accelerating | steady | decelerating |
|----------:|----------:|----------:|----------:|
| 0.0002 | 15650 (37.3%) | 4200 (10.0%) | 22149 (52.7%) |
| 0.0446 | 14247 (33.9%) | 8400 (20.0%) | 19352 (46.1%) |
| 0.1212 | 12133 (28.9%) | 12600 (30.0%) | 17266 (41.1%) |
| 0.2133 | 10077 (24.0%) | 16800 (40.0%) | 15122 (36.0%) |
| 0.2500 **<--** | 9390 (22.4%) | 18216 (43.4%) | 14393 (34.3%) |
| 0.3317 | 8066 (19.2%) | 21000 (50.0%) | 12933 (30.8%) |
| 0.4766 | 6120 (14.6%) | 25199 (60.0%) | 10680 (25.4%) |
| 0.6479 | 4310 (10.3%) | 29399 (70.0%) | 8290 (19.7%) |

---

## 7. mean_speed_low

**Question:** Is the mean speed below 5 m/s (18 km/h)?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `threshold_classification`  
*Tests understanding of average speed*

**Feature:** `mean_speed` &ensp; **Threshold:** `5.0`

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 5.9257 |
| Std | 4.6213 |
| Min | 0.0000 |
| p5 | 0.0001 |
| p10 | 0.0044 |
| p25 | 2.2738 |
| p50 | 5.7562 |
| p75 | 8.5087 |
| p90 | 11.3892 |
| p95 | 14.1526 |
| Max | 35.9765 |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| yes | 18203 | 43.3% |
| no | 23796 | 56.7% |

#### Threshold Sensitivity

| Threshold | Percentile | yes | no |
|----------:|:----------:|------:|------:|
| 0.0044 | p10 | 4200 (10.0%) | 37799 (90.0%) |
| 1.2122 | p20 | 8400 (20.0%) | 33599 (80.0%) |
| 3.1404 | p30 | 12600 (30.0%) | 29399 (70.0%) |
| 4.5979 | p40 | 16800 (40.0%) | 25199 (60.0%) |
| 5.7562 | p50 | 20999 (50.0%) | 21000 (50.0%) |
| 6.7961 | p60 | 25199 (60.0%) | 16800 (40.0%) |
| 7.8871 | p70 | 29399 (70.0%) | 12600 (30.0%) |
| 9.1893 | p80 | 33599 (80.0%) | 8400 (20.0%) |
| 11.3892 | p90 | 37799 (90.0%) | 4200 (10.0%) |

---

## 8. significant_heading_change

**Question:** Does the vehicle change heading by more than 15 degrees?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `threshold_classification`  
*Tests understanding of cumulative heading change*

**Feature:** `total_heading_change` &ensp; **Threshold:** `0.2618`

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 0.1239 |
| Std | 0.2111 |
| Min | 0.0000 |
| p5 | 0.0001 |
| p10 | 0.0005 |
| p25 | 0.0118 |
| p50 | 0.0388 |
| p75 | 0.1320 |
| p90 | 0.3512 |
| p95 | 0.6060 |
| Max | 1.6362 |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| no | 36305 | 86.4% |
| yes | 5694 | 13.6% |

#### Threshold Sensitivity

| Threshold | Percentile | no | yes |
|----------:|:----------:|------:|------:|
| 0.0005 | p10 | 4200 (10.0%) | 37799 (90.0%) |
| 0.0041 | p20 | 8400 (20.0%) | 33599 (80.0%) |
| 0.0162 | p30 | 12600 (30.0%) | 29399 (70.0%) |
| 0.0244 | p40 | 16800 (40.0%) | 25199 (60.0%) |
| 0.0388 | p50 | 20999 (50.0%) | 21000 (50.0%) |
| 0.0628 | p60 | 25199 (60.0%) | 16800 (40.0%) |
| 0.1021 | p70 | 29399 (70.0%) | 12600 (30.0%) |
| 0.1726 | p80 | 33599 (80.0%) | 8400 (20.0%) |
| 0.3512 | p90 | 37799 (90.0%) | 4200 (10.0%) |

---

## 9. emergency_maneuver

**Question:** Does the vehicle perform an emergency maneuver (high jerk or hard braking)?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `or_threshold_event`  
*Tests detection of emergency maneuvers via high jerk OR hard braking*

#### Condition 1: `max_abs_jerk` greater_than `20.0`

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 5.1676 |
| Std | 3.8816 |
| Min | 0.0000 |
| p5 | 0.0002 |
| p10 | 0.0025 |
| p25 | 3.0777 |
| p50 | 4.9575 |
| p75 | 6.6938 |
| p90 | 9.1432 |
| p95 | 11.6521 |
| Max | 50.1924 |

Triggered: 335/41999 (0.8%)

| Threshold | Triggered |
|----------:|----------:|
| 6.257 (p70) | 12599 (30.0%) |
| 7.213 (p80) | 8400 (20.0%) |
| 7.950 (p85) | 6300 (15.0%) |
| 9.143 (p90) | 4200 (10.0%) |
| 11.652 (p95) | 2100 (5.0%) |

#### Condition 2: `min_accel` less_than `-3.924`

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | -1.2356 |
| Std | 1.5194 |
| Min | -9.8025 |
| p5 | -4.5331 |
| p10 | -2.9359 |
| p25 | -1.5858 |
| p50 | -0.8881 |
| p75 | -0.1809 |
| p90 | -0.0000 |
| p95 | 0.0110 |
| Max | 2.3604 |

Triggered: 2689/41999 (6.4%)

| Threshold | Triggered |
|----------:|----------:|
| -4.533 (p5) | 2100 (5.0%) |
| -2.936 (p10) | 4198 (10.0%) |
| -2.234 (p15) | 6300 (15.0%) |
| -1.834 (p20) | 8400 (20.0%) |
| -1.401 (p30) | 12600 (30.0%) |

#### Combined (OR)

| Class | Count | Fraction |
|-------|------:|:--------:|
| yes | 2720 | 6.5% |
| no | 39279 | 93.5% |

---

## 10. dominant_motion_axis

**Question:** Is the vehicle's motion primarily longitudinal (speeding up/slowing down) or lateral (turning)?  
**Category:** direct_dynamics &ensp; **Answer type:** multiclass &ensp; **Rule:** `dominant_axis_comparison`  
*Tests understanding of longitudinal vs lateral dynamics*

*Evaluated on 5000 clips (arrays loaded).*

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| lateral | 645 | 12.9% |
| longitudinal | 3586 | 71.7% |
| none | 769 | 15.4% |

#### Lateral / Longitudinal Ratio Distribution

| Statistic | Value |
|-----------|------:|
| Count | 4946 |
| Mean | 15.0362 |
| Std | 363.4132 |
| Min | 0.0000 |
| p5 | 0.0001 |
| p10 | 0.0004 |
| p25 | 0.0249 |
| p50 | 0.1388 |
| p75 | 0.4823 |
| p90 | 1.4184 |
| p95 | 2.4595 |
| Max | 16603.8035 |

#### Ratio Threshold Sensitivity

| Threshold | Longitudinal | Lateral |
|----------:|-------------:|--------:|
| 0.5 | 3741 (75.6%) | 1205 (24.4%) |
| 0.75 | 4044 (81.8%) | 902 (18.2%) |
| 1.0 **<--** | 4231 (85.5%) | 715 (14.5%) |
| 1.25 | 4368 (88.3%) | 578 (11.7%) |
| 1.5 | 4481 (90.6%) | 465 (9.4%) |
| 2.0 | 4620 (93.4%) | 326 (6.6%) |

---

## 11. high_lateral_accel

**Question:** Does the vehicle experience high lateral acceleration?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `lateral_accel_threshold`  
*Tests detection of high lateral acceleration (a_lat = speed * yaw_rate)*

**Feature:** `max_lateral_accel` &ensp; **Threshold:** `2.0` m/s²

#### Feature Distribution

| Statistic | Value |
|-----------|------:|
| Count | 41999 |
| Mean | 0.6360 m/s² |
| Std | 1.1780 m/s² |
| Min | 0.0000 m/s² |
| p5 | 0.0000 m/s² |
| p10 | 0.0000 m/s² |
| p25 | 0.0330 m/s² |
| p50 | 0.1881 m/s² |
| p75 | 0.6997 m/s² |
| p90 | 1.7203 m/s² |
| p95 | 2.6026 m/s² |
| Max | 9.7998 m/s² |

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| yes | 3252 | 7.7% |
| no | 38747 | 92.3% |

#### Threshold Sensitivity

| Threshold | Percentile | no | yes |
|----------:|:----------:|------:|------:|
| 0.0000 | p10 | 4200 (10.0%) | 37799 (90.0%) |
| 0.0060 | p20 | 8400 (20.0%) | 33599 (80.0%) |
| 0.0645 | p30 | 12600 (30.0%) | 29399 (70.0%) |
| 0.1186 | p40 | 16799 (40.0%) | 25200 (60.0%) |
| 0.1881 | p50 | 20999 (50.0%) | 21000 (50.0%) |
| 0.3038 | p60 | 25199 (60.0%) | 16800 (40.0%) |
| 0.5182 | p70 | 29399 (70.0%) | 12600 (30.0%) |
| 0.9560 | p80 | 33599 (80.0%) | 8400 (20.0%) |
| 1.7203 | p90 | 37799 (90.0%) | 4200 (10.0%) |

---

## 12. brake_then_turn

**Question:** Does the vehicle brake and then turn (sequential maneuver)?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `sequential_event`  
*Tests detection of sequential brake-then-turn maneuvers*

*Evaluated on 5000 clips (arrays loaded).*

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| no | 4620 | 92.4% |
| yes | 380 | 7.6% |

#### Event Trigger Rates

| Event | Threshold | Clips with event |
|-------|----------:|:----------------:|
| 1st: `accel` less_than | -1.5 | 1349/5000 (27.0%) |
| 2nd: `yaw_rate` abs_greater_than | 0.1 | 1264/5000 (25.3%) |

#### Gap Distribution (clips with sequence)

| Statistic | Value |
|-----------|------:|
| Count | 380 |
| Mean | 0.5953 s |
| Std | 0.5765 s |
| Min | 0.1000 s |
| p5 | 0.1000 s |
| p10 | 0.1000 s |
| p25 | 0.1000 s |
| p50 | 0.4000 s |
| p75 | 0.9000 s |
| p90 | 1.6000 s |
| p95 | 1.9000 s |
| Max | 2.0000 s |

---

## 13. speed_peak_half

**Question:** Does the maximum speed occur in the first or second half of the clip?  
**Category:** comparative &ensp; **Answer type:** multiclass &ensp; **Rule:** `peak_half_detection`  
*Tests temporal localization of peak speed within the clip*

*Evaluated on 5000 clips (arrays loaded).*

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| first_half | 2360 | 47.2% |
| no_peak | 1206 | 24.1% |
| second_half | 1434 | 28.7% |

*No adjustable thresholds for peak-half detection.*

---

## 14. stop_and_go

**Question:** Does the vehicle exhibit stop-and-go behavior?  
**Category:** direct_dynamics &ensp; **Answer type:** binary &ensp; **Rule:** `stop_and_go_detection`  
*Tests detection of stop-and-go driving patterns*

*Evaluated on 5000 clips (arrays loaded).*

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| no | 4917 | 98.3% |
| yes | 83 | 1.7% |

#### Stop Threshold Sensitivity

| Stop (m/s) | Go (m/s) | Yes | No |
|----------:|--------:|----:|---:|
| 0.3 | 2.0 | 65 (1.3%) | 4935 (98.7%) |
| 0.5 **<--** | 2.0 | 83 (1.7%) | 4917 (98.3%) |
| 0.8 | 2.0 | 103 (2.1%) | 4897 (97.9%) |
| 1.0 | 2.0 | 111 (2.2%) | 4889 (97.8%) |
| 1.5 | 2.0 | 148 (3.0%) | 4852 (97.0%) |

---

## 15. contrastive_sequence

**Question:** Comparing the first and second halves of the clip, which half has more dynamic driving?  
**Category:** comparative &ensp; **Answer type:** multiclass &ensp; **Rule:** `half_comparison`  
*Tests ability to compare driving dynamics across temporal segments*

*Evaluated on 5000 clips (arrays loaded).*

#### Current Class Balance

| Class | Count | Fraction |
|-------|------:|:--------:|
| first_half | 2043 | 40.9% |
| second_half | 1322 | 26.4% |
| similar | 1635 | 32.7% |

#### Relative Difference Distribution

| Statistic | Value |
|-----------|------:|
| Count | 5000 |
| Mean | 0.3636 |
| Std | 0.2639 |
| Min | 0.0000 |
| p5 | 0.0275 |
| p10 | 0.0608 |
| p25 | 0.1558 |
| p50 | 0.3180 |
| p75 | 0.5111 |
| p90 | 0.7837 |
| p95 | 0.9388 |
| Max | 1.0000 |

#### Similarity Threshold Sensitivity

| Threshold | Similar | First Half | Second Half |
|----------:|--------:|-----------:|------------:|
| 0.05 | 425 (8.5%) | 2676 (53.5%) | 1899 (38.0%) |
| 0.1 | 813 (16.3%) | 2486 (49.7%) | 1701 (34.0%) |
| 0.15 | 1199 (24.0%) | 2286 (45.7%) | 1515 (30.3%) |
| 0.2 **<--** | 1632 (32.6%) | 2046 (40.9%) | 1322 (26.4%) |
| 0.25 | 1996 (39.9%) | 1850 (37.0%) | 1154 (23.1%) |
| 0.3 | 2351 (47.0%) | 1653 (33.1%) | 996 (19.9%) |
| 0.4 | 3170 (63.4%) | 1130 (22.6%) | 700 (14.0%) |
| 0.5 | 3701 (74.0%) | 822 (16.4%) | 477 (9.5%) |

---

*Report generated by `calibrate_thresholds.py`*
