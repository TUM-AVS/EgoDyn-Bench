#!/usr/bin/env python3
"""
Export benchmark distribution data as CSV tables for TikZ/pgfplots.

Generates clean CSV files that can be directly loaded in LaTeX via:
  \\pgfplotstableread{data/answer_balance.csv}\\balancedata

Output files:
  answer_balance.csv         — per-question positive-class fractions
  balance_comparison.csv     — nuScenes-only vs augmented benchmark
  feature_distributions.csv  — raw feature values per clip (for histograms/KDE)
  feature_summary.csv        — percentile summaries per source
  source_composition.csv     — clip counts by source and behavior
  correlation_matrix.csv     — feature correlation matrix
  speed_trend.csv            — 3-class speed trend breakdown

Usage:
    python scripts/export_tikz_data.py \
        --selected selected_clips.json \
        --output-dir assets/tikz_data
"""

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

FEATURES = [
    "mean_speed", "max_speed", "min_accel", "mean_abs_jerk",
    "max_lateral_accel", "max_abs_yaw_rate", "total_heading_change",
]

FEATURE_LABELS = {
    "mean_speed": "Mean Speed (m/s)",
    "max_speed": "Max Speed (m/s)",
    "min_accel": "Min Acceleration (m/s²)",
    "mean_abs_jerk": "Mean |Jerk| (m/s³)",
    "max_lateral_accel": "Max Lateral Accel (m/s²)",
    "max_abs_yaw_rate": "Max |Yaw Rate| (rad/s)",
    "total_heading_change": "Total Heading Change (rad)",
}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path}")


def export_answer_balance(clips, output_dir):
    """Per-question answer fractions for the selected benchmark."""
    labels_list = [c["answers"] for c in clips]
    questions = sorted(set(q for lab in labels_list for q in lab))

    rows = []
    for q in questions:
        answers = [lab.get(q) for lab in labels_list if q in lab]
        total = len(answers)
        counts = Counter(answers)
        for cls in sorted(counts.keys()):
            rows.append([q, cls, counts[cls], f"{counts[cls]/total:.4f}"])

    write_csv(
        os.path.join(output_dir, "answer_balance.csv"),
        ["question", "answer", "count", "fraction"],
        rows,
    )


def export_balance_comparison(clips, output_dir):
    """Side-by-side: nuScenes-only vs augmented mix positive-class fractions."""
    ns_clips = [c for c in clips if c["source"] == "nuscenes"]
    ns_labels = [c["answers"] for c in ns_clips]
    all_labels = [c["answers"] for c in clips]

    questions = sorted(set(q for lab in all_labels for q in lab))
    # Exclude speed_trend (3-class) — it has its own export
    questions = [q for q in questions if q != "speed_trend"]
    positive_classes = {"yes", "aggressive", "turning"}

    def pos_frac(labels_list, question):
        answers = [lab.get(question) for lab in labels_list if question in lab]
        if not answers:
            return 0
        return sum(1 for a in answers if a in positive_classes) / len(answers)

    rows = []
    for q in questions:
        rows.append([
            q,
            f"{pos_frac(ns_labels, q):.4f}",
            f"{pos_frac(all_labels, q):.4f}",
        ])

    write_csv(
        os.path.join(output_dir, "balance_comparison.csv"),
        ["question", "nuscenes_only", "augmented_benchmark"],
        rows,
    )


def export_feature_distributions(clips, output_dir):
    """Raw feature values per clip — for pgfplots histograms or KDE."""
    rows = []
    for c in clips:
        row = [c["source"]]
        for f in FEATURES:
            row.append(f"{c['features'].get(f, '')}")
        rows.append(row)

    write_csv(
        os.path.join(output_dir, "feature_distributions.csv"),
        ["source"] + FEATURES,
        rows,
    )


def export_feature_summary(clips, output_dir):
    """Percentile summary per source — compact table for pgfplots bar/box plots."""
    percentiles = [5, 25, 50, 75, 95]
    rows = []

    for source in ["nuscenes", "carla", "combined"]:
        if source == "combined":
            subset = clips
        else:
            subset = [c for c in clips if c["source"] == source]

        for feat in FEATURES:
            vals = [c["features"][feat] for c in subset if feat in c["features"]]
            if not vals:
                continue
            arr = np.array(vals)
            pcts = np.percentile(arr, percentiles)
            row = [source, feat, f"{np.mean(arr):.4f}", f"{np.std(arr):.4f}"]
            row += [f"{p:.4f}" for p in pcts]
            rows.append(row)

    write_csv(
        os.path.join(output_dir, "feature_summary.csv"),
        ["source", "feature", "mean", "std", "p5", "p25", "p50", "p75", "p95"],
        rows,
    )


def export_source_composition(clips, output_dir):
    """Clip counts by source and CARLA behavior."""
    n_ns = sum(1 for c in clips if c["source"] == "nuscenes")
    n_ca = sum(1 for c in clips if c["source"] == "carla")

    rows = [
        ["nuscenes", "all", str(n_ns), f"{n_ns/len(clips):.4f}"],
        ["carla", "all", str(n_ca), f"{n_ca/len(clips):.4f}"],
    ]

    # CARLA behavior breakdown
    behaviors = Counter()
    for c in clips:
        if c["source"] == "carla":
            parts = c["id"].split("__")
            if len(parts) >= 2:
                behaviors[parts[1]] += 1

    for beh in sorted(behaviors.keys()):
        cnt = behaviors[beh]
        rows.append(["carla", beh, str(cnt), f"{cnt/n_ca:.4f}"])

    write_csv(
        os.path.join(output_dir, "source_composition.csv"),
        ["source", "behavior", "count", "fraction"],
        rows,
    )


def export_correlation_matrix(clips, output_dir):
    """Feature correlation matrix — symmetric, for pgfplots matrix plot."""
    data = []
    for c in clips:
        data.append([c["features"].get(f, np.nan) for f in FEATURES])
    data = np.array(data)
    corr = np.corrcoef(data.T)

    rows = []
    for i, f1 in enumerate(FEATURES):
        row = [f1] + [f"{corr[i, j]:.4f}" for j in range(len(FEATURES))]
        rows.append(row)

    write_csv(
        os.path.join(output_dir, "correlation_matrix.csv"),
        ["feature"] + FEATURES,
        rows,
    )


def export_speed_trend(clips, output_dir):
    """3-class speed trend breakdown for grouped bar chart."""
    all_labels = [c["answers"] for c in clips]
    ns_labels = [c["answers"] for c in clips if c["source"] == "nuscenes"]
    ca_labels = [c["answers"] for c in clips if c["source"] == "carla"]

    rows = []
    for name, labels in [("nuscenes", ns_labels), ("carla", ca_labels), ("combined", all_labels)]:
        answers = [lab.get("speed_trend") for lab in labels if "speed_trend" in lab]
        total = len(answers)
        counts = Counter(answers)
        for cls in ["accelerating", "steady", "decelerating"]:
            cnt = counts.get(cls, 0)
            rows.append([name, cls, str(cnt), f"{cnt/total:.4f}" if total else "0"])

    write_csv(
        os.path.join(output_dir, "speed_trend.csv"),
        ["source", "class", "count", "fraction"],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selected", type=str, default="selected_clips.json")
    parser.add_argument("--output-dir", type=str, default="assets/tikz_data")
    args = parser.parse_args()

    with open(args.selected) as f:
        clips = json.load(f)

    n_ns = sum(1 for c in clips if c["source"] == "nuscenes")
    n_ca = sum(1 for c in clips if c["source"] == "carla")
    print(f"Loaded {len(clips)} clips: {n_ns} nuScenes + {n_ca} CARLA\n")

    os.makedirs(args.output_dir, exist_ok=True)
    print("Exporting CSV tables for pgfplots:")

    export_answer_balance(clips, args.output_dir)
    export_balance_comparison(clips, args.output_dir)
    export_feature_distributions(clips, args.output_dir)
    export_feature_summary(clips, args.output_dir)
    export_source_composition(clips, args.output_dir)
    export_correlation_matrix(clips, args.output_dir)
    export_speed_trend(clips, args.output_dir)

    print(f"\nAll CSV files written to {args.output_dir}/")
    print("\nLaTeX usage example:")
    print(r"  \pgfplotstableread[col sep=comma]{tikz_data/balance_comparison.csv}\compdata")
    print(r"  \begin{axis}[ybar, symbolic x coords={...}]")
    print(r"    \addplot table[x=question, y=nuscenes_only] {\compdata};")
    print(r"    \addplot table[x=question, y=augmented_benchmark] {\compdata};")
    print(r"  \end{axis}")


if __name__ == "__main__":
    main()
