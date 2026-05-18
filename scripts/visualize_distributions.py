#!/usr/bin/env python3
"""
Visualize data distributions for the selected benchmark clips.

Generates publication-quality figures showing:
  1. Answer balance across all questions (stacked bar chart)
  2. Feature distributions by source (nuScenes vs CARLA)
  3. Source composition + CARLA behavior breakdown
  4. Feature correlation heatmap
  5. Answer balance comparison (nuScenes-only vs augmented mix)

Usage:
    python scripts/visualize_distributions.py \
        --selected selected_clips.json \
        --output-dir assets/figures
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# TUM Corporate Design colours
# ---------------------------------------------------------------------------
TUM = {
    "blue":         "#0065bd",
    "dark_blue":    "#005293",
    "light_blue":   "#64a0c8",
    "lighter_blue": "#98c6ea",
    "orange":       "#e37222",
    "green":        "#a2ad00",
    "brand_blue":   "#3070B3",
    "blue_dark_1":  "#0A2D57",
    "grey_4":       "#6A757E",
    "grey_7":       "#DDE2E6",
    "grey_8":       "#EBECEF",
    "grey_9":       "#FBF9FA",
}

SOURCE_COLORS = {
    "nuscenes": TUM["orange"],
    "carla":    TUM["blue"],
    "combined": TUM["light_blue"],
}

BEHAVIOR_COLORS = {
    "Balanced":             TUM["brand_blue"],
    "Comfort":              TUM["green"],
    "Default":              TUM["grey_4"],
    "Efficiency-Sporty":    TUM["orange"],
    "Safety-Conservative":  TUM["blue"],
}

QUESTION_LABELS = {
    "brake_then_turn":            "Brake→Turn",
    "braking_intensity":          "Braking\nIntensity",
    "contrastive_sequence":       "Contrastive\nSequence",
    "dominant_motion_axis":       "Dominant\nAxis",
    "driving_smoothness":         "Driving\nSmoothness",
    "extreme_maneuver":           "Extreme\nManeuver",
    "high_lateral_accel":         "High Lat.\nAccel.",
    "lateral_stability":          "Lateral\nStability",
    "peak_decel_timing":          "Peak Decel.\nTiming",
    "peak_speed_timing":          "Peak Speed\nTiming",
    "speed_trend":                "Speed\nTrend",
    "speed_variance_comparison":  "Speed Var.\nComparison",
    "yaw_rate_comparison":        "Yaw Rate\nComparison",
    "yaw_rate_turn_direction":    "Turn\nDirection",
}

FEATURE_LABELS = {
    "mean_speed":            ("Mean Speed", "m/s"),
    "max_speed":             ("Max Speed", "m/s"),
    "min_accel":             ("Min Acceleration", "m/s²"),
    "mean_abs_jerk":         ("Mean |Jerk|", "m/s³"),
    "max_lateral_accel":     ("Max Lateral Accel.", "m/s²"),
    "max_abs_yaw_rate":      ("Max |Yaw Rate|", "rad/s"),
    "total_heading_change":  ("Total Heading Change", "rad"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style():
    """Set publication-quality matplotlib defaults (matching website script)."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial",
                              "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   9.5,
        "figure.dpi":        150,
        "savefig.dpi":       250,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "axes.linewidth":    0.8,
    })


def _save(fig, output_dir: str, name: str):
    """Save figure as PNG, SVG, and PDF."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}")


def _q_label(q: str) -> str:
    """Human-readable question label with line breaks for x-axes."""
    return QUESTION_LABELS.get(q, q.replace("_", "\n"))


# ---------------------------------------------------------------------------
# Figure 1: Feature distributions by source (KDE)
# ---------------------------------------------------------------------------

def plot_feature_distributions(clips: list[dict], output_dir: str):
    """KDE comparing nuScenes vs CARLA dynamics feature distributions."""
    features = [f for f in FEATURE_LABELS if any(f in c["features"] for c in clips)]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    ns_clips = [c for c in clips if c["source"] == "nuscenes"]
    ca_clips = [c for c in clips if c["source"] == "carla"]

    for i, feat in enumerate(features):
        ax = axes[i]
        label, unit = FEATURE_LABELS[feat]

        ns_vals = [c["features"][feat] for c in ns_clips
                   if feat in c["features"]]
        ca_vals = [c["features"][feat] for c in ca_clips
                   if feat in c["features"]]
        all_vals = ns_vals + ca_vals

        if not all_vals:
            ax.set_visible(False)
            continue

        p1, p99 = np.percentile(all_vals, [1, 99])
        margin = 0.1 * max(abs(p1), abs(p99), 0.01)
        xlim = (p1 - margin, p99 + margin)

        sns.kdeplot(ns_vals, ax=ax, color=SOURCE_COLORS["nuscenes"],
                    fill=True, alpha=0.25, linewidth=1.5,
                    label=f"nuScenes (n={len(ns_vals)})", clip=xlim)
        sns.kdeplot(ca_vals, ax=ax, color=SOURCE_COLORS["carla"],
                    fill=True, alpha=0.25, linewidth=1.5,
                    label=f"CARLA (n={len(ca_vals)})", clip=xlim)

        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=TUM["grey_7"])
        ax.set_xlim(xlim)
        # grid removed for clean white background
        ax.set_axisbelow(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions: nuScenes vs CARLA",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "feature_distributions")


# ---------------------------------------------------------------------------
# Figure 3: Source composition + CARLA behavior breakdown
# ---------------------------------------------------------------------------

def plot_source_composition(clips: list[dict], output_dir: str):
    """Donut chart of sources + horizontal bar chart of CARLA behaviours."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # --- Donut chart ---
    n_ns = sum(1 for c in clips if c["source"] == "nuscenes")
    n_ca = sum(1 for c in clips if c["source"] == "carla")
    wedges, texts, autotexts = ax1.pie(
        [n_ns, n_ca],
        labels=[f"nuScenes\n(n={n_ns})", f"CARLA\n(n={n_ca})"],
        colors=[SOURCE_COLORS["nuscenes"], SOURCE_COLORS["carla"]],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2.5, "width": 0.45},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)
    ax1.set_title("Data Source Composition", pad=14)

    # --- CARLA behaviour breakdown ---
    ca_clips = [c for c in clips if c["source"] == "carla"]
    behaviors: list[str] = []
    for c in ca_clips:
        parts = c["id"].split("__")
        if len(parts) >= 2:
            behaviors.append(parts[1])

    counts = Counter(behaviors)
    beh_names = sorted(counts.keys())
    beh_counts = [counts[b] for b in beh_names]
    colors = [BEHAVIOR_COLORS.get(b, TUM["grey_4"]) for b in beh_names]

    bars = ax2.barh(
        beh_names, beh_counts,
        color=colors, edgecolor="white", linewidth=0.8, height=0.6,
    )
    for bar, cnt in zip(bars, beh_counts):
        ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va="center", fontsize=10, fontweight="bold")

    ax2.set_xlabel("Number of Clips")
    ax2.set_title("CARLA Behaviour Breakdown", pad=14)
    # grid removed for clean white background
    ax2.set_axisbelow(True)
    max_cnt = max(beh_counts) if beh_counts else 1
    ax2.set_xlim(0, max_cnt * 1.12)

    fig.tight_layout(w_pad=3)
    _save(fig, output_dir, "source_composition")


# ---------------------------------------------------------------------------
# Figure 4: Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(clips: list[dict], output_dir: str):
    """Lower-triangle correlation matrix of dynamics features."""
    # Only include features that actually exist in the data
    all_features = list(FEATURE_LABELS.keys())
    features = [
        f for f in all_features
        if any(f in c["features"] for c in clips)
    ]
    labels = [FEATURE_LABELS[f][0] for f in features]

    data = np.array([
        [c["features"].get(f, np.nan) for f in features]
        for c in clips
    ])

    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # blue–red

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        xticklabels=labels, yticklabels=labels,
        cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=1.2, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                       fontsize=10, rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
    ax.set_title("Feature Correlation Matrix", pad=14)

    _save(fig, output_dir, "correlation_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Balance comparison (nuScenes-only vs augmented mix)
# ---------------------------------------------------------------------------

def _balance_score(labels_list: list[dict], question: str) -> float:
    """Normalised entropy: 1.0 = perfectly uniform, 0.0 = single class."""
    answers = [lab.get(question) for lab in labels_list if question in lab]
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = sum(counts.values())
    n_classes = len(counts)
    if n_classes <= 1:
        return 0.0
    entropy = -sum((c / total) * np.log(c / total) for c in counts.values())
    return float(entropy / np.log(n_classes))


def plot_balance_comparison(clips: list[dict], output_dir: str):
    """Grouped bars comparing balance scores: nuScenes-only vs full mix."""
    ns_clips = [c for c in clips if c["source"] == "nuscenes"]
    ns_labels = [c["answers"] for c in ns_clips]
    all_labels = [c["answers"] for c in clips]
    questions = sorted(set(q for lab in all_labels for q in lab))

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(len(questions))
    width = 0.36

    ns_scores = [_balance_score(ns_labels, q) for q in questions]
    all_scores = [_balance_score(all_labels, q) for q in questions]

    ax.bar(x - width / 2, ns_scores, width,
           color=SOURCE_COLORS["nuscenes"], edgecolor="white", linewidth=0.8,
           label="nuScenes Only", alpha=0.9)
    ax.bar(x + width / 2, all_scores, width,
           color=SOURCE_COLORS["combined"], edgecolor="white", linewidth=0.8,
           label="Augmented Benchmark Mix", alpha=0.9)

    ax.axhline(y=1.0, color=TUM["grey_4"], linestyle="--", alpha=0.4,
               linewidth=0.8, label="Perfect Balance")

    ax.set_xticks(x)
    ax.set_xticklabels([_q_label(q) for q in questions], fontsize=9,
                       linespacing=0.9)
    ax.set_ylabel("Balance Score (Normalised Entropy)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Answer Balance: nuScenes Only vs. Augmented Benchmark",
                 pad=12)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor=TUM["grey_7"])
    # grid removed for clean white background
    ax.set_axisbelow(True)

    _save(fig, output_dir, "balance_comparison")


def export_balance_comparison_page_json(clips: list[dict], output_dir: str) -> None:
    """Write per-question answer-balance scores in project page format.

    Format:
        {
          "questions": [{"id": "...", "label": "..."}, ...],
          "nuScenesOnly": [0.xx, ...],     # normalised entropy, nuScenes clips
          "augmentedMix": [0.xx, ...]      # normalised entropy, all clips
        }
    """
    ns_labels = [c["answers"] for c in clips if c["source"] == "nuscenes"]
    all_labels = [c["answers"] for c in clips]
    questions = sorted(set(q for lab in all_labels for q in lab))

    payload = {
        "questions":    [{"id": q, "label": _q_label(q).replace("\n", " ")}
                         for q in questions],
        "nuScenesOnly": [round(_balance_score(ns_labels, q), 4)
                         for q in questions],
        "augmentedMix": [round(_balance_score(all_labels, q), 4)
                         for q in questions],
    }
    out_path = os.path.join(output_dir, "balance_comparison_page.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  balance_comparison_page.json → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selected", type=str, default="selected_clips.json",
                        help="Path to selected_clips.json")
    parser.add_argument("--output-dir", type=str, default="assets/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    _apply_style()

    with open(args.selected) as f:
        clips = json.load(f)

    n_ns = sum(1 for c in clips if c["source"] == "nuscenes")
    n_ca = sum(1 for c in clips if c["source"] == "carla")
    print(f"Loaded {len(clips)} clips: {n_ns} nuScenes + {n_ca} CARLA")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating figures...")
    plot_feature_distributions(clips, args.output_dir)
    plot_source_composition(clips, args.output_dir)
    plot_correlation_heatmap(clips, args.output_dir)
    plot_balance_comparison(clips, args.output_dir)
    export_balance_comparison_page_json(clips, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
