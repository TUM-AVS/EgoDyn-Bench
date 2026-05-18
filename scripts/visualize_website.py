"""Generate website-ready benchmark visualisation figures.

Reads all ``results/*.json`` files produced by the evaluation pipeline and
produces publication-quality plots using the TUM corporate colour palette.

Usage:
    python scripts/visualize_website.py
    python scripts/visualize_website.py --results_dir results --output_dir assets/figures/website
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# TUM colour palette (2017 presentation + 2022 web accents)
# ---------------------------------------------------------------------------
TUM = {
    "blue":         "#0065bd",
    "dark_blue":    "#005293",
    "light_blue":   "#64a0c8",
    "lighter_blue": "#98c6ea",
    "orange":       "#e37222",
    "green":        "#a2ad00",
    "gray":         "#999999",
    "light_gray":   "#dad7cb",
    "black":        "#000000",
    "white":        "#ffffff",
    # 2022 web extras
    "pink":         "#B55CA5",
    "yellow":       "#FED702",
    "red":          "#EA7237",
    "bright_blue":  "#8F81EA",
    "brand_blue":   "#3070B3",
    "grey_4":       "#6A757E",
    "grey_7":       "#DDE2E6",
    "grey_8":       "#EBECEF",
}

# Semantic dataset colours (consistent with LaTeX \colorlet definitions)
C_NUSCENES = TUM["orange"]      # cNuScenes = TUMOrange
C_CARLA    = TUM["blue"]        # cCARLA    = TUMBlue
C_COMBINED = TUM["light_blue"]  # cCombined = TUMLightBlue

# Per-model colour assignments (fixed so colours stay consistent across plots)
MODEL_STYLE: dict[str, dict] = {
    "gemini3_pro":         {"color": "#0065bd", "short": "Gemini 3 Pro"},
    "gemini3_pro_w_traj":  {"color": "#0065bd", "short": "Gemini 3 Pro\n+ Traj."},
    "gpt51_v2":            {"color": "#e37222", "short": "GPT-5.1"},
    "gpt51_w_traj":        {"color": "#e37222", "short": "GPT-5.1\n+ Traj."},
    "claude_sonnet":       {"color": "#a2ad00", "short": "Claude\nSonnet 4.5"},
    "claude_sonnet_w_traj":{"color": "#a2ad00", "short": "Claude Sonnet 4.5\n+ Traj."},
    "gemini2_flash":       {"color": "#64a0c8", "short": "Gemini 2.0\nFlash"},
    "gemini2_flash_w_traj":{"color": "#64a0c8", "short": "Gemini 2.0 Flash\n+ Traj."},
    "qwen3vl_30b":         {"color": "#B55CA5", "short": "Qwen3-VL\n30B-MoE"},
    "qwen3vl_8b":          {"color": "#8F81EA", "short": "Qwen3-VL\n8B"},
    "qwen3vl_4B":          {"color": "#3070B3", "short": "Qwen3-VL\n4B"},
    "internvl35_4b":       {"color": "#EA7237", "short": "InternVL3.5\n4B"},
    "internvl3_2b":        {"color": "#999999", "short": "InternVL3\n2B"},
    "cosmos_reason2_8b":   {"color": "#005293", "short": "Cosmos-\nReason2-8B"},
    "kimi_k25":            {"color": "#FED702", "short": "Kimi K2.5"},
    "qwen3vl_8b_thinking": {"color": "#6A757E", "short": "Qwen3-VL-8B\nThinking"},
    "vod_baseline":        {"color": "#dad7cb", "short": "Visual Odometry\nBaseline"},
    "flow_heuristic_baseline": {"color": "#6A757E", "short": "Flow Heuristic\nBaseline"},
}

# Full human-readable labels (single-line, for axes with enough room)
MODEL_LABELS = {k: v["short"].replace("\n", " ") for k, v in MODEL_STYLE.items()}

# Preferred display order (models not listed here are appended alphabetically)
MODEL_ORDER_PREFERENCE = [
    "gemini3_pro", "gpt51_v2", "claude_sonnet", "gemini2_flash",
    "qwen3vl_30b", "qwen3vl_8b", "qwen3vl_4B",
    "internvl35_4b", "internvl3_2b",
    "cosmos_reason2_8b", "kimi_k25",
    "qwen3vl_8b_thinking",
]

MODEL_ORDER_TRAJ_PREFERENCE = [
    "gemini3_pro_w_traj", "claude_sonnet_w_traj",
    "gemini2_flash_w_traj", "gpt51_w_traj",
]

# Fallback colours for models not in MODEL_STYLE (cycled if >8 unknown models)
_FALLBACK_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]

QUESTION_LABELS = {
    "brake_then_turn":            "Brake-then-Turn",
    "braking_intensity":          "Braking Intensity",
    "contrastive_sequence":       "Contrastive Sequence",
    "dominant_motion_axis":       "Dominant Motion Axis",
    "driving_smoothness":         "Driving Smoothness",
    "extreme_maneuver":           "Extreme Maneuver",
    "high_lateral_accel":         "High Lateral Accel.",
    "mean_speed_low":             "Low Mean Speed",
    "significant_heading_change": "Heading Change",
    "speed_peak_half":            "Speed Peak Half",
    "speed_regime":               "Speed Regime",
    "speed_trend":                "Speed Trend",
    "stop_and_go":                "Stop-and-Go",
    "yaw_rate_turn_direction":    "Turn Direction",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 250,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.8,
    })


def _build_model_list(results: dict, preferred: list[str]) -> list[str]:
    """Build display-ordered model list: preferred order first, then extras."""
    ordered = [m for m in preferred if m in results]
    remaining = sorted(k for k in results if k not in ordered)
    return ordered + remaining


def _label(m: str) -> str:
    if m in MODEL_LABELS:
        return MODEL_LABELS[m]
    return m.replace("_", " ").title()


def _color(m: str) -> str:
    if m in MODEL_STYLE:
        return MODEL_STYLE[m]["color"]
    # Deterministic fallback: hash model key to pick from palette
    idx = hash(m) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def _save(fig, output_dir: Path, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def load_results(results_dir: Path) -> dict[str, dict]:
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        name = f.stem
        if "broken" in name:
            continue
        with open(f) as fp:
            data = json.load(fp)
        if data.get("n_total", 0) < 1000:
            continue
        results[name] = data
    return results


# ---------------------------------------------------------------------------
# Figure 1: Overall model comparison
# ---------------------------------------------------------------------------

def plot_overall_comparison(results: dict, output_dir: Path,
                            vision_models: list[str] | None = None):
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    f1 = [results[m]["global"]["macro_f1"] for m in models]
    labels = [_label(m) for m in models]
    colors = [_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(models) + 0.8))
    y = np.arange(len(models))

    bars = ax.barh(y, f1, height=0.55, color=colors, edgecolor="white",
                   linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 0.60)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    # ax.xaxis.grid — removed for clean white background

    for bar, score in zip(bars, f1):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.1%}", va="center", fontsize=10, fontweight="bold",
                color=TUM["black"])

    ax.set_title("Overall Performance (Vision Only)", pad=10)
    _save(fig, output_dir, "overall_comparison")


# ---------------------------------------------------------------------------
# Figure 2: Trajectory ablation
# ---------------------------------------------------------------------------

def plot_trajectory_ablation(results: dict, output_dir: Path):
    pairs = []
    for base in ["gemini3_pro", "claude_sonnet", "gemini2_flash", "gpt51_v2"]:
        traj = f"{base}_w_traj"
        if base in results and traj in results:
            pairs.append((base, traj))
    if not pairs:
        return

    labels = [_label(b).replace(" ", "\n", 1) for b, _ in pairs]
    vo = [results[b]["global"]["macro_f1"] for b, _ in pairs]
    tr = [results[t]["global"]["macro_f1"] for _, t in pairs]

    x = np.arange(len(pairs))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, vo, w, label="Vision Only",
                color=TUM["light_blue"], edgecolor=TUM["dark_blue"], linewidth=0.6)
    b2 = ax.bar(x + w / 2, tr, w, label="Vision + Trajectory",
                color=TUM["orange"], edgecolor="#b85a1a", linewidth=0.6)

    # Delta annotations
    for i in range(len(pairs)):
        delta = tr[i] - vo[i]
        mid_x = x[i] + w / 2
        ax.annotate(f"+{delta:.0%}", xy=(mid_x, tr[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=TUM["dark_blue"])

    # Value labels inside bars
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2, h - 0.025,
                        f"{h:.0%}", ha="center", va="top", fontsize=9,
                        color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, linespacing=0.9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax.yaxis.grid — removed for clean white background
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Effect of Trajectory Features", pad=10)
    _save(fig, output_dir, "trajectory_ablation")


# ---------------------------------------------------------------------------
# Figure 3: Per-question heatmap (seaborn)
# ---------------------------------------------------------------------------

# Model grouping for structured heatmap ordering
_CLOSED_SOURCE = {
    "gemini3_pro", "gemini3_pro_w_traj",
    "gpt51_v2", "gpt51_w_traj",
    "claude_sonnet", "claude_sonnet_w_traj",
    "gemini2_flash", "gemini2_flash_w_traj",
}

_BASELINES = {
    "flow_heuristic_baseline", "flow_heuristic_simulation",
    "flow_heuristic_transferred",
    "raft_flow_heuristic_simulation", "raft_flow_heuristic_transferred",
    "tartanvo_simulation", "tartanvo_transferred",
    "vo_proxy_simulation", "vo_proxy_transferred",
    "vod_baseline",
}


def _sort_models_grouped(models: list[str], results: dict,
                         qtypes: list[str]) -> tuple[list[str], list[int]]:
    """Sort models into groups (closed → open-source → baselines), each by
    descending mean Macro F1. Returns sorted model list and row indices
    where group separators should be drawn."""

    def _mean_f1(m: str) -> float:
        pq = results.get(m, {}).get("per_question", {})
        vals = [pq.get(q, {}).get("macro_f1", 0.0) for q in qtypes]
        return float(np.nanmean(vals)) if vals else 0.0

    closed = sorted([m for m in models if m in _CLOSED_SOURCE],
                    key=_mean_f1, reverse=True)
    baselines = sorted([m for m in models if m in _BASELINES],
                       key=_mean_f1, reverse=True)
    open_src = sorted([m for m in models
                       if m not in _CLOSED_SOURCE and m not in _BASELINES],
                      key=_mean_f1, reverse=True)

    ordered: list[str] = []
    separators: list[int] = []
    for group in [closed, open_src, baselines]:
        if not group:
            continue
        if ordered:
            separators.append(len(ordered))
        ordered.extend(group)
    return ordered, separators


def _plot_heatmap(results: dict, model_order: list[str], title: str,
                  filename: str, output_dir: Path):
    """Shared helper for per-question-type heatmaps."""
    models = [m for m in model_order if m in results]
    if not models:
        return

    qtypes = sorted(results[models[0]].get("per_question", {}).keys())

    # Re-order models: closed-source → open-source → baselines, by perf.
    models, separators = _sort_models_grouped(models, results, qtypes)

    matrix = np.array([
        [results[m].get("per_question", {}).get(q, {}).get("macro_f1", np.nan)
         for q in qtypes]
        for m in models
    ])

    q_labels = [QUESTION_LABELS.get(q, q) for q in qtypes]
    m_labels = [_label(m) for m in models]

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(models) + 1.5))

    # Custom diverging colormap anchored at TUM colours
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".0%", annot_kws={"fontsize": 9, "fontweight": "bold"},
        xticklabels=q_labels, yticklabels=m_labels,
        linewidths=1.5, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Macro F1",
                  "format": mticker.PercentFormatter(xmax=1.0, decimals=0)},
    )

    # Draw thicker separator lines between groups
    for sep in separators:
        ax.axhline(y=sep, color=TUM["black"], linewidth=3)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=10, rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10.5, rotation=0)
    ax.set_title(title, pad=14)
    _save(fig, output_dir, filename)


def plot_question_heatmap(results: dict, output_dir: Path,
                          vision_models: list[str] | None = None):
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    _plot_heatmap(results, models,
                  "Performance by Question Type (Vision Only)",
                  "question_heatmap", output_dir)


def plot_question_heatmap_traj(results: dict, output_dir: Path,
                               traj_models: list[str] | None = None):
    models = traj_models or _build_model_list(results, MODEL_ORDER_TRAJ_PREFERENCE)
    _plot_heatmap(results, models,
                  "Performance by Question Type (Vision + Trajectory)",
                  "question_heatmap_traj", output_dir)


# ---------------------------------------------------------------------------
# Figure 4: Radar chart
# ---------------------------------------------------------------------------

def plot_radar_chart(results: dict, output_dir: Path,
                     vision_models: list[str] | None = None):
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if len(models) < 2:
        return

    qtypes = sorted(results[models[0]].get("per_question", {}).keys())
    q_labels = [QUESTION_LABELS.get(q, q) for q in qtypes]
    n = len(qtypes)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for m in models:
        pq = results[m].get("per_question", {})
        vals = [pq.get(q, {}).get("macro_f1", 0) for q in qtypes]
        vals += vals[:1]
        c = _color(m)
        ax.plot(angles, vals, "-", linewidth=2.2, label=_label(m), color=c)
        ax.fill(angles, vals, alpha=0.06, color=c)
        # Dot markers only (no "o-" which clutters)
        ax.scatter(angles[:-1], vals[:-1], s=28, color=c, zorder=4)

    ax.set_xticks(angles[:-1])
    # Pad labels away from the chart
    ax.set_xticklabels(q_labels, fontsize=12, fontweight="medium")
    ax.tick_params(axis="x", pad=20)

    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=10,
                       color=TUM["grey_4"])

    # Style the grid
    ax.spines["polar"].set_visible(False)
    ax.grid(color=TUM["grey_7"], linewidth=0.6)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08),
              framealpha=0.95, edgecolor=TUM["grey_7"], fontsize=12)
    ax.set_title("Per-Question Performance Comparison", y=1.10, fontsize=13,
                 fontweight="bold")
    _save(fig, output_dir, "radar_chart")


# ---------------------------------------------------------------------------
# Figure 5: Category breakdown
# ---------------------------------------------------------------------------

def plot_category_breakdown(results: dict, output_dir: Path,
                            vision_models: list[str] | None = None):
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    cats = ["direct_dynamics", "comparative"]
    cat_labels = ["Direct Dynamics", "Comparative / Temporal"]
    cat_colors = [TUM["blue"], TUM["orange"]]

    x = np.arange(len(models))
    w = 0.34

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.95), 4.5))
    for ci, (cat, clabel) in enumerate(zip(cats, cat_labels)):
        scores = [results[m].get("per_category", {}).get(cat, {}).get("macro_f1", 0)
                  for m in models]
        bars = ax.bar(x + ci * w - w / 2, scores, w, label=clabel,
                      color=cat_colors[ci], edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            if h > 0.06:
                ax.text(bar.get_x() + bar.get_width() / 2, h - 0.018,
                        f"{h:.0%}", ha="center", va="top", fontsize=7.5,
                        color="white", fontweight="bold")

    ax.set_xticks(x)
    xlabels = [_label(m) for m in models]
    ax.set_xticklabels(xlabels, fontsize=9, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax.yaxis.grid — removed for clean white background
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Performance by Question Category (Vision Only)", pad=10)
    _save(fig, output_dir, "category_breakdown")


# ---------------------------------------------------------------------------
# Figure 6: Trajectory delta
# ---------------------------------------------------------------------------

def plot_trajectory_delta(results: dict, output_dir: Path):
    for base in ["gemini3_pro", "claude_sonnet", "gemini2_flash"]:
        traj = f"{base}_w_traj"
        if base in results and traj in results:
            break
    else:
        return

    pq_base = results[base].get("per_question", {})
    pq_traj = results[traj].get("per_question", {})
    qtypes = sorted(pq_base.keys())

    data = []
    for q in qtypes:
        f1_b = pq_base.get(q, {}).get("macro_f1", 0)
        f1_t = pq_traj.get(q, {}).get("macro_f1", 0)
        data.append((QUESTION_LABELS.get(q, q), f1_t - f1_b))

    data.sort(key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in data]
    deltas = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.5))
    y = np.arange(len(labels))
    colors = [TUM["green"] if d >= 0 else TUM["red"] for d in deltas]

    bars = ax.barh(y, deltas, height=0.55, color=colors, edgecolor="white",
                   linewidth=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20)
    ax.set_xlabel("Macro F1 Improvement (pp)", fontsize=18)
    ax.axvline(x=0, color=TUM["black"], linewidth=0.8, zorder=1)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f}" if v != 0 else "0"))
    ax.tick_params(axis="x", labelsize=18)
    ax.set_axisbelow(True)

    for i, d in enumerate(deltas):
        offset = 0.004 if d >= 0 else -0.004
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.0%}", va="center", ha=ha,
                fontsize=18, fontweight="bold", color=TUM["dark_blue"])

    ax.set_title(f"Trajectory Feature Impact — {_label(base)}", pad=10,
                 fontsize=14)
    _save(fig, output_dir, "trajectory_delta")


# ---------------------------------------------------------------------------
# Figure 6b: Trajectory delta heatmap (per question × per model)
# ---------------------------------------------------------------------------

def plot_trajectory_delta_heatmap(results: dict, output_dir: Path):
    """Heatmap showing per-question F1 delta when adding trajectory features."""
    # Auto-discover all model pairs (base + base_w_traj)
    # Explicit mapping for non-standard naming (gpt51_v2 → gpt51_w_traj)
    _EXPLICIT_PAIRS = [
        ("gemini3_pro", "gemini3_pro_w_traj"),
        ("claude_sonnet", "claude_sonnet_w_traj"),
        ("gemini2_flash", "gemini2_flash_w_traj"),
        ("gpt51_v2", "gpt51_w_traj"),
        ("cosmos_reason2_8b", "cosmos_reason2_8b_w_traj"),
    ]
    pairs = [(b, t) for b, t in _EXPLICIT_PAIRS
             if b in results and t in results]
    # Also discover any additional pairs not already listed
    seen_traj = {t for _, t in pairs}
    for m in results:
        if "_w_traj" in m and m not in seen_traj:
            base = m.replace("_w_traj", "")
            if base in results:
                pairs.append((base, m))
    if not pairs:
        return

    # Build question types from first available pair
    qtypes = sorted(results[pairs[0][0]].get("per_question", {}).keys())
    if not qtypes:
        return

    # Build delta matrix: rows = model pairs, cols = question types
    matrix = []
    row_labels = []
    for base, traj in pairs:
        pq_base = results[base].get("per_question", {})
        pq_traj = results[traj].get("per_question", {})
        row = []
        for q in qtypes:
            f1_b = pq_base.get(q, {}).get("macro_f1", 0)
            f1_t = pq_traj.get(q, {}).get("macro_f1", 0)
            row.append(f1_t - f1_b)
        matrix.append(row)
        row_labels.append(_label(base))

    matrix = np.array(matrix)
    q_labels = [QUESTION_LABELS.get(q, q) for q in qtypes]

    # Symmetric color range centred on zero
    vabs = max(abs(matrix.min()), abs(matrix.max()), 0.05)

    fig, ax = plt.subplots(figsize=(12, 0.7 * len(pairs) + 2.0))

    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red ↔ green

    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=-vabs, vmax=vabs, center=0,
        annot=True, fmt="+.0%",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        xticklabels=q_labels, yticklabels=row_labels,
        linewidths=1.5, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Macro F1 Change (pp)",
                  "format": mticker.FuncFormatter(
                      lambda v, _: f"{v:+.0%}" if v != 0 else "0%")},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=10, rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
    ax.set_title("Impact of Trajectory Features by Question Type", pad=14)
    _save(fig, output_dir, "trajectory_delta_heatmap")


# ---------------------------------------------------------------------------
# Figure 7: Temporal vs. global scatter
# ---------------------------------------------------------------------------

def plot_temporal_vs_global(results: dict, output_dir: Path,
                            all_models: list[str] | None = None):
    if all_models is None:
        all_models = _build_model_list(results, MODEL_ORDER_PREFERENCE + MODEL_ORDER_TRAJ_PREFERENCE)
    if not all_models:
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))

    texts = []
    for m in all_models:
        g = results[m]["global"]["macro_f1"]
        t = results[m].get("temporal", {}).get("macro_f1")
        if t is None:
            continue
        is_traj = "_w_traj" in m
        marker = "D" if is_traj else "o"
        color = TUM["orange"] if is_traj else _color(m)
        ax.scatter(g, t, s=90, c=color, marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.8)
        texts.append(ax.text(g, t, _label(m), fontsize=11, color=TUM["black"],
                             zorder=4))

    # Set limits before adjust_text so it respects them
    x_min, x_max = 0.0, 0.88
    y_min, y_max = 0.05, 0.82
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color=TUM["gray"], alpha=0.4, zorder=1)

    # Smart label placement — constrain labels to stay within axis bounds
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=TUM["gray"],
                                               alpha=0.4, lw=0.6),
                expand=(1.6, 1.8), force_text=(1.2, 1.2),
                lim=200)
    ax.set_xlabel("Global Macro F1", fontsize=13)
    ax.set_ylabel("Temporal Macro F1", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_aspect("equal")
    # ax.xaxis.grid — removed for clean white background
    # ax.yaxis.grid — removed for clean white background
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TUM["blue"],
               markersize=8, markeredgecolor="white", label="Vision Only"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TUM["orange"],
               markersize=8, markeredgecolor="white", label="Vision + Trajectory"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95,
              edgecolor=TUM["grey_7"])
    ax.set_title("Global vs. Temporal Reasoning", pad=10)
    _save(fig, output_dir, "temporal_vs_global")


# ---------------------------------------------------------------------------
# Figure 8: Parsability
# ---------------------------------------------------------------------------

def plot_parsability(results: dict, output_dir: Path,
                     vision_models: list[str] | None = None):
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    coverages = [results[m].get("parsable_coverage", 0) for m in models]
    labels = [_label(m) for m in models]
    colors = [_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(models) + 0.8))
    y = np.arange(len(models))

    bars = ax.barh(y, coverages, height=0.55, color=colors, edgecolor="white",
                   linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Parsable Coverage")
    ax.set_xlim(0.82, 1.005)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.invert_yaxis()
    # ax.xaxis.grid — removed for clean white background
    ax.set_axisbelow(True)

    for bar, cov in zip(bars, coverages):
        # Place label inside or outside depending on bar width
        bw = bar.get_width() - 0.82  # relative to xlim start
        if bw > 0.06:
            ax.text(bar.get_width() - 0.004,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cov:.1%}", va="center", ha="right", fontsize=9.5,
                    fontweight="bold", color="white")
        else:
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cov:.1%}", va="center", ha="left", fontsize=9.5,
                    fontweight="bold", color=TUM["black"])

    ax.set_title("Answer Parsability by Model", pad=10)
    _save(fig, output_dir, "parsability")


# ---------------------------------------------------------------------------
# Figure 9b: Per-source comparison (nuScenes vs CARLA)
# ---------------------------------------------------------------------------

def plot_source_comparison(results: dict, output_dir: Path,
                           vision_models: list[str] | None = None):
    """Grouped bar chart comparing Macro F1 on nuScenes vs CARLA per model."""
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    # Keep only models that have per_source data
    models = [m for m in models if "per_source" in results.get(m, {})]
    if not models:
        return

    nu_f1 = [results[m]["per_source"].get("nuscenes", {}).get("macro_f1", 0)
             for m in models]
    ca_f1 = [results[m]["per_source"].get("carla", {}).get("macro_f1", 0)
             for m in models]
    labels = [_label(m) for m in models]

    x = np.arange(len(models))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(models) + 2), 5.5))
    bars_nu = ax.bar(x - bar_w / 2, nu_f1, bar_w, label="nuScenes",
                     color=C_NUSCENES, edgecolor="white", linewidth=0.6,
                     zorder=2)
    bars_ca = ax.bar(x + bar_w / 2, ca_f1, bar_w, label="CARLA",
                     color=C_CARLA, edgecolor="white", linewidth=0.6,
                     zorder=2)

    # Value labels on bars — only on the taller bar per model to reduce clutter
    for nu_bar, ca_bar in zip(bars_nu, bars_ca):
        for bar, is_taller in [(nu_bar, nu_bar.get_height() >= ca_bar.get_height()),
                                (ca_bar, ca_bar.get_height() > nu_bar.get_height())]:
            h = bar.get_height()
            if h > 0.02 and is_taller:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor",
                       fontsize=10)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(max(nu_f1), max(ca_f1)) * 1.18)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_axisbelow(True)
    ax.set_title("Performance by Data Source", pad=10)
    fig.tight_layout()
    _save(fig, output_dir, "source_comparison")


# ---------------------------------------------------------------------------
# Figure 10: Inference timing (latency + throughput)
# ---------------------------------------------------------------------------

def _collect_timing_entries(results: dict,
                            vision_models: list[str] | None = None) -> list[dict]:
    """Collect timing entries with GPU metadata from all results."""
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    _REQUIRED_TIMING_KEYS = {"mean_latency_s", "median_latency_s",
                              "min_latency_s", "p95_latency_s", "max_latency_s"}
    entries = []
    for m in models:
        if m not in results or "inference_timing" not in results[m]:
            continue
        timing = results[m]["inference_timing"]
        if not _REQUIRED_TIMING_KEYS.issubset(timing):
            continue
        gpu = timing.get("gpu", "GPU")
        entries.append({
            "model_key": m, "label": _label(m), "color": _color(m),
            "gpu": gpu, "timing": timing,
        })
    return entries


def _plot_timing_pair(items: list[dict], output_dir: Path,
                      suffix: str, title_extra: str = ""):
    """Render a latency + throughput figure for a list of timing entries."""
    n = len(items)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.6 * n + 1.5))

    labels = [e["label"] for e in items]
    colors = [e["color"] for e in items]
    y = np.arange(n)

    # Dot-and-whisker: colored mean dot, line from min → P95, median tick
    cap_h = 0.15
    max_val = 0
    for yi, e in enumerate(items):
        t = e["timing"]
        mn = t["min_latency_s"]
        med = t["median_latency_s"]
        mean = t["mean_latency_s"]
        p95 = t["p95_latency_s"]
        c = e["color"]
        max_val = max(max_val, p95)

        # Range line: min → P95
        ax1.hlines(yi, mn, p95, color=c, linewidth=2, alpha=0.5, zorder=2)
        # Caps
        for xp in (mn, p95):
            ax1.vlines(xp, yi - cap_h, yi + cap_h,
                       color=c, linewidth=1.5, alpha=0.5, zorder=2)
        # Median tick (slightly taller)
        ax1.vlines(med, yi - cap_h * 1.3, yi + cap_h * 1.3,
                   color=c, linewidth=2, zorder=3)
        # Mean dot (large, colored)
        ax1.scatter(mean, yi, color=c, s=80, zorder=5,
                    edgecolors="white", linewidths=1)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Latency (seconds)")
    ax1.set_title(f"Per-Request Latency Distribution{title_extra}", pad=10)
    # ax1.xaxis.grid — removed for clean white background
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, max_val * 1.1)
    ax1.legend(
        [Line2D([0], [0], color=TUM["grey_4"], marker="o", markersize=8,
                linewidth=0, markeredgecolor="white", markeredgewidth=1),
         Line2D([0], [0], color=TUM["grey_4"], marker="|", markersize=10,
                linewidth=0, markeredgewidth=2),
         Line2D([0], [0], color=TUM["grey_4"], linewidth=2, alpha=0.5)],
        ["Mean", "Median", "Min → P95"],
        loc="lower right", fontsize=8.5, framealpha=0.95,
        edgecolor=TUM["grey_7"],
    )

    throughput = [e["timing"]["throughput_req_per_s"] for e in items]
    bars2 = ax2.barh(y, throughput, height=0.5, color=colors,
                     edgecolor="white", linewidth=0.8)
    for bar, tp in zip(bars2, throughput):
        ax2.text(bar.get_width() + 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{tp:.2f}", va="center", fontsize=9.5, fontweight="bold")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Requests / second")
    ax2.set_title(f"Throughput{title_extra}", pad=10)
    # ax2.xaxis.grid — removed for clean white background
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, max(throughput) * 1.25)

    fig.tight_layout(w_pad=3)
    _save(fig, output_dir, f"inference_timing{suffix}")


def plot_inference_timing(results: dict, output_dir: Path,
                          vision_models: list[str] | None = None):
    """Generate inference timing figures, grouped by GPU when applicable."""
    entries = _collect_timing_entries(results, vision_models)
    if len(entries) < 2:
        print("  (skipped inference_timing — fewer than 2 models with data)")
        return

    # Always generate the combined figure
    _plot_timing_pair(entries, output_dir, "")

    # If multiple GPUs, also generate per-GPU figures
    gpus = sorted(set(e["gpu"] for e in entries))
    if len(gpus) > 1:
        for gpu in gpus:
            gpu_entries = [e for e in entries if e["gpu"] == gpu]
            if len(gpu_entries) >= 2:
                slug = gpu.replace(" ", "_").replace("/", "_").lower()
                _plot_timing_pair(gpu_entries, output_dir, f"_{slug}",
                                  f" ({gpu})")


# ---------------------------------------------------------------------------
# Figure 11: Accuracy vs. speed tradeoff scatter
# ---------------------------------------------------------------------------

def plot_accuracy_vs_speed(results: dict, output_dir: Path,
                           vision_models: list[str] | None = None):
    """Scatter of Macro F1 vs mean latency for models with timing data."""
    # Exclude baselines and ablation variants (text-only, 1-frame) from this plot
    _EXCLUDE_FROM_SPEED = {"flow_heuristic_baseline", "vod_baseline",
                           "qwen3_8B_text_only", "qwen3vl_8b_1_frame"}
    all_vision = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    models = [m for m in all_vision if m in results
              and "inference_timing" in results[m]
              and m not in _EXCLUDE_FROM_SPEED]
    if len(models) < 2:
        print("  (skipped accuracy_vs_speed — fewer than 2 models with data)")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for m in models:
        lat = results[m]["inference_timing"]["mean_latency_s"]
        f1 = results[m]["global"]["macro_f1"]
        color = _color(m)
        ax.scatter(lat, f1, color=color, s=120, zorder=5, edgecolors="white",
                   linewidth=1.2)
        ax.annotate(f"  {_label(m)}", (lat, f1), fontsize=9,
                    color=color, fontweight="bold", va="center")

    ax.set_xlabel("Mean Latency (s)")
    ax.set_ylabel("Macro F1")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Accuracy vs. Inference Speed", pad=12)
    # ax.xaxis.grid — removed for clean white background
    # ax.yaxis.grid — removed for clean white background
    ax.set_axisbelow(True)

    _save(fig, output_dir, "accuracy_vs_speed")


# ---------------------------------------------------------------------------
# Figure 12: Modality ablation (Qwen3-VL-8B)
# ---------------------------------------------------------------------------

# Conditions: (label, results file). Ordered from least-informative input to most.
_MODALITY_ABLATION = [
    ("No Vision",                 "qwen3_8B_text_only"),
    ("Static Frame",              "qwen3vl_8b_1_frame"),
    ("Shuffled Frames",           "qwen3vl_8b_shuffled"),
    ("Temporal Frames",           "qwen3vl_8b"),
    ("Vision + Trajectory Text",  "qwen3vl_8b_w_traj"),
    ("Text-Only + Trajectory",    "qwen3vl_8b_text_only_w_traj"),
]


def _collect_modality_rows(results: dict) -> list[tuple[str, str, float, float]]:
    rows = []
    for label, key in _MODALITY_ABLATION:
        if key not in results:
            continue
        g = results[key].get("global", {})
        rows.append((label, key,
                     float(g.get("balanced_acc", 0.0)),
                     float(g.get("macro_f1", 0.0))))
    return rows


def plot_modality_ablation(results: dict, output_dir: Path):
    """Grouped bar chart over Qwen3-VL-8B modality conditions."""
    rows = _collect_modality_rows(results)
    if len(rows) < 2:
        return

    labels = [r[0] for r in rows]
    baccs  = [r[2] for r in rows]
    f1s    = [r[3] for r in rows]

    x = np.arange(len(rows))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(8.5, 1.1 * len(rows) + 2), 5.0))
    bars_b = ax.bar(x - bar_w / 2, baccs, bar_w, label="Balanced Accuracy",
                    color=TUM["blue"], edgecolor="white", linewidth=0.6, zorder=2)
    bars_f = ax.bar(x + bar_w / 2, f1s,   bar_w, label="Macro F1",
                    color=TUM["orange"], edgecolor="white", linewidth=0.6, zorder=2)

    for bar, val in zip(list(bars_b) + list(bars_f), baccs + f1s):
        if val > 0.015:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor",
                       fontsize=10)
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(max(baccs), max(f1s)) * 1.20)
    ax.legend(fontsize=10.5, loc="upper left")
    ax.set_axisbelow(True)
    ax.set_title("Modality Ablation — Qwen3-VL-8B", pad=10)
    fig.tight_layout()
    _save(fig, output_dir, "modality_ablation")


def export_modality_ablation_page_json(results: dict, output_dir: Path) -> None:
    """Per-condition modality ablation in project page format.

    Format:
        {
          "model": "Qwen3-VL-8B",
          "conditions": [{"name": "...", "key": "...",
                          "balancedAcc": 0.xx, "macroF1": 0.xx}, ...]
        }
    """
    rows = _collect_modality_rows(results)
    if not rows:
        return

    conditions_out = [
        {"name": label, "key": key,
         "balancedAcc": round(bacc, 4), "macroF1": round(f1, 4)}
        for (label, key, bacc, f1) in rows
    ]
    payload = {"model": "Qwen3-VL-8B", "conditions": conditions_out}
    out_path = output_dir / "modality_ablation_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  modality_ablation_page.json → {out_path}")


# ---------------------------------------------------------------------------
# Figure 13: Trajectory-encoding ablation (Qwen3-VL-8B vs InternVL3.5-8B)
# ---------------------------------------------------------------------------

# encoding -> list of candidate result keys, first existing one wins
_ENCODING_LOOKUP = {
    "Qwen3-VL-8B": {
        "Summary":    ["qwen3vl_8b_w_traj"],
        "Timeseries": ["qwen3_vl_8B_timeseries"],
        "Coordinates":["qwen3vl_8b_coordinates", "qwen3vl_coordinates_8b"],
        "Full":       ["qwen3-vl-8B-full-trajectory"],
    },
    "InternVL3.5-8B": {
        "Summary":    ["internvl35_8b_w_traj", "InternVL35_w_traj_summary"],
        "Timeseries": ["InternVL3_5_8B_timeseries"],
        "Coordinates":["InternVL3_5_8B_coordinates_trajectory",
                       "InternVL3_5_coordinates_trajectory"],
        "Full":       ["InternVL3_5_w_trajectory_full"],
    },
}

_ENCODING_ORDER = ["Summary", "Timeseries", "Coordinates", "Full"]

_ENCODING_MODEL_COLOR = {
    "Qwen3-VL-8B":   "#8F81EA",
    "InternVL3.5-8B": "#0065bd",
}


def _resolve_encoding_key(results: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in results:
            return k
    return None


def _collect_encoding_rows(results: dict) -> dict[str, dict[str, dict]]:
    """Returns {model_label: {encoding: {bacc, f1, key}}}."""
    out: dict[str, dict[str, dict]] = {}
    for model_label, enc_dict in _ENCODING_LOOKUP.items():
        per_enc = {}
        for enc, cands in enc_dict.items():
            key = _resolve_encoding_key(results, cands)
            if key is None:
                continue
            g = results[key].get("global", {})
            per_enc[enc] = {
                "key":         key,
                "balancedAcc": float(g.get("balanced_acc", 0.0)),
                "macroF1":     float(g.get("macro_f1", 0.0)),
            }
        if per_enc:
            out[model_label] = per_enc
    return out


def plot_encoding_ablation(results: dict, output_dir: Path):
    """Grouped bar chart: 4 encodings × {Qwen3-VL-8B, InternVL3.5-8B}."""
    data = _collect_encoding_rows(results)
    if not data:
        return

    encodings = _ENCODING_ORDER
    models = list(data.keys())

    x = np.arange(len(encodings))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, mlabel in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * bar_w
        baccs = [data[mlabel].get(e, {}).get("balancedAcc", 0.0)
                 for e in encodings]
        color = _ENCODING_MODEL_COLOR.get(mlabel, _FALLBACK_COLORS[i])
        bars = ax.bar(x + offset, baccs, bar_w, label=mlabel,
                      color=color, edgecolor="white", linewidth=0.6, zorder=2)
        for bar, val in zip(bars, baccs):
            if val > 0.015:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(encodings, fontsize=11)
    ax.set_ylabel("Balanced Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(
        max(data[m].get(e, {}).get("balancedAcc", 0.0)
            for e in encodings) for m in models
    ) * 1.20)
    ax.legend(fontsize=10.5, loc="upper right")
    ax.set_axisbelow(True)
    ax.set_title("Trajectory-Encoding Ablation", pad=10)
    fig.tight_layout()
    _save(fig, output_dir, "encoding_ablation")


def export_encoding_ablation_page_json(results: dict, output_dir: Path) -> None:
    """Trajectory-encoding ablation in project page format.

    Format:
        {
          "encodings": ["Summary", "Timeseries", "Coordinates", "Full"],
          "models": [{"name": "...", "color": "...",
                      "balancedAcc": [...], "macroF1": [...]}, ...]
        }
    Arrays align positionally with the `encodings` axis.
    """
    data = _collect_encoding_rows(results)
    if not data:
        return

    encodings = _ENCODING_ORDER
    models_out = []
    for mlabel, enc_dict in data.items():
        baccs = [round(enc_dict.get(e, {}).get("balancedAcc", 0.0), 4)
                 for e in encodings]
        f1s   = [round(enc_dict.get(e, {}).get("macroF1",     0.0), 4)
                 for e in encodings]
        models_out.append({
            "name":        mlabel,
            "color":       _ENCODING_MODEL_COLOR.get(mlabel, "#999999"),
            "balancedAcc": baccs,
            "macroF1":     f1s,
        })

    payload = {"encodings": encodings, "models": models_out}
    out_path = output_dir / "encoding_ablation_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  encoding_ablation_page.json → {out_path}")


# ---------------------------------------------------------------------------
# Project page JSON exports
# ---------------------------------------------------------------------------

def _model_type(m: str) -> str:
    closed = {"gemini3_pro", "gemini3_pro_w_traj", "gpt51_v2", "gpt51_w_traj",
              "claude_sonnet", "claude_sonnet_w_traj", "gemini2_flash",
              "gemini2_flash_w_traj", "kimi_k25"}
    return "Closed" if m in closed else "Open"


def _model_input(m: str) -> str:
    return "Vision + Trajectory" if "_w_traj" in m else "Vision"


def export_radar_page_json(
    results: dict,
    output_dir: Path,
    vision_models: list[str] | None = None,
) -> None:
    """Write per-question macro F1 for the radar chart in project page format.

    Format:
        {
          "questions": [{"id": "...", "label": "..."}, ...],
          "models": [{"name": "...", "type": "...", "input": "...",
                      "color": "...", "perQuestionF1": [...]}, ...]
        }
    """
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    qtypes = sorted(results[models[0]].get("per_question", {}).keys())
    questions_out = [
        {"id": q, "label": QUESTION_LABELS.get(q, q)} for q in qtypes
    ]

    models_out = []
    for m in models:
        pq = results[m].get("per_question", {})
        models_out.append({
            "name":          MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":          _model_type(m),
            "input":         _model_input(m),
            "color":         _color(m),
            "perQuestionF1": [round(pq.get(q, {}).get("macro_f1", 0.0), 4)
                              for q in qtypes],
        })

    payload = {"questions": questions_out, "models": models_out}
    out_path = output_dir / "radar_chart_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  radar_chart_page.json → {out_path}")


def export_temporal_vs_global_page_json(
    results: dict,
    output_dir: Path,
    all_models: list[str] | None = None,
) -> None:
    """Write global vs. temporal macro F1 per model in project page format.

    Format:
        {
          "models": [{"name": "...", "type": "...", "input": "...",
                      "color": "...", "globalF1": 0.xx, "temporalF1": 0.xx}, ...]
        }
    Only models that have both global and temporal macro_f1 are included.
    """
    if all_models is None:
        all_models = _build_model_list(
            results, MODEL_ORDER_PREFERENCE + MODEL_ORDER_TRAJ_PREFERENCE
        )

    models_out = []
    for m in all_models:
        g = results[m].get("global", {}).get("macro_f1")
        t = results[m].get("temporal", {}).get("macro_f1")
        if g is None or t is None:
            continue
        models_out.append({
            "name":      MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":      _model_type(m),
            "input":     _model_input(m),
            "color":     _color(m),
            "globalF1":  round(float(g), 4),
            "temporalF1": round(float(t), 4),
        })

    payload = {"models": models_out}
    out_path = output_dir / "temporal_vs_global_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  temporal_vs_global_page.json → {out_path}")


def export_source_comparison_page_json(
    results: dict,
    output_dir: Path,
    vision_models: list[str] | None = None,
) -> None:
    """Write per-source macro F1 (nuScenes vs CARLA) in project page format.

    Format:
        {"models": [{"name": "...", "type": "...", "input": "...", "color": "...",
                     "nuScenesF1": 0.xx, "carlaF1": 0.xx}, ...]}
    Only models with per_source data are included.
    """
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    models = [m for m in models if "per_source" in results.get(m, {})]
    if not models:
        return

    models_out = []
    for m in models:
        ps = results[m]["per_source"]
        models_out.append({
            "name":      MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":      _model_type(m),
            "input":     _model_input(m),
            "color":     _color(m),
            "nuScenesF1": round(float(ps.get("nuscenes", {}).get("macro_f1", 0.0)), 4),
            "carlaF1":    round(float(ps.get("carla",    {}).get("macro_f1", 0.0)), 4),
        })

    payload = {"models": models_out}
    out_path = output_dir / "source_comparison_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  source_comparison_page.json → {out_path}")


def export_parsability_page_json(
    results: dict,
    output_dir: Path,
    vision_models: list[str] | None = None,
) -> None:
    """Write parsable coverage per model in project page format.

    Format:
        {"models": [{"name": "...", "type": "...", "input": "...", "color": "...",
                     "parsableCoverage": 0.xx}, ...]}
    """
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    models_out = []
    for m in models:
        cov = results[m].get("parsable_coverage")
        if cov is None:
            continue
        models_out.append({
            "name":             MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":             _model_type(m),
            "input":            _model_input(m),
            "color":            _color(m),
            "parsableCoverage": round(float(cov), 4),
        })

    payload = {"models": models_out}
    out_path = output_dir / "parsability_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  parsability_page.json → {out_path}")


def export_overall_comparison_page_json(
    results: dict,
    output_dir: Path,
    vision_models: list[str] | None = None,
) -> None:
    """Write global macro F1 per vision-only model in project page format.

    Format:
        {"models": [{"name": "...", "type": "...", "input": "...", "color": "...",
                     "globalF1": 0.xx}, ...]}
    """
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    if not models:
        return

    models_out = []
    for m in models:
        f1 = results[m].get("global", {}).get("macro_f1")
        if f1 is None:
            continue
        models_out.append({
            "name":     MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":     _model_type(m),
            "input":    _model_input(m),
            "color":    _color(m),
            "globalF1": round(float(f1), 4),
        })

    payload = {"models": models_out}
    out_path = output_dir / "overall_comparison_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  overall_comparison_page.json → {out_path}")


def _export_question_heatmap_page_json(
    results: dict,
    output_dir: Path,
    model_order: list[str],
    filename: str,
) -> None:
    """Shared helper to export a per-question heatmap as page JSON.

    Models are sorted into groups (closed → open → baselines), each by
    descending mean macro F1 — matching the heatmap plot. Separator indices
    indicate where one group ends and the next begins.

    Format:
        {
          "questions": [{"id": "...", "label": "..."}, ...],
          "groupSeparators": [N1, N2, ...],
          "models": [{"name": "...", "type": "...", "input": "...", "color": "...",
                      "group": "closed"|"open"|"baseline",
                      "perQuestionF1": [...]}, ...]
        }
    """
    models = [m for m in model_order if m in results]
    if not models:
        return

    qtypes = sorted(results[models[0]].get("per_question", {}).keys())
    ordered, separators = _sort_models_grouped(models, results, qtypes)

    def _group(m: str) -> str:
        if m in _CLOSED_SOURCE:
            return "closed"
        if m in _BASELINES:
            return "baseline"
        return "open"

    questions_out = [
        {"id": q, "label": QUESTION_LABELS.get(q, q)} for q in qtypes
    ]

    models_out = []
    for m in ordered:
        pq = results[m].get("per_question", {})
        models_out.append({
            "name":          MODEL_LABELS.get(m, m.replace("_", " ").title()),
            "type":          _model_type(m),
            "input":         _model_input(m),
            "color":         _color(m),
            "group":         _group(m),
            "perQuestionF1": [round(float(pq.get(q, {}).get("macro_f1", 0.0)), 4)
                              for q in qtypes],
        })

    payload = {
        "questions":       questions_out,
        "groupSeparators": separators,
        "models":          models_out,
    }
    out_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  {filename} → {out_path}")


def export_question_heatmap_page_json(
    results: dict,
    output_dir: Path,
    vision_models: list[str] | None = None,
) -> None:
    """Per-question heatmap (vision-only models) in project page format."""
    models = vision_models or _build_model_list(results, MODEL_ORDER_PREFERENCE)
    _export_question_heatmap_page_json(
        results, output_dir, models, "question_heatmap_page.json"
    )


def export_question_heatmap_traj_page_json(
    results: dict,
    output_dir: Path,
    traj_models: list[str] | None = None,
) -> None:
    """Per-question heatmap (vision + trajectory models) in project page format."""
    models = traj_models or _build_model_list(results, MODEL_ORDER_TRAJ_PREFERENCE)
    _export_question_heatmap_page_json(
        results, output_dir, models, "question_heatmap_traj_page.json"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate website figures")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str,
                        default="assets/figures/website")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files: {list(results.keys())}")

    # Auto-discover model lists from results
    all_models = _build_model_list(results, MODEL_ORDER_PREFERENCE + MODEL_ORDER_TRAJ_PREFERENCE)
    vision_only = [m for m in all_models if "_w_traj" not in m]
    traj_models = [m for m in all_models if "_w_traj" in m]
    print(f"  Vision-only: {vision_only}")
    print(f"  Trajectory:  {traj_models}")

    plot_overall_comparison(results, output_dir, vision_only)
    plot_trajectory_ablation(results, output_dir)
    plot_modality_ablation(results, output_dir)
    plot_encoding_ablation(results, output_dir)
    plot_question_heatmap(results, output_dir, vision_only)
    plot_question_heatmap_traj(results, output_dir, traj_models)
    plot_radar_chart(results, output_dir, vision_only)
    plot_category_breakdown(results, output_dir, vision_only)
    plot_trajectory_delta(results, output_dir)
    plot_trajectory_delta_heatmap(results, output_dir)
    plot_temporal_vs_global(results, output_dir, all_models)
    plot_parsability(results, output_dir, vision_only)
    plot_source_comparison(results, output_dir, vision_only)
    plot_inference_timing(results, output_dir, vision_only)
    plot_accuracy_vs_speed(results, output_dir, vision_only)

    export_radar_page_json(results, output_dir, vision_only)
    export_temporal_vs_global_page_json(results, output_dir, all_models)
    export_source_comparison_page_json(results, output_dir, vision_only)
    export_parsability_page_json(results, output_dir, vision_only)
    export_overall_comparison_page_json(results, output_dir, vision_only)
    export_question_heatmap_page_json(results, output_dir, vision_only)
    export_question_heatmap_traj_page_json(results, output_dir, traj_models)
    export_modality_ablation_page_json(results, output_dir)
    export_encoding_ablation_page_json(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
