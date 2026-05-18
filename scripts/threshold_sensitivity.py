#!/usr/bin/env python3
"""Threshold sensitivity analysis for EgoDyn-Bench metrics.

Perturbs all labeling thresholds by multiplicative factors, re-derives oracle
labels from the raw dynamics arrays, and re-evaluates every model to show
that accuracy-based metrics and model rankings are stable.

Also verifies that:
- Oracle labels remain self-consistent (EMCR) under perturbation
- Model EMCR/WEMCR are threshold-invariant (they depend only on predictions)

Usage:
    python scripts/threshold_sensitivity.py
    python scripts/threshold_sensitivity.py --factors 0.8 0.9 1.0 1.1 1.2
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.generation.labeling_rules import apply_rule
from evaluation.metrics import (
    CONSISTENCY_RULES,
    compute_consistency,
    _compute_group_metrics,
)
from evaluation.parsers import load_question_config, parse_answer

import yaml


# ---------------------------------------------------------------------------
# Model metadata for project page export
# ---------------------------------------------------------------------------

_MODEL_META: dict[str, dict] = {
    "gemini3_pro":               {"name": "Gemini 3 Pro",              "type": "Closed",   "input": "Vision"},
    "gemini3_pro_w_traj":        {"name": "Gemini 3 Pro + Traj.",       "type": "Closed",   "input": "Vision+Traj"},
    "gpt51_v2":                  {"name": "GPT-5.1",                    "type": "Closed",   "input": "Vision"},
    "gpt51_w_traj":              {"name": "GPT-5.1 + Traj.",            "type": "Closed",   "input": "Vision+Traj"},
    "claude_sonnet":             {"name": "Claude Sonnet 4.5",          "type": "Closed",   "input": "Vision"},
    "claude_sonnet_w_traj":      {"name": "Claude Sonnet 4.5 + Traj.",  "type": "Closed",   "input": "Vision+Traj"},
    "gemini2_flash":             {"name": "Gemini 2.0 Flash",           "type": "Closed",   "input": "Vision"},
    "gemini2_flash_w_traj":      {"name": "Gemini 2.0 Flash + Traj.",   "type": "Closed",   "input": "Vision+Traj"},
    "qwen3vl_30b":               {"name": "Qwen3-VL 30B-MoE",          "type": "Open",     "input": "Vision"},
    "qwen3vl_8b":                {"name": "Qwen3-VL 8B",                "type": "Open",     "input": "Vision"},
    "qwen3vl_4B":                {"name": "Qwen3-VL 4B",                "type": "Open",     "input": "Vision"},
    "internvl35_4b":             {"name": "InternVL3.5 4B",             "type": "Open",     "input": "Vision"},
    "internvl3_2b":              {"name": "InternVL3 2B",               "type": "Open",     "input": "Vision"},
    "cosmos_reason2_8b":         {"name": "Cosmos-Reason2 8B",          "type": "Open",     "input": "Vision"},
    "cosmos_reason2_8b_w_traj":  {"name": "Cosmos-Reason2 8B + Traj.",  "type": "Open",     "input": "Vision+Traj"},
    "kimi_k25":                  {"name": "Kimi K2.5",                  "type": "Closed",   "input": "Vision"},
    "qwen3vl_8b_thinking":       {"name": "Qwen3-VL 8B Thinking",       "type": "Open",     "input": "Vision"},
    "vod_baseline":              {"name": "VO Baseline",                 "type": "Baseline", "input": "Vision"},
    "flow_heuristic_baseline":   {"name": "Flow Heuristic",             "type": "Baseline", "input": "Vision"},
}


def _model_meta(key: str) -> dict:
    if key in _MODEL_META:
        return _MODEL_META[key]
    return {
        "name": key.replace("_", " ").title(),
        "type": "Open",
        "input": "Vision+Traj" if "_w_traj" in key else "Vision",
    }


# ---------------------------------------------------------------------------
# Threshold perturbation
# ---------------------------------------------------------------------------

# Keys that represent classification thresholds (to be scaled)
_THRESHOLD_KEYS = {
    "threshold",
    "threshold_positive",
    "threshold_negative",
    "stop_speed_threshold",
    "go_speed_threshold",
    "similarity_threshold",
    "ratio_threshold",
    "min_peak_value",
    "deadzone_long",
    "deadzone_lat",
}


def perturb_params(params: dict, factor: float) -> dict:
    """Deep-copy params and scale all threshold values by *factor*."""
    out: dict = {}
    for key, value in params.items():
        if key in _THRESHOLD_KEYS and isinstance(value, (int, float)):
            out[key] = value * factor
        elif key == "thresholds" and isinstance(value, list):
            # multi_threshold_classification: scale every boundary
            out[key] = [v * factor for v in value]
        elif key == "conditions" and isinstance(value, list):
            # or_threshold_event: recurse into each condition
            out[key] = [perturb_params(c, factor) for c in value]
        elif key in ("first_event", "second_event") and isinstance(value, dict):
            # sequential_event: recurse
            out[key] = perturb_params(value, factor)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ---------------------------------------------------------------------------
# Re-labeling
# ---------------------------------------------------------------------------

def load_question_definitions(
    yaml_path: str | Path,
) -> list[dict[str, Any]]:
    """Load raw question definitions from YAML (with rule params)."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["questions"]


def relabel_clip(
    clip_features: dict[str, float],
    clip_arrays: dict[str, np.ndarray],
    questions: list[dict[str, Any]],
    factor: float,
) -> dict[str, str]:
    """Re-derive oracle labels for one clip with perturbed thresholds.

    Returns {question_id: oracle_label}.
    """
    labels: dict[str, str] = {}
    for q in questions:
        qid = q["question_id"]
        rule = q["rule"]
        perturbed = perturb_params(rule["params"], factor)
        try:
            answer, _evidence = apply_rule(
                rule["name"], clip_features, clip_arrays, perturbed,
            )
            labels[qid] = str(answer).lower()
        except Exception:
            # Skip questions that fail (e.g. missing array)
            pass
    return labels


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clips_and_arrays(
    selected_clips_path: Path,
    nuscenes_arrays_dir: Path,
    carla_arrays_dir: Path,
) -> list[dict[str, Any]]:
    """Load selected clips with their features and NPZ arrays."""
    with open(selected_clips_path) as f:
        clips = json.load(f)

    loaded = []
    for clip in clips:
        clip_id = clip["id"]
        source = clip["source"]

        # Load NPZ arrays
        if source == "nuscenes":
            npz_path = nuscenes_arrays_dir / f"{clip_id}.npz"
        else:
            npz_path = carla_arrays_dir / f"{clip_id}.npz"

        if not npz_path.exists():
            continue

        arrays = dict(np.load(npz_path))

        # Build clip_features from selected_clips.json features + derived
        features = dict(clip.get("features", {}))
        # Add max_speed if not present (needed by speed_regime)
        if "max_speed" not in features and "speed" in arrays:
            features["max_speed"] = float(np.max(arrays["speed"]))

        loaded.append({
            "clip_id": clip_id,
            "source": source,
            "features": features,
            "arrays": arrays,
            "original_answers": clip.get("answers", {}),
        })

    return loaded


def load_model_predictions(generated_dir: Path) -> dict[str, list[dict]]:
    """Load all model prediction JSONL files.

    Returns {model_name: [records]}.
    """
    import re
    models: dict[str, list[dict]] = {}
    for jsonl_path in sorted(generated_dir.glob("*.jsonl")):
        stem = jsonl_path.stem
        model_name = re.sub(r"_answers$", "", stem)
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if records:
            models[model_name] = records
    return models


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model_with_oracles(
    records: list[dict],
    new_oracles: dict[str, dict[str, str]],
    question_config: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Evaluate a model's predictions against perturbed oracle labels.

    Returns a dict with accuracy, balanced_acc, macro_f1.
    """
    oracle_list: list[str] = []
    pred_list: list[str] = []
    per_question: dict[str, dict[str, list]] = defaultdict(
        lambda: {"oracle": [], "pred": [], "labels": None}
    )

    for rec in records:
        clip_id = rec["clip_id"]
        qid = rec["question_id"]
        model_answer = rec.get("model_answer", "")

        # Get new oracle label
        clip_oracles = new_oracles.get(clip_id, {})
        oracle_label = clip_oracles.get(qid)
        if oracle_label is None:
            continue

        # Parse model answer
        qcfg = question_config.get(qid)
        if qcfg is None:
            continue
        parsed = parse_answer(model_answer, qcfg["choices"], qcfg["answer_type"])
        if parsed is None:
            continue

        oracle_list.append(oracle_label)
        pred_list.append(parsed)

        pq = per_question[qid]
        pq["oracle"].append(oracle_label)
        pq["pred"].append(parsed)
        if pq["labels"] is None:
            pq["labels"] = qcfg["choices"]

    if not oracle_list:
        return {"accuracy": 0.0, "balanced_acc": 0.0, "macro_f1": 0.0, "n": 0}

    # Use the same _compute_group_metrics from metrics.py
    metrics = _compute_group_metrics(oracle_list, pred_list, labels=None)
    return metrics


def compute_oracle_consistency(
    new_oracles: dict[str, dict[str, str]],
) -> dict[str, float]:
    """Compute EMCR/WEMCR for oracle labels themselves."""
    result = compute_consistency(new_oracles)
    return {
        "emcr": result["rate"],
        "wemcr": result["wemcr"],
        "rule_coverage": result["rule_coverage"],
        "n_evaluable": result["n_evaluable"],
    }


def kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Compute Kendall's tau-b between two rankings of model names."""
    # Build rank maps
    if not ranking_a or not ranking_b:
        return 0.0
    common = set(ranking_a) & set(ranking_b)
    if len(common) < 2:
        return 1.0

    common = sorted(common)
    rank_a = {m: i for i, m in enumerate(ranking_a) if m in common}
    rank_b = {m: i for i, m in enumerate(ranking_b) if m in common}

    n = len(common)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = common[i], common[j]
            diff_a = rank_a[mi] - rank_a[mj]
            diff_b = rank_b[mi] - rank_b[mj]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
            # ties ignored for tau-b simplification

    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    factors: list[float],
    model_metrics: dict[str, dict[float, dict]],
    oracle_consistency: dict[float, dict],
    rank_correlations: dict[float, float],
    output_dir: Path,
) -> None:
    """Generate sensitivity plots (TUM paper style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── TUM colour palette (matches visualize_website.py) ──
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
        "pink":         "#B55CA5",
        "yellow":       "#FED702",
        "red":          "#EA7237",
        "bright_blue":  "#8F81EA",
        "brand_blue":   "#3070B3",
        "grey_4":       "#6A757E",
        "grey_7":       "#DDE2E6",
        "grey_8":       "#EBECEF",
    }

    # ── Apply global rcParams (same as visualize_website.py) ──
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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model colour palette: cycle through distinguishable TUM colours
    _M_COLORS = [
        TUM["blue"], TUM["orange"], TUM["green"], TUM["pink"],
        TUM["bright_blue"], TUM["red"], TUM["brand_blue"], TUM["dark_blue"],
        TUM["light_blue"], TUM["grey_4"], TUM["lighter_blue"], TUM["yellow"],
        TUM["gray"], TUM["light_gray"],
    ]

    def _mcolor(i: int) -> str:
        return _M_COLORS[i % len(_M_COLORS)]

    def _save(fig, name: str) -> None:
        for ext in ("png", "svg", "pdf"):
            fig.savefig(output_dir / f"{name}.{ext}")
        plt.close(fig)
        print(f"  {name}")

    # ── Short model labels for legend ──
    _MODEL_SHORT = {
        "gemini3_pro": "Gemini 3 Pro",
        "gemini3_pro_w_traj": "Gemini 3 Pro + Traj.",
        "gpt51_v2": "GPT-5.1",
        "gpt51_w_traj": "GPT-5.1 + Traj.",
        "claude_sonnet": "Claude Sonnet 4.5",
        "claude_sonnet_w_traj": "Claude Sonnet 4.5 + Traj.",
        "claude_sonnet_answers_w_traj": "Claude Sonnet 4.5 + Traj.",
        "gemini2_flash": "Gemini 2.0 Flash",
        "gemini2_flash_w_traj": "Gemini 2.0 Flash + Traj.",
        "cosmos_reason2_8b": "Cosmos-Reason2 8B",
        "cosmos_reason2_8b_w_traj": "Cosmos-Reason2 8B + Traj.",
        "cosmos_reason2_2b": "Cosmos-Reason2 2B",
        "drivemm": "DriveMM",
    }

    def _mlabel(m: str) -> str:
        if m in _MODEL_SHORT:
            return _MODEL_SHORT[m]
        return m.replace("_", " ").title()

    # ── Fig 1: 2×2 grid — BAcc, Acc, F1, Ranking Stability ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    metric_keys = [
        ("balanced_acc", "Balanced Accuracy"),
        ("accuracy", "Raw Accuracy"),
        ("macro_f1", "Macro F1"),
    ]

    sorted_models = sorted(model_metrics.items())
    for ax, (mkey, mlabel) in zip(axes.flat[:3], metric_keys):
        for i, (model_name, factor_metrics) in enumerate(sorted_models):
            values = [factor_metrics[f].get(mkey, 0.0) for f in factors]
            ax.plot(factors, values, "-", alpha=0.75, linewidth=2.2,
                    color=_mcolor(i), label=_mlabel(model_name))
            ax.scatter(factors, values, color=_mcolor(i), s=20, zorder=4,
                       edgecolors="white", linewidths=0.5, alpha=0.75)
        ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
                   linewidth=0.8, zorder=1)
        ax.set_xlabel(r"Perturbation Factor $\alpha$")
        ax.set_ylabel(mlabel)
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_axisbelow(True)
        ax.set_title(mlabel, pad=8)

    # Add shared legend from first subplot (all have the same models)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # De-duplicate (claude_sonnet_answers_w_traj / claude_sonnet_w_traj)
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # Ranking stability subplot
    ax = axes.flat[3]
    tau_values = [rank_correlations[f] for f in factors]
    ax.plot(factors, tau_values, "-", color=TUM["blue"], linewidth=2.5)
    ax.scatter(factors, tau_values, color=TUM["blue"], s=50, zorder=4,
               edgecolors="white", linewidths=0.8)
    ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
               linewidth=0.8, zorder=1)
    ax.axhline(y=1.0, color=TUM["gray"], linestyle=":", alpha=0.4,
               linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Perturbation Factor $\alpha$")
    ax.set_ylabel(r"Kendall's $\tau$")
    ax.set_ylim(0.5, 1.05)
    ax.set_axisbelow(True)
    ax.set_title("Ranking Stability (BAcc)", pad=8)

    # Layout first, then place legend in the remaining space
    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    fig.subplots_adjust(right=0.76)
    fig.legend(unique_handles, unique_labels,
               bbox_to_anchor=(0.77, 0.5), loc="center left",
               fontsize=12, framealpha=0.95, edgecolor=TUM["grey_7"])
    _save(fig, "threshold_sensitivity_metrics")

    # ── Fig 2: Oracle consistency ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    oracle_emcr = [oracle_consistency[f]["emcr"] for f in factors]
    oracle_wemcr = [oracle_consistency[f]["wemcr"] for f in factors]
    oracle_rcov = [oracle_consistency[f]["rule_coverage"] for f in factors]

    ax.plot(factors, oracle_emcr, "-", label="Oracle EMCR",
            color=TUM["orange"], linewidth=2.5)
    ax.scatter(factors, oracle_emcr, color=TUM["orange"], s=50, zorder=4,
               edgecolors="white", linewidths=0.8, marker="s")
    ax.plot(factors, oracle_wemcr, "-", label="Oracle WEMCR",
            color=TUM["blue"], linewidth=2.5)
    ax.scatter(factors, oracle_wemcr, color=TUM["blue"], s=50, zorder=4,
               edgecolors="white", linewidths=0.8, marker="D")
    ax.plot(factors, oracle_rcov, "--", label="Rule Coverage",
            color=TUM["gray"], linewidth=2)
    ax.scatter(factors, oracle_rcov, color=TUM["gray"], s=40, zorder=4,
               edgecolors="white", linewidths=0.8, marker="^")

    ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
               linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Perturbation Factor $\alpha$")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.0, 1.05)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Oracle Consistency under Threshold Perturbation", pad=10)
    fig.tight_layout()
    _save(fig, "threshold_sensitivity_oracle")

    # ── Fig 3: BAcc spread (box plot per factor) ──
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bacc_spread = []
    for f in factors:
        vals = [
            model_metrics[m][f].get("balanced_acc", 0.0)
            for m in model_metrics
        ]
        bacc_spread.append(vals)

    positions = list(range(len(factors)))
    bp = ax.boxplot(
        bacc_spread, positions=positions, widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=TUM["dark_blue"], linewidth=1.5),
        whiskerprops=dict(color=TUM["grey_4"], linewidth=0.8),
        capprops=dict(color=TUM["grey_4"], linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(TUM["lighter_blue"])
        patch.set_edgecolor(TUM["blue"])
        patch.set_linewidth(0.8)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.2f}" for f in factors])
    ax.set_xlabel(r"Perturbation Factor $\alpha$")
    ax.set_ylabel("Balanced Accuracy")
    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_axisbelow(True)
    ax.set_title("Distribution of Model BAcc across Perturbations", pad=10)
    fig.tight_layout()
    _save(fig, "threshold_sensitivity_spread")

    print(f"  All plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Project page helpers
# ---------------------------------------------------------------------------

def get_temporal_qids(questions: list[dict[str, Any]]) -> set[str]:
    """Return question IDs marked temporal: true in the YAML metadata."""
    return {
        q["question_id"]
        for q in questions
        if q.get("metadata", {}).get("temporal", False)
    }


def evaluate_model_temporal(
    records: list[dict],
    new_oracles: dict[str, dict[str, str]],
    question_config: dict[str, dict[str, Any]],
    temporal_qids: set[str],
) -> dict[str, float]:
    """Evaluate a model on temporal questions only against perturbed oracles."""
    oracle_list: list[str] = []
    pred_list: list[str] = []
    for rec in records:
        if rec["question_id"] not in temporal_qids:
            continue
        clip_id = rec["clip_id"]
        qid = rec["question_id"]
        model_answer = rec.get("model_answer", "")
        oracle_label = new_oracles.get(clip_id, {}).get(qid)
        if oracle_label is None:
            continue
        qcfg = question_config.get(qid)
        if qcfg is None:
            continue
        parsed = parse_answer(model_answer, qcfg["choices"], qcfg["answer_type"])
        if parsed is None:
            continue
        oracle_list.append(oracle_label)
        pred_list.append(parsed)
    if not oracle_list:
        return {"accuracy": 0.0, "balanced_acc": 0.0, "macro_f1": 0.0, "n": 0}
    return _compute_group_metrics(oracle_list, pred_list, labels=None)


def compute_model_wemcr(
    records: list[dict],
    question_config: dict[str, dict[str, Any]],
) -> float:
    """Compute WEMCR for a model's parsed predictions (alpha-independent)."""
    pred_dict: dict[str, dict[str, str]] = {}
    for rec in records:
        clip_id = rec["clip_id"]
        qid = rec["question_id"]
        qcfg = question_config.get(qid)
        if qcfg is None:
            continue
        parsed = parse_answer(
            rec.get("model_answer", ""), qcfg["choices"], qcfg["answer_type"]
        )
        if parsed is None:
            continue
        pred_dict.setdefault(clip_id, {})[qid] = parsed
    if not pred_dict:
        return 0.0
    return compute_consistency(pred_dict)["wemcr"]


def compute_model_pcov(
    records: list[dict],
    question_config: dict[str, dict[str, Any]],
) -> float:
    """Compute parsable coverage (fraction of answers that parse cleanly)."""
    total = parsable = 0
    for rec in records:
        qcfg = question_config.get(rec["question_id"])
        if qcfg is None:
            continue
        total += 1
        if parse_answer(
            rec.get("model_answer", ""), qcfg["choices"], qcfg["answer_type"]
        ) is not None:
            parsable += 1
    return parsable / total if total > 0 else 0.0


def export_project_page_json(
    factors: list[float],
    model_metrics: dict[str, dict[float, dict]],
    model_temporal_metrics: dict[str, dict[float, dict]],
    model_wemcr: dict[str, float],
    model_pcov: dict[str, float],
    output_path: Path,
) -> None:
    """Write alpha-sensitivity data in project page format."""
    models_out = []
    for model_key in sorted(model_metrics.keys()):
        meta = _model_meta(model_key)
        fm = model_metrics[model_key]
        tm = model_temporal_metrics[model_key]
        wemcr_val = round(model_wemcr[model_key], 4)
        pcov_val = round(model_pcov[model_key], 4)
        models_out.append({
            "name":    meta["name"],
            "type":    meta["type"],
            "input":   meta["input"],
            "semBAcc": [round(fm[f].get("balanced_acc", 0.0), 4) for f in factors],
            "semF1":   [round(fm[f].get("macro_f1",     0.0), 4) for f in factors],
            "tmpAcc":  [round(tm[f].get("balanced_acc", 0.0), 4) for f in factors],
            "tmpF1":   [round(tm[f].get("macro_f1",     0.0), 4) for f in factors],
            # wemcr and pcov depend only on predictions, not on alpha — constant arrays
            "wemcr":   [wemcr_val] * len(factors),
            "pcov":    [pcov_val]  * len(factors),
        })

    payload = {
        "alphas": [round(f, 2) for f in factors],
        "models": models_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  Project page JSON saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--factors", nargs="+", type=float,
        default=[0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20],
        help="Threshold perturbation factors (1.0 = nominal)",
    )
    parser.add_argument(
        "--selected_clips", type=str,
        default=str(PROJECT_ROOT / "selected_clips.json"),
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"),
    )
    parser.add_argument(
        "--generated_dir", type=str,
        default=str(PROJECT_ROOT / "generated"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "assets" / "figures" / "sensitivity"),
    )
    parser.add_argument("--top_n", type=int, default=None, help="Only evaluate top-N models by record count")
    args = parser.parse_args()

    factors = sorted(args.factors)
    print(f"Perturbation factors: {factors}", file=sys.stderr)

    # --- Load question definitions ---
    questions = load_question_definitions(args.config)
    question_config = load_question_config(args.config)
    temporal_qids = get_temporal_qids(questions)
    print(f"Loaded {len(questions)} question definitions "
          f"({len(temporal_qids)} temporal)", file=sys.stderr)

    # --- Load clips + arrays ---
    print("Loading clips and arrays ...", file=sys.stderr)
    clips = load_clips_and_arrays(
        Path(args.selected_clips),
        PROJECT_ROOT / "output" / "nuscenes_clips" / "arrays",
        PROJECT_ROOT / "output" / "carla_clips" / "arrays",
    )
    print(f"  {len(clips)} clips loaded with arrays", file=sys.stderr)

    # --- Load model predictions ---
    print("Loading model predictions ...", file=sys.stderr)
    all_models = load_model_predictions(Path(args.generated_dir))
    if args.top_n:
        # Keep top_n by record count
        sorted_models = sorted(all_models.items(), key=lambda x: len(x[1]), reverse=True)
        all_models = dict(sorted_models[:args.top_n])
    print(f"  {len(all_models)} models loaded", file=sys.stderr)

    # --- Re-label at each perturbation factor ---
    # {factor: {clip_id: {qid: oracle_label}}}
    perturbed_oracles: dict[float, dict[str, dict[str, str]]] = {}
    oracle_consistency_results: dict[float, dict] = {}

    print("Re-labeling clips at each perturbation factor ...", file=sys.stderr)
    for factor in factors:
        oracles: dict[str, dict[str, str]] = {}
        for clip in clips:
            labels = relabel_clip(
                clip["features"], clip["arrays"], questions, factor,
            )
            oracles[clip["clip_id"]] = labels
        perturbed_oracles[factor] = oracles

        # Oracle consistency
        oc = compute_oracle_consistency(oracles)
        oracle_consistency_results[factor] = oc
        print(
            f"  factor={factor:.2f}: oracle EMCR={oc['emcr']:.4f}, "
            f"WEMCR={oc['wemcr']:.4f}, rule_cov={oc['rule_coverage']:.2f}, "
            f"n_eval={oc['n_evaluable']}",
            file=sys.stderr,
        )

    # --- Evaluate each model at each factor ---
    print("Evaluating models ...", file=sys.stderr)
    # {model: {factor: metrics_dict}}
    model_metrics: dict[str, dict[float, dict]] = {}
    model_temporal_metrics: dict[str, dict[float, dict]] = {}
    model_wemcr: dict[str, float] = {}
    model_pcov: dict[str, float] = {}

    for model_name, records in sorted(all_models.items()):
        model_metrics[model_name] = {}
        model_temporal_metrics[model_name] = {}
        for factor in factors:
            model_metrics[model_name][factor] = evaluate_model_with_oracles(
                records, perturbed_oracles[factor], question_config,
            )
            model_temporal_metrics[model_name][factor] = evaluate_model_temporal(
                records, perturbed_oracles[factor], question_config, temporal_qids,
            )
        # alpha-independent metrics
        model_wemcr[model_name] = compute_model_wemcr(records, question_config)
        model_pcov[model_name] = compute_model_pcov(records, question_config)
        # Progress
        nominal = model_metrics[model_name][1.0]
        print(
            f"  {model_name}: BAcc@1.0={nominal.get('balanced_acc', 0):.3f}, "
            f"wemcr={model_wemcr[model_name]:.3f}, "
            f"pcov={model_pcov[model_name]:.3f}, "
            f"n={nominal.get('n', 0)}",
            file=sys.stderr,
        )

    # --- Ranking stability (Kendall's tau) ---
    nominal_ranking = sorted(
        model_metrics.keys(),
        key=lambda m: model_metrics[m][1.0].get("balanced_acc", 0),
        reverse=True,
    )
    rank_correlations: dict[float, float] = {}
    for factor in factors:
        perturbed_ranking = sorted(
            model_metrics.keys(),
            key=lambda m: model_metrics[m][factor].get("balanced_acc", 0),
            reverse=True,
        )
        tau = kendall_tau(nominal_ranking, perturbed_ranking)
        rank_correlations[factor] = tau

    # --- Summary table ---
    print(f"\n{'Factor':<8}", end="")
    for model_name in sorted(model_metrics.keys()):
        short = model_name[:15]
        print(f" {short:>15}", end="")
    print(f" {'τ(rank)':>8} {'OrcEMCR':>8} {'OrcWEMCR':>9}")
    print("-" * (8 + 16 * len(model_metrics) + 28))

    for factor in factors:
        print(f"{factor:<8.2f}", end="")
        for model_name in sorted(model_metrics.keys()):
            bacc = model_metrics[model_name][factor].get("balanced_acc", 0)
            print(f" {bacc:>15.3f}", end="")
        tau = rank_correlations[factor]
        oc = oracle_consistency_results[factor]
        print(f" {tau:>8.3f} {oc['emcr']:>8.3f} {oc['wemcr']:>9.3f}")

    # --- Compute relative change statistics ---
    print(f"\nMetric stability summary (relative change from nominal):")
    print(f"{'Factor':<8} {'BAcc mean':>10} {'BAcc max':>10} {'Acc mean':>10} {'F1 mean':>10} {'τ':>8}")
    print("-" * 56)
    for factor in factors:
        bacc_changes = []
        acc_changes = []
        f1_changes = []
        for model_name in model_metrics:
            nom_bacc = model_metrics[model_name][1.0].get("balanced_acc", 0)
            pert_bacc = model_metrics[model_name][factor].get("balanced_acc", 0)
            if nom_bacc > 0:
                bacc_changes.append(abs(pert_bacc - nom_bacc) / nom_bacc)

            nom_acc = model_metrics[model_name][1.0].get("accuracy", 0)
            pert_acc = model_metrics[model_name][factor].get("accuracy", 0)
            if nom_acc > 0:
                acc_changes.append(abs(pert_acc - nom_acc) / nom_acc)

            nom_f1 = model_metrics[model_name][1.0].get("macro_f1", 0)
            pert_f1 = model_metrics[model_name][factor].get("macro_f1", 0)
            if nom_f1 > 0:
                f1_changes.append(abs(pert_f1 - nom_f1) / nom_f1)

        mean_bacc = np.mean(bacc_changes) if bacc_changes else 0
        max_bacc = np.max(bacc_changes) if bacc_changes else 0
        mean_acc = np.mean(acc_changes) if acc_changes else 0
        mean_f1 = np.mean(f1_changes) if f1_changes else 0
        tau = rank_correlations[factor]
        print(
            f"{factor:<8.2f} {mean_bacc:>10.2%} {max_bacc:>10.2%} "
            f"{mean_acc:>10.2%} {mean_f1:>10.2%} {tau:>8.3f}"
        )

    # --- Plots ---
    print("\nGenerating plots ...", file=sys.stderr)
    plot_results(
        factors,
        model_metrics,
        oracle_consistency_results,
        rank_correlations,
        Path(args.output_dir),
    )

    # --- Save raw data as JSON ---
    raw_output = {
        "factors": factors,
        "oracle_consistency": {
            str(f): oracle_consistency_results[f] for f in factors
        },
        "rank_correlations": {str(f): rank_correlations[f] for f in factors},
        "model_metrics": {
            model: {str(f): metrics for f, metrics in fmetrics.items()}
            for model, fmetrics in model_metrics.items()
        },
    }
    json_path = Path(args.output_dir) / "threshold_sensitivity.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(raw_output, f, indent=2)
        f.write("\n")
    print(f"  Raw data saved to {json_path}", file=sys.stderr)

    # --- Project page export ---
    export_project_page_json(
        factors,
        model_metrics,
        model_temporal_metrics,
        model_wemcr,
        model_pcov,
        Path(args.output_dir) / "threshold_sensitivity_page.json",
    )


if __name__ == "__main__":
    main()
