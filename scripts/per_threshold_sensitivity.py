#!/usr/bin/env python3
"""Per-threshold sensitivity analysis for EgoDyn-Bench.

Unlike the global analysis (threshold_sensitivity.py) which scales ALL
thresholds by the same factor, this script perturbs each question's
thresholds **individually** while holding all others at their nominal
values.  This produces per-question sensitivity curves.

For each of the 14 questions:
  1. Perturb only that question's threshold(s) by α ∈ [0.5, 1.5]
  2. Re-derive oracle labels for all clips (only that question changes)
  3. Re-evaluate all models and compute BAcc / Acc / F1 for that question
  4. Plot per-question sensitivity curves

Also reports:
  - Per-question oracle label-flip rate (how many clips change label)
  - Per-question model ranking stability (Kendall τ)
  - Heatmap: question × factor → mean ΔBAcc

Usage:
    python scripts/per_threshold_sensitivity.py
    python scripts/per_threshold_sensitivity.py --top_n 15
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
from evaluation.metrics import _compute_group_metrics
from evaluation.parsers import load_question_config, parse_answer

import yaml


# ---------------------------------------------------------------------------
# Reuse helpers from threshold_sensitivity.py
# ---------------------------------------------------------------------------

from scripts.threshold_sensitivity import (
    _THRESHOLD_KEYS,
    perturb_params,
    load_clips_and_arrays,
    load_model_predictions,
    load_question_definitions,
    kendall_tau,
)


# ---------------------------------------------------------------------------
# Per-question re-labeling
# ---------------------------------------------------------------------------

def relabel_one_question(
    clips: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    target_qid: str,
    factor: float,
) -> dict[str, dict[str, str]]:
    """Re-derive oracle labels, perturbing only *target_qid*'s thresholds.

    Other questions keep their nominal (factor=1.0) thresholds.

    Returns {clip_id: {qid: oracle_label}}.
    """
    oracles: dict[str, dict[str, str]] = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        features = clip["features"]
        arrays = clip["arrays"]
        labels: dict[str, str] = {}

        for q in questions:
            qid = q["question_id"]
            rule = q["rule"]
            f = factor if qid == target_qid else 1.0
            params = perturb_params(rule["params"], f)
            try:
                answer, _evidence = apply_rule(
                    rule["name"], features, arrays, params,
                )
                labels[qid] = str(answer).lower()
            except Exception:
                pass
        oracles[clip_id] = labels
    return oracles


def count_label_flips(
    nominal_oracles: dict[str, dict[str, str]],
    perturbed_oracles: dict[str, dict[str, str]],
    qid: str,
) -> int:
    """Count how many clips have a different oracle label for *qid*."""
    flips = 0
    for clip_id in nominal_oracles:
        nom = nominal_oracles[clip_id].get(qid)
        pert = perturbed_oracles[clip_id].get(qid)
        if nom is not None and pert is not None and nom != pert:
            flips += 1
    return flips


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_per_question(
    records: list[dict],
    oracles: dict[str, dict[str, str]],
    question_config: dict[str, dict[str, Any]],
    target_qid: str,
) -> dict[str, float]:
    """Evaluate only *target_qid* predictions against perturbed oracles."""
    oracle_list: list[str] = []
    pred_list: list[str] = []

    for rec in records:
        if rec["question_id"] != target_qid:
            continue
        clip_id = rec["clip_id"]
        model_answer = rec.get("model_answer", "")

        clip_oracles = oracles.get(clip_id, {})
        oracle_label = clip_oracles.get(target_qid)
        if oracle_label is None:
            continue

        qcfg = question_config.get(target_qid)
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


def evaluate_all_questions(
    records: list[dict],
    oracles: dict[str, dict[str, str]],
    question_config: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Evaluate all questions together (global BAcc) against perturbed oracles."""
    oracle_list: list[str] = []
    pred_list: list[str] = []

    for rec in records:
        clip_id = rec["clip_id"]
        qid = rec["question_id"]
        model_answer = rec.get("model_answer", "")

        clip_oracles = oracles.get(clip_id, {})
        oracle_label = clip_oracles.get(qid)
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_threshold(
    factors: list[float],
    questions: list[dict[str, Any]],
    question_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate per-threshold sensitivity plots (TUM paper style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

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
    qids = [q["question_id"] for q in questions]

    # Question labels (consistent with QUESTION_LABELS in visualize_website.py)
    QUESTION_LABELS = {
        "brake_then_turn":            "Brake-then-Turn",
        "braking_intensity":          "Braking Intensity",
        "contrastive_sequence":       "Contrastive Seq.",
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

    # Per-question colour palette: cycle through distinguishable TUM colours
    _Q_COLORS = [
        TUM["blue"], TUM["orange"], TUM["green"], TUM["pink"],
        TUM["bright_blue"], TUM["red"], TUM["brand_blue"], TUM["dark_blue"],
        TUM["light_blue"], TUM["grey_4"], TUM["lighter_blue"], TUM["yellow"],
        TUM["gray"], TUM["light_gray"],
    ]

    def _qcolor(i: int) -> str:
        return _Q_COLORS[i % len(_Q_COLORS)]

    def _save(fig, name: str) -> None:
        for ext in ("png", "svg", "pdf"):
            fig.savefig(output_dir / f"{name}.{ext}")
        plt.close(fig)
        print(f"  {name}")

    # ── Fig 1: Per-question BAcc sensitivity (model-averaged) ──
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, qid in enumerate(qids):
        qr = question_results[qid]
        mean_bacc = []
        for f in factors:
            model_baccs = qr["per_factor_model_metrics"][f]
            vals = [m.get("balanced_acc", 0.0) for m in model_baccs.values()]
            mean_bacc.append(np.mean(vals) if vals else 0.0)

        label = QUESTION_LABELS.get(qid, qid)
        ax.plot(factors, mean_bacc, "-", color=_qcolor(i),
                linewidth=2.5, label=label)
        ax.scatter(factors, mean_bacc, color=_qcolor(i), s=50, zorder=4,
                   edgecolors="white", linewidths=0.8)

    ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
               linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Threshold Perturbation Factor $\alpha$")
    ax.set_ylabel("Mean Balanced Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=12, framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Per-Question Threshold Sensitivity", pad=10)
    fig.tight_layout()
    _save(fig, "per_threshold_bacc")

    # ── Fig 2: Heatmap of ΔBAcc (question × factor) ──
    n_q = len(qids)
    n_f = len(factors)
    delta_matrix = np.zeros((n_q, n_f))

    for i, qid in enumerate(qids):
        qr = question_results[qid]
        for j, f in enumerate(factors):
            model_baccs = qr["per_factor_model_metrics"][f]
            nom_baccs = qr["per_factor_model_metrics"][1.0]
            deltas = []
            for m in model_baccs:
                nom = nom_baccs.get(m, {}).get("balanced_acc", 0)
                pert = model_baccs[m].get("balanced_acc", 0)
                deltas.append(abs(pert - nom))
            delta_matrix[i, j] = np.mean(deltas) if deltas else 0.0

    q_labels = [QUESTION_LABELS.get(q, q) for q in qids]
    f_labels = [f"{f:.2f}" for f in factors]

    fig, ax = plt.subplots(figsize=(12, 0.6 * n_q + 1.5))
    sns.heatmap(
        delta_matrix, ax=ax, cmap="YlOrRd",
        vmin=0, vmax=max(0.05, delta_matrix.max()),
        annot=True, fmt=".2%",
        annot_kws={"fontsize": 8, "fontweight": "bold"},
        xticklabels=f_labels, yticklabels=q_labels,
        linewidths=1.5, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": r"Mean $|\Delta$BAcc$|$",
                  "format": mticker.PercentFormatter(xmax=1.0, decimals=1)},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(r"Threshold Perturbation Factor $\alpha$")
    ax.set_title(r"Mean $|\Delta$BAcc$|$ per Question", pad=14)
    fig.tight_layout()
    _save(fig, "per_threshold_heatmap")

    # ── Fig 3: Oracle label flip rate per question ──
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, qid in enumerate(qids):
        qr = question_results[qid]
        flip_rates = [qr["label_flip_rates"][f] for f in factors]
        label = QUESTION_LABELS.get(qid, qid)
        ax.plot(factors, flip_rates, "-", color=_qcolor(i),
                linewidth=2.5, label=label)
        ax.scatter(factors, flip_rates, color=_qcolor(i), s=50, zorder=4,
                   edgecolors="white", linewidths=0.8)

    ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
               linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Threshold Perturbation Factor $\alpha$")
    ax.set_ylabel("Oracle Label Flip Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.02, None)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=12, framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Oracle Label Flip Rate per Question", pad=10)
    fig.tight_layout()
    _save(fig, "per_threshold_flip_rate")

    # ── Fig 4: Global ranking stability ──
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, qid in enumerate(qids):
        qr = question_results[qid]
        taus = [qr["global_rank_tau"][f] for f in factors]
        label = QUESTION_LABELS.get(qid, qid)
        ax.plot(factors, taus, "-", color=_qcolor(i),
                linewidth=2.5, label=label)
        ax.scatter(factors, taus, color=_qcolor(i), s=50, zorder=4,
                   edgecolors="white", linewidths=0.8)

    ax.axvline(x=1.0, color=TUM["black"], linestyle="--", alpha=0.25,
               linewidth=0.8, zorder=1)
    ax.axhline(y=1.0, color=TUM["gray"], linestyle=":", alpha=0.4,
               linewidth=0.8, zorder=1)
    ax.set_xlabel(r"Threshold Perturbation Factor $\alpha$")
    ax.set_ylabel(r"Kendall's $\tau$ (vs. nominal ranking)")
    ax.set_ylim(0.7, 1.05)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=12, framealpha=0.95, edgecolor=TUM["grey_7"])
    ax.set_title("Global Model Ranking Stability", pad=10)
    fig.tight_layout()
    _save(fig, "per_threshold_ranking")

    print(f"  All plots saved to {output_dir}/")


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
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50],
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
    parser.add_argument(
        "--top_n", type=int, default=None,
        help="Only evaluate top-N models by record count",
    )
    args = parser.parse_args()

    factors = sorted(args.factors)
    if 1.0 not in factors:
        factors.append(1.0)
        factors.sort()
    print(f"Perturbation factors: {factors}", file=sys.stderr)

    # --- Load data ---
    questions = load_question_definitions(args.config)
    question_config = load_question_config(args.config)
    qids = [q["question_id"] for q in questions]
    print(f"Loaded {len(questions)} question definitions", file=sys.stderr)

    print("Loading clips and arrays ...", file=sys.stderr)
    clips = load_clips_and_arrays(
        Path(args.selected_clips),
        PROJECT_ROOT / "output" / "nuscenes_clips" / "arrays",
        PROJECT_ROOT / "output" / "carla_clips" / "arrays",
    )
    print(f"  {len(clips)} clips loaded", file=sys.stderr)

    print("Loading model predictions ...", file=sys.stderr)
    all_models = load_model_predictions(Path(args.generated_dir))
    if args.top_n:
        sorted_models = sorted(
            all_models.items(), key=lambda x: len(x[1]), reverse=True
        )
        all_models = dict(sorted_models[: args.top_n])
    print(f"  {len(all_models)} models loaded", file=sys.stderr)

    # --- Nominal oracles ---
    print("Computing nominal oracle labels ...", file=sys.stderr)
    nominal_oracles = relabel_one_question(clips, questions, "__none__", 1.0)

    # --- Nominal global model ranking (by BAcc) ---
    nominal_global: dict[str, float] = {}
    for model_name, records in all_models.items():
        metrics = evaluate_all_questions(records, nominal_oracles, question_config)
        nominal_global[model_name] = metrics.get("balanced_acc", 0.0)
    nominal_ranking = sorted(
        nominal_global.keys(),
        key=lambda m: nominal_global[m],
        reverse=True,
    )

    # --- Per-question analysis ---
    question_results: dict[str, dict] = {}
    n_clips = len(clips)

    for qi, qid in enumerate(qids):
        print(
            f"\n[{qi+1}/{len(qids)}] Analysing: {qid}",
            file=sys.stderr,
        )
        qr: dict[str, Any] = {
            "label_flip_rates": {},
            "per_factor_model_metrics": {},
            "global_rank_tau": {},
        }

        for factor in factors:
            # Re-label with only this question perturbed
            oracles = relabel_one_question(clips, questions, qid, factor)

            # Label flip rate
            flips = count_label_flips(nominal_oracles, oracles, qid)
            qr["label_flip_rates"][factor] = flips / n_clips if n_clips > 0 else 0.0

            # Per-model metrics for this question
            model_metrics: dict[str, dict] = {}
            for model_name, records in all_models.items():
                m = evaluate_per_question(records, oracles, question_config, qid)
                model_metrics[model_name] = m
            qr["per_factor_model_metrics"][factor] = model_metrics

            # Global ranking stability: re-evaluate ALL questions (only qid changed)
            global_baccs: dict[str, float] = {}
            for model_name, records in all_models.items():
                gm = evaluate_all_questions(records, oracles, question_config)
                global_baccs[model_name] = gm.get("balanced_acc", 0.0)

            perturbed_ranking = sorted(
                global_baccs.keys(),
                key=lambda m: global_baccs[m],
                reverse=True,
            )
            tau = kendall_tau(nominal_ranking, perturbed_ranking)
            qr["global_rank_tau"][factor] = tau

        # Summary for this question
        flip_at_edges = max(
            qr["label_flip_rates"].get(factors[0], 0),
            qr["label_flip_rates"].get(factors[-1], 0),
        )
        tau_at_edges = min(
            qr["global_rank_tau"].get(factors[0], 1),
            qr["global_rank_tau"].get(factors[-1], 1),
        )
        print(
            f"  max flip rate: {flip_at_edges:.1%}, "
            f"min global τ: {tau_at_edges:.3f}",
            file=sys.stderr,
        )

        question_results[qid] = qr

    # --- Summary table ---
    print(f"\n{'Question':<30} {'Max Flip%':>10} {'Min τ':>8} {'Max ΔBAcc':>10}")
    print("-" * 60)
    for qid in qids:
        qr = question_results[qid]
        max_flip = max(qr["label_flip_rates"].values())
        min_tau = min(qr["global_rank_tau"].values())

        # Max mean ΔBAcc across factors
        max_delta = 0.0
        nom_metrics = qr["per_factor_model_metrics"][1.0]
        for f in factors:
            pert_metrics = qr["per_factor_model_metrics"][f]
            deltas = []
            for m in pert_metrics:
                nom = nom_metrics.get(m, {}).get("balanced_acc", 0)
                pert = pert_metrics[m].get("balanced_acc", 0)
                deltas.append(abs(pert - nom))
            mean_d = np.mean(deltas) if deltas else 0.0
            max_delta = max(max_delta, mean_d)

        print(f"{qid:<30} {max_flip:>10.1%} {min_tau:>8.3f} {max_delta:>10.2%}")

    # --- Plots ---
    print("\nGenerating plots ...", file=sys.stderr)
    plot_per_threshold(factors, questions, question_results, Path(args.output_dir))

    # --- Save raw data ---
    raw_output: dict[str, Any] = {
        "factors": factors,
        "questions": qids,
        "per_question": {},
    }
    for qid in qids:
        qr = question_results[qid]
        raw_output["per_question"][qid] = {
            "label_flip_rates": {str(f): v for f, v in qr["label_flip_rates"].items()},
            "global_rank_tau": {str(f): v for f, v in qr["global_rank_tau"].items()},
            "per_factor_mean_bacc": {},
        }
        for f in factors:
            model_metrics = qr["per_factor_model_metrics"][f]
            baccs = [m.get("balanced_acc", 0) for m in model_metrics.values()]
            raw_output["per_question"][qid]["per_factor_mean_bacc"][str(f)] = (
                float(np.mean(baccs)) if baccs else 0.0
            )

    json_path = Path(args.output_dir) / "per_threshold_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(raw_output, f, indent=2)
        f.write("\n")
    print(f"  Raw data saved to {json_path}", file=sys.stderr)

    # --- Project page exports ---
    out_dir = Path(args.output_dir)
    export_per_question_page_json(factors, qids, question_results, out_dir)
    export_flip_rate_page_json(factors, qids, question_results, out_dir)


def export_per_question_page_json(
    factors: list[float],
    qids: list[str],
    question_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Write per-question mean BAcc sensitivity in project page format.

    Format: {"alphas": [...], "questions": [{"id": ..., "meanBAcc": [...]}]}
    """
    questions_out = []
    for qid in qids:
        qr = question_results[qid]
        mean_baccs = []
        for f in factors:
            model_metrics = qr["per_factor_model_metrics"][f]
            baccs = [m.get("balanced_acc", 0.0) for m in model_metrics.values()]
            mean_baccs.append(round(float(np.mean(baccs)) if baccs else 0.0, 4))
        questions_out.append({"id": qid, "meanBAcc": mean_baccs})

    payload = {
        "alphas": [round(f, 2) for f in factors],
        "questions": questions_out,
    }
    out_path = output_dir / "per_question_sensitivity_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  Per-question sensitivity page JSON saved to {out_path}", file=sys.stderr)


def export_flip_rate_page_json(
    factors: list[float],
    qids: list[str],
    question_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Write per-question oracle label flip rates in project page format.

    Format: {"alphas": [...], "questions": [{"id": ..., "flipRate": [...]}]}
    """
    questions_out = []
    for qid in qids:
        flip_rates = question_results[qid]["label_flip_rates"]
        questions_out.append({
            "id": qid,
            "flipRate": [round(float(flip_rates[f]), 4) for f in factors],
        })

    payload = {
        "alphas": [round(f, 2) for f in factors],
        "questions": questions_out,
    }
    out_path = output_dir / "flip_rate_page.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"  Flip rate page JSON saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
