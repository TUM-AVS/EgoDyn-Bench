#!/usr/bin/env python3
"""
Visualize evaluation JSON outputs from scripts/evaluate.py

Usage:
  python scripts/viz_eval.py results/*_eval.json --save_dir results/figs

If --save_dir is omitted, figures are shown interactively.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class ModelEval:
    name: str
    path: str
    data: Dict[str, Any]


def infer_model_name(path: str) -> str:
    base = os.path.basename(path)
    # e.g., gemini_eval.json -> gemini
    for suffix in ["_eval.json", ".json"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def load_evals(paths: List[str]) -> List[ModelEval]:
    out: List[ModelEval] = []
    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)
        out.append(ModelEval(name=infer_model_name(p), path=p, data=d))
    return out


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_models(evals: List[ModelEval]):
    rows = []
    for e in evals:
        rows.append(
            dict(
                model=e.name,
                n_total=safe_get(e.data, ["n_total"], None),
                n_parsed=safe_get(e.data, ["n_parsed"], None),
                coverage=safe_get(e.data, ["parsable_coverage"], None),
                global_acc=safe_get(e.data, ["global", "accuracy"], None),
                global_f1=safe_get(e.data, ["global", "macro_f1"], None),
                temporal_acc=safe_get(e.data, ["temporal", "accuracy"], None),
                temporal_f1=safe_get(e.data, ["temporal", "macro_f1"], None),
            )
        )

    if pd is not None:
        df = pd.DataFrame(rows).sort_values("model")
        print("\n=== Model summary ===")
        print(df.to_string(index=False))
        return df
    else:
        print("\n=== Model summary (install pandas for nicer table) ===")
        for r in rows:
            print(r)
        return rows


def plot_global_bars(df, save_dir: str | None):
    models = list(df["model"])
    acc = list(df["global_acc"])
    cov = list(df["coverage"])

    x = range(len(models))
    width = 0.38

    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width / 2 for i in x], acc, width=width, label="Global accuracy")
    plt.bar([i + width / 2 for i in x], cov, width=width, label="Parsable coverage")
    plt.xticks(list(x), models, rotation=0)
    plt.ylim(0, 1.05)
    plt.title("Global accuracy vs parsable coverage")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "global_accuracy_vs_coverage.png")
        plt.savefig(out, dpi=200)
        print(f"Saved {out}")
        plt.close()
    else:
        plt.show()


def extract_per_question(e: ModelEval) -> Dict[str, Dict[str, Any]]:
    pq = safe_get(e.data, ["per_question"], default={})
    return pq if isinstance(pq, dict) else {}


def build_per_question_table(evals: List[ModelEval]):
    # Collect all question keys
    questions = sorted(
        {q for e in evals for q in extract_per_question(e).keys()}
    )

    rows = []
    for e in evals:
        pq = extract_per_question(e)
        for q in questions:
            qd = pq.get(q)
            if not isinstance(qd, dict):
                rows.append(
                    dict(model=e.name, question=q, n=0, acc=None, f1=None, has_cm=False)
                )
                continue
            rows.append(
                dict(
                    model=e.name,
                    question=q,
                    n=safe_get(qd, ["n"], 0),
                    acc=safe_get(qd, ["accuracy"], None),
                    f1=safe_get(qd, ["macro_f1"], None),
                    has_cm=isinstance(safe_get(qd, ["confusion_matrix"], None), dict),
                )
            )

    if pd is None:
        return rows, questions

    dfq = pd.DataFrame(rows).sort_values(["question", "model"])
    return dfq, questions


def plot_per_question_heatmap_like(dfq, metric: str, save_dir: str | None):
    """
    Simple matplotlib "heatmap-like" grid without seaborn.
    metric: 'acc' or 'f1'
    """
    if metric == "acc":
        value_col = "acc"
        title = "Per-question accuracy"
        fname = "per_question_accuracy.png"
    else:
        value_col = "f1"
        title = "Per-question macro-F1"
        fname = "per_question_macro_f1.png"

    questions = list(dfq["question"].unique())
    models = list(dfq["model"].unique())

    # Build matrix [questions x models]
    mat = []
    ann = []
    for q in questions:
        row = []
        row_ann = []
        for m in models:
            sub = dfq[(dfq["question"] == q) & (dfq["model"] == m)]
            v = sub[value_col].iloc[0]
            n = int(sub["n"].iloc[0])
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                row.append(float("nan"))
                row_ann.append(f"—\n(n={n})")
            else:
                row.append(float(v))
                row_ann.append(f"{v:.2f}\n(n={n})")
        mat.append(row)
        ann.append(row_ann)

    plt.figure(figsize=(max(7, 1.2 * len(models)), max(4.5, 0.45 * len(questions))))
    im = plt.imshow(mat, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.035, pad=0.02)

    plt.xticks(range(len(models)), models, rotation=0)
    plt.yticks(range(len(questions)), questions)
    plt.title(title)

    # annotate cells
    for i in range(len(questions)):
        for j in range(len(models)):
            plt.text(j, i, ann[i][j], ha="center", va="center", fontsize=9)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, fname)
        plt.savefig(out, dpi=200)
        print(f"Saved {out}")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrices(evals: List[ModelEval], save_dir: str | None, max_plots: int = 50):
    """
    For each model & question that has a confusion_matrix, plot it.
    """
    count = 0
    for e in evals:
        pq = extract_per_question(e)
        for q, qd in pq.items():
            cm = safe_get(qd, ["confusion_matrix"], None)
            if not isinstance(cm, dict):
                continue
            labels = cm.get("labels")
            matrix = cm.get("matrix")
            if not (isinstance(labels, list) and isinstance(matrix, list)):
                continue

            # matrix is list of lists
            try:
                import numpy as np
                mat = np.array(matrix, dtype=float)
            except Exception:
                continue

            plt.figure(figsize=(5.5, 4.8))
            plt.imshow(mat, aspect="auto")
            plt.colorbar(fraction=0.04, pad=0.02)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{e.name} — {q}")

            # annotate
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=10)

            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(save_dir, f"cm_{e.name}_{q}.png".replace("/", "_"))
                plt.savefig(out, dpi=200)
                plt.close()
            else:
                plt.show()

            count += 1
            if count >= max_plots:
                print(f"Stopped after {max_plots} confusion matrices (max_plots).")
                return

    if save_dir:
        print(f"Saved {count} confusion matrix plots into {save_dir}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", nargs="+", help="Paths to *_eval.json files")
    ap.add_argument("--save_dir", default=None, help="Directory to save PNGs. If omitted, shows plots.")
    ap.add_argument("--no_confusions", action="store_true", help="Skip confusion matrix plots")
    args = ap.parse_args()

    evals = load_evals(args.eval_json)

    if pd is None:
        raise SystemExit("Please install pandas: pip install pandas")

    df = summarize_models(evals)

    # Main global view
    plot_global_bars(df, args.save_dir)

    # Per-question grid
    dfq, _questions = build_per_question_table(evals)
    print("\n=== Per-question table (first 50 rows) ===")
    print(dfq.head(50).to_string(index=False))

    plot_per_question_heatmap_like(dfq, metric="acc", save_dir=args.save_dir)
    plot_per_question_heatmap_like(dfq, metric="f1", save_dir=args.save_dir)

    if not args.no_confusions:
        plot_confusion_matrices(evals, save_dir=args.save_dir)

    # Small reminder about coverage
    print("\nNote: interpret 'perfect' scores together with coverage (parsable_coverage).")


if __name__ == "__main__":
    main()
