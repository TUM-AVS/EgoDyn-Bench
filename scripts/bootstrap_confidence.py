#!/usr/bin/env python3
"""Compute bootstrap 95% confidence intervals for key evaluation metrics.

Resamples clips (with replacement) and recomputes balanced accuracy and WPCR
for each bootstrap iteration.  Produces a LaTeX-ready table for supplementary
material.

Usage:
    python scripts/bootstrap_confidence.py
    python scripts/bootstrap_confidence.py --n_bootstrap 5000 --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import balanced_accuracy, compute_consistency
from evaluation.parsers import load_question_config, normalize_numeric, parse_answer


def load_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def parse_records(
    records: list[dict],
    question_config: dict,
) -> dict[str, list[tuple[str, str, str]]]:
    """Parse records into per-clip data: {clip_id: [(qid, oracle, pred), ...]}."""
    clips: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for rec in records:
        qid = rec.get("question_id")
        clip_id = rec.get("clip_id")
        if qid is None or clip_id is None:
            continue
        oracle_raw = rec.get("oracle_label", rec.get("answer"))
        if oracle_raw is None:
            continue
        oracle_label = str(oracle_raw).lower().strip()
        qcfg = question_config.get(qid)
        if qcfg is None:
            continue
        if qcfg["answer_type"] == "numeric":
            oracle_label = normalize_numeric(oracle_label)
        pred_label = parse_answer(
            rec.get("model_answer", ""),
            qcfg["choices"],
            qcfg["answer_type"],
        )
        if pred_label is None:
            continue
        clips[clip_id].append((qid, oracle_label, pred_label))
    return dict(clips)


def compute_metrics_from_clips(
    clip_data: dict[str, list[tuple[str, str, str]]],
) -> tuple[float, float]:
    """Compute BAcc and WPCR from clip-level data."""
    all_oracle: list[str] = []
    all_pred: list[str] = []
    clip_answers: dict[str, dict[str, str]] = {}
    for clip_id, entries in clip_data.items():
        clip_answers[clip_id] = {}
        for qid, oracle, pred in entries:
            all_oracle.append(oracle)
            all_pred.append(pred)
            clip_answers[clip_id][qid] = pred
    bacc = balanced_accuracy(all_oracle, all_pred) if all_oracle else 0.0
    cons = compute_consistency(clip_answers)
    wpcr = cons["wemcr"]
    return bacc, wpcr


def bootstrap_ci(
    clip_data: dict[str, list[tuple[str, str, str]]],
    n_bootstrap: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CIs for BAcc and WPCR."""
    clip_ids = list(clip_data.keys())
    n = len(clip_ids)
    bacc_samples: list[float] = []
    wpcr_samples: list[float] = []

    for _ in range(n_bootstrap):
        # Resample clips with replacement
        sampled_ids = rng.choices(clip_ids, k=n)
        sampled_data: dict[str, list[tuple[str, str, str]]] = {}
        for i, cid in enumerate(sampled_ids):
            # Use index as key to allow duplicates
            sampled_data[f"{cid}__{i}"] = clip_data[cid]
        bacc, wpcr = compute_metrics_from_clips(sampled_data)
        bacc_samples.append(bacc)
        wpcr_samples.append(wpcr)

    lo = alpha / 2
    hi = 1 - alpha / 2

    def percentile(data: list[float], p: float) -> float:
        s = sorted(data)
        idx = p * (len(s) - 1)
        lo_i = int(idx)
        hi_i = min(lo_i + 1, len(s) - 1)
        frac = idx - lo_i
        return s[lo_i] * (1 - frac) + s[hi_i] * frac

    return {
        "balanced_acc": {
            "mean": sum(bacc_samples) / len(bacc_samples),
            "ci_lo": percentile(bacc_samples, lo),
            "ci_hi": percentile(bacc_samples, hi),
        },
        "wpcr": {
            "mean": sum(wpcr_samples) / len(wpcr_samples),
            "ci_lo": percentile(wpcr_samples, lo),
            "ci_hi": percentile(wpcr_samples, hi),
        },
    }


def derive_model_name(jsonl_path: Path) -> str:
    import re
    stem = jsonl_path.stem
    return re.sub(r"_answers$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir", type=str,
        default=str(PROJECT_ROOT / "generated"),
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"),
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for JSON results (default: print table only)",
    )
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    jsonl_files = sorted(generated_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files in {generated_dir}", file=sys.stderr)
        sys.exit(1)

    question_config = load_question_config(args.config)
    rng = random.Random(args.seed)

    # Skip ablation/variant models for cleaner table
    skip_patterns = {"broken", "10fps", "1_frame", "shuffled", "text_only",
                     "coordinates", "timeseries", "no_images", "full-trajectory"}

    results: list[dict] = []
    print(f"Computing {args.n_bootstrap}-iteration bootstrap CIs ...\n", file=sys.stderr)
    print(f"{'Model':<40} {'BAcc':>12}  {'WPCR':>12}")
    print("-" * 68)

    for jsonl_path in jsonl_files:
        model_name = derive_model_name(jsonl_path)
        if any(p in model_name for p in skip_patterns):
            continue

        records = load_records(jsonl_path)
        if not records:
            continue

        clip_data = parse_records(records, question_config)
        if not clip_data:
            continue

        # Point estimates
        bacc_point, wpcr_point = compute_metrics_from_clips(clip_data)

        # Bootstrap
        ci = bootstrap_ci(clip_data, args.n_bootstrap, rng)

        bacc_str = f"{bacc_point:.1%} [{ci['balanced_acc']['ci_lo']:.1%}, {ci['balanced_acc']['ci_hi']:.1%}]"
        wpcr_str = f"{wpcr_point:.1%} [{ci['wpcr']['ci_lo']:.1%}, {ci['wpcr']['ci_hi']:.1%}]"
        print(f"{model_name:<40} {bacc_str:>12}  {wpcr_str:>12}")

        results.append({
            "model": model_name,
            "balanced_acc": round(bacc_point, 4),
            "balanced_acc_ci_lo": round(ci["balanced_acc"]["ci_lo"], 4),
            "balanced_acc_ci_hi": round(ci["balanced_acc"]["ci_hi"], 4),
            "wpcr": round(wpcr_point, 4),
            "wpcr_ci_lo": round(ci["wpcr"]["ci_lo"], 4),
            "wpcr_ci_hi": round(ci["wpcr"]["ci_hi"], 4),
        })

    # LaTeX output
    print("\n\n% LaTeX table (copy to supplementary)")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Bootstrap 95\% confidence intervals (" + str(args.n_bootstrap) + r" iterations, clip-level resampling).}")
    print(r"\label{tab:bootstrap_ci}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Model & BAcc (\%) & WPCR (\%) \\")
    print(r"\midrule")
    for r in sorted(results, key=lambda x: x["balanced_acc"], reverse=True):
        name = r["model"].replace("_", r"\_")
        bacc = f"{r['balanced_acc']:.1%}".rstrip("%")
        bacc_ci = f"[{r['balanced_acc_ci_lo']:.1%}, {r['balanced_acc_ci_hi']:.1%}]".replace("%", "")
        wpcr = f"{r['wpcr']:.1%}".rstrip("%")
        wpcr_ci = f"[{r['wpcr_ci_lo']:.1%}, {r['wpcr_ci_hi']:.1%}]".replace("%", "")
        print(f"{name} & {bacc} {bacc_ci} & {wpcr} {wpcr_ci} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
