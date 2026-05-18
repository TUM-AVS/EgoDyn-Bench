#!/usr/bin/env python3
"""Build a consolidated leaderboard by evaluating generated answer JSONLs.

Reads every ``*.jsonl`` in ``generated/``, runs the full evaluation pipeline
(parsers + metrics + consistency), and writes one consolidated leaderboard
to ``leaderboard/results.json``.  Also updates individual ``results/*.json``
files so they stay in sync.

Inference timing is preserved from existing ``results/*.json`` if present.

Usage:
    python scripts/build_leaderboard.py
    python scripts/build_leaderboard.py --generated-dir generated/ --output leaderboard/results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.parsers import load_question_config
from evaluation.metrics import evaluate, CONSISTENCY_RULES


def derive_model_name(jsonl_path: Path) -> str:
    """Derive a clean model name from a JSONL filename.

    Strips common suffixes like ``_answers`` from the stem.
    """
    stem = jsonl_path.stem
    # Remove trailing _answers
    stem = re.sub(r"_answers$", "", stem)
    return stem


def load_records(jsonl_path: Path) -> list[dict]:
    """Load JSONL records, skipping malformed lines."""
    records: list[dict] = []
    with open(jsonl_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"  Warning: skipping line {line_no} in {jsonl_path.name}: {e}",
                    file=sys.stderr,
                )
    return records


def extract_entry(model_name: str, data: dict) -> dict:
    """Extract a flat leaderboard row from a full evaluation result."""
    g = data.get("global", {})
    t = data.get("temporal", {})
    c = data.get("consistency", {})

    entry: dict = {
        "model": model_name,
        # Coverage
        "n_total": data.get("n_total", 0),
        "n_parsed": data.get("n_parsed", 0),
        "parsable_coverage": data.get("parsable_coverage", 0.0),
        # Global metrics
        "accuracy": g.get("accuracy", 0.0),
        "balanced_acc": g.get("balanced_acc", 0.0),
        "macro_f1": g.get("macro_f1", 0.0),
        # Temporal subset
        "temporal_accuracy": t.get("accuracy", 0.0),
        "temporal_balanced_acc": t.get("balanced_acc", 0.0),
        "temporal_macro_f1": t.get("macro_f1", 0.0),
        # Consistency
        "emcr": c.get("rate", 0.0),
        "wemcr": c.get("wemcr", 0.0),
        "mean_compliance": c.get("mean_compliance", 0.0),
        "rule_coverage": c.get("rule_coverage", 0.0),
        "n_rules_triggered": c.get("n_rules_triggered", 0),
        "consistency_coverage": c.get("consistency_coverage", 0.0),
        "n_evaluable": c.get("n_evaluable", 0),
        "n_consistent": c.get("n_consistent", 0),
        "mean_violations": c.get("mean_violations", 0.0),
    }

    # Per-rule consistency diagnostics
    per_rule = c.get("per_rule", {})
    for rule_name, stats in sorted(per_rule.items()):
        entry[f"rule_{rule_name}_trigger"] = stats.get("n_applicable", 0)
        entry[f"rule_{rule_name}_violations"] = stats.get("n_violations", 0)
        entry[f"rule_{rule_name}_compliance"] = stats.get("compliance", 1.0)

    # Per-category metrics
    for cat_name, cat_data in sorted(data.get("per_category", {}).items()):
        entry[f"cat_{cat_name}_accuracy"] = cat_data.get("accuracy", 0.0)
        entry[f"cat_{cat_name}_balanced_acc"] = cat_data.get("balanced_acc", 0.0)
        entry[f"cat_{cat_name}_macro_f1"] = cat_data.get("macro_f1", 0.0)

    # Per-source metrics
    for src_name, src_data in sorted(data.get("per_source", {}).items()):
        entry[f"src_{src_name}_accuracy"] = src_data.get("accuracy", 0.0)
        entry[f"src_{src_name}_balanced_acc"] = src_data.get("balanced_acc", 0.0)
        entry[f"src_{src_name}_macro_f1"] = src_data.get("macro_f1", 0.0)
        src_c = src_data.get("consistency", {})
        entry[f"src_{src_name}_emcr"] = src_c.get("rate", 0.0)
        entry[f"src_{src_name}_mean_compliance"] = src_c.get("mean_compliance", 0.0)

    # Per-question accuracy (flat)
    for qid, q_data in sorted(data.get("per_question", {}).items()):
        entry[f"q_{qid}_accuracy"] = q_data.get("accuracy", 0.0)
        entry[f"q_{qid}_balanced_acc"] = q_data.get("balanced_acc", 0.0)

    # Inference timing (if present)
    timing = data.get("inference_timing")
    if timing:
        entry["inference_timing"] = timing

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=str(PROJECT_ROOT / "generated"),
        help="Directory containing answer JSONL files (default: generated/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"),
        help="Path to questions_template.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "leaderboard" / "results.json"),
        help="Output path for consolidated leaderboard (default: leaderboard/results.json)",
    )
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    if not generated_dir.is_dir():
        print(f"Error: {generated_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(generated_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: no .jsonl files found in {generated_dir}", file=sys.stderr)
        sys.exit(1)

    question_config = load_question_config(args.config)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load existing inference_timing from results/*.json
    existing_timing: dict[str, dict] = {}
    for rpath in results_dir.glob("*.json"):
        try:
            with open(rpath) as f:
                rdata = json.load(f)
            if "inference_timing" in rdata:
                existing_timing[rpath.stem] = rdata["inference_timing"]
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Evaluating {len(jsonl_files)} answer files ...", file=sys.stderr)

    leaderboard: list[dict] = []
    for jsonl_path in jsonl_files:
        model_name = derive_model_name(jsonl_path)
        records = load_records(jsonl_path)
        if not records:
            print(f"  Warning: no records in {jsonl_path.name}, skipping.", file=sys.stderr)
            continue

        result = evaluate(records, question_config)

        # Attach inference timing from existing results if available
        if model_name in existing_timing:
            result["inference_timing"] = existing_timing[model_name]

        # Write individual result JSON
        out_path = results_dir / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

        entry = extract_entry(model_name, result)
        leaderboard.append(entry)
        print(f"  {jsonl_path.name} -> {model_name} (n={result['n_parsed']}/{result['n_total']})", file=sys.stderr)

    # Sort by balanced accuracy descending
    leaderboard.sort(key=lambda e: e.get("balanced_acc", 0.0), reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
        f.write("\n")

    print(f"\nLeaderboard: {len(leaderboard)} models -> {output_path}", file=sys.stderr)

    # Print summary table
    print(f"\n{'Model':<40} {'Parse':>6} {'Acc':>6} {'SAcc':>6} {'SF1':>6} {'TAcc':>6} {'TF1':>6} {'EMCR':>6} {'WEMCR':>6} {'MCmpl':>6} {'ECov':>6}")
    print("-" * 112)
    for e in leaderboard:
        print(
            f"{e['model']:<40} "
            f"{e['parsable_coverage']:>6.1%} "
            f"{e['accuracy']:>6.1%} "
            f"{e['balanced_acc']:>6.1%} "
            f"{e['macro_f1']:>6.1%} "
            f"{e['temporal_balanced_acc']:>6.1%} "
            f"{e['temporal_macro_f1']:>6.1%} "
            f"{e['emcr']:>6.1%} "
            f"{e['wemcr']:>6.1%} "
            f"{e['mean_compliance']:>6.1%} "
            f"{e['consistency_coverage']:>6.1%}"
        )

    # Per-rule compliance summary (transposed: models as rows, rules as columns)
    rule_names = [r["name"] for r in CONSISTENCY_RULES]
    print(f"\nPer-Rule Compliance (top-15 by BAcc)")
    # Short rule labels for column headers
    short_labels = [
        "hdg→trn", "lat→trn", "str→¬hdg", "str→¬lat",
        "hwy→¬lo", "stp→lo", "stp→¬acc",
        "b+t→brk", "b+t→trn", "s&g→¬stp",
    ]
    mcol = 8  # column width
    header = f"{'Model':<35}"
    for sl in short_labels:
        header += f" {sl:>{mcol}}"
    print(header)
    print("-" * len(header))
    for e in leaderboard[:15]:
        row = f"{e['model']:<35}"
        for rn in rule_names:
            compliance = e.get(f"rule_{rn}_compliance", 1.0)
            row += f" {compliance:>{mcol}.0%}"
        print(row)
    # Legend
    print(f"\n  Rule key:")
    for sl, rn in zip(short_labels, rule_names):
        desc = next(r["description"] for r in CONSISTENCY_RULES if r["name"] == rn)
        print(f"    {sl:<10} {desc}")


if __name__ == "__main__":
    main()
