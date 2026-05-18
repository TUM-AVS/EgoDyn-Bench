#!/usr/bin/env python3
"""Evaluate model predictions against oracle labels for EgoDyn-Bench.

Reads JSONL prediction files and computes Semantic Accuracy and Macro F1
at global, per-category, and per-question granularity.

``--predictions`` accepts either a single JSONL file or a directory.
When a directory is given, every ``.jsonl`` file inside it is evaluated
and the metrics are written to ``results/{stem}_eval.json``.

Usage:
    # Single file
    python scripts/evaluate.py --predictions results/gpt4o_test.jsonl

    # All predictions in a folder
    python scripts/evaluate.py --predictions results/
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.parsers import load_question_config
from evaluation.metrics import evaluate


def evaluate_file(
    pred_path: Path,
    question_config: dict,
    output_path: Path | None,
) -> dict | None:
    """Evaluate a single predictions JSONL file.

    Returns the metrics dict, or *None* if the file has no valid records.
    """
    records: list[dict] = []
    with open(pred_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: skipping malformed line {line_no} in {pred_path}: {e}",
                    file=sys.stderr,
                )

    if not records:
        print(f"Warning: no valid records in {pred_path}, skipping.", file=sys.stderr)
        return None

    result = evaluate(records, question_config)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve extra keys (e.g. inference_timing) from existing file
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)
            for k, v in existing.items():
                if k not in result:
                    result[k] = v
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"  {pred_path.name} -> {output_path}", file=sys.stderr)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to a single JSONL file or a directory containing JSONL prediction files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "configs" / "questions_template.yaml"),
        help="Path to questions_template.yaml (default: dataset/configs/questions_template.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. For a single file: write metrics JSON here. "
        "For a directory: ignored (results are written to results/<name>_eval.json).",
    )
    args = parser.parse_args()

    question_config = load_question_config(args.config)
    pred_path = Path(args.predictions)

    # --- single file ---------------------------------------------------
    if pred_path.is_file():
        output_path = Path(args.output) if args.output else None
        result = evaluate_file(pred_path, question_config, output_path)
        if result is None:
            sys.exit(1)
        if output_path is None:
            print(json.dumps(result, indent=2))
        return

    # --- directory -----------------------------------------------------
    if not pred_path.is_dir():
        print(f"Error: {pred_path} is not a file or directory.", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(pred_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: no .jsonl files found in {pred_path}", file=sys.stderr)
        sys.exit(1)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(jsonl_files)} prediction files ...", file=sys.stderr)

    n_ok = 0
    for jsonl_file in jsonl_files:
        model_name = jsonl_file.stem  # e.g. "gpt4o_test"
        out_path = results_dir / f"{model_name}_eval.json"
        result = evaluate_file(jsonl_file, question_config, out_path)
        if result is not None:
            n_ok += 1

    print(f"Done: {n_ok}/{len(jsonl_files)} files evaluated.", file=sys.stderr)

    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
