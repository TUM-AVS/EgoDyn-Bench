#!/usr/bin/env python3
"""
Select a balanced subset of clips from pre-generated QA files (nuScenes + CARLA).

Loads QA pairs from JSONL files, groups them by clip, and uses a greedy algorithm
to select N clips that minimize answer-class imbalance across all categorical questions.

Usage:
    python scripts/improved_clip_selection.py --target 3000 --output selected_clips.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_qa_data(qa_paths: list[tuple[str, str]]) -> list[dict]:
    """
    Load QA pairs from multiple files and group them by clip.
    
    Args:
        qa_paths: List of (source_name, file_path) tuples.
        
    Returns:
        List of clip dictionaries. Each clip has:
        - id: str
        - source: str
        - answers: dict {question_id: answer}
        - qas: list of full QA objects associated with this clip
    """
    clips_map = {}  # clip_id -> clip_dict

    for source, path in qa_paths:
        path = Path(path)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
            
        print(f"Loading {source} QA from {path}...")
        count = 0
        with open(path, 'r') as f:
            for line in f:
                try:
                    qa = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                clip_id = qa['clip_id']
                
                if clip_id not in clips_map:
                    clips_map[clip_id] = {
                        "id": clip_id,
                        "source": source,
                        "answers": {},
                        "qas": [],
                        "valid_for_balancing": True
                    }
                
                # Add QA info
                clips_map[clip_id]["qas"].append(qa)
                
                # We only balance on non-numeric answers
                if qa.get('answer_type') != 'numeric':
                    q_id = qa['question_id']
                    clips_map[clip_id]["answers"][q_id] = qa['answer']

                # Extract statistics from evidence
                ev = qa.get("evidence", {})
                if qa['question_id'] == "mean_speed_low" and "value_mean_speed" in ev:
                    clips_map[clip_id].setdefault("stats", {})["mean_speed"] = ev["value_mean_speed"]
                elif qa['question_id'] == "driving_smoothness" and "mean_mean_abs_jerk" in ev:
                    clips_map[clip_id].setdefault("stats", {})["mean_abs_jerk"] = ev["mean_mean_abs_jerk"]
                elif qa['question_id'] == "high_lateral_accel" and "max_lateral_accel" in ev:
                    clips_map[clip_id].setdefault("stats", {})["max_lateral_accel"] = ev["max_lateral_accel"]
                elif qa['question_id'] == "significant_heading_change" and "value_total_heading_change" in ev:
                    clips_map[clip_id].setdefault("stats", {})["total_heading_change"] = ev["value_total_heading_change"]
                elif qa['question_id'] == "braking_intensity" and "min_accel" in ev:
                    clips_map[clip_id].setdefault("stats", {})["min_accel"] = ev["min_accel"]
                elif qa['question_id'] == "max_speed_kmh" and "raw_value" in ev:
                     clips_map[clip_id].setdefault("stats", {})["max_speed"] = ev["raw_value"]
                elif qa['question_id'] == "yaw_rate_turn_direction" and "abs_max_yaw_rate" in ev:
                     clips_map[clip_id].setdefault("stats", {})["max_abs_yaw_rate"] = ev["abs_max_yaw_rate"]
                
                count += 1
        print(f"  Loaded {count} QA pairs for {source}")

    clips = list(clips_map.values())
    print(f"Total unique clips loaded: {len(clips)}")
    return clips


# ---------------------------------------------------------------------------
# Greedy balanced selection
# ---------------------------------------------------------------------------
def compute_imbalance(counts: dict, target_frac: dict) -> float:
    """
    Compute imbalance score for one question.
    Returns the maximum deviation of any answer class from its target fraction.
    """
    total = sum(counts.values())
    if total == 0:
        return 1.0
    max_dev = 0.0
    for cls, target in target_frac.items():
        actual = counts.get(cls, 0) / total
        max_dev = max(max_dev, abs(actual - target))
    return max_dev


def greedy_select(
    pool: list[dict],
    target_n: int,
    seed: int = 42,
    min_nuscenes_frac: float = 0.0,
    min_carla_frac: float = 0.0,
    ignored_questions: list[str] = None
) -> list[int]:
    """
    Greedily select clips to minimize answer imbalance.
    """
    if ignored_questions is None:
        ignored_questions = []
        
    rng = np.random.default_rng(seed)

    # 1. Determine questions and target distributions from the ENTIRE pool
    #    (This assumes the pool covers the full possibility space)
    question_classes = defaultdict(set)
    for clip in pool:
        for q, ans in clip["answers"].items():
            if q not in ignored_questions:
                question_classes[q].add(ans)

    # Target: uniform distribution across classes for each question
    targets = {}
    for q, classes in question_classes.items():
        n_classes = len(classes)
        targets[q] = {cls: 1.0 / n_classes for cls in classes}

    questions = sorted(targets.keys())
    print(f"\n  Balancing on {len(questions)} questions: {questions}")
    # for q in questions:
    #     print(f"    {q}: target {targets[q]}")

    # 2. Pre-index the pool
    #    q_ans_indices[q][ans] -> list of pool indices
    q_ans_indices = defaultdict(lambda: defaultdict(list))
    for idx, clip in enumerate(pool):
        for q, ans in clip["answers"].items():
            if q in targets:
                q_ans_indices[q][ans].append(idx)

    # Source-ratio constraint info
    if min_nuscenes_frac > 0 or min_carla_frac > 0:
        print(f"\n  Source constraints: min nuScenes >= {min_nuscenes_frac:.0%}, "
              f"min CARLA >= {min_carla_frac:.0%}")

    # Selection loop variables
    selected = []
    selected_set = set()
    running_counts = {q: Counter() for q in questions}
    source_counts = Counter()

    n_pool = len(pool)
    print(f"\n  Selecting {target_n} clips from pool of {n_pool}...")

    # Pre-calculate caps
    ns_cap = target_n - int(np.ceil(min_carla_frac * target_n))   # max nuScenes allowed
    ca_cap = target_n - int(np.ceil(min_nuscenes_frac * target_n))  # max CARLA allowed

    for step in range(target_n):
        # A. Find the most imbalanced question based on current selection
        worst_q = None
        worst_imb = -1
        
        # If we have no clips yet, pick randomly or based on first question
        if not selected:
             # Just pick the first question to start
            if questions:
                worst_q = questions[0]
        else:
            for q in questions:
                imb = compute_imbalance(running_counts[q], targets[q])
                if imb > worst_imb:
                    worst_imb = imb
                    worst_q = q
        
        if worst_q is None and questions:
             worst_q = questions[0]

        # B. Find the most underrepresented class for that question
        if worst_q:
            total = max(sum(running_counts[worst_q].values()), 1)
            worst_cls = None
            worst_deficit = float("inf")
            for cls, target_f in targets[worst_q].items():
                actual_f = running_counts[worst_q].get(cls, 0) / total
                deficit = actual_f - target_f  # negative = underrepresented
                if deficit < worst_deficit:
                    worst_deficit = deficit
                    worst_cls = cls
        else:
            # Should not happen if there are questions
            worst_cls = None

        # C. Find candidate clips with that answer
        candidates = []
        if worst_q and worst_cls:
            candidates = [
                idx for idx in q_ans_indices[worst_q][worst_cls]
                if idx not in selected_set
            ]

        # Fallback 1: If no candidates for the worst class (or no questions), 
        # try to find any unselected clip.
        if not candidates:
            candidates = [i for i in range(n_pool) if i not in selected_set]
            if not candidates:
                print("    Pool exhausted.")
                break

        # D. Apply source-ratio constraints (Caps)
        excluded_sources = set()
        if source_counts.get("nuscenes", 0) >= ns_cap:
            excluded_sources.add("nuscenes")
        if source_counts.get("carla", 0) >= ca_cap:
            excluded_sources.add("carla")

        if excluded_sources:
            filtered_candidates = [
                idx for idx in candidates
                if pool[idx]["source"] not in excluded_sources
            ]
            if filtered_candidates:
                candidates = filtered_candidates
            else:
                # Our preferred candidates (helping the balance) are all from excluded sources.
                # We must pick ANY clip from an allowed source to satisfy source constraints,
                # even if it hurts balance.
                fallback_candidates = [
                    i for i in range(n_pool)
                    if i not in selected_set
                    and pool[i]["source"] not in excluded_sources
                ]
                if fallback_candidates:
                    candidates = fallback_candidates
                else:
                    # No allowed clips left at all?
                    print("    Pool exhausted (constraints met).")
                    break

        # E. Secondary Optimization: Pick best candidate among valid ones
        #    "Best" means it helps balance other questions too.
        if len(candidates) > 1000:
            candidates = list(rng.choice(candidates, size=1000, replace=False))

        best_idx = None
        best_secondary = -1.0

        for idx in candidates:
            secondary_score = 0.0
            clip_answers = pool[idx]["answers"]
            
            for q in questions:
                if q == worst_q: 
                    continue
                ans = clip_answers.get(q)
                if ans is None:
                    continue
                
                # Check if this answer helps this question
                q_total = max(sum(running_counts[q].values()), 1)
                actual_f = running_counts[q].get(ans, 0) / q_total
                target_f = targets[q].get(ans, 0.0) # 0.0 if unknown class? shouldn't happen
                
                if actual_f < target_f:
                    # Helps an underrepresented class
                    secondary_score += (target_f - actual_f)
            
            if secondary_score > best_secondary:
                best_secondary = secondary_score
                best_idx = idx

        if best_idx is None:
            best_idx = rng.choice(candidates)

        # F. Add to selection
        selected.append(best_idx)
        selected_set.add(best_idx)
        source_counts[pool[best_idx]["source"]] += 1
        
        for q, ans in pool[best_idx]["answers"].items():
            if q in running_counts:
                running_counts[q][ans] += 1

        if (step + 1) % 500 == 0 or step == target_n - 1:
            print(f"    Step {step+1}/{target_n}")

    return selected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_selection(pool: list[dict], selected_indices: list[int]):
    """Print distribution report for the selected subset."""
    selected = [pool[i] for i in selected_indices]
    n = len(selected)

    # Source breakdown
    n_ns = sum(1 for c in selected if c["source"] == "nuscenes")
    n_ca = sum(1 for c in selected if c["source"] == "carla")
    print(f"\n{'='*80}")
    print(f"SELECTED {n} CLIPS: {n_ns} nuScenes ({100*n_ns/n:.1f}%) + {n_ca} CARLA ({100*n_ca/n:.1f}%)")
    print(f"{'='*80}")

    # Collect all questions
    all_qs = set()
    for c in selected:
        all_qs.update(c["answers"].keys())
    
    questions = sorted(all_qs)

    print(f"\n  {'Question':28s} {'Class':25s} {'Count':>7s} {'Pct':>7s}")
    print("  " + "-" * 70)
    for q in questions:
        # Collect answers for this question from selected clips
        answers = [c["answers"].get(q) for c in selected if q in c["answers"]]
        if not answers:
            continue
            
        counts = Counter(answers)
        total = sum(counts.values())
        
        # Skip if single class or mostly empty (optional, but good for noise reduction)
        # if len(counts) < 2: continue

        print(f"  [ {q} ]")
        for cls, cnt in sorted(counts.items(), key=lambda x: str(x[0])):
            print(f"  {'':28s} {str(cls):25s} {cnt:7d} {100*cnt/total:6.1f}%")
        print()
    
    # Feature distribution summary
    feature_keys = [
        "mean_speed", "mean_abs_jerk", "max_lateral_accel", "min_accel", "total_heading_change"
    ]
    print(f"  {'Feature':28s} {'Mean':>9s} {'P50':>9s} {'P95':>9s}")
    print("  " + "-" * 60)
    for fk in feature_keys:
        vals = [c.get("stats", {}).get(fk) for c in selected if c.get("stats", {}).get(fk) is not None]
        if not vals:
            continue
        vals = np.array(vals)
        print(f"  {fk:28s} {np.mean(vals):9.4f} {np.percentile(vals, 50):9.4f} {np.percentile(vals, 95):9.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=3000,
                        help="Target number of clips to select (default: 3000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Input paths
    parser.add_argument("--nuscenes-qa", type=str,
                        default="output/nuscenes_clips/qa.jsonl",
                        help="Path to nuScenes QA JSONL")
    parser.add_argument("--carla-qa", type=str,
                        default="output/carla_clips/qa.jsonl",
                        help="Path to CARLA QA JSONL")
    
    # Constraints
    parser.add_argument("--min-nuscenes-frac", type=float, default=0.0,
                        help="Minimum fraction of nuScenes clips (e.g. 0.5)")
    parser.add_argument("--min-carla-frac", type=float, default=0.0,
                        help="Minimum fraction of CARLA clips (e.g. 0.2)")
    
    parser.add_argument("--output", type=str, default="selected_clips.json",
                        help="Path to save selected clips JSON")
    
    args = parser.parse_args()

    # Load data
    qa_paths = [
        ("nuscenes", args.nuscenes_qa),
        ("carla", args.carla_qa)
    ]
    
    pool = load_qa_data(qa_paths)
    if not pool:
        print("No clips loaded. Exiting.")
        sys.exit(1)

    # Run selection
    selected_indices = greedy_select(
        pool, args.target, args.seed,
        min_nuscenes_frac=args.min_nuscenes_frac,
        min_carla_frac=args.min_carla_frac,
    )

    if not selected_indices:
        print("No clips selected.")
        sys.exit(0)

    # Report
    report_selection(pool, selected_indices)

    # Save
    if args.output:
        selected_clips = []
        for idx in selected_indices:
            c = pool[idx]
            # Create a clean record for export
            # We can export the answers and the ID. 
            # We can also export the full list of QAs if needed, 
            # but for now let's keep it lightweight: ID + answers.
            record = {
                "id": c["id"],
                "source": c["source"],
                "answers": c["answers"],
                "features": c.get("stats", {})
            }
            selected_clips.append(record)
            
        with open(args.output, "w") as f:
            json.dump(selected_clips, f, indent=2)
        print(f"\nSaved {len(selected_clips)} selected clips to {args.output}")

if __name__ == "__main__":
    main()