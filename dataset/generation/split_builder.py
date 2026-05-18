"""
Dataset split builder for train/val partitioning.

Creates reproducible, balanced splits at the clip level using stratified sampling
based on QA-derived tags (turning, braking, smoothness).
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import numpy as np


logger = logging.getLogger(__name__)


class ClipTags:
    """Container for clip-level tags derived from QA labels."""

    def __init__(self, has_turn: bool = False, has_braking: bool = False, has_aggressive: bool = False):
        self.has_turn = has_turn
        self.has_braking = has_braking
        self.has_aggressive = has_aggressive

    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple for stratification binning."""
        return (int(self.has_turn), int(self.has_braking), int(self.has_aggressive))

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return {
            "has_turn": self.has_turn,
            "has_braking": self.has_braking,
            "has_aggressive": self.has_aggressive
        }

    @staticmethod
    def from_tuple(t: Tuple[int, int, int]) -> 'ClipTags':
        """Create from tuple."""
        return ClipTags(bool(t[0]), bool(t[1]), bool(t[2]))


class SplitBuilder:
    """Builds train/val splits with stratified sampling."""

    def __init__(
        self,
        seed: int = 42,
        tag_question_ids: Dict[str, str] = None
    ):
        """
        Initialize split builder.

        Args:
            seed: Random seed for reproducibility
            tag_question_ids: Question IDs for tag extraction
                {
                    'turn': 'yaw_rate_turn_direction',
                    'braking': 'braking_intensity',
                    'aggressive': 'driving_smoothness'
                }
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Default tag question IDs
        if tag_question_ids is None:
            tag_question_ids = {
                'turn': 'yaw_rate_turn_direction',
                'braking': 'braking_intensity',
                'aggressive': 'driving_smoothness'
            }
        self.tag_question_ids = tag_question_ids

        logger.info(f"Initialized SplitBuilder with seed={seed}")
        logger.info(f"Tag question IDs: {tag_question_ids}")

    def load_data(
        self,
        clips_index_path: str,
        qa_jsonl_path: str
    ) -> Tuple[List[dict], List[dict]]:
        """
        Load clips and QA data.

        Args:
            clips_index_path: Path to clips_index.jsonl
            qa_jsonl_path: Path to qa.jsonl

        Returns:
            Tuple of (clips, qa_items)
        """
        logger.info(f"Loading clips from {clips_index_path}")
        clips = []
        with open(clips_index_path, 'r') as f:
            for line in f:
                clips.append(json.loads(line))

        logger.info(f"Loading QA items from {qa_jsonl_path}")
        qa_items = []
        with open(qa_jsonl_path, 'r') as f:
            for line in f:
                qa_items.append(json.loads(line))

        logger.info(f"Loaded {len(clips)} clips and {len(qa_items)} QA items")

        return clips, qa_items

    def derive_clip_tags(
        self,
        qa_items: List[dict]
    ) -> Dict[str, ClipTags]:
        """
        Derive clip-level tags from QA answers.

        Uses "any positive evidence" logic: if any QA for a clip indicates
        the tag condition, the clip is tagged.

        Clips with all-negative tags (straight, no braking, smooth) are still
        included with default False values.

        Args:
            qa_items: List of QA item dicts

        Returns:
            Dictionary mapping clip_id to ClipTags
        """
        logger.info("Deriving clip-level tags from QA answers...")

        # First pass: collect all unique clip IDs and initialize with default tags
        unique_clip_ids = set(qa['clip_id'] for qa in qa_items)
        clip_tags = {clip_id: ClipTags() for clip_id in unique_clip_ids}

        # Second pass: set tags to True where positive evidence exists
        for qa in qa_items:
            clip_id = qa['clip_id']
            question_id = qa['question_id']
            answer = qa['answer']

            # Check for turning (left or right)
            if question_id == self.tag_question_ids['turn']:
                if answer in ['left', 'right']:
                    clip_tags[clip_id].has_turn = True

            # Check for braking event (any intensity above "none")
            elif question_id == self.tag_question_ids['braking']:
                if answer != 'none':
                    clip_tags[clip_id].has_braking = True

            # Check for aggressive driving
            elif question_id == self.tag_question_ids['aggressive']:
                if answer == 'aggressive':
                    clip_tags[clip_id].has_aggressive = True

        logger.info(f"Derived tags for {len(clip_tags)} clips")

        # Log tag distribution
        turn_count = sum(1 for t in clip_tags.values() if t.has_turn)
        brake_count = sum(1 for t in clip_tags.values() if t.has_braking)
        aggressive_count = sum(1 for t in clip_tags.values() if t.has_aggressive)

        # Log (0,0,0) bin explicitly
        all_negative_count = sum(
            1 for t in clip_tags.values()
            if not t.has_turn and not t.has_braking and not t.has_aggressive
        )

        logger.info("Tag distribution:")
        logger.info(f"  has_turn: {turn_count}/{len(clip_tags)} ({100*turn_count/len(clip_tags):.1f}%)")
        logger.info(f"  has_braking: {brake_count}/{len(clip_tags)} ({100*brake_count/len(clip_tags):.1f}%)")
        logger.info(f"  has_aggressive: {aggressive_count}/{len(clip_tags)} ({100*aggressive_count/len(clip_tags):.1f}%)")
        logger.info(f"  all_negative (0,0,0): {all_negative_count}/{len(clip_tags)} ({100*all_negative_count/len(clip_tags):.1f}%)")

        return clip_tags

    def stratify_clips(
        self,
        clip_ids: List[str],
        clip_tags: Dict[str, ClipTags]
    ) -> Dict[Tuple[int, int, int], List[str]]:
        """
        Stratify clips into bins based on tags.

        Args:
            clip_ids: List of clip IDs to stratify
            clip_tags: Clip tags dictionary

        Returns:
            Dictionary mapping bin tuple to list of clip IDs
        """
        bins = defaultdict(list)

        for clip_id in clip_ids:
            if clip_id in clip_tags:
                bin_key = clip_tags[clip_id].to_tuple()
                bins[bin_key].append(clip_id)
            else:
                # Clip with no tags (shouldn't happen if QA exists)
                logger.warning(f"Clip {clip_id} has no tags (no QA?), using default bin")
                bins[(0, 0, 0)].append(clip_id)

        logger.info(f"Stratified {len(clip_ids)} clips into {len(bins)} bins")

        # Log bin sizes
        for bin_key, clips_in_bin in sorted(bins.items()):
            tags = ClipTags.from_tuple(bin_key)
            logger.info(
                f"  Bin {bin_key} (turn={tags.has_turn}, brake={tags.has_braking}, "
                f"agg={tags.has_aggressive}): {len(clips_in_bin)} clips"
            )

        return dict(bins)

    def sample_validation_set(
        self,
        bins: Dict[Tuple[int, int, int], List[str]],
        num_val_clips: int,
        min_per_bin: int = 10
    ) -> Tuple[Set[str], Dict]:
        """
        Sample validation set with proportional + minimum coverage strategy.

        Args:
            bins: Stratification bins
            num_val_clips: Target number of validation clips
            min_per_bin: Minimum clips per bin (if available)

        Returns:
            Tuple of (val_clip_ids, bin_metadata)
        """
        logger.info(f"Sampling validation set: target={num_val_clips}, min_per_bin={min_per_bin}")

        total_clips = sum(len(clips) for clips in bins.values())

        # Phase 1: Ensure minimum coverage per bin
        val_clips = set()
        bin_metadata = {}

        for bin_key, clips_in_bin in bins.items():
            n_available = len(clips_in_bin)
            n_min = min(min_per_bin, n_available)

            # Sample minimum
            sampled = self.rng.choice(clips_in_bin, size=n_min, replace=False).tolist()
            val_clips.update(sampled)

            bin_metadata[str(bin_key)] = {
                'total': n_available,
                'sampled_min': n_min,
                'sampled_proportional': 0,
                'sampled_total': n_min
            }

        logger.info(f"Phase 1 (minimum coverage): {len(val_clips)} clips sampled")

        # Phase 2: Proportional sampling to reach target
        remaining_quota = num_val_clips - len(val_clips)

        if remaining_quota > 0:
            # Calculate proportional allocation for remaining quota
            for bin_key, clips_in_bin in bins.items():
                # Available clips not yet sampled
                available = [c for c in clips_in_bin if c not in val_clips]

                if not available:
                    continue

                # Proportional share
                proportion = len(clips_in_bin) / total_clips
                n_proportional = int(remaining_quota * proportion)

                # Cap at available
                n_sample = min(n_proportional, len(available))

                if n_sample > 0:
                    sampled = self.rng.choice(available, size=n_sample, replace=False).tolist()
                    val_clips.update(sampled)

                    bin_metadata[str(bin_key)]['sampled_proportional'] = n_sample
                    bin_metadata[str(bin_key)]['sampled_total'] += n_sample

        logger.info(f"Phase 2 (proportional): {len(val_clips)} total clips sampled")

        # Phase 3: Top-up to reach exact target if needed
        remaining_quota = num_val_clips - len(val_clips)

        if remaining_quota > 0:
            # Collect all clips not yet sampled
            all_clips = []
            for clips_in_bin in bins.values():
                all_clips.extend(clips_in_bin)

            available_for_topup = [c for c in all_clips if c not in val_clips]

            if available_for_topup:
                n_topup = min(remaining_quota, len(available_for_topup))
                topup_sampled = self.rng.choice(available_for_topup, size=n_topup, replace=False).tolist()
                val_clips.update(topup_sampled)

                logger.info(f"Phase 3 (top-up): sampled {n_topup} additional clips to reach target")
                logger.info(f"Final validation set size: {len(val_clips)}")

        # Handle under/over sampling
        if len(val_clips) < num_val_clips:
            logger.warning(f"Could not reach target {num_val_clips}, sampled {len(val_clips)} (insufficient eligible clips)")
        elif len(val_clips) > num_val_clips:
            # Trim excess (shouldn't happen with correct calculation)
            excess = len(val_clips) - num_val_clips
            val_clips_list = list(val_clips)
            self.rng.shuffle(val_clips_list)
            val_clips = set(val_clips_list[:num_val_clips])
            logger.warning(f"Trimmed {excess} excess clips to reach target")

        return val_clips, bin_metadata

    def build_splits(
        self,
        clips: List[dict],
        qa_items: List[dict],
        num_val_clips: int = 500,
        num_train_clips: int = 3000,
        min_per_bin: int = 10,
        val_ratio: float = None,
        exclude_scenes: Set[str] = None,
    ) -> Dict:
        """
        Build train/val splits.

        Args:
            clips: List of clip records
            qa_items: List of QA items
            num_val_clips: Target validation set size (ignored if val_ratio is set)
            num_train_clips: Target training set size (ignored if val_ratio is set)
            min_per_bin: Minimum clips per stratification bin
            val_ratio: If set, use this ratio for validation (e.g. 0.2 for 80/20).
                       Overrides num_val_clips and num_train_clips.
            exclude_scenes: Set of scene names to exclude (e.g. benchmark scenes)

        Returns:
            Dictionary containing split data and metadata
        """
        logger.info("="*80)
        logger.info("BUILDING DATASET SPLITS")
        logger.info("="*80)

        # Exclude scenes (e.g. overlapping with benchmark)
        excluded_count = 0
        if exclude_scenes:
            before = len(clips)
            clips = [c for c in clips if c.get('scene') not in exclude_scenes]
            excluded_count = before - len(clips)
            qa_clip_ids = {c['clip_id'] for c in clips}
            qa_items = [qa for qa in qa_items if qa['clip_id'] in qa_clip_ids]
            logger.info(f"Excluded {excluded_count} clips from {len(exclude_scenes)} scenes")

        # Derive clip tags
        clip_tags = self.derive_clip_tags(qa_items)

        # Filter clips: only include clips with QA
        clips_with_qa = [c for c in clips if c['clip_id'] in clip_tags]
        clips_without_qa = [c for c in clips if c['clip_id'] not in clip_tags]

        if clips_without_qa:
            logger.warning(f"Dropped {len(clips_without_qa)} clips without QA")

        logger.info(f"Eligible clips: {len(clips_with_qa)}")

        # Compute targets from ratio if provided
        if val_ratio is not None:
            num_val_clips = int(len(clips_with_qa) * val_ratio)
            num_train_clips = len(clips_with_qa) - num_val_clips
            logger.info(f"Using ratio split: {(1 - val_ratio)*100:.0f}% train / {val_ratio*100:.0f}% val")
            logger.info(f"  → {num_train_clips} train, {num_val_clips} val")

        # Stratify clips
        clip_ids = [c['clip_id'] for c in clips_with_qa]
        bins = self.stratify_clips(clip_ids, clip_tags)

        # Sample validation set
        val_clip_ids, bin_metadata = self.sample_validation_set(
            bins, num_val_clips, min_per_bin
        )

        # Remaining clips for training
        train_pool = [cid for cid in clip_ids if cid not in val_clip_ids]

        # Subsample training set if needed
        if len(train_pool) > num_train_clips:
            logger.info(f"Subsampling training set: {len(train_pool)} → {num_train_clips}")
            train_clip_ids = set(self.rng.choice(train_pool, size=num_train_clips, replace=False))
        else:
            train_clip_ids = set(train_pool)
            if len(train_clip_ids) < num_train_clips:
                logger.warning(
                    f"Training set smaller than target: {len(train_clip_ids)} < {num_train_clips}"
                )

        # Partition clips and QA
        train_clips = [c for c in clips if c['clip_id'] in train_clip_ids]
        val_clips = [c for c in clips if c['clip_id'] in val_clip_ids]

        train_qa = [qa for qa in qa_items if qa['clip_id'] in train_clip_ids]
        val_qa = [qa for qa in qa_items if qa['clip_id'] in val_clip_ids]

        # Sort for determinism
        train_clips.sort(key=lambda c: c['clip_id'])
        val_clips.sort(key=lambda c: c['clip_id'])
        train_qa.sort(key=lambda qa: qa['qa_id'])
        val_qa.sort(key=lambda qa: qa['qa_id'])

        # Build metadata
        metadata = {
            'seed': self.seed,
            'balance_strategy': 'qa_tags_stratified',
            'tag_question_ids': self.tag_question_ids,
            'timestamp': datetime.utcnow().isoformat(),
            'target_num_val_clips': num_val_clips,
            'target_num_train_clips': num_train_clips,
            'val_ratio': val_ratio,
            'min_per_bin': min_per_bin,
            'actual_counts': {
                'total_clips': len(clips),
                'total_qa': len(qa_items),
                'eligible_clips': len(clips_with_qa),
                'dropped_clips_no_qa': len(clips_without_qa),
                'excluded_scenes': excluded_count,
                'val_clips': len(val_clips),
                'train_clips': len(train_clips),
                'val_qa': len(val_qa),
                'train_qa': len(train_qa)
            },
            'stratification_bins': bin_metadata
        }

        logger.info("="*80)
        logger.info("SPLIT SUMMARY")
        logger.info("="*80)
        logger.info(f"Total clips: {len(clips)}")
        if excluded_count:
            logger.info(f"Excluded (benchmark overlap): {excluded_count}")
        logger.info(f"Eligible clips (with QA): {len(clips_with_qa)}")
        logger.info(f"Validation: {len(val_clips)} clips, {len(val_qa)} QA items")
        logger.info(f"Training: {len(train_clips)} clips, {len(train_qa)} QA items")
        logger.info("="*80)

        return {
            'train_clips': train_clips,
            'val_clips': val_clips,
            'train_qa': train_qa,
            'val_qa': val_qa,
            'metadata': metadata,
            'clip_tags': clip_tags
        }

    def write_splits(
        self,
        output_dir: Path,
        split_data: Dict
    ):
        """
        Write split files to disk.

        Args:
            output_dir: Output directory
            split_data: Split data from build_splits()
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing splits to {output_dir}")

        # Write clips (shallow-copy each record to avoid mutating caller's data)
        with open(output_dir / 'train_clips.jsonl', 'w') as f:
            for clip in split_data['train_clips']:
                f.write(json.dumps({**clip, 'split': 'train'}) + '\n')

        with open(output_dir / 'val_clips.jsonl', 'w') as f:
            for clip in split_data['val_clips']:
                f.write(json.dumps({**clip, 'split': 'val'}) + '\n')

        # Write QA
        with open(output_dir / 'train_qa.jsonl', 'w') as f:
            for qa in split_data['train_qa']:
                f.write(json.dumps({**qa, 'split': 'train'}) + '\n')

        with open(output_dir / 'val_qa.jsonl', 'w') as f:
            for qa in split_data['val_qa']:
                f.write(json.dumps({**qa, 'split': 'val'}) + '\n')

        # Write metadata
        with open(output_dir / 'split_metadata.json', 'w') as f:
            # Remove clip_tags from metadata (too large)
            metadata = {k: v for k, v in split_data['metadata'].items()}
            json.dump(metadata, f, indent=2)

        logger.info("Split files written successfully")
        logger.info(f"  train_clips.jsonl: {len(split_data['train_clips'])} records")
        logger.info(f"  val_clips.jsonl: {len(split_data['val_clips'])} records")
        logger.info(f"  train_qa.jsonl: {len(split_data['train_qa'])} records")
        logger.info(f"  val_qa.jsonl: {len(split_data['val_qa'])} records")
        logger.info("  split_metadata.json")
