#!/usr/bin/env bash
#
# End-to-end pipeline: extract all clips, generate QA, build splits, validate.
#
# Usage:
#   bash scripts/run_pipeline_all.sh /path/to/nuscenes
#
# Requires NUSCENES_ROOT as first argument (or set as environment variable).

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
NUSCENES_ROOT="${1:-${NUSCENES_ROOT:-}}"
if [ -z "$NUSCENES_ROOT" ]; then
    echo "Usage: $0 /path/to/nuscenes"
    echo "  or:  NUSCENES_ROOT=/path/to/nuscenes $0"
    exit 1
fi

SEED=42
CLIP_DURATION=3
OUTPUT_DIR="output/pipeline_all"
CLIPS_DIR="${OUTPUT_DIR}/clips"
QA_JSONL="${OUTPUT_DIR}/qa.jsonl"
SPLITS_DIR="${OUTPUT_DIR}/splits"

echo "============================================================"
echo "  nuScenes root: ${NUSCENES_ROOT}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# ── Step 1: Extract clips ────────────────────────────────────────────────────
echo "[1/5] Extracting clips..."
python dataset/scripts/extract_nuscenes_clips.py \
    --nuscenes_root "${NUSCENES_ROOT}" \
    --nuscenes_version v1.0-trainval \
    --output_dir "${CLIPS_DIR}" \
    --clip_seconds "${CLIP_DURATION}" \
    --seed "${SEED}"
echo ""

# ── Step 2: Validate clips ──────────────────────────────────────────────────
echo "[2/5] Validating clips..."
python dataset/validation/validate_clips.py \
    --clips_dir "${CLIPS_DIR}"
echo ""

# ── Step 3: Generate QA ─────────────────────────────────────────────────────
echo "[3/5] Generating QA..."
python dataset/scripts/generate_qa.py \
    --clips_index "${CLIPS_DIR}/clips_index.jsonl" \
    --questions_config dataset/configs/questions_template.yaml \
    --output_qa_jsonl "${QA_JSONL}" \
    --seed "${SEED}"
echo ""

# ── Step 4: Validate QA ─────────────────────────────────────────────────────
echo "[4/5] Validating QA..."
python dataset/validation/validate_qa.py \
    --qa_jsonl "${QA_JSONL}" \
    --clips_index "${CLIPS_DIR}/clips_index.jsonl"
echo ""

# ── Step 5: Build & validate splits ─────────────────────────────────────────
echo "[5/5] Building splits..."
python dataset/scripts/build_splits.py \
    --clips_index "${CLIPS_DIR}/clips_index.jsonl" \
    --qa_jsonl "${QA_JSONL}" \
    --output_dir "${SPLITS_DIR}" \
    --num_val_clips 500 \
    --num_train_clips 3000 \
    --min_per_bin 2 \
    --seed "${SEED}"
echo ""

echo "Validating splits..."
python dataset/validation/validate_splits.py \
    --splits_dir "${SPLITS_DIR}"
echo ""

echo "============================================================"
echo "  Pipeline complete!"
echo "  Clips:  ${CLIPS_DIR}/clips_index.jsonl"
echo "  QA:     ${QA_JSONL}"
echo "  Splits: ${SPLITS_DIR}/"
echo "============================================================"
