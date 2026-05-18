#!/bin/bash
# ==========================================================================
# CARLA preparation pipeline:
#   Select clips → Chunk videos → Prepare Cosmos Transfer 2.5 batch
# ==========================================================================
#
# Run locally to prepare everything before transferring to your GPU cluster.
#
# Required data paths (override the env vars or edit the defaults below).
# Both must point to actual directories on your machine before running.
#
# Usage:
#   bash scripts/prepare_carla_cosmos.sh
#
set -euo pipefail

CONDA_ENV="dynamics-benchmark"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

# Data paths — adapt to your machine, or set as environment variables.
: "${EGODYN_CARLA_LOGS_DIR:=./data/carla/frenetix_logs}"
: "${EGODYN_CARLA_VIDEO_DIR:=./data/carla/video_frenetix_replay_physics}"

echo "Using CARLA logs:  ${EGODYN_CARLA_LOGS_DIR}"
echo "Using CARLA video: ${EGODYN_CARLA_VIDEO_DIR}"
echo ""

echo "============================================"
echo "Step 1: Balanced clip selection (3000 clips)"
echo "============================================"
conda run -n ${CONDA_ENV} python scripts/select_balanced_clips.py \
    --target 3000 \
    --min-nuscenes-frac 0.5 \
    --carla-logs "${EGODYN_CARLA_LOGS_DIR}" \
    --carla-video-dir "${EGODYN_CARLA_VIDEO_DIR}" \
    --output selected_clips.json

echo ""
echo "============================================"
echo "Step 2: Chunk CARLA videos (1280x720)"
echo "============================================"
conda run -n ${CONDA_ENV} python scripts/chunk_carla_videos.py \
    --selected selected_clips.json \
    --video-dir "${EGODYN_CARLA_VIDEO_DIR}" \
    --output-dir output/carla_chunks \
    --cosmos-ready

echo ""
echo "============================================"
echo "Step 3: Generate Cosmos Transfer 2.5 specs"
echo "============================================"
python scripts/prepare_cosmos_batch.py \
    --chunks-dir output/carla_chunks \
    --cosmos-dir output/cosmos_batch

echo ""
echo "============================================"
echo "Pipeline complete"
echo "============================================"
echo ""
echo "Local outputs:"
echo "  selected_clips.json         — selected clip IDs + features"
echo "  output/carla_chunks/        — 3s video chunks (1280x720)"
echo "  output/cosmos_batch/specs/  — per-clip Cosmos inference specs"
echo "  output/cosmos_batch/all_specs.txt — master spec list"
echo ""
echo "Next: transfer to your GPU cluster and submit:"
echo "  rsync -avP output/carla_chunks output/cosmos_batch <user>@<cluster>:~/dynamic-trajectory-understanding/output/"
echo "  sbatch scripts/slurm_cosmos_transfer.sh"
