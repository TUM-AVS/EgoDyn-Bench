#!/bin/bash
#SBATCH --job-name=cosmos-transfer
#SBATCH --output=logs/cosmos_%A_%a.out
#SBATCH --error=logs/cosmos_%A_%a.err
# === Cluster-specific — adapt to your scheduler ===========================
# Original used --partition=<your_a100_partition> --qos=<your_qos> on an
# 80GB A100 GPU node. Set both to whatever your site uses.
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
# ===========================================================================
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-14
#
# Cosmos Transfer 2.5 — SLURM array job (originally written for an LRZ AI cluster)
#
# Processes CARLA video chunks through Cosmos Transfer 2.5 vis (blur) mode
# for sim-to-real conversion.
#
# Each array task processes a batch of ~100 clips (1500 clips / 15 tasks).
# Adjust --array range based on total clips and BATCH_SIZE.
#
# Prerequisites on the cluster:
#   1. Clone cosmos-transfer2.5 repo
#   2. Set up environment (uv sync)
#   3. Run chunk_carla_videos.py and prepare_cosmos_batch.py locally
#   4. Transfer chunked videos + spec files to cluster
#
# Usage:
#   sbatch scripts/slurm_cosmos_transfer.sh
#
# NOTE: Adjust the --partition and --qos lines above to match your cluster.
#       Cosmos Transfer 2.5 needs a single 80GB+ GPU per task (A100 or H100).

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
# Adjust these paths to match your cluster setup
COSMOS_REPO="${HOME}/cosmos-transfer2.5"
PROJECT_DIR="${HOME}/dynamic-trajectory-understanding"
SPECS_DIR="${PROJECT_DIR}/cosmos_batch/specs"
OUTPUT_DIR="${PROJECT_DIR}/cosmos_batch/outputs"
BATCH_SIZE=100

# ── Derived paths ─────────────────────────────────────────────────────────
SPECS_LIST="${PROJECT_DIR}/cosmos_batch/all_specs.txt"

# ── Environment setup ─────────────────────────────────────────────────────
module load cuda 2>/dev/null || module load nvidia/cuda 2>/dev/null || true

cd "${COSMOS_REPO}"

# Ensure CUDA extras are installed (requires GPU node for compilation)
uv sync --extra=cu128 2>&1 | tail -5
source .venv/bin/activate 2>/dev/null || true

# Add pip-installed NVIDIA CUDA libraries to LD_LIBRARY_PATH
# (transformer_engine needs libnvrtc which pip puts in site-packages/nvidia/*/lib/)
for _nv_lib in .venv/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="${_nv_lib}:${LD_LIBRARY_PATH:-}"
done

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ── Determine this task's batch of spec files ─────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$((TASK_ID * BATCH_SIZE))
END_IDX=$(( (TASK_ID + 1) * BATCH_SIZE ))

# Read spec file paths for this batch
mapfile -t ALL_SPECS < "${SPECS_LIST}"
TOTAL=${#ALL_SPECS[@]}

if [ ${START_IDX} -ge ${TOTAL} ]; then
    echo "Task ${TASK_ID}: START_IDX=${START_IDX} >= TOTAL=${TOTAL}, nothing to do."
    exit 0
fi

if [ ${END_IDX} -gt ${TOTAL} ]; then
    END_IDX=${TOTAL}
fi

BATCH_SPECS=("${ALL_SPECS[@]:${START_IDX}:$((END_IDX - START_IDX))}")
N_CLIPS=${#BATCH_SPECS[@]}

echo "================================================================"
echo "Cosmos Transfer 2.5 — Array Task ${TASK_ID}"
echo "  Clips: ${START_IDX} to $((END_IDX - 1)) (${N_CLIPS} clips)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "================================================================"

# ── Run inference ─────────────────────────────────────────────────────────
# Process all specs in this batch in a single inference call.
# The model is loaded once and processes all clips sequentially.
python examples/inference.py \
    -i "${BATCH_SPECS[@]}" \
    -o "${OUTPUT_DIR}" \
    control:vis

echo "================================================================"
echo "Task ${TASK_ID} completed at $(date)"
echo "  Processed ${N_CLIPS} clips"
echo "================================================================"
