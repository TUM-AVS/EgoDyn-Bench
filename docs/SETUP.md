# Environment Setup

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | 3.12 may work but is untested |
| Conda | any | Miniconda or Anaconda |
| Storage | ~50 GB | Full nuScenes v1.0-trainval |
| RAM | 16 GB+ | For clip extraction and QA generation |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<org>/dynamic-trajectory-understanding.git
cd dynamic-trajectory-understanding
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml   # creates "dynamics-benchmark"
conda activate dynamics-benchmark
```

**Alternative (pip + venv):**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the installation

```bash
python --version                        # 3.11.x
python -c "import numpy; print(numpy.__version__)"
python -c "from nuscenes import NuScenes; print('nuScenes devkit OK')"
```

### 4. Set API keys (for VLM evaluation only)

```bash
export OPENAI_API_KEY="sk-..."          # GPT-4o
# Optional:
# export GOOGLE_API_KEY="..."           # Gemini
# export ANTHROPIC_API_KEY="sk-ant-..." # Claude
```

Add to `~/.bashrc` or `~/.zshrc` to persist across sessions.

## Updating the environment

```bash
# After pulling changes that modify environment.yml
conda env update -f environment.yml --prune

# After installing new packages manually, update the lock
conda env export --from-history > environment.yml
```

## Data sources

EgoDyn-Bench combines two data sources. Both must be downloaded separately
to your own machine; the released code only ships derived QA and arrays.

### nuScenes

Download from <https://www.nuscenes.org/nuscenes#download>. The pipeline
requires **v1.0-trainval** (full) or **v1.0-mini** (for quick testing).
Extract so the directory looks like:

```
/path/to/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
    ├── attribute.json
    ├── ...
    └── visibility.json
```

### CARLA Frenetix replays

The CARLA half is recorded with the [Frenetix](https://github.com/TUM-AVS/Frenetix)
planner. Place the replay outputs at:

```
/path/to/carla/
├── frenetix_logs/           # per-scene CSV planner logs
├── video_frenetix_replay_physics/   # raw FPV videos (used for chunking)
└── benchmark_transferred/   # Cosmos-Transfer 2.5 sim-to-real video (optional)
```

### Telling the code where the data lives

Two patterns. Pick whichever is easier in your workflow:

**1. Environment variables (recommended for repeated use).** Set these once
in your shell profile and every script picks them up:

```bash
export EGODYN_NUSCENES_ROOT=/path/to/nuscenes
export EGODYN_CARLA_LOGS_DIR=/path/to/carla/frenetix_logs
export EGODYN_CARLA_VIDEO_DIR=/path/to/carla/video_frenetix_replay_physics
export EGODYN_CARLA_TRANSFERRED_DIR=/path/to/carla/benchmark_transferred
```

| Variable | Used by |
|---|---|
| `EGODYN_CARLA_LOGS_DIR` | `scripts/plot_trajectories.py`, `scripts/prepare_carla_cosmos.sh` |
| `EGODYN_CARLA_VIDEO_DIR` | `scripts/prepare_carla_cosmos.sh` |
| `EGODYN_CARLA_TRANSFERRED_DIR` | `evaluation/evaluator_common.py`, `scripts/clip_viewer.py` |

> The `EGODYN_NUSCENES_ROOT` variable is *not* currently read by the code —
> nuScenes paths are passed via `--nuscenes_root` on the CLI. The variable
> is listed here as a convenience: shell scripts in `scripts/*.sh` (and your
> own wrapper scripts) can forward it.

**2. Explicit CLI flags (one-off invocations).** Every script accepts
`--nuscenes_root`, `--carla_logs`, `--carla-video-dir`, `--carla_video_dir`
etc. — see `--help` on any individual script.

If neither is provided where a path is needed, the relevant script exits
with a `TypeError: argument should be a str or an os.PathLike object`. That
is the signal that the data path wasn't set.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nuscenes-devkit` install fails | Run `pip install setuptools wheel` first |
| `conda activate` fails | Run `conda init bash`, restart terminal |
| `openai` version conflicts | `pip install --upgrade openai` |
| GUI/rendering errors on Linux | `sudo apt install libgl1-mesa-glx` |
