# GOGGLES

Pre-decoder latent extraction from StreamVGGT for 3DGS coverage analysis.

GOGGLES extracts aggregator-level representations from [StreamVGGT](https://github.com/wzzheng/StreamVGGT) and uses them to analyze scene coverage, predict next-best views, and benchmark camera pose estimation against ground-truth 3DGS training data.

## Prerequisites

GOGGLES runs inside Docker and depends on sibling repositories that should live alongside it on the host:

```
StanfordMSL/
├── GOGGLES/                  # this repo
├── StreamVGGT/               # github.com/wzzheng/StreamVGGT (raw git clone)
├── FiGS-Standalone/          # 3DGS integration (provides the base Docker image)
└── coverage_view_selection/  # nbv-splat + custom nerfstudio fork
```

### 1. Clone StreamVGGT

```bash
cd /path/to/StanfordMSL
git clone https://github.com/wzzheng/StreamVGGT.git
```

### 2. Download StreamVGGT checkpoint

The model weights need to be at `StreamVGGT/ckpt/checkpoints.pth`. You can either download manually or let the code auto-download from HuggingFace on first run.

To download manually:

```bash
mkdir -p StreamVGGT/ckpt
# Download from HuggingFace (huggingface_hub CLI or browser)
huggingface-cli download lch01/StreamVGGT checkpoints.pth --local-dir StreamVGGT/ckpt
```

### 3. Build the base image

GOGGLES extends the `figs:latest` Docker image. Build it first:

```bash
cd /path/to/StanfordMSL/FiGS-Standalone
docker compose build
```

### 4. Build the GOGGLES image

```bash
cd /path/to/StanfordMSL/GOGGLES
docker compose build
```

This installs StreamVGGT-specific dependencies (huggingface_hub, transformers, h5py, etc.) on top of the FiGS base image.

## Usage

### Start the container

```bash
docker compose run --rm goggles
```

On startup the container automatically:
- Bind-mounts GOGGLES, StreamVGGT, FiGS-Standalone, and coverage_view_selection into `/workspace/`
- Adds `StreamVGGT/src/` to `PYTHONPATH` (so `import streamvggt` and `import vggt` work)
- Installs all packages in editable mode (`pip install -e`)
- Verifies imports succeed before dropping you into a shell

### Environment variables

Override default sibling repo paths or GPU selection via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SVGGT_PATH` | `../StreamVGGT` | Path to StreamVGGT repo on host |
| `FIGS_PATH` | `../FiGS-Standalone` | Path to FiGS-Standalone repo on host |
| `CVS_PATH` | `../coverage_view_selection` | Path to coverage_view_selection repo on host |
| `DATA_PATH` | `/media/admin/data/StanfordMSL` | Data directory (mounted at same path in container) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

Example with custom paths:

```bash
SVGGT_PATH=/home/user/StreamVGGT DATA_PATH=/data/nerf docker compose run --rm goggles
```

## Scripts

All scripts run from inside the container at `/workspace/GOGGLES`.

### Extract latents

Extract pre-decoder aggregator tokens from a directory of images:

```bash
python scripts/extract_latents.py /path/to/images/ -o latents.h5
python scripts/extract_latents.py /path/to/images/ -o latents.h5 --streaming  # causal context via KV cache
python scripts/extract_latents.py /path/to/images/ -o latents.h5 --all-layers  # all 24 layers (default: DPT taps [4,11,17,23])
```

### Extract NBV-splat latents

Extract latents aligned with nbv-splat training metrics (view-selection order + PSNR/SSIM/LPIPS/coverage):

```bash
python scripts/extract_nbv_latents.py /path/to/outputs/scene/nbv-splat/timestamp --num-views 50
python scripts/extract_nbv_latents.py /path/to/outputs/scene/nbv-splat/timestamp --streaming
```

### Evaluate camera poses

Benchmark StreamVGGT pose predictions against ground-truth poses from nerfstudio `transforms.json` (3DGS training data). Uses all-pairs relative pose errors — no coordinate frame alignment needed:

```bash
# From a transforms.json directly (20 uniformly-spaced frames)
python scripts/eval_poses.py \
    --transforms /path/to/nerf_data/scene/transforms.json \
    --num-frames 20

# Save metrics JSON + CDF plot
python scripts/eval_poses.py \
    --transforms /path/to/transforms.json \
    --num-frames 20 \
    --output data/eval/results.json \
    --plot

# From an nbv-splat training directory
python scripts/eval_poses.py \
    --training-dir /path/to/outputs/scene/nbv-splat/timestamp

# With intrinsics comparison (predicted vs GT focal lengths)
python scripts/eval_poses.py \
    --transforms /path/to/transforms.json \
    --compare-intrinsics
```

### Linear probe

Test whether reconstruction metrics are linearly decodable from camera tokens:

```bash
python scripts/linear_probe.py /path/to/latents.h5
```

### Saturation analysis

Analyze convergence of aggregator features over the image sequence:

```bash
python scripts/saturation_analysis.py /path/to/latents.h5
```

## Project structure

```
GOGGLES/
├── src/goggles/
│   ├── __init__.py
│   ├── latent_extractor.py   # StreamVGGT wrapper for batch/streaming extraction
│   └── pose_eval.py          # Relative pose error metrics (adapted from StreamVGGT eval)
├── scripts/
│   ├── extract_latents.py    # Extract aggregator tokens from images
│   ├── extract_nbv_latents.py # Extract tokens aligned with nbv-splat metrics
│   ├── eval_poses.py         # Benchmark pose predictions vs GT
│   ├── linear_probe.py       # Ridge regression on camera tokens
│   └── saturation_analysis.py # Feature convergence detection
├── notes/                    # Analysis notes and results
├── data/                     # Symlink to data storage (gitignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## How StreamVGGT is integrated

StreamVGGT is **not** pip-installed. It is a raw git clone that gets bind-mounted into the container and made importable via `PYTHONPATH`:

```
Host:      StanfordMSL/StreamVGGT/src/          (streamvggt/, vggt/ packages)
Container: /workspace/StreamVGGT/src/           (added to PYTHONPATH)
```

This means edits to StreamVGGT source on the host are immediately reflected inside the container. Model weights at `StreamVGGT/ckpt/checkpoints.pth` are also accessible via the mount.
