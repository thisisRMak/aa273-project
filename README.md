# GOGGLES

Pre-decoder latent extraction from StreamVGGT for 3DGS coverage analysis.

GOGGLES extracts aggregator-level representations from [StreamVGGT](https://github.com/wzzheng/StreamVGGT) and uses them to analyze scene coverage, predict next-best views, and benchmark camera pose estimation against ground-truth 3DGS training data.

## Prerequisites

GOGGLES runs inside Docker and depends on sibling repositories that should live alongside it on the host:

```
StanfordMSL/
├── GOGGLES/                  # this repo
├── StreamVGGT/               # github.com/wzzheng/StreamVGGT (raw git clone)
├── Depth-Anything-3/         # github.com/ByteDance-Seed/Depth-Anything-3 (raw git clone)
├── reloc3r/                  # https://github.com/ffrivera0/reloc3r (git clone, build croco)
├── open_vins/                # https://github.com/rpng/open_vins.git + our ROS-free additions
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
cd /path/to/StanfordMSL
git clone https://github.com/madang6/FiGS-Standalone.git
cd FiGS-Standalone
CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') docker compose build
```

### 4. Clone reloc3r

GOGGLES uses reloc3r. Clone the repo:

```bash
cd /path/to/StanfordMSL
git clone https://github.com/ffrivera0/reloc3r.git
cd reloc3r
git submodule update --init --recursive
```

### 5. Clone OpenVINS

GOGGLES uses OpenVINS for visual-inertial odometry pose estimation:

```bash
cd /path/to/StanfordMSL
git clone https://github.com/rpng/open_vins.git
```

Then apply our custom ROS-free build files (stored in `notes/openvins_setup/`):

```bash
cd open_vins
cp ../GOGGLES/notes/openvins_setup/Dockerfile.rosfree .
cp ../GOGGLES/notes/openvins_setup/docker-compose.yml .
cp ../GOGGLES/notes/openvins_setup/run_from_files.cpp ov_msckf/src/
cp -r ../GOGGLES/notes/openvins_setup/config_flightroom config/flightroom
mkdir -p scripts
cp ../GOGGLES/notes/openvins_setup/euroc_to_files.py scripts/
cp ../GOGGLES/notes/openvins_setup/uzhfpv_to_files.py scripts/
git apply ../GOGGLES/notes/openvins_setup/ROS1.cmake.patch
docker compose build
```

Review/edit paths inside `docker-compose.yml` as needed

This builds `openvins:rosfree` — a minimal Ubuntu 22.04 image with the `run_from_files` binary (no ROS). See `notes/openvins_setup/README.md` for details on what we changed from upstream.

### 6. Clone Depth-Anything-3

GOGGLES uses Depth-Anything-3 for pose estimation. Clone the repo:

```bash
cd /path/to/StanfordMSL
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

No checkpoint download is required — models are fetched automatically from HuggingFace on first use.

### 7. Build the GOGGLES image

```bash
cd /path/to/StanfordMSL/GOGGLES
docker compose build
```

This installs all extra dependencies (huggingface_hub, transformers, h5py, typer, moviepy, evo, pillow_heif, plyfile, etc.) on top of the FiGS base image. Note that `xformers` and `e3nn` are intentionally omitted to avoid breaking the pinned torch 2.1.2 environment; DA3 falls back gracefully to pure-PyTorch equivalents.

## Usage

### Start the container

```bash
docker compose run --rm goggles
```

If you don't have `coverage_view_selection`, use the base config:
```bash
docker compose -f docker-compose.base.yml run --rm goggles
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
| `RELOC3R_PATH` | `../reloc3r` | Path to reloc3r repo on host |
| `DA3_PATH` | `../Depth-Anything-3` | Path to Depth-Anything-3 repo on host |
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

Benchmark pose predictions against ground-truth poses from nerfstudio `transforms.json`.  Uses all-pairs relative pose errors — no coordinate frame alignment needed.

Supported models: `streamvggt` (default), `da3`, `da3_chunked`, `da3_pairwise`, `reloc3r`, `open_vins`.

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
│   ├── latent_extractor.py        # StreamVGGT wrapper for batch/streaming extraction
│   ├── pose_eval.py               # Relative pose error metrics (adapted from StreamVGGT eval)
│   ├── da3_predictor.py           # DA3 batch pose predictor
│   ├── da3_pairwise_predictor.py  # DA3 pairwise chaining with depth scale correction
│   ├── da3_chunked_predictor.py   # DA3 chunked inference with SIM(3) alignment
│   ├── imu_ekf.py                 # Error-state EKF fusing IMU + foundation model poses
│   ├── tum_utils.py               # TUM trajectory I/O + camera-to-IMU extrinsic conversion
│   ├── sim3_utils.py              # Umeyama SIM(3) alignment utilities
│   └── visualization.py           # SE(3) Procrustes alignment + trajectory plotting
├── scripts/
│   ├── extract_latents.py         # Extract aggregator tokens from images
│   ├── extract_nbv_latents.py     # Extract tokens aligned with nbv-splat metrics
│   ├── eval_poses.py              # Benchmark pose predictions vs GT
│   ├── eval_poses_experiment.py   # End-to-end: simulate → synthesize IMU → run models → evaluate
│   ├── run_sweep.py               # Parameter sweep runner
│   ├── linear_probe.py            # Ridge regression on camera tokens
│   └── saturation_analysis.py     # Feature convergence detection
├── notes/                         # Analysis notes and debugging records
├── data/                          # Symlink to data storage (gitignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## OpenVINS integration

GOGGLES can run [OpenVINS](https://docs.openvins.com/) as a visual-inertial odometry baseline. OpenVINS runs in a separate ROS-free Docker container and is invoked cross-container from the GOGGLES shell.

### Setup

1. **Build the OpenVINS container** (one-time):

```bash
cd /path/to/StanfordMSL/open_vins
docker compose build
```

This builds `openvins:rosfree` — a minimal Ubuntu 22.04 image with Eigen3, Ceres, OpenCV, and the `run_from_files` binary (no ROS dependency). The Dockerfile is `open_vins/Dockerfile.rosfree`.

2. **Verify the build**:

```bash
docker run --rm openvins:rosfree /opt/open_vins/ov_msckf/build/run_from_files --help
```

3. **GOGGLES container needs Docker socket access** (already configured in `docker-compose.yml`):

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock   # cross-container calls
  - ${OPENVINS_PATH:-../open_vins}:/workspace/open_vins  # config access
```

The startup script auto-detects the Docker socket GID and grants the container user access.

### Configuration

OpenVINS config lives in `open_vins/config/flightroom/` (three files):

| File | Purpose |
|------|---------|
| `estimator_config.yaml` | Filter settings, feature tracking, initialization |
| `kalibr_imucam_chain.yaml` | Camera intrinsics + camera-to-IMU extrinsic (from carl.json) |
| `kalibr_imu_chain.yaml` | IMU noise parameters (EuRoC ADIS16448 defaults) |

**Key configuration choices:**

- **Monocular**: `use_stereo: false`, `max_cameras: 1`
- **Calibration fixed**: All `calib_*: false` — intrinsics and extrinsics are known from carl.json
- **Dynamic initialization**: `init_imu_thresh: 0.0`, `init_max_disparity: 10.0` — forces dynamic init because the drone is always moving (see `notes/openvins_initialization_fix.md` for why this matters)
- **Camera-to-IMU extrinsic**: Derived from carl.json's `camera_to_body_transform` (OpenGL cam → FRD body), converted through FRD→FLU and OpenGL→OpenCV

### Pipeline

When `eval_poses_experiment.py` runs with `--model openvins`:

1. **IMU synthesis** (Python, runs in GOGGLES container):
   - Reads `tXUd.npy` from the simulation stage
   - `python -m figs.utilities.imu_synthesizer tXUd.npy -o imu.csv --noise euroc`
   - Produces 200 Hz IMU CSV in FLU body frame: `timestamp, ax, ay, az, wx, wy, wz`

2. **Image timestamps** (Python):
   - Generates `image_timestamps.csv` mapping sequential PNG filenames to timestamps

3. **OpenVINS execution** (Docker cross-container call):
   ```bash
   docker run --rm \
     -v /media/admin/data/StanfordMSL:/media/admin/data/StanfordMSL \
     openvins:rosfree \
     /opt/open_vins/ov_msckf/build/run_from_files \
       <config.yaml> <imu.csv> <images/> <timestamps.csv> <poses_tum.txt>
   ```
   Config is copied into the experiment directory (on the shared data mount) so paths work in both containers.

4. **Output**: TUM-format trajectory (`timestamp tx ty tz qx qy qz qw`) — IMU body poses in world frame, converted to camera w2c via the camera-to-IMU extrinsic.

### Standalone usage

You can also run OpenVINS directly without the experiment pipeline:

```bash
# From host (or any shell with docker access)
docker run --rm \
  -v /media/admin/data/StanfordMSL:/media/admin/data/StanfordMSL \
  openvins:rosfree \
  /opt/open_vins/ov_msckf/build/run_from_files \
    /media/admin/data/.../openvins/config/estimator_config.yaml \
    /media/admin/data/.../openvins/imu.csv \
    /media/admin/data/.../images/ \
    /media/admin/data/.../openvins/image_timestamps.csv \
    /media/admin/data/.../openvins/poses_tum.txt
```

### Validation

The build was validated on the UZH-FPV drone racing dataset: 0.59m RMSE on a 124m trajectory (0.5% error). On synthetic FiGS data (after the initialization fix): 0.063m ATE RMSE, 1.17° median rotation error.

See `notes/openvins_initialization_fix.md` for the full debugging story.

## How StreamVGGT is integrated

StreamVGGT is **not** pip-installed. It is a raw git clone that gets bind-mounted into the container and made importable via `PYTHONPATH`:

```
Host:      StanfordMSL/StreamVGGT/src/          (streamvggt/, vggt/ packages)
Container: /workspace/StreamVGGT/src/           (added to PYTHONPATH)
```

This means edits to StreamVGGT source on the host are immediately reflected inside the container. Model weights at `StreamVGGT/ckpt/checkpoints.pth` are also accessible via the mount.

## How Depth-Anything-3 is integrated

Depth-Anything-3 is a proper Python package (`pyproject.toml`) and is installed in editable mode at container startup:

```
Host:      StanfordMSL/Depth-Anything-3/        (git clone, no build step needed)
Container: /workspace/Depth-Anything-3/         (pip install -e, src layout)
```

The `depth_anything_3` package is importable as a regular Python import. Model weights are downloaded automatically from HuggingFace on first use and cached at `~/.cache/huggingface/`.

Two dependencies from DA3's `requirements.txt` are intentionally omitted from the Dockerfile:

| Package | Reason omitted |
|---------|---------------|
| `xformers` | Would replace the pinned torch 2.1.2+cu118, breaking FiGS/gsplat. DA3 falls back to plain-PyTorch SwiGLU. |
| `e3nn` | Requires torch ≥ 2.2.0. Only used for Spherical Harmonic rotation; not needed for pose or depth extraction. |

The `da3_streaming/` sub-pipeline (long-video streaming inference) has additional dependencies (`faiss-gpu`, `pypose`, `numba`, `pandas`) that are also not installed — add them to the Dockerfile if that pipeline is needed.


## How reloc3r is integrated

reloc3r is **not** pip-installed. It is a raw git clone that gets bind-mounted into the container and made importable via `PYTHONPATH`:

```
Host:      StanfordMSL/reloc3r/                 (reloc3r/ package + croco/ submodule)
Container: /workspace/reloc3r/                  (root added to PYTHONPATH)
```

Unlike StreamVGGT (which lives under a `src/` subdirectory), reloc3r's package directory sits at the repo root, so the repo root itself is added to `PYTHONPATH`. The `croco/` submodule must be initialised before building (`git submodule update --init --recursive`). Model weights are downloaded automatically from HuggingFace on first use.


## How OpenVINS is integrated

Unlike the other models, OpenVINS runs in its own Docker container (`openvins:rosfree`), not inside the GOGGLES container. This is because OpenVINS is a C++ application with its own system dependencies (Ceres, Eigen3, OpenCV) that would conflict with the Python/CUDA environment.

The upstream `rpng/open_vins` repo only supports ROS1/ROS2 builds. We added:

- **`run_from_files.cpp`** — Standalone C++ driver that reads IMU CSV + image timestamps instead of ROS bags
- **`Dockerfile.rosfree`** — Minimal build with `-DENABLE_ROS=OFF`
- **`config/flightroom/`** — Camera/IMU config derived from FiGS `carl.json`

All of these files are stored in `notes/openvins_setup/` for portability. The GOGGLES container invokes OpenVINS via the Docker socket (`/var/run/docker.sock`), passing data through the shared data mount.

See `notes/openvins_setup/README.md` for full setup instructions from a fresh clone.