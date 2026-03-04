# Pose Estimation Evaluation Scripts

## Overview

Two scripts for benchmarking camera pose estimation models against ground-truth poses from 3DGS training data:

- **`eval_poses_experiment.py`** — End-to-end orchestrator (simulate flight + evaluate poses)
- **`eval_poses.py`** — Standalone evaluation (pose prediction + metrics on existing data)

## Models

| Model | Type | Description |
|---|---|---|
| `streamvggt` | Autoregressive | StreamVGGT with KV cache. Processes frames sequentially. |
| `da3` | Batch | DA3 on all frames at once. Best accuracy, but not online-compatible. |
| `da3_chunked` | Chunked online | Overlapping chunks with SIM(3) alignment on 3D point maps. |
| `da3_pairwise` | Pairwise online | Consecutive frame pairs with depth-based scale correction. |

### Online methods for robotics

`da3_chunked` and `da3_pairwise` are designed for onboard use where frames arrive incrementally:

- **Chunked**: Buffers `--chunk-size` frames (default 60), runs DA3 batch per chunk, aligns adjacent chunks via SIM(3) on overlapping depth-derived point clouds. Best accuracy of the online methods.
- **Pairwise**: Processes each new frame immediately with just the previous frame. Lowest latency but most drift.

## Quick Start

All commands run inside the GOGGLES Docker container (`docker compose run --rm goggles`).

### Full experiment (simulate + evaluate)

```bash
python scripts/eval_poses_experiment.py \
    --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
    --course square_toward_center \
    --model da3 -n 70
```

### Re-evaluate with a different model (skip simulation)

```bash
python scripts/eval_poses_experiment.py \
    --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
    --course square_toward_center \
    --skip-to evaluate -n 70 \
    --model da3_chunked \
    --experiment-name flightroom_ssv_exp_square_toward_center_2026-03-04_044709
```

Use `--experiment-name` to point at an existing experiment directory when skipping simulation.

### Compare all models on same data

```bash
EXP=flightroom_ssv_exp_square_toward_center_2026-03-04_044709

for model in streamvggt da3 da3_chunked da3_pairwise; do
    python scripts/eval_poses_experiment.py \
        --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
        --course square_toward_center \
        --skip-to evaluate -n 70 \
        --model $model \
        --experiment-name $EXP
done
```

Each model writes its own output files (`metrics_{model}_{n}f.json`, `.png`, `_trajectory.png`).

### Standalone evaluation

```bash
python scripts/eval_poses.py \
    --transforms /path/to/transforms.json \
    --model da3_chunked \
    -n 70 \
    -o results.json \
    --plot --visualize
```

## CLI Reference

### eval_poses_experiment.py

| Flag | Default | Description |
|---|---|---|
| `--scene` | *required* | GSplat model path (relative to `3dgs/workspace/outputs/`) |
| `--course` | *required* | Course config name (from FiGS `configs/course/`) |
| `--model` | `streamvggt` | Model: `streamvggt`, `da3`, `da3_chunked`, `da3_pairwise` |
| `-n` / `--num-frames` | `20` | Number of frames to evaluate |
| `--skip-to` | — | Resume from stage: `simulate` or `evaluate` |
| `--experiment-name` | auto | Explicit experiment directory name |
| `--da3-model-name` | — | HuggingFace model ID for DA3 variants |
| `--window-size` | — | KV-cache sliding window (StreamVGGT only) |
| `--chunk-size` | `60` | Frames per chunk (`da3_chunked` only) |
| `--overlap` | `20` | Overlap between chunks (`da3_chunked` only) |
| `--force` | — | Re-run all stages |

### eval_poses.py

| Flag | Default | Description |
|---|---|---|
| `--transforms` | — | Path to transforms.json with GT poses |
| `--model` | `streamvggt` | Model: `streamvggt`, `da3`, `da3_chunked`, `da3_pairwise` |
| `-n` / `--num-frames` | `20` | Number of frames to subsample |
| `-o` / `--output` | — | Save metrics JSON to this path |
| `--plot` | — | Save error distribution CDF plot |
| `--visualize` | — | Save 3D trajectory visualization |
| `--sparse-pc` | auto | Path to sparse point cloud for visualization |
| `--da3-model-name` | — | HuggingFace model ID for DA3 variants |
| `--chunk-size` | `60` | Frames per chunk (`da3_chunked` only) |
| `--overlap` | `20` | Overlap between chunks (`da3_chunked` only) |
| `--window-size` | — | KV-cache sliding window (StreamVGGT only) |
| `--verbose` | — | Print per-frame diagnostic output |

## Output Files

Each evaluation produces (in the experiment directory):

```
metrics_{model}_{n}f.json          # AUC, rotation/translation errors, accuracies
metrics_{model}_{n}f.png           # Error distribution CDF plot
metrics_{model}_{n}f_trajectory.png # 3D trajectory (predicted vs GT)
```

## Metrics

- **AUC@k**: Area under the CDF of max(rotation_error, translation_error) up to k degrees
- **R/T median**: Median pairwise rotation/translation error in degrees
- **R/T Acc@k**: Fraction of pairs with error < k degrees
- Evaluation uses **all-pairs relative pose errors** — no coordinate frame alignment needed
