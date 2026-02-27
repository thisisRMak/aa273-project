#!/usr/bin/env python3
"""
Extract StreamVGGT pre-decoder latents for an nbv-splat training run,
aligned with reconstruction metrics for time-history correlation.

Reads the nbv-splat output directory to determine:
  - Which images were used (from transforms_sorted.json via config.yml)
  - What order they were added (from view_selection_log.json)
  - Reconstruction metrics over time (from eval_all_images_metrics.json)

Then runs StreamVGGT aggregator on the first --num-views images in
view-selection order, and saves latents + metrics to a single HDF5 file.

Usage:
    # Test with 10 views
    python scripts/extract_nbv_latents.py \\
        /path/to/outputs/flightroom_ssv_exp/nbv-splat/2026-02-20_051708 \\
        --num-views 10

    # All views
    python scripts/extract_nbv_latents.py \\
        /path/to/outputs/flightroom_ssv_exp/nbv-splat/2026-02-20_051708

    # Custom output location
    python scripts/extract_nbv_latents.py \\
        /path/to/outputs/flightroom_ssv_exp/nbv-splat/2026-02-20_051708 \\
        --output /path/to/output.h5

    # Streaming mode (full causal context)
    python scripts/extract_nbv_latents.py \\
        /path/to/outputs/flightroom_ssv_exp/nbv-splat/2026-02-20_051708 \\
        --num-views 10 --streaming
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_nbv_latents")


def parse_training_dir(training_dir: str) -> dict:
    """Parse an nbv-splat training output directory.

    Returns dict with:
        workspace_dir:    str   — workspace root (parent of outputs/)
        scene_name:       str   — e.g. "flightroom_ssv_exp"
        timestamp:        str   — e.g. "2026-02-20_051708"
        transforms_path:  str   — absolute path to transforms_sorted.json
        images_dir:       str   — absolute path to images/
        view_order:       list[int]  — frame indices in view-selection order
        metrics:          list[dict] — eval_all_images_metrics entries
        config:           dict  — parsed config.yml
    """
    training_dir = Path(training_dir).resolve()

    # Load config.yml
    config_path = training_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yml in {training_dir}")

    with open(config_path) as f:
        config_text = f.read()

    # Extract data path from config (avoid full YAML load due to custom tags)
    # The data field can appear at top level or nested in the dataparser config.
    # We collect ALL "data: PosixPath" occurrences and use the last one with
    # actual path parts (the nested dataparser one takes precedence).
    all_data_sections = []
    current_parts = []
    in_data = False
    for line in config_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("data:") and "PosixPath" in stripped:
            if in_data and current_parts:
                all_data_sections.append(current_parts)
            current_parts = []
            in_data = True
            continue
        if in_data:
            if stripped.startswith("- "):
                current_parts.append(stripped[2:])
            else:
                if current_parts:
                    all_data_sections.append(current_parts)
                    current_parts = []
                in_data = False
    if in_data and current_parts:
        all_data_sections.append(current_parts)

    # Use the last section with path parts (dataparser-level overrides top-level)
    data_parts = all_data_sections[-1] if all_data_sections else []
    if not data_parts:
        raise ValueError("Could not parse data path from config.yml")

    data_relpath = os.path.join(*data_parts)  # e.g. "flightroom_ssv_exp/transforms_sorted.json"

    # Resolve workspace directory
    # Output structure: workspace/outputs/<scene>/<method>/<timestamp>/
    # So workspace = training_dir / ../../../../
    # But more robustly, find "outputs" in the path
    parts = training_dir.parts
    try:
        outputs_idx = len(parts) - 1 - list(reversed(parts)).index("outputs")
        workspace_dir = Path(*parts[:outputs_idx])
    except ValueError:
        raise ValueError(
            f"Cannot find 'outputs' in training dir path: {training_dir}\n"
            "Expected structure: <workspace>/outputs/<scene>/<method>/<timestamp>/"
        )

    transforms_path = workspace_dir / data_relpath
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

    images_dir = transforms_path.parent / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load transforms to get frame file paths
    with open(transforms_path) as f:
        transforms = json.load(f)

    # Load view selection log — each entry is [index] of the view added
    vsl_path = training_dir / "view_selection_log.json"
    if not vsl_path.exists():
        raise FileNotFoundError(f"No view_selection_log.json in {training_dir}")

    with open(vsl_path) as f:
        view_selection_log = json.load(f)

    # Flatten: each entry is a list of indices added at that step
    view_order = [idx for entry in view_selection_log for idx in entry]

    # Load metrics
    metrics_path = training_dir / "eval_all_images_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        logger.warning("No eval_all_images_metrics.json found, metrics will be empty")
        metrics = []

    # Also load converged_metrics if available
    converged_path = training_dir / "converged_metrics.json"
    if converged_path.exists():
        with open(converged_path) as f:
            converged_metrics = json.load(f)
    else:
        converged_metrics = []

    # Also load secondary eval metrics if available
    secondary_metrics_path = training_dir / "eval_all_images_secondary_metrics.json"
    if secondary_metrics_path.exists():
        with open(secondary_metrics_path) as f:
            secondary_metrics = json.load(f)
        logger.info("Loaded %d secondary eval metric entries", len(secondary_metrics))
    else:
        secondary_metrics = []

    scene_name = parts[outputs_idx + 1] if outputs_idx + 1 < len(parts) else "unknown"
    timestamp = training_dir.name

    return {
        "workspace_dir": str(workspace_dir),
        "scene_name": scene_name,
        "timestamp": timestamp,
        "transforms_path": str(transforms_path),
        "transforms": transforms,
        "images_dir": str(images_dir),
        "view_order": view_order,
        "metrics": metrics,
        "converged_metrics": converged_metrics,
        "secondary_metrics": secondary_metrics,
    }


def resolve_image_paths(parsed: dict, num_views: int) -> list[str]:
    """Get absolute image paths for the first num_views in view-selection order."""
    transforms = parsed["transforms"]
    frames = transforms["frames"]
    view_order = parsed["view_order"]
    workspace_dir = parsed["workspace_dir"]

    if num_views is None:
        num_views = len(view_order)
    else:
        num_views = min(num_views, len(view_order))

    image_paths = []
    for i in range(num_views):
        frame_idx = view_order[i]
        file_path = frames[frame_idx]["file_path"]  # e.g. "images/frame_00001.png"
        # Resolve relative to transforms file's parent directory
        abs_path = os.path.join(os.path.dirname(parsed["transforms_path"]), file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image not found: {abs_path}")
        image_paths.append(abs_path)

    return image_paths


def save_nbv_latents(
    result: dict,
    parsed: dict,
    num_views: int,
    output_path: str,
    streaming: bool,
) -> str:
    """Save latents + correlated nbv-splat metrics to HDF5.

    File layout:
        # Latents
        tokens:               float16 [num_views, n_layers, n_tokens, 2048]
        layer_indices:        int     [n_layers]
        image_paths:          str     [num_views]
        patch_start_idx:      int     scalar
        patch_grid:           int     [2]
        image_size:           int     [2]
        view_order:           int     [num_views]  (indices into transforms)

        # Reconstruction metrics (from eval_all_images_metrics)
        metrics/step:         int     [n_eval_steps]
        metrics/num_active_views: int [n_eval_steps]
        metrics/psnr:         float   [n_eval_steps]
        metrics/ssim:         float   [n_eval_steps]
        metrics/lpips:        float   [n_eval_steps]
        metrics/coverage_mean: float  [n_eval_steps]

        # Converged metrics (from converged_metrics)
        converged/step:       int     [n_converge_steps]
        converged/num_active_views: int [n_converge_steps]
        converged/psnr:       float   [n_converge_steps]
        converged/ssim:       float   [n_converge_steps]
        converged/lpips:      float   [n_converge_steps]
        converged/coverage_mean: float [n_converge_steps]

        # Secondary eval metrics (from eval_all_images_secondary_metrics, optional)
        secondary_metrics/step:       int     [n_secondary_steps]
        secondary_metrics/psnr:       float   [n_secondary_steps]
        secondary_metrics/ssim:       float   [n_secondary_steps]
        secondary_metrics/lpips:      float   [n_secondary_steps]
        secondary_metrics/coverage_mean: float [n_secondary_steps]

        # Metadata
        attrs: scene_name, timestamp, streaming, num_views_requested
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tokens = result["tokens"]
    n_frames, n_layers, n_tokens, feat_dim = tokens.shape

    with h5py.File(output_path, "w") as f:
        # Latent tokens
        f.create_dataset(
            "tokens",
            data=tokens,
            dtype="float16",
            chunks=(1, n_layers, n_tokens, feat_dim),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("layer_indices", data=result["layer_indices"])
        f.create_dataset("patch_start_idx", data=result["patch_start_idx"])
        f.create_dataset("patch_grid", data=result["patch_grid"])
        f.create_dataset("image_size", data=result["image_size"])
        f.create_dataset("view_order", data=parsed["view_order"][:num_views])

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_paths", data=result["image_paths"], dtype=dt)

        # Reconstruction metrics
        if parsed["metrics"]:
            mg = f.create_group("metrics")
            metrics = parsed["metrics"]
            for key in ["step", "num_active_views", "psnr", "ssim", "lpips", "coverage_mean"]:
                values = [m[key] for m in metrics if key in m]
                if values:
                    mg.create_dataset(key, data=values)

        if parsed["converged_metrics"]:
            cg = f.create_group("converged")
            cmetrics = parsed["converged_metrics"]
            for key in ["step", "num_active_views", "psnr", "ssim", "lpips", "coverage_mean"]:
                values = [m[key] for m in cmetrics if key in m]
                if values:
                    cg.create_dataset(key, data=values)

        if parsed.get("secondary_metrics"):
            sg = f.create_group("secondary_metrics")
            smetrics = parsed["secondary_metrics"]
            for key in ["step", "num_active_views", "psnr", "ssim", "lpips", "coverage_mean"]:
                values = [m[key] for m in smetrics if key in m]
                if values:
                    sg.create_dataset(key, data=values)

        # Metadata
        f.attrs["scene_name"] = parsed["scene_name"]
        f.attrs["timestamp"] = parsed["timestamp"]
        f.attrs["streaming"] = streaming
        f.attrs["num_views_extracted"] = n_frames
        f.attrs["total_views_available"] = len(parsed["view_order"])

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        "Saved %d frames x %d layers + metrics to %s (%.1f MB)",
        n_frames, n_layers, output_path, size_mb,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract StreamVGGT latents for an nbv-splat training run.",
    )
    parser.add_argument(
        "training_dir",
        help="Path to nbv-splat training output directory "
             "(contains config.yml, view_selection_log.json, etc.)",
    )
    parser.add_argument(
        "--num-views", "-n",
        type=int,
        default=10,
        help="Number of views to extract (in view-selection order). "
             "Use -1 for all views. Default: 10.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HDF5 path. Default: data/<scene>/<timestamp>_<n>views.h5",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (KV cache) for full causal context.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Frames per batch in non-streaming mode (default: 16).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Aggregator layer indices to extract. Default: DPT taps [4,11,17,23].",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Extract all 24 aggregator layers.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include camera + register tokens (indices 0-4) before patch tokens.",
    )
    parser.add_argument(
        "--max-cache-frames",
        type=int,
        default=50,
        help="Max past frames in KV cache for streaming mode. "
             "Oldest frames evicted when exceeded. Default: 50 (fits 24GB GPU). "
             "During forward pass, peak memory is ~2x cache size due to "
             "cat + RoPE temporaries.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to StreamVGGT checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )

    args = parser.parse_args()

    # Parse training directory
    logger.info("Parsing training directory: %s", args.training_dir)
    parsed = parse_training_dir(args.training_dir)
    logger.info("Scene: %s, timestamp: %s", parsed["scene_name"], parsed["timestamp"])
    logger.info("Total views available: %d", len(parsed["view_order"]))
    logger.info("Metrics entries: %d eval, %d converged",
                len(parsed["metrics"]), len(parsed["converged_metrics"]))

    # Resolve number of views
    num_views = args.num_views
    if num_views == -1:
        num_views = len(parsed["view_order"])
    num_views = min(num_views, len(parsed["view_order"]))
    logger.info("Extracting latents for %d views", num_views)

    # Resolve image paths in view-selection order
    image_paths = resolve_image_paths(parsed, num_views)
    logger.info("First image: %s", image_paths[0])
    logger.info("Last image:  %s", image_paths[-1])

    # Resolve layer indices
    if args.all_layers:
        from goggles.latent_extractor import ALL_LAYER_INDICES
        layer_indices = ALL_LAYER_INDICES
    elif args.layers:
        layer_indices = args.layers
    else:
        layer_indices = None  # defaults to DPT_LAYER_INDICES

    # Extract
    from goggles.latent_extractor import LatentExtractor

    extractor = LatentExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.streaming:
        logger.info("Extracting in streaming mode (KV cache)")
        result = extractor.extract_streaming(
            image_paths,
            layer_indices=layer_indices,
            include_special_tokens=args.include_special_tokens,
            max_cache_frames=args.max_cache_frames,
        )
    else:
        logger.info("Extracting in batch mode (chunk_size=%d)", args.chunk_size)
        result = extractor.extract(
            image_paths,
            layer_indices=layer_indices,
            chunk_size=args.chunk_size,
            include_special_tokens=args.include_special_tokens,
        )

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        # Default: data/<scene>/<timestamp>_<n>views[_full].h5
        goggles_root = Path(__file__).resolve().parent.parent
        suffix = "_full" if args.include_special_tokens else ""
        output_path = (
            goggles_root / "data" / parsed["scene_name"]
            / f"{parsed['timestamp']}_{num_views}views{suffix}.h5"
        )

    save_nbv_latents(result, parsed, num_views, str(output_path), args.streaming)

    tokens = result["tokens"]
    logger.info(
        "Done: %d frames, %d layers, %d tokens/frame, %d-dim",
        *tokens.shape,
    )
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
