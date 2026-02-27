#!/usr/bin/env python3
"""
Extract pre-decoder latent tokens from StreamVGGT for a directory of images.

Usage:
    # DPT tap layers [4, 11, 17, 23], batch mode, chunk_size=16
    python scripts/extract_latents.py /path/to/images/ -o latents.h5

    # All 24 layers
    python scripts/extract_latents.py /path/to/images/ -o latents.h5 --all-layers

    # Specific layers
    python scripts/extract_latents.py /path/to/images/ -o latents.h5 --layers 11 23

    # Streaming mode (full causal context, lower memory)
    python scripts/extract_latents.py /path/to/images/ -o latents.h5 --streaming

    # Custom checkpoint path
    python scripts/extract_latents.py /path/to/images/ -o latents.h5 --checkpoint /path/to/checkpoints.pth

    # Larger chunks (more cross-frame context, more GPU memory)
    python scripts/extract_latents.py /path/to/images/ -o latents.h5 --chunk-size 32
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_latents")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(image_dir: str) -> list[str]:
    """Find and sort image files in a directory."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        logger.error("Not a directory: %s", image_dir)
        sys.exit(1)

    paths = sorted(
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        logger.error("No images found in %s", image_dir)
        sys.exit(1)

    logger.info("Found %d images in %s", len(paths), image_dir)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-decoder latents from StreamVGGT aggregator.",
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to StreamVGGT checkpoint. Default: /workspace/StreamVGGT/ckpt/checkpoints.pth",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Aggregator layer indices to extract (0-23). Default: DPT taps [4, 11, 17, 23].",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Extract all 24 aggregator layers.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (KV cache) for full causal context. Slower but richer.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Frames per batch in non-streaming mode (default: 16).",
    )
    parser.add_argument(
        "--max-cache-frames",
        type=int,
        default=None,
        help="Max past frames in KV cache for streaming mode. "
             "Oldest frames evicted when exceeded. Default: None (unlimited). "
             "Set smaller (e.g. 50) if GPU OOMs on long sequences.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include camera + register tokens (indices 0-4) in output.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )

    args = parser.parse_args()

    # Resolve layer indices
    if args.all_layers:
        from goggles.latent_extractor import ALL_LAYER_INDICES
        layer_indices = ALL_LAYER_INDICES
    elif args.layers:
        layer_indices = args.layers
    else:
        layer_indices = None  # will default to DPT_LAYER_INDICES

    # Find images
    image_paths = find_images(args.image_dir)

    # Load model and extract
    from goggles.latent_extractor import LatentExtractor, save_latents

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

    # Save
    save_latents(result, args.output)

    tokens = result["tokens"]
    logger.info(
        "Done: %d frames, %d layers, %d tokens/frame, %d-dim features",
        *tokens.shape,
    )
    logger.info("Patch grid: %d x %d", *result["patch_grid"])
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
