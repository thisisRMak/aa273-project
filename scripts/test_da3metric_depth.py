#!/usr/bin/env python3
"""
Sanity check: compare DA3Metric-Large (metric depth) vs DA3-Large (relative depth).

Tests whether DA3Metric produces plausible metric-scale depth (in meters) and
whether the scale ratio between metric and relative depth is consistent across frames.

Usage (inside Docker):
    python scripts/test_da3metric_depth.py \
        --experiment-dir experiments/4_sweep_.../packardpark_circle_toward_center \
        --num-frames 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_da3metric")


def main():
    parser = argparse.ArgumentParser(description="DA3Metric depth sanity check")
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to experiment directory with images/ and transforms.json",
    )
    parser.add_argument(
        "--num-frames", type=int, default=5,
        help="Number of frames to test (evenly spaced, default: 5)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    images_dir = exp_dir / "images"
    transforms_path = exp_dir / "transforms.json"

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    # Load GT intrinsics
    with open(transforms_path) as f:
        transforms = json.load(f)

    gt_fl_x = transforms["fl_x"]
    gt_fl_y = transforms["fl_y"]
    gt_focal = (gt_fl_x + gt_fl_y) / 2
    gt_w, gt_h = transforms["w"], transforms["h"]
    logger.info("GT intrinsics: focal=%.1f, size=%dx%d", gt_focal, gt_w, gt_h)

    # Select frames
    all_frames = sorted(images_dir.glob("frame_*.png"))
    n_total = len(all_frames)
    indices = np.linspace(0, n_total - 1, args.num_frames, dtype=int)
    selected = [all_frames[i] for i in indices]
    logger.info("Testing %d frames from %d total", len(selected), n_total)

    # Load models
    from depth_anything_3.api import DepthAnything3

    logger.info("Loading DA3Metric-Large...")
    model_metric = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model_metric = model_metric.to(device=args.device)
    logger.info("DA3Metric-Large loaded")

    logger.info("Loading DA3-Large-1.1...")
    model_rel = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
    model_rel = model_rel.to(device=args.device)
    logger.info("DA3-Large-1.1 loaded")

    # Per-frame results
    results = []

    for i, img_path in enumerate(selected):
        frame_name = img_path.name
        logger.info("--- Frame %d/%d: %s ---", i + 1, len(selected), frame_name)

        # Run DA3Metric (monocular, single frame)
        with torch.no_grad():
            pred_metric = model_metric.inference([str(img_path)])
            torch.cuda.empty_cache()

        raw_metric = np.squeeze(pred_metric.depth)  # (H, W)
        h_m, w_m = raw_metric.shape

        # DA3Metric needs focal length for metric scaling
        # Scale GT focal to DA3's processing resolution
        scale_x = w_m / gt_w
        scale_y = h_m / gt_h
        da3_focal_gt = gt_focal * (scale_x + scale_y) / 2
        scaled_metric_gt = raw_metric * (da3_focal_gt / 300.0)

        # Run DA3-Large (needs pair for extrinsics, but single frame works for depth)
        # Use same frame twice as a pair to get intrinsics prediction
        with torch.no_grad():
            pred_rel = model_rel.inference([str(img_path), str(img_path)])
            torch.cuda.empty_cache()

        raw_rel = np.squeeze(pred_rel.depth[0])  # (H, W) from first of pair
        da3_intr = pred_rel.intrinsics[0]  # (3, 3) predicted intrinsics
        da3_focal_pred = (da3_intr[0, 0] + da3_intr[1, 1]) / 2

        # Also compute metric depth using DA3-predicted focal
        scaled_metric_pred = raw_metric * (float(da3_focal_pred) / 300.0)

        # Compute scale ratio (metric / relative) over valid pixels
        valid = (raw_rel > 1e-3) & (raw_metric > 1e-3)
        if np.sum(valid) > 100:
            ratio_gt = scaled_metric_gt[valid] / raw_rel[valid]
            ratio_pred = scaled_metric_pred[valid] / raw_rel[valid]
            scale_ratio_gt = float(np.median(ratio_gt))
            scale_ratio_pred = float(np.median(ratio_pred))
        else:
            scale_ratio_gt = float("nan")
            scale_ratio_pred = float("nan")

        result = {
            "frame": frame_name,
            "raw_metric": {
                "min": float(raw_metric.min()),
                "max": float(raw_metric.max()),
                "mean": float(raw_metric.mean()),
                "median": float(np.median(raw_metric)),
            },
            "scaled_metric_gt_focal": {
                "min": float(scaled_metric_gt.min()),
                "max": float(scaled_metric_gt.max()),
                "mean": float(scaled_metric_gt.mean()),
                "median": float(np.median(scaled_metric_gt)),
                "focal_used": float(da3_focal_gt),
            },
            "scaled_metric_pred_focal": {
                "min": float(scaled_metric_pred.min()),
                "max": float(scaled_metric_pred.max()),
                "mean": float(scaled_metric_pred.mean()),
                "median": float(np.median(scaled_metric_pred)),
                "focal_used": float(da3_focal_pred),
            },
            "relative": {
                "min": float(raw_rel.min()),
                "max": float(raw_rel.max()),
                "mean": float(raw_rel.mean()),
                "median": float(np.median(raw_rel)),
            },
            "scale_ratio_gt_focal": scale_ratio_gt,
            "scale_ratio_pred_focal": scale_ratio_pred,
            "da3_pred_focal": float(da3_focal_pred),
            "gt_focal_at_da3_res": float(da3_focal_gt),
        }
        results.append(result)

        # Print
        print(f"\n  Frame: {frame_name}")
        print(f"  DA3Metric raw:         min={raw_metric.min():.4f}  max={raw_metric.max():.4f}  median={np.median(raw_metric):.4f}")
        print(f"  DA3Metric (GT focal):  min={scaled_metric_gt.min():.2f}m  max={scaled_metric_gt.max():.2f}m  median={np.median(scaled_metric_gt):.2f}m  (focal={da3_focal_gt:.1f})")
        print(f"  DA3Metric (pred focal):min={scaled_metric_pred.min():.2f}m  max={scaled_metric_pred.max():.2f}m  median={np.median(scaled_metric_pred):.2f}m  (focal={da3_focal_pred:.1f})")
        print(f"  DA3-Large relative:    min={raw_rel.min():.4f}  max={raw_rel.max():.4f}  median={np.median(raw_rel):.4f}")
        print(f"  Scale ratio (GT focal):   {scale_ratio_gt:.4f}")
        print(f"  Scale ratio (pred focal): {scale_ratio_pred:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    scale_ratios_gt = [r["scale_ratio_gt_focal"] for r in results]
    scale_ratios_pred = [r["scale_ratio_pred_focal"] for r in results]
    metric_medians = [r["scaled_metric_gt_focal"]["median"] for r in results]

    print(f"\n  Metric depth medians (GT focal): {[f'{m:.2f}m' for m in metric_medians]}")
    print(f"  Scale ratios (GT focal):   {[f'{s:.4f}' for s in scale_ratios_gt]}")
    print(f"    mean={np.mean(scale_ratios_gt):.4f}  std={np.std(scale_ratios_gt):.4f}  CV={np.std(scale_ratios_gt)/np.mean(scale_ratios_gt)*100:.1f}%")
    print(f"  Scale ratios (pred focal): {[f'{s:.4f}' for s in scale_ratios_pred]}")
    print(f"    mean={np.mean(scale_ratios_pred):.4f}  std={np.std(scale_ratios_pred):.4f}  CV={np.std(scale_ratios_pred)/np.mean(scale_ratios_pred)*100:.1f}%")

    # GT camera positions for reference
    frames_data = transforms["frames"]
    gt_positions = []
    for idx in indices:
        c2w = np.array(frames_data[idx]["transform_matrix"])
        gt_positions.append(c2w[:3, 3])
    gt_positions = np.array(gt_positions)
    print(f"\n  GT camera positions (x,y,z):")
    for j, (idx, pos) in enumerate(zip(indices, gt_positions)):
        print(f"    Frame {idx}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    extents = gt_positions.max(axis=0) - gt_positions.min(axis=0)
    print(f"  GT position extents: [{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}]m")
    print(f"  Expected scene depth range: ~{np.linalg.norm(extents[:2]):.1f}m across")

    print("\n  Plausibility check:")
    med_depth = np.mean(metric_medians)
    scene_scale = np.linalg.norm(extents[:2])
    if 0.5 < med_depth < 50.0:
        print(f"    Metric depth median {med_depth:.1f}m is in plausible range (0.5-50m)")
    else:
        print(f"    WARNING: Metric depth median {med_depth:.1f}m seems implausible")

    cv_gt = np.std(scale_ratios_gt) / np.mean(scale_ratios_gt) * 100
    if cv_gt < 20:
        print(f"    Scale ratio CV={cv_gt:.1f}% < 20% — CONSISTENT across frames")
    else:
        print(f"    WARNING: Scale ratio CV={cv_gt:.1f}% >= 20% — NOT consistent")


if __name__ == "__main__":
    main()
