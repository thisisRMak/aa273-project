#!/usr/bin/env python3
"""
Benchmark StreamVGGT camera pose predictions against ground-truth poses.

Ground truth comes from nerfstudio transforms.json (3DGS training data).
Evaluation uses relative pose errors (all-pairs), so no coordinate frame
alignment is needed.

Usage:
    # Default: flightroom_ssv_exp nbv-splat run, 20 frames
    python scripts/eval_poses.py

    # Custom nbv-splat training directory
    python scripts/eval_poses.py /path/to/outputs/scene/nbv-splat/timestamp

    # Direct transforms.json
    python scripts/eval_poses.py --transforms /path/to/transforms.json -n 20

    # Save metrics + plot
    python scripts/eval_poses.py -n 30 -o data/eval/results.json --plot

    # With intrinsics comparison
    python scripts/eval_poses.py --transforms /path/to/transforms.json --compare-intrinsics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_poses")


# ---------------------------------------------------------------------------
# transforms.json loading
# ---------------------------------------------------------------------------

def load_transforms(transforms_path):
    """Load and validate a nerfstudio transforms.json file."""
    with open(transforms_path) as f:
        transforms = json.load(f)

    assert "frames" in transforms, "transforms.json must contain 'frames'"
    assert len(transforms["frames"]) > 0, "transforms.json has no frames"
    for key in ["fl_x", "fl_y", "w", "h"]:
        assert key in transforms, f"transforms.json missing '{key}'"

    return transforms


def resolve_image_paths_from_transforms(transforms, transforms_path):
    """Get absolute image paths from transforms.json frames."""
    transforms_dir = os.path.dirname(os.path.abspath(transforms_path))
    paths = []
    for frame in transforms["frames"]:
        abs_path = os.path.join(transforms_dir, frame["file_path"])
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image not found: {abs_path}")
        paths.append(abs_path)
    return paths


def extract_gt_poses_w2c(transforms):
    """Extract ground truth w2c 4x4 SE(3) matrices from transforms.json.

    transforms.json stores c2w (camera-to-world) matrices in nerfstudio's
    OpenGL camera convention (y-up, z-backward). StreamVGGT uses OpenCV
    convention (y-down, z-forward). We convert by negating columns 1 and 2
    of the c2w rotation, then invert to get w2c.

    Returns:
        [N, 4, 4] float64 tensor of w2c poses in OpenCV camera convention.
    """
    #TODO: Remove vggt dependency
    from vggt.utils.geometry import closed_form_inverse_se3

    c2w_list = []
    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        assert c2w.shape == (4, 4), f"Expected 4x4 matrix, got {c2w.shape}"
        # OpenGL camera → OpenCV camera: flip y and z axes
        c2w[:3, 1] *= -1  # negate camera y-axis (up → down)
        c2w[:3, 2] *= -1  # negate camera z-axis (backward → forward)
        c2w_list.append(c2w)

    c2w = torch.from_numpy(np.stack(c2w_list))  # [N, 4, 4]
    #TODO: implement closed_form_inverse_se3 directly
    w2c = closed_form_inverse_se3(c2w)  # [N, 4, 4]
    return w2c


# ---------------------------------------------------------------------------
# Frame subsampling
# ---------------------------------------------------------------------------

def subsample_indices(total, num_frames, method="uniform"):
    """Select a subset of frame indices.

    Args:
        total: total number of available frames.
        num_frames: desired number of frames.
        method: "uniform" for evenly spaced, "random" for random.

    Returns:
        sorted list of indices.
    """
    if num_frames >= total:
        return list(range(total))

    if method == "uniform":
        return sorted(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    else:
        return sorted(np.random.choice(total, num_frames, replace=False).tolist())


# ---------------------------------------------------------------------------
# StreamVGGT inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_poses(model, images, device, dtype):
    """Run StreamVGGT streaming inference and return w2c extrinsics.

    Args:
        model: StreamVGGT model instance.
        images: [N, 3, H, W] preprocessed image tensor.
        device: torch device.
        dtype: autocast dtype.

    Returns:
        pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
        pred_intrinsics: [1, N, 3, 3] predicted intrinsics or None.
    """
    frames = [{"img": images[i : i + 1].to(device)} for i in range(images.shape[0])]

    with torch.cuda.amp.autocast(dtype=dtype):
        output = model.inference(frames)

    with torch.cuda.amp.autocast(dtype=torch.float64):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_enc = torch.stack(
            [r["camera_pose"] for r in output.ress], dim=1
        )  # [1, N, 9]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )
        pred_3x4 = extrinsics[0]  # [N, 3, 4]

    # Pad to 4x4
    N = pred_3x4.shape[0]
    bottom = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float64, device=device
    ).expand(N, 1, 4)
    pred_w2c = torch.cat([pred_3x4.double(), bottom], dim=1)  # [N, 4, 4]

    return pred_w2c, intrinsics


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def _trim_kv_cache(past_key_values, window_size):
    """Trim KV cache to keep only the last window_size frames (dim=2).

    KV tensors have shape (B, heads, num_frames, seq_per_frame, head_dim).
    """
    for i, kv in enumerate(past_key_values):
        if kv is not None:
            k, v = kv
            if k.shape[2] > window_size:
                past_key_values[i] = (
                    k[:, :, -window_size:, :, :],
                    v[:, :, -window_size:, :, :],
                )


@torch.no_grad()
def predict_poses_windowed(model, images, device, dtype, window_size):
    """StreamVGGT inference with sliding window KV-cache trimming.

    Replicates model.inference() but trims both aggregator and camera_head
    KV caches to the most recent `window_size` frames after each step.
    Only computes camera poses (skips depth/point/track heads).

    Args:
        model: StreamVGGT model instance.
        images: [N, 3, H, W] preprocessed image tensor.
        device: torch device.
        dtype: autocast dtype.
        window_size: Number of frames to keep in the KV cache.

    Returns:
        pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
        pred_intrinsics: None (not computed in windowed mode).
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    N = images.shape[0]
    past_key_values = [None] * model.aggregator.depth
    past_key_values_camera = [None] * model.camera_head.trunk_depth

    all_pose_enc = []
    for i in range(N):
        img = images[i : i + 1].unsqueeze(0).to(device)  # [1, 1, 3, H, W]

        with torch.cuda.amp.autocast(dtype=dtype):
            aggregator_output = model.aggregator(
                img,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
            )
            aggregated_tokens, patch_start_idx, past_key_values = aggregator_output

            with torch.cuda.amp.autocast(enabled=False):
                pose_enc, past_key_values_camera = model.camera_head(
                    aggregated_tokens,
                    past_key_values_camera=past_key_values_camera,
                    use_cache=True,
                )
                pose_enc = pose_enc[-1]
                camera_pose = pose_enc[:, 0, :]  # [1, 9]

        all_pose_enc.append(camera_pose)

        # Trim aggregator KV cache to sliding window (this is the memory saver).
        # Camera head cache is NOT trimmed — it's tiny (4 layers × 1 token/frame)
        # but needs the full history to maintain a consistent pose coordinate frame.
        _trim_kv_cache(past_key_values, window_size)

        if (i + 1) % 50 == 0 or i == N - 1:
            logger.info("  Frame %d/%d", i + 1, N)

    # Decode pose encodings to extrinsics
    with torch.cuda.amp.autocast(dtype=torch.float64):
        pose_enc = torch.stack(all_pose_enc, dim=1)  # [1, N, 9]
        extrinsics, _ = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )
        pred_3x4 = extrinsics[0]  # [N, 3, 4]

    bottom = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float64, device=device
    ).expand(N, 1, 4)
    pred_w2c = torch.cat([pred_3x4.double(), bottom], dim=1)

    return pred_w2c, None


# ---------------------------------------------------------------------------
# Intrinsics comparison
# ---------------------------------------------------------------------------

def compute_intrinsic_errors(pred_intrinsics, gt_fl_x, gt_fl_y, gt_w, gt_h, preprocessed_hw):
    """Compare predicted vs GT focal lengths.

    StreamVGGT predicts intrinsics for the preprocessed image size.
    GT intrinsics are for the original resolution — we scale them down.
    """
    scale_x = preprocessed_hw[1] / gt_w
    scale_y = preprocessed_hw[0] / gt_h
    gt_fx_scaled = gt_fl_x * scale_x
    gt_fy_scaled = gt_fl_y * scale_y

    # pred_intrinsics is [1, N, 3, 3]
    pred_fx = pred_intrinsics[0, :, 0, 0].cpu().numpy()
    pred_fy = pred_intrinsics[0, :, 1, 1].cpu().numpy()

    return {
        "gt_fx_scaled": float(gt_fx_scaled),
        "gt_fy_scaled": float(gt_fy_scaled),
        "pred_fx_mean": float(np.mean(pred_fx)),
        "pred_fy_mean": float(np.mean(pred_fy)),
        "fx_abs_error_mean": float(np.mean(np.abs(pred_fx - gt_fx_scaled))),
        "fy_abs_error_mean": float(np.mean(np.abs(pred_fy - gt_fy_scaled))),
        "fx_rel_error_mean": float(np.mean(np.abs(pred_fx - gt_fx_scaled) / gt_fx_scaled)),
        "fy_rel_error_mean": float(np.mean(np.abs(pred_fy - gt_fy_scaled) / gt_fy_scaled)),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_error_distributions(r_error, t_error, metrics, output_path):
    """CDF plots of rotation and translation errors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sorted_r = np.sort(r_error)
    cdf_r = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax1.plot(sorted_r, cdf_r, "b-", linewidth=1.5)
    for thresh in [3, 5, 15, 30]:
        ax1.axvline(thresh, color="gray", linestyle="--", alpha=0.5, label=f"{thresh}\u00b0")
    ax1.set_xlabel("Relative Rotation Error (degrees)")
    ax1.set_ylabel("CDF")
    ax1.set_title(
        f"Rotation Error (median={metrics['rotation_error_median_deg']:.2f}\u00b0)"
    )
    ax1.set_xlim(0, min(60, np.percentile(r_error, 99)))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    sorted_t = np.sort(t_error)
    cdf_t = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax2.plot(sorted_t, cdf_t, "r-", linewidth=1.5)
    for thresh in [3, 5, 15, 30]:
        ax2.axvline(thresh, color="gray", linestyle="--", alpha=0.5, label=f"{thresh}\u00b0")
    ax2.set_xlabel("Relative Translation Error (degrees)")
    ax2.set_ylabel("CDF")
    ax2.set_title(
        f"Translation Error (median={metrics['translation_error_median_deg']:.2f}\u00b0)"
    )
    ax2.set_xlim(0, min(90, np.percentile(t_error, 99)))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    auc_text = (
        f"AUC@3={metrics['auc_at_3']:.4f}  "
        f"AUC@5={metrics['auc_at_5']:.4f}  "
        f"AUC@15={metrics['auc_at_15']:.4f}  "
        f"AUC@30={metrics['auc_at_30']:.4f}"
    )
    fig.suptitle(auc_text, fontsize=10, y=0.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved error distribution plot to %s", output_path)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(metrics, intrinsic_metrics=None):
    """Pretty-print evaluation metrics to console."""
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(f"\n{'=' * 60}")
    print(f"{BOLD}Relative Pose Evaluation Results{RESET}")
    print(f"{'=' * 60}")
    print(f"  Pairs evaluated: {metrics['num_pairs']}")
    print()
    print(f"  {BOLD}AUC Metrics:{RESET}")
    print(f"    AUC@3:  {GREEN}{metrics['auc_at_3']:.4f}{RESET}")
    print(f"    AUC@5:  {GREEN}{metrics['auc_at_5']:.4f}{RESET}")
    print(f"    AUC@15: {GREEN}{metrics['auc_at_15']:.4f}{RESET}")
    print(f"    AUC@30: {GREEN}{metrics['auc_at_30']:.4f}{RESET}")
    print()
    print(f"  {BOLD}Rotation Error:{RESET}")
    print(f"    Mean:   {BLUE}{metrics['rotation_error_mean_deg']:.3f} deg{RESET}")
    print(f"    Median: {BLUE}{metrics['rotation_error_median_deg']:.3f} deg{RESET}")
    print(f"    Acc@5:  {metrics['r_acc_at_5']:.4f}")
    print(f"    Acc@15: {metrics['r_acc_at_15']:.4f}")
    print()
    print(f"  {BOLD}Translation Error:{RESET}")
    print(f"    Mean:   {BLUE}{metrics['translation_error_mean_deg']:.3f} deg{RESET}")
    print(f"    Median: {BLUE}{metrics['translation_error_median_deg']:.3f} deg{RESET}")
    print(f"    Acc@5:  {metrics['t_acc_at_5']:.4f}")
    print(f"    Acc@15: {metrics['t_acc_at_15']:.4f}")

    if intrinsic_metrics:
        print()
        print(f"  {BOLD}Intrinsics (focal length):{RESET}")
        print(f"    GT fx (scaled): {intrinsic_metrics['gt_fx_scaled']:.1f}")
        print(f"    Pred fx (mean): {intrinsic_metrics['pred_fx_mean']:.1f}")
        print(f"    fx rel error:   {intrinsic_metrics['fx_rel_error_mean']:.4f}")
        print(f"    fy rel error:   {intrinsic_metrics['fy_rel_error_mean']:.4f}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_DIR = (
    "/media/admin/data/StanfordMSL/nerf_data/amber/3dgs/workspace"
    "/outputs/flightroom_ssv_exp/nbv-splat/2026-02-20_051708"
)
DEFAULT_NUM_FRAMES = 20


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark StreamVGGT pose predictions against GT from transforms.json",
    )

    parser.add_argument(
        "training_dir", nargs="?", default=None,
        help="Path to nbv-splat training output directory. "
        "Default: flightroom_ssv_exp 2026-02-20_051708.",
    )
    parser.add_argument(
        "--transforms",
        help="Path to nerfstudio transforms.json (alternative to positional arg).",
    )

    parser.add_argument(
        "--num-frames", "-n", type=int, default=None,
        help=f"Number of frames to evaluate "
        f"(default: {DEFAULT_NUM_FRAMES} for default scene, all otherwise).",
    )
    parser.add_argument(
        "--subsample-method", choices=["uniform", "random"], default="uniform",
        help="How to subsample frames (default: uniform spacing).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--model",
        choices=["streamvggt", "da3", "da3_chunked", "da3_pairwise", "openvins", "reloc3r"],
        default="streamvggt",
        help="Pose prediction model (default: streamvggt).\n"\
             "reloc3r uses first+last frames as anchors (batch). "
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to StreamVGGT checkpoint.",
    )
    parser.add_argument(
        "--da3-model-name", default=None,
        help="DA3 HuggingFace model ID (default: depth-anything/DA3-LARGE-1.1).",
    )
    parser.add_argument(
        "--img-reso", type=int, default=512, choices=[224, 512],
        help="Reloc3r image resolution (default: 512).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save metrics to JSON file.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save error distribution CDF plot.",
    )
    parser.add_argument(
        "--compare-intrinsics", action="store_true",
        help="Also compare predicted vs GT focal lengths.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save 3D trajectory visualization on sparse point cloud.",
    )
    parser.add_argument(
        "--sparse-pc", default=None,
        help="Path to sparse_pc.ply (auto-discovered from transforms.json if omitted).",
    )
    parser.add_argument(
        "--z-band", type=float, default=2.0,
        help="Z-band half-width in meters for point cloud filtering (default: 2.0).",
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Sliding window size for KV-cache trimming (StreamVGGT only). "
        "Processes all frames with bounded memory. Default: None (unbounded).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=60,
        help="Chunk size for da3_chunked model (default: 60).",
    )
    parser.add_argument(
        "--overlap", type=int, default=20,
        help="Overlap between chunks for da3_chunked model (default: 20).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-frame diagnostic output.",
    )
    parser.add_argument(
        "--openvins-trajectory", default=None,
        help="Path to TUM-format trajectory file (openvins model only).",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load transforms and resolve image paths
    # ------------------------------------------------------------------
    if args.transforms:
        transforms_path = os.path.abspath(args.transforms)
        transforms = load_transforms(transforms_path)
        image_paths = resolve_image_paths_from_transforms(transforms, transforms_path)
        scene_name = Path(transforms_path).parent.name
    else:
        # nbv-splat training directory mode (positional arg or default)
        training_dir = args.training_dir
        using_default = training_dir is None
        if using_default:
            training_dir = DEFAULT_TRAINING_DIR
            logger.info("No input specified, using default: %s", training_dir)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from extract_nbv_latents import parse_training_dir, resolve_image_paths

        parsed = parse_training_dir(training_dir)
        transforms = parsed["transforms"]
        transforms_path = parsed["transforms_path"]
        scene_name = parsed["scene_name"]

        num_to_use = args.num_frames
        if num_to_use is None and using_default:
            num_to_use = DEFAULT_NUM_FRAMES
        if num_to_use is None:
            num_to_use = len(parsed["view_order"])
        num_to_use = min(num_to_use, len(parsed["view_order"]))
        image_paths = resolve_image_paths(parsed, num_to_use)

        # Reorder transforms frames to match view-selection order
        view_order = parsed["view_order"][:num_to_use]
        transforms["frames"] = [transforms["frames"][i] for i in view_order]

    # ------------------------------------------------------------------
    # Subsample frames (only in --transforms mode, batch models only)
    # ------------------------------------------------------------------
    total_frames = len(transforms["frames"])
    _streaming_models = {"da3_pairwise", "openvins"}
    _is_streaming = (
        args.model in _streaming_models
        or (args.model == "streamvggt" and args.window_size is not None)
    )

    if args.num_frames and args.transforms:
        if _is_streaming:
            logger.info(
                "Streaming model '%s': processing all %d frames (ignoring -n %d)",
                args.model, total_frames, args.num_frames,
            )
        else:
            indices = subsample_indices(total_frames, args.num_frames, args.subsample_method)
            transforms["frames"] = [transforms["frames"][i] for i in indices]
            image_paths = [image_paths[i] for i in indices]

    num_frames = len(transforms["frames"])
    num_pairs = num_frames * (num_frames - 1) // 2
    logger.info("Scene: %s", scene_name)
    logger.info("Evaluating %d frames (%d pairs)", num_frames, num_pairs)

    if num_frames > 200:
        logger.warning(
            "Large frame count (%d) produces %d pairs. "
            "Consider using --num-frames to subsample.",
            num_frames, num_pairs,
        )

    # ------------------------------------------------------------------
    # Extract GT w2c poses
    # ------------------------------------------------------------------
    gt_w2c = extract_gt_poses_w2c(transforms)  # [N, 4, 4] float64
    logger.info("GT poses loaded: %d frames", gt_w2c.shape[0])

    # ------------------------------------------------------------------
    # Load model and run inference
    # ------------------------------------------------------------------
    if args.model == "streamvggt":
        from goggles.latent_extractor import LatentExtractor

        extractor = LatentExtractor(checkpoint_path=args.checkpoint, device=args.device)
        model = extractor.model
        device = extractor.device
        dtype = extractor.dtype

        from streamvggt.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(image_paths)  # [N, 3, H, W]
        logger.info("Preprocessed images: %s", list(images.shape))

        preprocessed_hw = images.shape[-2:]

        if args.window_size is not None:
            logger.info(
                "Running StreamVGGT sliding-window inference "
                "(%d frames, window=%d)...",
                num_frames, args.window_size,
            )
            pred_w2c, pred_intrinsics = predict_poses_windowed(
                model, images, device, dtype, args.window_size,
            )
        else:
            logger.info("Running StreamVGGT inference (%d frames)...", num_frames)
            pred_w2c, pred_intrinsics = predict_poses(model, images, device, dtype)

    elif args.model == "da3":
        from goggles.da3_predictor import DA3PosePredictor

        da3 = DA3PosePredictor(model_name=args.da3_model_name, device=args.device)
        device = da3.device

        # DA3 default processing resolution (upper_bound_resize to 504)
        preprocessed_hw = (504, 504)

        logger.info("Running DA3 inference (%d frames)...", num_frames)
        pred_w2c, pred_intrinsics = da3.predict_poses(image_paths)

    elif args.model == "da3_chunked":
        from goggles.da3_chunked_predictor import DA3ChunkedPredictor

        predictor = DA3ChunkedPredictor(
            model_name=args.da3_model_name,
            device=args.device,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        device = predictor.device
        preprocessed_hw = (504, 504)

        logger.info(
            "Running DA3 chunked inference (%d frames, chunk=%d, overlap=%d)...",
            num_frames, args.chunk_size, args.overlap,
        )
        pred_w2c, pred_intrinsics = predictor.predict_poses(image_paths)

    elif args.model == "da3_pairwise":
        from goggles.da3_pairwise_predictor import DA3PairwisePredictor

        predictor = DA3PairwisePredictor(
            model_name=args.da3_model_name,
            device=args.device,
        )
        device = predictor.device
        preprocessed_hw = (504, 504)

        logger.info("Running DA3 pairwise inference (%d frames)...", num_frames)
        pred_w2c, pred_intrinsics = predictor.predict_poses(image_paths)

    elif args.model == "openvins":
        from goggles.tum_utils import load_tum_trajectory

        if not args.openvins_trajectory:
            logger.error("--openvins-trajectory required for openvins model")
            sys.exit(1)

        logger.info(
            "Loading OpenVINS trajectory from %s", args.openvins_trajectory
        )
        all_pred_w2c, ov_timestamps = load_tum_trajectory(
            args.openvins_trajectory, device=args.device
        )

        # OpenVINS produces poses only after initialization, so we may have
        # fewer predicted poses than GT frames.  Match by timestamp using
        # image_timestamps.csv (written by the experiment pipeline).
        ov_dir = Path(args.openvins_trajectory).parent
        image_ts_csv = ov_dir / "image_timestamps.csv"
        if image_ts_csv.is_file():
            gt_image_ts = []
            with open(image_ts_csv) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    gt_image_ts.append(float(parts[0]))
            gt_image_ts = np.array(gt_image_ts)

            # For each GT frame, find nearest OpenVINS pose (50 ms tolerance)
            matched_pred = []
            matched_gt_indices = []
            tolerance = 0.05  # seconds
            for gi, gt_t in enumerate(gt_image_ts):
                diffs = np.abs(ov_timestamps - gt_t)
                best = np.argmin(diffs)
                if diffs[best] < tolerance:
                    matched_pred.append(all_pred_w2c[best])
                    matched_gt_indices.append(gi)

            if len(matched_pred) == 0:
                logger.error(
                    "No OpenVINS poses matched GT timestamps "
                    "(OV: %.3f–%.3f, GT: %.3f–%.3f)",
                    ov_timestamps[0], ov_timestamps[-1],
                    gt_image_ts[0], gt_image_ts[-1],
                )
                sys.exit(1)

            pred_w2c = torch.stack(matched_pred)
            # Subsample GT to matched frames only
            transforms["frames"] = [transforms["frames"][i] for i in matched_gt_indices]
            image_paths = [image_paths[i] for i in matched_gt_indices]
            gt_w2c = extract_gt_poses_w2c(transforms)
            num_frames = len(matched_gt_indices)
            logger.info(
                "Matched %d/%d OpenVINS poses to GT (%d total GT frames)",
                len(matched_pred), len(ov_timestamps), len(gt_image_ts),
            )
        else:
            # Fallback: assume 1:1 correspondence
            pred_w2c = all_pred_w2c
            logger.warning(
                "No image_timestamps.csv — assuming 1:1 correspondence "
                "(%d predicted, %d GT)", pred_w2c.shape[0], gt_w2c.shape[0],
            )

        pred_intrinsics = None
        device = torch.device(args.device)
        # OpenVINS doesn't preprocess images — use GT resolution
        preprocessed_hw = None

    elif args.model == "reloc3r":
        
        from goggles.reloc3r_predictor import Reloc3rPredictor

        predictor = Reloc3rPredictor(
            img_reso=args.img_reso,
            device=args.device,
        )
        device = predictor.device
        preprocessed_hw = (args.img_reso, args.img_reso)

        logger.info(
            "Running Reloc3r-%d inference (%d frames)...", args.img_reso, num_frames
        )
        pred_w2c, pred_intrinsics = predictor.predict_poses(image_paths)

    logger.info("Predicted poses: %s", list(pred_w2c.shape))

    # ------------------------------------------------------------------
    # Diagnostic: print poses for convention debugging (--verbose)
    # ------------------------------------------------------------------
    if args.verbose:
        diag_frames = [0, min(1, num_frames - 1), min(num_frames - 1, 19)]
        diag_frames = sorted(set(diag_frames))
        print("\n--- Diagnostic: pose samples ---")
        for fi in diag_frames:
            pred_np = pred_w2c[fi].cpu().numpy()
            gt_np = gt_w2c[fi].numpy() if not gt_w2c.is_cuda else gt_w2c[fi].cpu().numpy()
            pred_t = np.linalg.norm(pred_np[:3, 3])
            gt_t = np.linalg.norm(gt_np[:3, 3])
            pred_r_trace = np.clip((np.trace(pred_np[:3, :3]) - 1) / 2, -1, 1)
            gt_r_trace = np.clip((np.trace(gt_np[:3, :3]) - 1) / 2, -1, 1)
            pred_angle = np.degrees(np.arccos(pred_r_trace))
            gt_angle = np.degrees(np.arccos(gt_r_trace))
            img_name = Path(image_paths[fi]).name
            print(f"  Frame {fi} ({img_name}):")
            print(f"    Pred w2c:  |t|={pred_t:.4f}  rot_from_I={pred_angle:.1f}deg")
            print(f"    GT   w2c:  |t|={gt_t:.4f}  rot_from_I={gt_angle:.1f}deg")
        print("---\n")

    # ------------------------------------------------------------------
    # Compute relative pose errors
    # ------------------------------------------------------------------
    from goggles.pose_eval import se3_to_relative_pose_error, compute_pose_metrics

    gt_w2c = gt_w2c.to(device)
    pred_w2c = pred_w2c.to(device)
    rel_r_err, rel_t_err = se3_to_relative_pose_error(pred_w2c, gt_w2c, num_frames)

    r_error_np = rel_r_err.cpu().numpy()
    t_error_np = rel_t_err.cpu().numpy()

    metrics = compute_pose_metrics(r_error_np, t_error_np)
    metrics["scene_name"] = scene_name
    metrics["num_frames"] = num_frames
    metrics["model"] = args.model

    # ------------------------------------------------------------------
    # Optional intrinsics comparison
    # ------------------------------------------------------------------
    intrinsic_metrics = None
    if args.compare_intrinsics and pred_intrinsics is not None:
        intrinsic_metrics = compute_intrinsic_errors(
            pred_intrinsics,
            transforms["fl_x"], transforms["fl_y"],
            transforms["w"], transforms["h"],
            preprocessed_hw,
        )
        metrics["intrinsics"] = intrinsic_metrics

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print_metrics(metrics, intrinsic_metrics)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", args.output)

    # ------------------------------------------------------------------
    # Save plot
    # ------------------------------------------------------------------
    if args.plot:
        if args.output:
            plot_path = args.output.replace(".json", ".png")
        else:
            plot_path = f"eval_poses_{scene_name}.png"
        plot_error_distributions(r_error_np, t_error_np, metrics, plot_path)

    # ------------------------------------------------------------------
    # Align predicted poses to GT (always, for .npz export)
    # ------------------------------------------------------------------
    from goggles.visualization import align_poses_first_frame

    aligned_c2w, _, gt_c2w = align_poses_first_frame(pred_w2c, gt_w2c)
    gt_pos = gt_c2w[:, :3, 3].numpy()
    pred_pos = aligned_c2w[:, :3, 3].numpy()

    # Save aligned positions for multi-method comparison plots
    if args.output:
        npz_path = args.output.replace(".json", "_trajectory.npz")
        np.savez(npz_path, gt=gt_pos, pred=pred_pos)
        logger.info("Saved aligned positions to %s", npz_path)

    # ------------------------------------------------------------------
    # Trajectory visualization on point cloud
    # ------------------------------------------------------------------
    if args.visualize:
        from goggles.visualization import (
            discover_sparse_pc,
            load_sparse_pointcloud,
            plot_trajectory_on_pointcloud,
        )

        # Load sparse point cloud
        pc_path = args.sparse_pc or discover_sparse_pc(transforms_path)
        if pc_path is None:
            logger.warning(
                "No sparse_pc.ply found near %s. "
                "Use --sparse-pc to specify the path. "
                "Plotting trajectories without point cloud.",
                transforms_path,
            )
            pcd_pts = np.empty((3, 0))
            pcd_colors = None
        else:
            pcd_pts, pcd_colors = load_sparse_pointcloud(pc_path)

        # Output path
        if args.output:
            vis_path = args.output.replace(".json", "_trajectory.png")
        else:
            vis_path = f"eval_poses_{scene_name}_trajectory.png"

        plot_trajectory_on_pointcloud(
            gt_pos, pred_pos, pcd_pts, pcd_colors,
            title=f"{scene_name} ({num_frames} frames)",
            output_path=vis_path,
            z_band=args.z_band,
        )

    return metrics


if __name__ == "__main__":
    main()
