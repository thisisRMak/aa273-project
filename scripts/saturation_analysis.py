"""
Saturation Analysis: Does StreamVGGT's latent representation reach steady state?

Analyzes the streaming-mode aggregator tokens to detect whether the
cross-frame representation converges as more views are added, and whether
this convergence correlates with reconstruction metrics (PSNR, coverage)
approaching their maxima.

Four analyses:
1. Camera token velocity — L2 norm of consecutive differences
2. Global vs local velocity — split dims [0:1024] and [1024:2048]
3. Cosine similarity ramp — consecutive token similarity
4. Overlay with reconstruction metrics — dual-axis correlation plots

Usage:
    python notebooks/saturation_analysis.py [path_to_h5]
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
SMOOTHING_WINDOW = 5  # rolling average window for velocity curves

# ── Load data ───────────────────────────────────────────────────────────

def load_h5(path: str) -> dict:
    """Load streaming extraction HDF5 into a flat dict."""
    data = {}
    with h5py.File(path, "r") as f:
        data["tokens"] = f["tokens"][:]  # [N, L, P, 2048]
        data["layer_indices"] = f["layer_indices"][:].tolist()
        data["view_order"] = f["view_order"][:]

        # Reconstruction metrics (logged at eval steps, not per-view)
        if "metrics" in f:
            mg = f["metrics"]
            data["metrics"] = {
                key: mg[key][:] for key in mg.keys()
            }
        if "converged" in f:
            cg = f["converged"]
            data["converged"] = {
                key: cg[key][:] for key in cg.keys()
            }

        data["streaming"] = f.attrs.get("streaming", False)
        data["scene_name"] = f.attrs.get("scene_name", "unknown")

    return data


def rolling_mean(x, w):
    """Simple rolling mean with same-length output (edge-padded)."""
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="same")
    return smoothed


# ── Analysis functions ──────────────────────────────────────────────────

def camera_token_velocity(tokens, layer_idx=-1):
    """L2 norm of consecutive camera token differences.

    Camera token is at position 0 (first special token).
    Args:
        tokens: [N, n_layers, P, 2048]
        layer_idx: which layer to use (-1 = last = layer 23)
    Returns:
        velocity: [N-1] array of L2 norms
    """
    cam = tokens[:, layer_idx, 0, :].astype(np.float32)  # [N, 2048]
    diffs = np.diff(cam, axis=0)  # [N-1, 2048]
    return np.linalg.norm(diffs, axis=1)


def camera_token_angular_distance(tokens, layer_idx=-1):
    """1 - cos_sim between consecutive camera tokens. Bounded [0, 2]."""
    cam = tokens[:, layer_idx, 0, :].astype(np.float32)
    norms = np.linalg.norm(cam, axis=1, keepdims=True)
    cam_normed = cam / np.clip(norms, 1e-8, None)
    cos_sim = np.sum(cam_normed[:-1] * cam_normed[1:], axis=1)
    return 1.0 - cos_sim


def split_velocity(tokens, layer_idx=-1, pool="mean"):
    """Velocity of frame-local (0:1024) vs global (1024:2048) halves.

    Computed on mean-pooled patch tokens (indices 5+, skipping special tokens).
    """
    patches = tokens[:, layer_idx, 5:, :].astype(np.float32)  # [N, n_patches, 2048]

    if pool == "mean":
        pooled = patches.mean(axis=1)  # [N, 2048]
    else:
        pooled = patches.max(axis=1)

    local_half = pooled[:, :1024]   # frame attention output
    global_half = pooled[:, 1024:]  # global attention output

    local_diff = np.diff(local_half, axis=0)
    global_diff = np.diff(global_half, axis=0)

    local_vel = np.linalg.norm(local_diff, axis=1)
    global_vel = np.linalg.norm(global_diff, axis=1)

    return local_vel, global_vel


def split_angular_distance(tokens, layer_idx=-1, pool="mean"):
    """1 - cos_sim for frame-local and global halves of mean-pooled patches."""
    patches = tokens[:, layer_idx, 5:, :].astype(np.float32)

    if pool == "mean":
        pooled = patches.mean(axis=1)
    else:
        pooled = patches.max(axis=1)

    local_half = pooled[:, :1024]
    global_half = pooled[:, 1024:]

    def _angular_dist(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_normed = x / np.clip(norms, 1e-8, None)
        cos_sim = np.sum(x_normed[:-1] * x_normed[1:], axis=1)
        return 1.0 - cos_sim

    return _angular_dist(local_half), _angular_dist(global_half)


def cosine_similarity_ramp(tokens, layer_idx=-1, offsets=(1, 5, 10)):
    """Cosine similarity between camera tokens at frame t and t+k."""
    cam = tokens[:, layer_idx, 0, :].astype(np.float32)  # [N, 2048]
    norms = np.linalg.norm(cam, axis=1, keepdims=True)
    cam_normed = cam / np.clip(norms, 1e-8, None)

    results = {}
    for k in offsets:
        if k >= len(cam):
            continue
        sims = np.sum(cam_normed[:-k] * cam_normed[k:], axis=1)  # [N-k]
        results[k] = sims
    return results


def interpolate_metrics_to_views(metrics, n_views):
    """Map eval-step metrics to per-view indices using num_active_views.

    Metrics are logged at training eval steps. Each eval step records
    num_active_views. We interpolate to get per-view metric estimates.
    """
    if "num_active_views" not in metrics:
        return None

    nav = metrics["num_active_views"]
    result = {}
    for key in ["psnr", "ssim", "lpips", "coverage_mean"]:
        if key not in metrics:
            continue
        vals = metrics[key]
        # Interpolate: for each view index 0..n_views-1, find the metric
        # value at the closest eval step by num_active_views
        view_indices = np.arange(n_views)
        interp_vals = np.interp(view_indices, nav, vals)
        result[key] = interp_vals
    return result


# ── Plotting ────────────────────────────────────────────────────────────

def _add_metric_overlay(ax, metrics_per_view):
    """Add PSNR + coverage to a twin axis. Returns (lines, labels) for legend."""
    if metrics_per_view is None or "psnr" not in metrics_per_view:
        return [], []
    ax2 = ax.twinx()
    psnr = metrics_per_view["psnr"]
    ax2.plot(np.arange(len(psnr)), psnr, color="coral", linewidth=2, label="PSNR")
    ax2.set_ylabel("PSNR (dB)", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    if "coverage_mean" in metrics_per_view:
        cov = metrics_per_view["coverage_mean"]
        ax2.plot(np.arange(len(cov)), cov * 30 + 10, color="green",
                 linewidth=1.5, linestyle="--", alpha=0.7, label="Coverage (scaled)")
    return ax2.get_legend_handles_labels()


def plot_camera_velocity(vel, ang_dist, metrics_per_view, scene_name, save_dir):
    """Plot 1: Camera token — L2 velocity (top) and angular distance (bottom) with metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    frames = np.arange(1, len(vel) + 1)

    # Top: L2 velocity
    ax = axes[0]
    ax.plot(frames, vel, alpha=0.25, color="steelblue", linewidth=0.8)
    ax.plot(frames, rolling_mean(vel, SMOOTHING_WINDOW), color="steelblue",
            linewidth=2, label="L2 velocity")
    ax.set_ylabel("L2 norm", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    lines_m, labels_m = _add_metric_overlay(ax, metrics_per_view)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines_m, labels1 + labels_m, loc="upper right", fontsize=9)
    ax.set_title(f"Camera Token: L2 Velocity vs Reconstruction — {scene_name}")
    ax.grid(True, alpha=0.3)

    # Bottom: angular distance (1 - cos_sim)
    ax = axes[1]
    ax.plot(frames, ang_dist, alpha=0.25, color="purple", linewidth=0.8)
    ax.plot(frames, rolling_mean(ang_dist, SMOOTHING_WINDOW), color="purple",
            linewidth=2, label="Angular distance (1 − cos_sim)")
    ax.set_ylabel("1 − cos_sim", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax.set_xlabel("View index")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    lines_m, labels_m = _add_metric_overlay(ax, metrics_per_view)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines_m, labels1 + labels_m, loc="upper right", fontsize=9)
    ax.set_title(f"Camera Token: Angular Distance vs Reconstruction — {scene_name}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "01_camera_token_velocity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_split_velocity(local_vel, global_vel, local_ang, global_ang,
                        scene_name, save_dir):
    """Plot 2: Frame-local vs global — L2 velocity (top) and angular distance (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    frames = np.arange(1, len(local_vel) + 1)

    # Top: L2 velocity
    ax = axes[0]
    ax.plot(frames, local_vel, alpha=0.15, color="orange", linewidth=0.8)
    ax.plot(frames, rolling_mean(local_vel, SMOOTHING_WINDOW), color="orange",
            linewidth=2, label="Frame-local (dims 0:1024)")
    ax.plot(frames, global_vel, alpha=0.15, color="purple", linewidth=0.8)
    ax.plot(frames, rolling_mean(global_vel, SMOOTHING_WINDOW), color="purple",
            linewidth=2, label="Global-temporal (dims 1024:2048)")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax.set_ylabel("L2 velocity (mean-pooled patches)")
    ax.set_title(f"Split L2 Velocity — {scene_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: angular distance
    ax = axes[1]
    ax.plot(frames, local_ang, alpha=0.15, color="orange", linewidth=0.8)
    ax.plot(frames, rolling_mean(local_ang, SMOOTHING_WINDOW), color="orange",
            linewidth=2, label="Frame-local (dims 0:1024)")
    ax.plot(frames, global_ang, alpha=0.15, color="purple", linewidth=0.8)
    ax.plot(frames, rolling_mean(global_ang, SMOOTHING_WINDOW), color="purple",
            linewidth=2, label="Global-temporal (dims 1024:2048)")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax.set_xlabel("View index")
    ax.set_ylabel("Angular distance (1 − cos_sim)")
    ax.set_title(f"Split Angular Distance — {scene_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "02_split_velocity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_cosine_ramp(cos_sims, scene_name, save_dir):
    """Plot 3: Cosine similarity between camera tokens at offset k."""
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {1: "steelblue", 5: "orange", 10: "green"}
    for k, sims in sorted(cos_sims.items()):
        frames = np.arange(len(sims))
        sims_smooth = rolling_mean(sims, SMOOTHING_WINDOW)
        ax.plot(frames, sims, alpha=0.15, color=colors.get(k, "gray"), linewidth=0.8)
        ax.plot(frames, sims_smooth, color=colors.get(k, "gray"), linewidth=2,
                label=f"cos_sim(t, t+{k})")

    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax.set_xlabel("View index t")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Camera Token Cosine Similarity Ramp — {scene_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()

    path = save_dir / "03_cosine_similarity_ramp.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_metric_overlay(local_vel, global_vel, metrics_per_view, scene_name, save_dir):
    """Plot 4: Global velocity ratio vs coverage — direct saturation test."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    frames = np.arange(1, len(local_vel) + 1)

    # Top: ratio of global to local velocity (normalized)
    # If global velocity drops while local stays high → saturation
    local_smooth = rolling_mean(local_vel, SMOOTHING_WINDOW)
    global_smooth = rolling_mean(global_vel, SMOOTHING_WINDOW)
    ratio = global_smooth / np.clip(local_smooth, 1e-8, None)

    ax1 = axes[0]
    ax1.plot(frames, ratio, color="purple", linewidth=2, label="Global / Local velocity ratio")
    ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Velocity ratio")
    ax1.set_title(f"Cross-Frame Information Gain Rate — {scene_name}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: reconstruction metrics
    ax2 = axes[1]
    if metrics_per_view is not None:
        if "psnr" in metrics_per_view:
            psnr = metrics_per_view["psnr"]
            ax2.plot(np.arange(len(psnr)), psnr, color="coral", linewidth=2, label="PSNR")
        if "coverage_mean" in metrics_per_view:
            cov = metrics_per_view["coverage_mean"]
            ax2_r = ax2.twinx()
            ax2_r.plot(np.arange(len(cov)), cov, color="green", linewidth=2,
                       linestyle="--", label="Coverage")
            ax2_r.set_ylabel("Coverage", color="green")
            ax2_r.tick_params(axis="y", labelcolor="green")
            lines_r, labels_r = ax2_r.get_legend_handles_labels()
        else:
            lines_r, labels_r = [], []
        if "ssim" in metrics_per_view:
            ssim = metrics_per_view["ssim"]
            ax2.plot(np.arange(len(ssim)), ssim * 30, color="teal",
                     linewidth=1.5, alpha=0.7, label="SSIM (x30)")
    else:
        lines_r, labels_r = [], []

    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax2.set_xlabel("View index")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Reconstruction Metrics")
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2 + lines_r, labels2 + labels_r, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "04_metric_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_norm_decomposition(tokens, layer_idx, scene_name, save_dir):
    """Plot 6: Disentangle L2 velocity from cosine similarity.

    L2 distance between consecutive tokens decomposes as:
        ||a - b||² = ||a||² + ||b||² - 2·||a||·||b||·cos(a,b)

    If norms are ~constant, then L2 ≈ norm * sqrt(2 - 2·cos_sim),
    and the two metrics are just nonlinear rescalings.  If norms vary,
    they carry independent information.
    """
    cam = tokens[:, layer_idx, 0, :].astype(np.float32)  # [N, 2048]
    norms = np.linalg.norm(cam, axis=1)  # [N]

    # Actual consecutive L2 velocity
    actual_l2 = np.linalg.norm(np.diff(cam, axis=0), axis=1)  # [N-1]

    # Cosine similarity
    cam_normed = cam / np.clip(norms[:, None], 1e-8, None)
    cos_sim = np.sum(cam_normed[:-1] * cam_normed[1:], axis=1)  # [N-1]

    # Predicted L2 from norms + cos_sim:
    # ||a-b|| = sqrt(||a||² + ||b||² - 2·||a||·||b||·cos(a,b))
    n_a, n_b = norms[:-1], norms[1:]
    predicted_l2 = np.sqrt(np.clip(n_a**2 + n_b**2 - 2*n_a*n_b*cos_sim, 0, None))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    frames = np.arange(len(norms))
    diff_frames = np.arange(1, len(actual_l2) + 1)

    # Panel 1: Token norm over time
    ax = axes[0]
    ax.plot(frames, norms, color="steelblue", linewidth=1.5, alpha=0.4)
    ax.plot(frames, rolling_mean(norms, SMOOTHING_WINDOW), color="steelblue", linewidth=2)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("||camera token||₂")
    ax.set_title(f"Camera Token Norm Over Time — {scene_name}")
    ax.grid(True, alpha=0.3)
    # Show coefficient of variation
    cv = norms.std() / norms.mean()
    ax.text(0.98, 0.95, f"CV = {cv:.3f}  (mean={norms.mean():.1f}, std={norms.std():.1f})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Panel 2: Actual L2 vs predicted L2 (should be identical — sanity check)
    ax = axes[1]
    ax.plot(diff_frames, rolling_mean(actual_l2, SMOOTHING_WINDOW),
            color="steelblue", linewidth=2, label="Actual L2 velocity")
    ax.plot(diff_frames, rolling_mean(predicted_l2, SMOOTHING_WINDOW),
            color="coral", linewidth=2, linestyle="--", label="Predicted from norm + cos_sim")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("L2 distance")
    ax.set_title("Actual vs Predicted L2 (sanity: should overlap)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Decompose — what fraction of L2 change comes from norm change vs direction change?
    # At constant norm N: L2_dir = N * sqrt(2 - 2·cos_sim)
    # At constant direction: L2_norm = |N_a - N_b|
    ax = axes[2]
    mean_norm = (n_a + n_b) / 2
    l2_from_direction = mean_norm * np.sqrt(np.clip(2 - 2*cos_sim, 0, None))
    l2_from_norm = np.abs(n_a - n_b)

    ax.plot(diff_frames, rolling_mean(l2_from_direction, SMOOTHING_WINDOW),
            color="purple", linewidth=2, label="Directional component (norm × angular dist)")
    ax.plot(diff_frames, rolling_mean(l2_from_norm, SMOOTHING_WINDOW),
            color="orange", linewidth=2, label="Norm component (|Δ norm|)")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("View index")
    ax.set_ylabel("L2 contribution")
    ax.set_title("L2 Velocity Decomposition: Direction vs Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "06_norm_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    return norms, cv


def plot_per_layer_velocity(tokens, scene_name, save_dir, layer_indices):
    """Plot 5: Per-layer — L2 velocity (top) and angular distance (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layer_indices)))

    # Top: L2 velocity
    ax = axes[0]
    for li_idx, (li, color) in enumerate(zip(layer_indices, colors)):
        vel = camera_token_velocity(tokens, layer_idx=li_idx)
        frames = np.arange(1, len(vel) + 1)
        ax.plot(frames, rolling_mean(vel, SMOOTHING_WINDOW), color=color,
                linewidth=2, label=f"Layer {li}")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax.set_ylabel("L2 velocity")
    ax.set_title(f"Per-Layer Camera Token L2 Velocity — {scene_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: angular distance
    ax = axes[1]
    for li_idx, (li, color) in enumerate(zip(layer_indices, colors)):
        ang = camera_token_angular_distance(tokens, layer_idx=li_idx)
        frames = np.arange(1, len(ang) + 1)
        ax.plot(frames, rolling_mean(ang, SMOOTHING_WINDOW), color=color,
                linewidth=2, label=f"Layer {li}")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Cache window (50)")
    ax.set_xlabel("View index")
    ax.set_ylabel("Angular distance (1 − cos_sim)")
    ax.set_title(f"Per-Layer Camera Token Angular Distance — {scene_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "05_per_layer_velocity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        # Default to latest streaming extraction
        h5_path = str(
            Path(__file__).resolve().parent.parent
            / "data" / "flightroom_ssv_exp"
            / "2026-02-20_051708_260views_full.h5"
        )

    print(f"Loading: {h5_path}")
    data = load_h5(h5_path)

    tokens = data["tokens"]
    n_frames, n_layers, n_tokens, feat_dim = tokens.shape
    print(f"Tokens: {n_frames} frames, {n_layers} layers, {n_tokens} tokens/frame, {feat_dim}-dim")
    print(f"Streaming: {data['streaming']}")

    # Output directory
    save_dir = Path(h5_path).parent / "saturation_analysis"
    save_dir.mkdir(exist_ok=True)
    print(f"Saving plots to: {save_dir}")

    scene_name = data["scene_name"]

    # Interpolate metrics to per-view
    metrics_per_view = None
    if "metrics" in data:
        metrics_per_view = interpolate_metrics_to_views(data["metrics"], n_frames)
        if metrics_per_view:
            print(f"Metrics interpolated to {n_frames} views")
            for k, v in metrics_per_view.items():
                print(f"  {k}: {v[0]:.3f} -> {v[-1]:.3f}")

    # ── Analysis 1: Camera token velocity ───────────────────────────
    print("\n── Analysis 1: Camera Token Velocity + Angular Distance ──")
    vel = camera_token_velocity(tokens, layer_idx=-1)
    ang_dist = camera_token_angular_distance(tokens, layer_idx=-1)
    print(f"  L2 velocity — mean: {vel.mean():.3f}, first 10: {vel[:10].mean():.3f}, last 10: {vel[-10:].mean():.3f}")
    print(f"  Angular dist — mean: {ang_dist.mean():.4f}, first 10: {ang_dist[:10].mean():.4f}, last 10: {ang_dist[-10:].mean():.4f}")
    plot_camera_velocity(vel, ang_dist, metrics_per_view, scene_name, save_dir)

    # ── Analysis 2: Frame-local vs global velocity ──────────────────
    print("\n── Analysis 2: Frame-Local vs Global (Velocity + Angular) ──")
    local_vel, global_vel = split_velocity(tokens, layer_idx=-1)
    local_ang, global_ang = split_angular_distance(tokens, layer_idx=-1)
    print(f"  Local  L2 — mean: {local_vel.mean():.3f}, first 10: {local_vel[:10].mean():.3f}, last 10: {local_vel[-10:].mean():.3f}")
    print(f"  Global L2 — mean: {global_vel.mean():.3f}, first 10: {global_vel[:10].mean():.3f}, last 10: {global_vel[-10:].mean():.3f}")
    print(f"  Local  ang — mean: {local_ang.mean():.4f}, first 10: {local_ang[:10].mean():.4f}, last 10: {local_ang[-10:].mean():.4f}")
    print(f"  Global ang — mean: {global_ang.mean():.4f}, first 10: {global_ang[:10].mean():.4f}, last 10: {global_ang[-10:].mean():.4f}")
    plot_split_velocity(local_vel, global_vel, local_ang, global_ang, scene_name, save_dir)

    # ── Analysis 3: Cosine similarity ramp ──────────────────────────
    print("\n── Analysis 3: Cosine Similarity Ramp ──")
    cos_sims = cosine_similarity_ramp(tokens, layer_idx=-1, offsets=(1, 5, 10))
    for k, sims in sorted(cos_sims.items()):
        print(f"  k={k}: mean={sims.mean():.4f}, first 10={sims[:10].mean():.4f}, last 10={sims[-10:].mean():.4f}")
    plot_cosine_ramp(cos_sims, scene_name, save_dir)

    # ── Analysis 4: Metric overlay ──────────────────────────────────
    print("\n── Analysis 4: Metric Overlay ──")
    plot_metric_overlay(local_vel, global_vel, metrics_per_view, scene_name, save_dir)

    # ── Analysis 5: Norm decomposition ─────────────────────────────
    print("\n── Analysis 5: Norm Decomposition ──")
    norms, cv = plot_norm_decomposition(tokens, -1, scene_name, save_dir)
    print(f"  Norm — mean: {norms.mean():.1f}, std: {norms.std():.1f}, CV: {cv:.3f}")
    if cv < 0.05:
        print("  → Norms are nearly constant — L2 velocity and cosine similarity are redundant")
    else:
        print("  → Norms vary meaningfully — L2 velocity carries information beyond cosine similarity")

    # ── Bonus: Per-layer velocity ───────────────────────────────────
    print("\n── Bonus: Per-Layer Velocity ──")
    plot_per_layer_velocity(tokens, scene_name, save_dir, data["layer_indices"])

    # ── Summary statistics ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SATURATION SUMMARY")
    print("=" * 60)

    # Compute velocity in thirds
    n = len(vel)
    third = n // 3
    v1, v2, v3 = vel[:third].mean(), vel[third:2*third].mean(), vel[2*third:].mean()
    print(f"Camera velocity by thirds: {v1:.3f} → {v2:.3f} → {v3:.3f}")

    if v3 < v1 * 0.5:
        print("  → Strong saturation signal (>50% velocity reduction)")
    elif v3 < v1 * 0.75:
        print("  → Moderate saturation signal (25-50% velocity reduction)")
    else:
        print("  → Weak/no saturation signal (<25% velocity reduction)")

    # Check if global half saturates more than local
    gl1 = global_vel[:third].mean()
    gl3 = global_vel[2*third:].mean()
    lo1 = local_vel[:third].mean()
    lo3 = local_vel[2*third:].mean()
    print(f"Global velocity reduction: {(1 - gl3/gl1)*100:.1f}%")
    print(f"Local velocity reduction:  {(1 - lo3/lo1)*100:.1f}%")

    if (1 - gl3/gl1) > (1 - lo3/lo1) + 0.1:
        print("  → Global half saturates faster than local (expected for scene understanding)")
    else:
        print("  → No clear differential saturation between halves")

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
