"""Visualization utilities for camera pose evaluation.

Provides SE(3) Procrustes alignment, point cloud loading, and SINGER-style
dark-theme isometric 3D trajectory plotting.
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE(3) alignment (Procrustes, no scale)
# ---------------------------------------------------------------------------

def align_poses_procrustes(pred_w2c, gt_w2c):
    """Rigid Procrustes alignment of predicted poses to GT world frame.

    Finds R, t minimizing  sum_i || gt_pos_i - (R @ pred_pos_i + t) ||^2
    with no scale (StreamVGGT is metric).

    Args:
        pred_w2c: [N, 4, 4] predicted w2c (float64).
        gt_w2c: [N, 4, 4] ground-truth w2c (float64).

    Returns:
        aligned_c2w: [N, 4, 4] predicted c2w in GT world frame.
        T_align: [4, 4] alignment transform (pred world -> GT world).
        gt_c2w: [N, 4, 4] GT c2w for reference.
    """
    from vggt.utils.geometry import closed_form_inverse_se3

    pred_c2w = closed_form_inverse_se3(pred_w2c).cpu().double()
    gt_c2w = closed_form_inverse_se3(gt_w2c.cpu()).double()

    pred_pos = pred_c2w[:, :3, 3].numpy()
    gt_pos = gt_c2w[:, :3, 3].numpy()

    # Centroids
    pred_mean = pred_pos.mean(axis=0)
    gt_mean = gt_pos.mean(axis=0)

    # SVD for optimal rotation
    H = (pred_pos - pred_mean).T @ (gt_pos - gt_mean)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # ensure proper rotation
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = gt_mean - R @ pred_mean

    # Build 4x4 alignment transform
    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R
    T_align[:3, 3] = t
    T_align = torch.from_numpy(T_align)

    aligned_c2w = T_align.unsqueeze(0) @ pred_c2w

    # Alignment quality
    aligned_pos = aligned_c2w[:, :3, 3].numpy()
    residuals = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    logger.info(
        "Procrustes alignment: mean residual=%.4fm, max=%.4fm",
        residuals.mean(), residuals.max(),
    )

    return aligned_c2w, T_align, gt_c2w


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def discover_sparse_pc(transforms_path):
    """Find a point cloud near the transforms.json location.

    Prefers sparse_pc_ned.ply (NED dynamics frame, saved by
    figs_simulate_flight_CLI.py) over raw sparse_pc.ply (COLMAP frame).
    """
    base = os.path.dirname(os.path.abspath(transforms_path))
    for candidate in [
        os.path.join(base, "sparse_pc_ned.ply"),  # NED frame (from simulate CLI)
        os.path.join(base, "sparse_pc.ply"),
        os.path.join(base, "sfm", "sparse_pc.ply"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def load_sparse_pointcloud(ply_path, subsample=3):
    """Load PLY point cloud.

    Args:
        ply_path: Path to .ply file.
        subsample: Take every Nth point (default 3).

    Returns:
        pts: (3, M) positions.
        colors: (M, 3) RGB in [0, 1], or None.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).T  # (3, M)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None  # (M, 3)
    if colors is not None and colors.size == 0:
        colors = None
    if subsample > 1:
        pts = pts[:, ::subsample]
        if colors is not None:
            colors = colors[::subsample]
    logger.info(
        "Loaded point cloud: %d pts (subsampled %dx from %s)",
        pts.shape[1], subsample, ply_path,
    )
    return pts, colors


# ---------------------------------------------------------------------------
# Trajectory visualization (SINGER-style isometric 3D)
# ---------------------------------------------------------------------------

def plot_trajectory_on_pointcloud(
    gt_pos, pred_pos, pcd_pts, pcd_colors=None,
    title="", output_path=None, z_band=2.0,
):
    """Plot GT and aligned predicted camera trajectories on a point cloud.

    Uses the SINGER dark-theme isometric style.

    Args:
        gt_pos: (N, 3) GT camera positions in COLMAP world frame.
        pred_pos: (N, 3) aligned predicted camera positions.
        pcd_pts: (3, M) point cloud XYZ.
        pcd_colors: (M, 3) RGB in [0,1] or None.
        title: plot title string.
        output_path: save path (PNG). If None, calls plt.show().
        z_band: half-width for z-band filtering around camera altitude.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Dark theme
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    # NED convention: Z-down, so negate Z for display (Z-up)
    gt_pos = gt_pos.copy()
    pred_pos = pred_pos.copy()
    gt_pos[:, 2] *= -1
    pred_pos[:, 2] *= -1
    pcd_pts = pcd_pts.copy()
    pcd_pts[2, :] *= -1

    # Z-band filter around trajectory altitude
    all_cam_z = np.concatenate([gt_pos[:, 2], pred_pos[:, 2]])
    altitude = float(np.median(all_cam_z))
    if pcd_pts.shape[1] > 0:
        z_mask = np.abs(pcd_pts[2, :] - altitude) <= z_band
        vis_pts = pcd_pts[:, z_mask]
        vis_colors = pcd_colors[z_mask] if pcd_colors is not None else None
        logger.info(
            "Z-band: altitude=%.2f +/-%.1fm -> %d / %d pts",
            altitude, z_band, vis_pts.shape[1], pcd_pts.shape[1],
        )
    else:
        vis_pts = pcd_pts
        vis_colors = pcd_colors

    # Point cloud scatter
    if vis_pts.shape[1] > 0:
        kw = dict(s=0.3, alpha=0.3, rasterized=True, depthshade=False)
        if vis_colors is not None:
            ax.scatter(vis_pts[0], vis_pts[1], vis_pts[2], c=vis_colors, **kw)
        else:
            ax.scatter(vis_pts[0], vis_pts[1], vis_pts[2], c="#888888", **kw)

    # GT trajectory (green)
    ax.plot(
        gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
        color="#44ff44", linewidth=1.5, alpha=0.9, label="GT",
    )
    ax.scatter(
        *gt_pos[0], color="#44ff44", s=60, marker="o",
        edgecolors="white", linewidths=0.5, zorder=5,
    )
    ax.scatter(
        *gt_pos[-1], color="#44ff44", s=80, marker="x",
        linewidths=2, zorder=5,
    )

    # Predicted trajectory (red, aligned)
    ax.plot(
        pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
        color="#ff4444", linewidth=1.5, alpha=0.9, label="Predicted (aligned)",
    )
    ax.scatter(
        *pred_pos[0], color="#ff4444", s=60, marker="o",
        edgecolors="white", linewidths=0.5, zorder=5,
    )
    ax.scatter(
        *pred_pos[-1], color="#ff4444", s=80, marker="x",
        linewidths=2, zorder=5,
    )

    # Correspondence lines between matched frames
    for i in range(len(gt_pos)):
        ax.plot(
            [gt_pos[i, 0], pred_pos[i, 0]],
            [gt_pos[i, 1], pred_pos[i, 1]],
            [gt_pos[i, 2], pred_pos[i, 2]],
            color="white", alpha=0.15, linewidth=0.5,
        )

    # Axis labels and ticks
    ax.set_xlabel("X (m)", color="white", fontsize=10)
    ax.set_ylabel("Y (m)", color="white", fontsize=10)
    ax.set_zlabel("Z (m)", color="white", fontsize=10)
    ax.tick_params(colors="white")

    # Legend
    ax.legend(
        loc="upper right", fontsize=9,
        facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white",
    )

    # Title
    if title:
        ax.set_title(title, color="white", fontweight="bold", fontsize=12)

    # Isometric equal-axis scaling
    all_pts = np.concatenate([gt_pos, pred_pos], axis=0)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    max_range = max((maxs - mins).max() * 1.2, 0.1)
    centers = (mins + maxs) / 2
    for setter, c in [
        (ax.set_xlim, centers[0]),
        (ax.set_ylim, centers[1]),
        (ax.set_zlim, centers[2]),
    ]:
        setter(c - max_range / 2, c + max_range / 2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)

    # Pane / grid styling
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.1, 0.1, 0.1, 0.3))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.05)
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis.line.set_color("#555555")

    fig.tight_layout()
    if output_path:
        fig.savefig(
            output_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        logger.info("Saved trajectory visualization to %s", output_path)
    else:
        plt.show()
    plt.close(fig)
