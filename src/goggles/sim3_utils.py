"""
Lightweight SIM(3) alignment utilities for chunk-based pose estimation.

Extracted from Depth-Anything-3/da3_streaming/ (VGGT-Long).
Pure numpy + torch — no numba, sklearn, or trimesh dependencies.
"""

import numpy as np
import torch


def estimate_sim3(source_points: np.ndarray, target_points: np.ndarray):
    """Estimate SIM(3) transform from source to target using Umeyama algorithm.

    Args:
        source_points: (N, 3) points in source frame.
        target_points: (N, 3) points in target frame.

    Returns:
        s: Scale factor.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        Such that target ≈ s * R @ source + t.
    """
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered**2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered**2).sum(axis=1).mean())
    s = scale_tgt / scale_src

    src_scaled = src_centered * s

    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def accumulate_sim3_transforms(transforms):
    """Accumulate adjacent SIM(3) transforms into cumulative transforms.

    Each transform maps from chunk k+1's coordinate frame to chunk k's frame.
    The output maps from each chunk's frame to chunk 0's frame.

    Args:
        transforms: List of (s, R, t) tuples — adjacent SIM(3) transforms.

    Returns:
        List of cumulative (s, R, t) tuples.
    """
    if not transforms:
        return []

    cumulative = [transforms[0]]

    for i in range(1, len(transforms)):
        s_prev, R_prev, t_prev = cumulative[i - 1]
        s_next, R_next, t_next = transforms[i]

        R_cum = R_prev @ R_next
        s_cum = s_prev * s_next
        t_cum = s_prev * (R_prev @ t_next) + t_prev

        cumulative.append((s_cum, R_cum, t_cum))

    return cumulative


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject depth maps to world-space 3D point clouds.

    Args:
        depth: (N, H, W) depth maps.
        intrinsics: (N, 3, 3) camera intrinsics.
        extrinsics: (N, 3, 4) or (N, 4, 4) w2c extrinsics.

    Returns:
        (N, H, W, 3) world-space point cloud as numpy array.
    """
    depth_t = torch.from_numpy(depth).float()
    intr_t = torch.from_numpy(intrinsics).float()

    # Ensure extrinsics are 4x4
    ext = torch.from_numpy(extrinsics).float()
    if ext.shape[-2:] == (3, 4):
        N = ext.shape[0]
        ext_4x4 = torch.zeros(N, 4, 4)
        ext_4x4[:, :3, :4] = ext
        ext_4x4[:, 3, 3] = 1.0
    else:
        ext_4x4 = ext

    N, H, W = depth_t.shape

    u = torch.arange(W).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones(N, H, W, 1)
    pixel_coords = torch.cat([u, v, ones], dim=-1)  # (N, H, W, 3)

    intr_inv = torch.inverse(intr_t)  # (N, 3, 3)
    cam_coords = torch.einsum("nij,nhwj->nhwi", intr_inv, pixel_coords)
    cam_coords = cam_coords * depth_t.unsqueeze(-1)

    cam_homo = torch.cat([cam_coords, ones], dim=-1)  # (N, H, W, 4)

    c2w = torch.inverse(ext_4x4)  # (N, 4, 4)
    world_homo = torch.einsum("nij,nhwj->nhwi", c2w, cam_homo)

    return world_homo[..., :3].numpy()
