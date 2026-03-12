"""
Minimal SE(3) and quaternion utilities — no external dependencies beyond
torch and numpy. Replaces vggt.utils.geometry and vggt.utils.rotation
imports so GOGGLES can run without the StreamVGGT/vggt package installed.
"""

import torch
import numpy as np


def closed_form_inverse_se3(se3):
    """Batch-invert SE(3) matrices via R^T, -R^T @ t.

    Args:
        se3: (..., 4, 4) or (..., 3, 4) tensor or ndarray.

    Returns:
        Inverted SE(3) matrices, same type/device as input.
    """
    is_numpy = isinstance(se3, np.ndarray)

    R = se3[..., :3, :3]
    t = se3[..., :3, 3:]

    if is_numpy:
        R_T = np.swapaxes(R, -2, -1)
        top_right = -np.matmul(R_T, t)
        inv = np.broadcast_to(np.eye(4), se3.shape[:-2] + (4, 4)).copy()
    else:
        R_T = R.transpose(-2, -1)
        top_right = -torch.matmul(R_T, t)
        inv = torch.eye(4, dtype=se3.dtype, device=se3.device).expand(
            se3.shape[:-2] + (4, 4)
        ).clone()

    inv[..., :3, :3] = R_T
    inv[..., :3, 3:] = top_right
    return inv


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (scalar-last: xyzw).

    Numerically stable Shepperd method — picks the quaternion component
    with the largest magnitude to avoid division by near-zero.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in [x, y, z, w] order.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {matrix.shape}")

    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    N = m.shape[0]

    m00 = m[:, 0, 0]; m01 = m[:, 0, 1]; m02 = m[:, 0, 2]
    m10 = m[:, 1, 0]; m11 = m[:, 1, 1]; m12 = m[:, 1, 2]
    m20 = m[:, 2, 0]; m21 = m[:, 2, 1]; m22 = m[:, 2, 2]

    # Four candidate traces
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ], dim=-1), min=0)) / 2.0  # (N, 4) = [w, x, y, z] magnitudes

    # For each candidate, compute full quaternion in wxyz order
    # Then pick the one with largest magnitude
    quat_wxyz = torch.zeros(N, 4, dtype=matrix.dtype, device=matrix.device)

    idx = q_abs.argmax(dim=-1)  # (N,)

    # Case 0: w is largest
    mask = idx == 0
    if mask.any():
        w = q_abs[mask, 0]
        denom = 4.0 * w
        quat_wxyz[mask, 0] = w
        quat_wxyz[mask, 1] = (m21[mask] - m12[mask]) / denom
        quat_wxyz[mask, 2] = (m02[mask] - m20[mask]) / denom
        quat_wxyz[mask, 3] = (m10[mask] - m01[mask]) / denom

    # Case 1: x is largest
    mask = idx == 1
    if mask.any():
        x = q_abs[mask, 1]
        denom = 4.0 * x
        quat_wxyz[mask, 0] = (m21[mask] - m12[mask]) / denom
        quat_wxyz[mask, 1] = x
        quat_wxyz[mask, 2] = (m01[mask] + m10[mask]) / denom
        quat_wxyz[mask, 3] = (m02[mask] + m20[mask]) / denom

    # Case 2: y is largest
    mask = idx == 2
    if mask.any():
        y = q_abs[mask, 2]
        denom = 4.0 * y
        quat_wxyz[mask, 0] = (m02[mask] - m20[mask]) / denom
        quat_wxyz[mask, 1] = (m01[mask] + m10[mask]) / denom
        quat_wxyz[mask, 2] = y
        quat_wxyz[mask, 3] = (m12[mask] + m21[mask]) / denom

    # Case 3: z is largest
    mask = idx == 3
    if mask.any():
        z = q_abs[mask, 3]
        denom = 4.0 * z
        quat_wxyz[mask, 0] = (m10[mask] - m01[mask]) / denom
        quat_wxyz[mask, 1] = (m02[mask] + m20[mask]) / denom
        quat_wxyz[mask, 2] = (m12[mask] + m21[mask]) / denom
        quat_wxyz[mask, 3] = z

    # Ensure w >= 0 (canonical form)
    quat_wxyz[quat_wxyz[:, 0] < 0] *= -1

    # Convert wxyz → xyzw (scalar-last)
    out = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim=-1)
    return out.reshape(batch_shape + (4,))
