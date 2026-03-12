"""
Relative pose error evaluation for camera pose benchmarking.

Adapted from StreamVGGT/src/eval/pose_evaluation/test_co3d.py.
All functions operate on world-to-camera (w2c) 4x4 SE(3) tensors.
"""

import torch
import numpy as np
from goggles.geometry import mat_to_quat, closed_form_inverse_se3


def build_pair_index(N, B=1):
    """Build indices for all unique pairs of N frames.

    Returns (i1, i2) index tensors, each of length N*(N-1)/2.
    """
    i1_, i2_ = torch.combinations(
        torch.arange(N), 2, with_replacement=False
    ).unbind(-1)
    i1, i2 = [
        (i[None] + torch.arange(B)[:, None] * N).reshape(-1)
        for i in [i1_, i2_]
    ]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    """Quaternion-based rotation error in degrees.

    Args:
        rot_gt: [N, 3, 3] ground truth rotation matrices.
        rot_pred: [N, 3, 3] predicted rotation matrices.

    Returns:
        [N] rotation angle error in degrees.
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, eps=1e-15):
    """Translation direction error in degrees (scale-invariant).

    Handles sign ambiguity by taking min(angle, 180 - angle).

    Args:
        tvec_gt: [N, 3] ground truth translations.
        tvec_pred: [N, 3] predicted translations.

    Returns:
        [N] translation angle error in degrees.
    """
    t = tvec_pred / (torch.norm(tvec_pred, dim=1, keepdim=True) + eps)
    t_gt = tvec_gt / (torch.norm(tvec_gt, dim=1, keepdim=True) + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6

    deg = err_t * 180.0 / np.pi
    return torch.min(deg, (180 - deg).abs())


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """Compute all-pairs relative rotation and translation errors.

    Args:
        pred_se3: [N, 4, 4] predicted w2c SE(3) matrices.
        gt_se3: [N, 4, 4] ground truth w2c SE(3) matrices.
        num_frames: N.

    Returns:
        (rotation_errors, translation_errors) each [N*(N-1)/2], in degrees.
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """AUC of the max(R_err, T_err) cumulative histogram.

    Args:
        r_error: [N] rotation errors in degrees (numpy).
        t_error: [N] translation errors in degrees (numpy).
        max_threshold: bin up to this value in degrees.

    Returns:
        (auc_value, normalized_histogram).
    """
    error_matrix = np.column_stack((r_error, t_error))
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized_histogram))), normalized_histogram


def compute_pose_metrics(r_error, t_error):
    """Compute a full set of relative pose evaluation metrics.

    Args:
        r_error: [N] rotation errors in degrees (numpy).
        t_error: [N] translation errors in degrees (numpy).

    Returns:
        dict with AUC@3/5/15/30, mean/median R/T errors, accuracy@thresholds.
    """
    auc_3, _ = calculate_auc_np(r_error, t_error, max_threshold=3)
    auc_5, _ = calculate_auc_np(r_error, t_error, max_threshold=5)
    auc_15, _ = calculate_auc_np(r_error, t_error, max_threshold=15)
    auc_30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)

    return {
        "auc_at_3": auc_3,
        "auc_at_5": auc_5,
        "auc_at_15": auc_15,
        "auc_at_30": auc_30,
        "rotation_error_mean_deg": float(np.mean(r_error)),
        "rotation_error_median_deg": float(np.median(r_error)),
        "translation_error_mean_deg": float(np.mean(t_error)),
        "translation_error_median_deg": float(np.median(t_error)),
        "r_acc_at_5": float(np.mean(r_error < 5)),
        "r_acc_at_15": float(np.mean(r_error < 15)),
        "t_acc_at_5": float(np.mean(t_error < 5)),
        "t_acc_at_15": float(np.mean(t_error < 15)),
        "num_pairs": len(r_error),
    }
