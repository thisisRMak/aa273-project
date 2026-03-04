"""
DA3 Pairwise pose predictor — frame-pair chaining with depth scale correction.

Runs DA3 on consecutive frame pairs, extracts relative poses, and chains them
into an absolute trajectory. Depth-based scale correction maintains metric
consistency between pairs. Follows the MASt3R-SLAM-style pairwise approach.

Returns the same (pred_w2c, pred_intrinsics) contract as DA3PosePredictor.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DA3PairwisePredictor:
    """Predict camera poses using DA3 with pairwise pose chaining.

    For each consecutive pair of frames, DA3 returns a relative pose.
    These relative poses are chained into an absolute trajectory with
    depth-based scale correction to reduce drift.

    Args:
        model_name: HuggingFace model ID (default: DA3-LARGE-1.1).
        device: Torch device string.
    """

    DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = self._load_model()

    def _load_model(self):
        from depth_anything_3.api import DepthAnything3

        logger.info("Loading DA3 model: %s", self.model_name)
        model = DepthAnything3.from_pretrained(self.model_name)
        model = model.to(device=self.device)
        logger.info("DA3 model loaded on %s", self.device)
        return model

    @staticmethod
    def _compute_depth_scale_ratio(
        depth_old: np.ndarray,
        conf_old: np.ndarray,
        depth_new: np.ndarray,
        conf_new: np.ndarray,
    ) -> float:
        """Compute scale ratio between two depth predictions of the same frame.

        Uses high-confidence pixels to estimate median(depth_old / depth_new).

        Args:
            depth_old: (H, W) depth of shared frame from previous pair.
            conf_old: (H, W) confidence from previous pair.
            depth_new: (H, W) depth of shared frame from current pair.
            conf_new: (H, W) confidence from current pair.

        Returns:
            Scale ratio to multiply current pair's translation by.
        """
        # Use pixels confident in both predictions
        conf_thresh = min(np.median(conf_old), np.median(conf_new)) * 0.3
        mask = (
            (conf_old > conf_thresh)
            & (conf_new > conf_thresh)
            & (depth_old > 1e-3)
            & (depth_new > 1e-3)
        )

        n_valid = np.sum(mask)
        if n_valid < 100:
            logger.warning(
                "Only %d valid pixels for scale correction, using 1.0", n_valid,
            )
            return 1.0

        ratios = depth_old[mask] / depth_new[mask]
        scale = float(np.median(ratios))

        # Reject unreasonable scales
        if not (0.1 < scale < 10.0):
            logger.warning("Unreasonable scale ratio %.4f, clamping to 1.0", scale)
            return 1.0

        return scale

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run DA3 pairwise inference and return chained w2c extrinsics.

        Args:
            image_paths: List of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
            pred_intrinsics: [1, N, 3, 3] predicted intrinsics (float64).
        """
        n_frames = len(image_paths)
        logger.info("DA3 pairwise: %d frames → %d pairs", n_frames, n_frames - 1)

        all_w2c = np.zeros((n_frames, 4, 4), dtype=np.float64)
        all_w2c[0] = np.eye(4)  # First frame at identity

        all_intr = np.zeros((n_frames, 3, 3), dtype=np.float64)

        # Track depth of the shared frame for scale correction
        prev_depth_of_shared = None
        prev_conf_of_shared = None

        for i in range(1, n_frames):
            pair_paths = [image_paths[i - 1], image_paths[i]]
            pred = self.model.inference(pair_paths)
            torch.cuda.empty_cache()

            # DA3 normalizes first frame to identity
            # pred.extrinsics[0] ≈ identity, pred.extrinsics[1] = relative w2c
            ext = pred.extrinsics.astype(np.float64)

            # Build 4x4 relative pose
            T_rel = np.eye(4, dtype=np.float64)
            if ext.shape[-2:] == (4, 4):
                T_rel = ext[1]
            else:
                T_rel[:3, :4] = ext[1, :3, :4]

            # Depth-based scale correction
            depth_shared_new = np.squeeze(pred.depth[0])  # depth of frame i-1 in this pair
            conf_shared_new = pred.conf[0]

            if prev_depth_of_shared is not None:
                scale_ratio = self._compute_depth_scale_ratio(
                    prev_depth_of_shared, prev_conf_of_shared,
                    depth_shared_new, conf_shared_new,
                )
                T_rel[:3, 3] *= scale_ratio
                if i <= 5 or i % 20 == 0:
                    logger.info(
                        "  Pair %d/%d: scale_ratio=%.4f",
                        i, n_frames - 1, scale_ratio,
                    )

            # Chain: T_abs_i = T_rel @ T_abs_{i-1}
            all_w2c[i] = T_rel @ all_w2c[i - 1]

            # Store intrinsics from the current frame (second in pair)
            all_intr[i] = pred.intrinsics[1].astype(np.float64)
            if i == 1:
                all_intr[0] = pred.intrinsics[0].astype(np.float64)

            # Save depth of frame i for next pair's scale correction
            prev_depth_of_shared = np.squeeze(pred.depth[1])
            prev_conf_of_shared = pred.conf[1]

        logger.info("DA3 pairwise: assembled %d poses", n_frames)

        pred_w2c = torch.from_numpy(all_w2c).to(self.device)
        pred_intrinsics = torch.from_numpy(all_intr).to(self.device).unsqueeze(0)
        return pred_w2c, pred_intrinsics
