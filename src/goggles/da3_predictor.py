"""
DA3 (Depth Anything 3) pose prediction wrapper.

Provides DA3-based camera pose estimation returning the same [N, 4, 4] w2c
format as the StreamVGGT path in eval_poses.py.

DA3 outputs extrinsics as [N, 4, 4] w2c (world-to-camera) in OpenCV convention
when no input extrinsics are provided (blind pose estimation). This matches
our eval pipeline directly — no axis flipping or inversion needed.

See: Depth-Anything-3/src/depth_anything_3/utils/io/output_processor.py
     Depth-Anything-3/src/depth_anything_3/api.py (_normalize_extrinsics)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DA3PosePredictor:
    """Predict camera poses using Depth Anything 3.

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

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run DA3 inference and return w2c extrinsics.

        Args:
            image_paths: List of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64) on self.device.
            pred_intrinsics: [1, N, 3, 3] predicted intrinsics (float64)
                on self.device, or None. Wrapped in batch dim for
                compatibility with compute_intrinsic_errors().
        """
        logger.info("Running DA3 inference on %d images...", len(image_paths))

        prediction = self.model.inference(image_paths)

        if prediction.extrinsics is None:
            raise RuntimeError("DA3 returned None extrinsics. Check input images.")

        # prediction.extrinsics: [N, 4, 4] numpy float32, w2c convention
        # (OutputProcessor extracts [B=1, N, 4, 4] -> [N, 4, 4])
        pred_w2c = torch.from_numpy(
            prediction.extrinsics.astype(np.float64)
        ).to(self.device)

        logger.info(
            "DA3 predicted %d poses, extrinsics shape: %s",
            pred_w2c.shape[0], list(pred_w2c.shape),
        )

        # Intrinsics: [N, 3, 3] -> [1, N, 3, 3] for compute_intrinsic_errors()
        pred_intrinsics = None
        if prediction.intrinsics is not None:
            intr = torch.from_numpy(
                prediction.intrinsics.astype(np.float64)
            ).to(self.device)
            pred_intrinsics = intr.unsqueeze(0)  # [1, N, 3, 3]

        return pred_w2c, pred_intrinsics
