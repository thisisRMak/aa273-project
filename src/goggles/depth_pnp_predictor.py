"""
Depth-augmented PnP pose predictor — KLT feature tracking + DA3 depth + PnP.

Runs DA3 on consecutive frame pairs to get dense depth maps, then:
1. Detects KLT features in frame i-1
2. Tracks them to frame i via optical flow
3. Unprojecs tracked features using DA3 depth at frame i-1 → 3D points
4. Solves PnP (3D-2D) to get relative pose WITH metric scale
5. Chains relative poses into absolute trajectory

The key insight: PnP with 3D-2D correspondences gives metric-scale translation
(up to DA3's depth scale), unlike Essential matrix decomposition which loses scale.

Returns the same (pred_w2c, pred_intrinsics) contract as DA3PairwisePredictor.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DepthPnPPredictor:
    """Predict camera poses using DA3 depth + KLT feature tracking + PnP.

    Args:
        model_name: HuggingFace model ID for DA3 (default: DA3-LARGE-1.1).
        device: Torch device string.
        max_features: Maximum KLT features to detect per frame.
        quality_level: goodFeaturesToTrack quality threshold.
        min_distance: Minimum pixel distance between features.
        ransac_reproj_threshold: PnP RANSAC reprojection threshold in pixels.
        min_conf_percentile: Reject depth below this confidence percentile.
    """

    DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        max_features: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        ransac_reproj_threshold: float = 3.0,
        min_conf_percentile: float = 30.0,
    ):
        self.device = torch.device(device)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_conf_percentile = min_conf_percentile
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

        Reuses the same logic as DA3PairwisePredictor.
        """
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

        if not (0.1 < scale < 10.0):
            logger.warning("Unreasonable scale ratio %.4f, clamping to 1.0", scale)
            return 1.0

        return scale

    def _detect_and_track_features(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect KLT features in img_prev and track to img_curr.

        Args:
            img_prev: Grayscale image (H, W) uint8.
            img_curr: Grayscale image (H, W) uint8.

        Returns:
            pts_prev: (M, 2) matched feature points in frame i-1 (x, y).
            pts_curr: (M, 2) matched feature points in frame i (x, y).
        """
        # Detect features
        corners = cv2.goodFeaturesToTrack(
            img_prev,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if corners is None or len(corners) < 10:
            return np.empty((0, 2)), np.empty((0, 2))

        # Track via Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, corners, None, **lk_params,
        )

        # Filter by tracking status
        good = status.ravel() == 1
        pts_prev = corners[good].reshape(-1, 2)
        pts_next = pts_next[good].reshape(-1, 2)

        # Filter by forward-backward consistency
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            img_curr, img_prev, pts_next.reshape(-1, 1, 2), None, **lk_params,
        )
        if pts_back is not None:
            fb_good = status_back.ravel() == 1
            # Compute FB distance for all points (failed backward tracks get 999)
            fb_dist = np.full(len(pts_prev), 999.0)
            fb_dist[fb_good] = np.linalg.norm(
                pts_prev[fb_good] - pts_back[fb_good].reshape(-1, 2), axis=1
            )
            # Keep features with < 1 pixel forward-backward error
            mask = fb_dist < 1.0
            pts_prev = pts_prev[mask]
            pts_next = pts_next[mask]

        return pts_prev, pts_next

    def _solve_pnp(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        K: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solve PnP with RANSAC to get relative pose.

        Args:
            pts_3d: (M, 3) 3D points in frame i-1's camera coordinate system.
            pts_2d: (M, 2) corresponding 2D observations in frame i.
            K: (3, 3) camera intrinsic matrix for frame i.

        Returns:
            R: (3, 3) rotation matrix (frame i-1 to frame i).
            t: (3,) translation vector with metric scale.
            None if PnP fails.
        """
        if len(pts_3d) < 6:
            logger.warning("Too few points for PnP: %d", len(pts_3d))
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            K.astype(np.float64),
            distCoeffs=None,
            iterationsCount=1000,
            reprojectionError=self.ransac_reproj_threshold,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 6:
            logger.warning(
                "PnP failed: success=%s, inliers=%s",
                success, len(inliers) if inliers is not None else 0,
            )
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        logger.debug(
            "PnP: %d/%d inliers, |t|=%.4f",
            len(inliers), len(pts_3d), np.linalg.norm(t),
        )
        return R, t

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run depth-augmented PnP inference and return chained w2c extrinsics.

        Args:
            image_paths: List of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
            pred_intrinsics: [1, N, 3, 3] predicted intrinsics (float64).
        """
        n_frames = len(image_paths)
        logger.info(
            "Depth-PnP: %d frames -> %d pairs, %d features/frame",
            n_frames, n_frames - 1, self.max_features,
        )

        all_w2c = np.zeros((n_frames, 4, 4), dtype=np.float64)
        all_w2c[0] = np.eye(4)

        all_intr = np.zeros((n_frames, 3, 3), dtype=np.float64)

        # Track depth of the shared frame for scale correction
        prev_depth_of_shared = None
        prev_conf_of_shared = None

        # Stats
        n_pnp_success = 0
        n_pnp_fallback = 0

        for i in range(1, n_frames):
            pair_paths = [image_paths[i - 1], image_paths[i]]

            # Run DA3 on the pair to get depth maps + intrinsics
            pred = self.model.inference(pair_paths)
            torch.cuda.empty_cache()

            # Extract DA3 outputs
            depth_prev = np.squeeze(pred.depth[0])  # (H, W)
            depth_curr = np.squeeze(pred.depth[1])  # (H, W)
            conf_prev = pred.conf[0]  # (H, W)
            conf_curr = pred.conf[1]  # (H, W)
            K_prev = pred.intrinsics[0].astype(np.float64)  # (3, 3)
            K_curr = pred.intrinsics[1].astype(np.float64)  # (3, 3)

            # DA3's extrinsics as fallback
            ext = pred.extrinsics.astype(np.float64)
            T_rel_da3 = np.eye(4, dtype=np.float64)
            if ext.shape[-2:] == (4, 4):
                T_rel_da3 = ext[1].copy()
            else:
                T_rel_da3[:3, :4] = ext[1, :3, :4]

            # Depth-based scale correction between pairs
            depth_shared_new = depth_prev  # depth of frame i-1 in this pair
            conf_shared_new = conf_prev

            scale_ratio = 1.0
            if prev_depth_of_shared is not None:
                scale_ratio = self._compute_depth_scale_ratio(
                    prev_depth_of_shared, prev_conf_of_shared,
                    depth_shared_new, conf_shared_new,
                )

            # Load images for feature tracking
            img_prev_bgr = cv2.imread(image_paths[i - 1])
            img_curr_bgr = cv2.imread(image_paths[i])

            # DA3 may process at different resolution than original image.
            # We need to work in DA3's coordinate system (depth map resolution).
            h_depth, w_depth = depth_prev.shape
            h_img, w_img = img_prev_bgr.shape[:2]

            # Resize images to DA3's processing resolution for consistent coordinates
            if (h_img, w_img) != (h_depth, w_depth):
                img_prev_resized = cv2.resize(img_prev_bgr, (w_depth, h_depth))
                img_curr_resized = cv2.resize(img_curr_bgr, (w_depth, h_depth))
            else:
                img_prev_resized = img_prev_bgr
                img_curr_resized = img_curr_bgr

            gray_prev = cv2.cvtColor(img_prev_resized, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(img_curr_resized, cv2.COLOR_BGR2GRAY)

            # Detect and track features
            pts_prev, pts_curr = self._detect_and_track_features(gray_prev, gray_curr)

            pnp_success = False
            T_rel = T_rel_da3.copy()  # default fallback

            if len(pts_prev) >= 10:
                # Unproject pts_prev to 3D using DA3 depth
                px = pts_prev[:, 0].astype(int).clip(0, w_depth - 1)
                py = pts_prev[:, 1].astype(int).clip(0, h_depth - 1)

                depths = depth_prev[py, px]
                confs = conf_prev[py, px]

                # Filter by confidence
                conf_threshold = np.percentile(conf_prev[conf_prev > 0], self.min_conf_percentile)
                valid = (depths > 1e-3) & (confs > conf_threshold)

                pts_prev_valid = pts_prev[valid]
                pts_curr_valid = pts_curr[valid]
                depths_valid = depths[valid]

                if len(pts_prev_valid) >= 10:
                    # Unproject to 3D in camera frame: p = depth * K^-1 @ [u, v, 1]
                    K_inv = np.linalg.inv(K_prev)
                    ones = np.ones(len(pts_prev_valid))
                    pixels_h = np.stack([pts_prev_valid[:, 0], pts_prev_valid[:, 1], ones], axis=1)  # (M, 3)
                    rays = (K_inv @ pixels_h.T).T  # (M, 3)
                    pts_3d = rays * depths_valid[:, None]  # (M, 3)

                    # Solve PnP
                    result = self._solve_pnp(pts_3d, pts_curr_valid, K_curr)
                    if result is not None:
                        R_pnp, t_pnp = result
                        T_rel = np.eye(4, dtype=np.float64)
                        T_rel[:3, :3] = R_pnp
                        T_rel[:3, 3] = t_pnp
                        pnp_success = True
                        n_pnp_success += 1

            if not pnp_success:
                n_pnp_fallback += 1
                logger.debug(
                    "Pair %d/%d: PnP failed (%d features), using DA3 pose",
                    i, n_frames - 1, len(pts_prev),
                )

            # Apply depth scale correction to translation
            T_rel[:3, 3] *= scale_ratio

            # Chain: T_abs_i = T_rel @ T_abs_{i-1}
            all_w2c[i] = T_rel @ all_w2c[i - 1]

            # Store intrinsics
            all_intr[i] = K_curr
            if i == 1:
                all_intr[0] = K_prev

            # Save depth of frame i for next pair's scale correction
            prev_depth_of_shared = depth_curr
            prev_conf_of_shared = conf_curr

            if i <= 5 or i % 20 == 0:
                src = "PnP" if pnp_success else "DA3"
                logger.info(
                    "  Pair %d/%d: %s, scale=%.4f, |t|=%.4f, %d features",
                    i, n_frames - 1, src, scale_ratio,
                    np.linalg.norm(T_rel[:3, 3]), len(pts_prev),
                )

        logger.info(
            "Depth-PnP: %d poses, PnP success: %d/%d (fallback: %d)",
            n_frames, n_pnp_success, n_frames - 1, n_pnp_fallback,
        )

        pred_w2c = torch.from_numpy(all_w2c).to(self.device)
        pred_intrinsics = torch.from_numpy(all_intr).to(self.device).unsqueeze(0)
        return pred_w2c, pred_intrinsics
