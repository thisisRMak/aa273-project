"""
DA3Metric + PnP pose predictor — metric depth + KLT feature tracking + PnP.

Uses DA3Metric-Large for single-frame metric depth estimation, then:
1. Detects KLT features in frame i-1
2. Tracks them to frame i via optical flow
3. Unprojecs tracked features using metric depth at frame i-1 → 3D points in meters
4. Solves PnP (3D-2D) to get relative pose with metric-scale translation
5. Chains relative poses into absolute trajectory

DA3Metric-Large outputs depth that converts to meters via:
    metric_depth = raw_output * (focal_length / 300)

Requires known camera intrinsics (from transforms.json).

Returns the same (pred_w2c, pred_intrinsics) contract as DA3PairwisePredictor.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DA3MetricPairwisePredictor:
    """Predict camera poses using DA3Metric depth + KLT + PnP.

    Args:
        intrinsics: Dict with 'fl_x', 'fl_y', 'cx', 'cy', 'w', 'h' from transforms.json.
        device: Torch device string.
        max_features: Maximum KLT features to detect per frame.
        quality_level: goodFeaturesToTrack quality threshold.
        min_distance: Minimum pixel distance between features.
        ransac_reproj_threshold: PnP RANSAC reprojection threshold in pixels.
    """

    DEFAULT_MODEL = "depth-anything/DA3METRIC-LARGE"

    def __init__(
        self,
        intrinsics: Dict[str, float],
        device: str = "cuda",
        max_features: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        ransac_reproj_threshold: float = 3.0,
    ):
        self.device = torch.device(device)
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_reproj_threshold = ransac_reproj_threshold

        # Store GT intrinsics (original image resolution)
        self.gt_fl_x = intrinsics["fl_x"]
        self.gt_fl_y = intrinsics["fl_y"]
        self.gt_cx = intrinsics["cx"]
        self.gt_cy = intrinsics["cy"]
        self.gt_w = intrinsics["w"]
        self.gt_h = intrinsics["h"]

        self.model = self._load_model()

    def _load_model(self):
        from depth_anything_3.api import DepthAnything3

        logger.info("Loading DA3Metric-Large...")
        model = DepthAnything3.from_pretrained(self.DEFAULT_MODEL)
        model = model.to(device=self.device)
        logger.info("DA3Metric-Large loaded on %s", self.device)
        return model

    def _build_intrinsics_at_resolution(self, h: int, w: int) -> np.ndarray:
        """Build 3x3 intrinsics matrix scaled to the given resolution."""
        sx = w / self.gt_w
        sy = h / self.gt_h
        K = np.array([
            [self.gt_fl_x * sx, 0.0, self.gt_cx * sx],
            [0.0, self.gt_fl_y * sy, self.gt_cy * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    def _detect_and_track_features(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect KLT features in img_prev and track to img_curr.

        Returns:
            pts_prev: (M, 2) matched feature points in frame i-1.
            pts_curr: (M, 2) matched feature points in frame i.
        """
        corners = cv2.goodFeaturesToTrack(
            img_prev,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if corners is None or len(corners) < 10:
            return np.empty((0, 2)), np.empty((0, 2))

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, corners, None, **lk_params,
        )

        good = status.ravel() == 1
        pts_prev = corners[good].reshape(-1, 2)
        pts_next = pts_next[good].reshape(-1, 2)

        # Forward-backward consistency check
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            img_curr, img_prev, pts_next.reshape(-1, 1, 2), None, **lk_params,
        )
        if pts_back is not None:
            fb_good = status_back.ravel() == 1
            fb_dist = np.full(len(pts_prev), 999.0)
            fb_dist[fb_good] = np.linalg.norm(
                pts_prev[fb_good] - pts_back[fb_good].reshape(-1, 2), axis=1
            )
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
        """Solve PnP with RANSAC to get relative pose."""
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
        return R, t

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run DA3Metric + PnP inference and return chained w2c extrinsics.

        Args:
            image_paths: List of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
            pred_intrinsics: [1, N, 3, 3] predicted intrinsics (float64).
        """
        n_frames = len(image_paths)
        logger.info(
            "DA3Metric-PnP: %d frames, %d features/frame",
            n_frames, self.max_features,
        )

        all_w2c = np.zeros((n_frames, 4, 4), dtype=np.float64)
        all_w2c[0] = np.eye(4)

        # Precompute: run DA3Metric on all frames to get metric depth maps
        # This avoids redundant inference (each frame appears in two pairs)
        logger.info("Computing metric depth for all %d frames...", n_frames)
        all_metric_depth = []
        depth_resolution = None

        for idx in range(n_frames):
            pred = self.model.inference([image_paths[idx]])
            torch.cuda.empty_cache()
            raw_depth = np.squeeze(pred.depth)  # (H, W)
            all_metric_depth.append(raw_depth)
            if depth_resolution is None:
                depth_resolution = raw_depth.shape  # (H, W)

        h_depth, w_depth = depth_resolution
        logger.info("Depth resolution: %dx%d", w_depth, h_depth)

        # Build intrinsics at depth resolution
        K = self._build_intrinsics_at_resolution(h_depth, w_depth)
        K_inv = np.linalg.inv(K)
        focal_at_depth = (K[0, 0] + K[1, 1]) / 2

        # Apply metric scaling: metric_depth = raw * (focal / 300)
        metric_scale = focal_at_depth / 300.0
        logger.info(
            "Metric scaling: focal=%.1f, scale_factor=%.4f",
            focal_at_depth, metric_scale,
        )
        for idx in range(n_frames):
            all_metric_depth[idx] = all_metric_depth[idx] * metric_scale

        # Fill intrinsics array (same for all frames — GT intrinsics)
        all_intr = np.tile(K, (n_frames, 1, 1))  # (N, 3, 3)

        # Stats
        n_pnp_success = 0
        n_pnp_fail = 0
        translation_norms = []

        logger.info("Running PnP on %d pairs...", n_frames - 1)
        for i in range(1, n_frames):
            metric_depth_prev = all_metric_depth[i - 1]

            # Load and resize images to depth resolution
            img_prev_bgr = cv2.imread(image_paths[i - 1])
            img_curr_bgr = cv2.imread(image_paths[i])
            h_img, w_img = img_prev_bgr.shape[:2]

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

            T_rel = np.eye(4, dtype=np.float64)
            pnp_success = False

            if len(pts_prev) >= 10:
                # Look up metric depth at feature locations
                px = pts_prev[:, 0].astype(int).clip(0, w_depth - 1)
                py = pts_prev[:, 1].astype(int).clip(0, h_depth - 1)
                depths = metric_depth_prev[py, px]

                valid = depths > 0.1  # at least 10cm
                pts_prev_valid = pts_prev[valid]
                pts_curr_valid = pts_curr[valid]
                depths_valid = depths[valid]

                if len(pts_prev_valid) >= 10:
                    # Unproject to 3D in camera frame (metric scale)
                    ones = np.ones(len(pts_prev_valid))
                    pixels_h = np.stack(
                        [pts_prev_valid[:, 0], pts_prev_valid[:, 1], ones], axis=1,
                    )
                    rays = (K_inv @ pixels_h.T).T
                    pts_3d = rays * depths_valid[:, None]  # meters

                    result = self._solve_pnp(pts_3d, pts_curr_valid, K)
                    if result is not None:
                        R_pnp, t_pnp = result
                        T_rel[:3, :3] = R_pnp
                        T_rel[:3, 3] = t_pnp
                        pnp_success = True
                        n_pnp_success += 1
                        translation_norms.append(np.linalg.norm(t_pnp))

            if not pnp_success:
                n_pnp_fail += 1

            # Chain
            all_w2c[i] = T_rel @ all_w2c[i - 1]

            if i <= 5 or i % 20 == 0:
                t_norm = np.linalg.norm(T_rel[:3, 3])
                status = "PnP" if pnp_success else "FAIL"
                logger.info(
                    "  Pair %d/%d: %s, |t|=%.4fm, %d features",
                    i, n_frames - 1, status, t_norm, len(pts_prev),
                )

        mean_t = np.mean(translation_norms) if translation_norms else 0.0
        logger.info(
            "DA3Metric-PnP: %d poses, PnP: %d/%d success, mean |t|=%.4fm",
            n_frames, n_pnp_success, n_frames - 1, mean_t,
        )

        pred_w2c = torch.from_numpy(all_w2c).to(self.device)
        pred_intrinsics = torch.from_numpy(all_intr).to(self.device).unsqueeze(0)
        return pred_w2c, pred_intrinsics
