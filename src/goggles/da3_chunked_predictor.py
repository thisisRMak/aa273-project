"""
DA3 Chunked pose predictor — overlapping submaps with SIM(3) alignment.

Divides a frame sequence into overlapping chunks, runs DA3 batch inference
per chunk, and aligns adjacent chunks via SIM(3) on overlapping 3D point maps.
Follows the VGGT-Long / DA3-Streaming approach but in a lightweight,
self-contained wrapper.

Returns the same (pred_w2c, pred_intrinsics) contract as DA3PosePredictor.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from goggles.sim3_utils import (
    accumulate_sim3_transforms,
    depth_to_point_cloud,
    estimate_sim3,
)

logger = logging.getLogger(__name__)


class DA3ChunkedPredictor:
    """Predict camera poses using DA3 with chunked submap alignment.

    Args:
        model_name: HuggingFace model ID (default: DA3-LARGE-1.1).
        device: Torch device string.
        chunk_size: Number of frames per chunk.
        overlap: Number of overlapping frames between adjacent chunks.
    """

    DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        chunk_size: int = 60,
        overlap: int = 20,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.device = torch.device(device)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = self._load_model()

    def _load_model(self):
        from depth_anything_3.api import DepthAnything3

        logger.info("Loading DA3 model: %s", self.model_name)
        model = DepthAnything3.from_pretrained(self.model_name)
        model = model.to(device=self.device)
        logger.info("DA3 model loaded on %s", self.device)
        return model

    def _get_chunk_indices(self, n_frames: int) -> List[Tuple[int, int]]:
        """Partition frames into overlapping chunks."""
        if n_frames <= self.chunk_size:
            return [(0, n_frames)]

        step = self.chunk_size - self.overlap
        chunks = []
        start = 0
        while start < n_frames:
            end = min(start + self.chunk_size, n_frames)
            chunks.append((start, end))
            if end == n_frames:
                break
            start += step
        return chunks

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run DA3 chunked inference and return globally-aligned w2c extrinsics.

        Args:
            image_paths: List of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64).
            pred_intrinsics: [1, N, 3, 3] predicted intrinsics (float64).
        """
        n_frames = len(image_paths)
        chunk_indices = self._get_chunk_indices(n_frames)
        n_chunks = len(chunk_indices)

        logger.info(
            "DA3 chunked: %d frames → %d chunks (size=%d, overlap=%d)",
            n_frames, n_chunks, self.chunk_size, self.overlap,
        )

        # --- Stage 1: Run DA3 inference per chunk ---
        predictions = []
        for ci, (start, end) in enumerate(chunk_indices):
            logger.info("  Chunk %d/%d: frames [%d, %d)", ci, n_chunks, start, end)
            chunk_paths = image_paths[start:end]
            pred = self.model.inference(chunk_paths)
            predictions.append(pred)
            torch.cuda.empty_cache()

        # --- Stage 2: Single chunk — no alignment needed ---
        if n_chunks == 1:
            pred = predictions[0]
            pred_w2c = self._extrinsics_to_4x4(pred.extrinsics)
            pred_w2c = torch.from_numpy(pred_w2c.astype(np.float64)).to(self.device)

            intr = torch.from_numpy(
                pred.intrinsics.astype(np.float64)
            ).to(self.device).unsqueeze(0)
            return pred_w2c, intr

        # --- Stage 3: Align adjacent chunks via SIM(3) on overlap point maps ---
        sim3_list = []
        for ci in range(n_chunks - 1):
            s, R, t = self._align_adjacent_chunks(
                predictions[ci], predictions[ci + 1],
                chunk_indices[ci], chunk_indices[ci + 1],
            )
            sim3_list.append((s, R, t))
            logger.info(
                "  SIM(3) chunk %d→%d: s=%.4f", ci + 1, ci, s,
            )

        # Accumulate: each entry maps chunk k+1 → chunk 0
        cum_sim3 = accumulate_sim3_transforms(sim3_list)

        # --- Stage 4: Assemble global poses ---
        pred_w2c, pred_intrinsics = self._assemble_global_poses(
            predictions, chunk_indices, cum_sim3,
        )

        logger.info(
            "DA3 chunked: assembled %d global poses", pred_w2c.shape[0],
        )

        pred_w2c = torch.from_numpy(pred_w2c).to(self.device)
        pred_intrinsics = torch.from_numpy(pred_intrinsics).to(self.device).unsqueeze(0)
        return pred_w2c, pred_intrinsics

    def _extrinsics_to_4x4(self, ext: np.ndarray) -> np.ndarray:
        """Convert (N, 3, 4) or (N, 4, 4) extrinsics to (N, 4, 4)."""
        if ext.shape[-2:] == (4, 4):
            return ext
        N = ext.shape[0]
        out = np.zeros((N, 4, 4), dtype=ext.dtype)
        out[:, :3, :4] = ext[:, :3, :4]
        out[:, 3, 3] = 1.0
        return out

    def _align_adjacent_chunks(self, pred_prev, pred_curr, idx_prev, idx_curr):
        """Estimate SIM(3) from current chunk's frame to previous chunk's frame.

        Uses overlapping frames' 3D point maps (from depth + intrinsics + extrinsics).
        """
        start_prev, end_prev = idx_prev
        start_curr, end_curr = idx_curr
        overlap_start_global = start_curr  # overlap begins at start of current chunk
        overlap_end_global = end_prev       # overlap ends at end of previous chunk

        n_overlap = overlap_end_global - overlap_start_global
        if n_overlap <= 0:
            raise ValueError(
                f"No overlap between chunks [{start_prev}, {end_prev}) "
                f"and [{start_curr}, {end_curr})"
            )

        # Indices within each chunk's prediction arrays
        prev_overlap_start = overlap_start_global - start_prev
        curr_overlap_end = n_overlap  # overlap is at the beginning of current chunk

        # Extract overlap depth, intrinsics, extrinsics
        depth_prev = np.squeeze(pred_prev.depth[prev_overlap_start:prev_overlap_start + n_overlap])
        depth_curr = np.squeeze(pred_curr.depth[:curr_overlap_end])
        intr_prev = pred_prev.intrinsics[prev_overlap_start:prev_overlap_start + n_overlap]
        intr_curr = pred_curr.intrinsics[:curr_overlap_end]

        ext_prev = self._extrinsics_to_4x4(pred_prev.extrinsics)
        ext_curr = self._extrinsics_to_4x4(pred_curr.extrinsics)
        ext_prev_overlap = ext_prev[prev_overlap_start:prev_overlap_start + n_overlap, :3, :4]
        ext_curr_overlap = ext_curr[:curr_overlap_end, :3, :4]

        # Build 3D point maps
        pm_prev = depth_to_point_cloud(depth_prev, intr_prev, ext_prev_overlap)
        pm_curr = depth_to_point_cloud(depth_curr, intr_curr, ext_curr_overlap)

        # Confidence-based filtering
        conf_prev = pred_prev.conf[prev_overlap_start:prev_overlap_start + n_overlap]
        conf_curr = pred_curr.conf[:curr_overlap_end]

        # DA3 conf is shifted by +1 in da3_streaming; raw conf here
        conf_threshold = min(np.median(conf_prev), np.median(conf_curr)) * 0.1

        # Collect confident matching points across all overlap frames
        src_pts = []
        tgt_pts = []
        for i in range(n_overlap):
            mask = (conf_prev[i] > conf_threshold) & (conf_curr[i] > conf_threshold)
            idx = np.where(mask)
            if len(idx[0]) == 0:
                continue
            src_pts.append(pm_curr[i][idx])  # source = current chunk
            tgt_pts.append(pm_prev[i][idx])  # target = previous chunk

        if not src_pts:
            logger.warning("No confident overlap points, using identity SIM(3)")
            return 1.0, np.eye(3), np.zeros(3)

        all_src = np.concatenate(src_pts, axis=0)
        all_tgt = np.concatenate(tgt_pts, axis=0)

        logger.info("    Overlap: %d frames, %d confident point pairs", n_overlap, len(all_src))

        s, R, t = estimate_sim3(all_src, all_tgt)
        return s, R, t

    def _assemble_global_poses(self, predictions, chunk_indices, cum_sim3):
        """Transform per-chunk w2c poses into global (chunk 0) coordinate frame.

        For chunk 0: poses used directly.
        For chunk k (k>0): apply cumulative SIM(3) to transform c2w, then invert to w2c.
        Non-overlap frames are taken from each chunk to avoid duplicates.
        """
        n_chunks = len(chunk_indices)
        n_frames = chunk_indices[-1][1]

        all_w2c = np.zeros((n_frames, 4, 4), dtype=np.float64)
        all_intr = np.zeros((n_frames, 3, 3), dtype=np.float64)
        assigned = np.zeros(n_frames, dtype=bool)

        for ci in range(n_chunks):
            start, end = chunk_indices[ci]
            ext = self._extrinsics_to_4x4(predictions[ci].extrinsics).astype(np.float64)
            intr = predictions[ci].intrinsics.astype(np.float64)

            # Determine which frames this chunk is responsible for
            # First chunk: everything except last `overlap` frames (shared with next chunk)
            # Middle chunks: skip first `overlap` frames (owned by prev), keep up to last `overlap`
            # Last chunk: skip first `overlap` frames, keep the rest
            if ci == 0:
                local_start = 0
                local_end = (end - start) if n_chunks == 1 else (end - start)
            else:
                local_start = self.overlap  # skip overlap region (owned by previous chunk)
                local_end = end - start

            if ci > 0:
                s, R, t = cum_sim3[ci - 1]
                # Build 4x4 SIM(3) matrix
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = s * R
                S[:3, 3] = t

                for li in range(local_start, local_end):
                    global_idx = start + li
                    if assigned[global_idx]:
                        continue

                    w2c = ext[li]
                    c2w = np.linalg.inv(w2c)
                    # Apply SIM(3): transform c2w into chunk 0's frame
                    c2w_global = S @ c2w
                    # Normalize rotation (remove scale from rotation block)
                    c2w_global[:3, :3] /= s
                    w2c_global = np.linalg.inv(c2w_global)

                    all_w2c[global_idx] = w2c_global
                    all_intr[global_idx] = intr[li]
                    assigned[global_idx] = True
            else:
                for li in range(local_start, local_end):
                    global_idx = start + li
                    if assigned[global_idx]:
                        continue
                    all_w2c[global_idx] = ext[li]
                    all_intr[global_idx] = intr[li]
                    assigned[global_idx] = True

        # Sanity check
        unassigned = np.where(~assigned)[0]
        if len(unassigned) > 0:
            logger.warning("Unassigned frames: %s", unassigned.tolist())

        return all_w2c, all_intr
