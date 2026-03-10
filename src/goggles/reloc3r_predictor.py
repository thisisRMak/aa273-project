"""
Reloc3r camera pose prediction wrapper.

Provides Reloc3r-based visual localization returning the same [N, 4, 4] w2c
format as the other predictors in eval_poses.py.

Reloc3r outputs c2w poses (camera-to-world); these are inverted to w2c here
to match the eval pipeline convention.

Requires the reloc3r source tree on PYTHONPATH:
    export PYTHONPATH=/workspace/reloc3r:$PYTHONPATH

Models are downloaded automatically from HuggingFace on first use:
    siyan824/reloc3r-224
    siyan824/reloc3r-512
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _invert_se3(c2w: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix without full matrix inverse."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=c2w.dtype)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -(R.T @ t)
    return w2c


def _load_reloc3r_model(img_reso: int, device):
    """Load (or download) a Reloc3r-{img_reso} model from HuggingFace."""
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model

    logger.info("Loading Reloc3r-%d model from HuggingFace...", img_reso)
    model = setup_reloc3r_relpose_model(model_args=str(img_reso), device=device)
    logger.info("Reloc3r model loaded on %s", device)
    return model


class Reloc3rPredictor:
    """Predict camera poses using Reloc3r visual localization.

    Args:
        img_reso: Image resolution for reloc3r inference (224 or 512).
        device: Torch device string.
        mode: 'seq' (sequential, uses previous frame as extra DB entry) or 'db' (db-only).
        use_amp: Use Automatic Mixed Precision (faster on Ampere+).
    """

    def __init__(
        self,
        img_reso: int = 512,
        device: str = "cuda",
        mode: str = "seq",
        use_amp: bool = False,
    ):
        self.img_reso = img_reso
        self.device = torch.device(device)
        self.mode = mode
        self.use_amp = use_amp
        self._relpose_model = None  # lazy-loaded on first predict_poses call

    def _load_model(self):
        return _load_reloc3r_model(self.img_reso, self.device)

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, None]:
        """Run Reloc3r visual localization and return w2c extrinsics.

        Implements the sequential visloc pipeline from wild_visloc.py,
        operating directly on image file paths (no video decode needed).

        Args:
            image_paths: Ordered list of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64) on CPU.
            None: intrinsics are not predicted by reloc3r.
        """
        from reloc3r.reloc3r_relpose import inference_relpose
        from reloc3r.reloc3r_visloc import Reloc3rVisloc
        from reloc3r.utils.device import to_numpy
        from reloc3r.utils.image import load_images, check_images_shape_format

        if self._relpose_model is None:
            self._relpose_model = self._load_model()

        reloc3r_visloc = Reloc3rVisloc()

        # reloc3r expects a directory; write symlinks into a temp dir
        # to avoid copying large image files
        with tempfile.TemporaryDirectory(prefix="reloc3r_eval_") as tmpdir:
            img_dir = Path(tmpdir) / "images"
            img_dir.mkdir()
            for i, p in enumerate(image_paths):
                dst = img_dir / f"{i:04d}{Path(p).suffix}"
                os.symlink(os.path.abspath(p), dst)

            images = load_images(str(img_dir), size=self.img_reso)

        images = check_images_shape_format(images, self.device)
        logger.info(
            "Reloc3r loaded %d images at resolution %d", len(images), self.img_reso
        )

        # Bootstrap: first and last frames define the coordinate frame + scale
        batch = [images[0], images[-1]]
        pose2to1 = to_numpy(
            inference_relpose(batch, self._relpose_model, self.device,
                              use_amp=self.use_amp)[0]
        )
        # Normalise baseline to 1 m (reloc3r has no absolute scale)
        pose2to1[:3, 3] /= np.linalg.norm(pose2to1[:3, 3])

        pose_beg = np.eye(4)
        pose_end = pose_beg @ pose2to1
        poses_c2w = [pose_beg, pose_end]

        for fid in tqdm(range(1, len(images) - 1), desc="Reloc3r localizing"):
            db1, db2, query = images[0], images[-1], images[fid]

            view1 = {
                "img":        torch.cat((db1["img"],        db2["img"]),        dim=0),
                "true_shape": torch.cat((db1["true_shape"], db2["true_shape"]), dim=0),
            }
            view2 = {
                "img":        torch.cat((query["img"],        query["img"]),        dim=0),
                "true_shape": torch.cat((query["true_shape"], query["true_shape"]), dim=0),
            }

            if self.mode == "seq" and fid > 1:
                db3 = images[fid - 1]
                view1["img"]        = torch.cat((view1["img"],        db3["img"]),        dim=0)
                view1["true_shape"] = torch.cat((view1["true_shape"], db3["true_shape"]), dim=0)
                view2["img"]        = torch.cat((view2["img"],        query["img"]),        dim=0)
                view2["true_shape"] = torch.cat((view2["true_shape"], query["true_shape"]), dim=0)

            poses2to1 = to_numpy(
                inference_relpose([view1, view2], self._relpose_model, self.device)
            )
            poses_db  = [poses_c2w[0], poses_c2w[-1]]
            poses_q2d = [poses2to1[0], poses2to1[1]]
            if self.mode == "seq" and fid > 1:
                poses_db.append(poses_c2w[fid - 1])
                poses_q2d.append(poses2to1[2])

            pose = reloc3r_visloc.motion_averaging(poses_db, poses_q2d)
            pose_end = poses_c2w.pop()
            poses_c2w.append(pose)
            poses_c2w.append(pose_end)

        # Convert c2w list → w2c [N, 4, 4] float64 tensor (CPU)
        w2c_np = np.stack([_invert_se3(p) for p in poses_c2w])
        pred_w2c = torch.from_numpy(w2c_np.astype(np.float64))

        logger.info("Reloc3r predicted %d poses", pred_w2c.shape[0])
        return pred_w2c, None


class Reloc3rWindowPredictor:
    """Causal windowed Reloc3r visual localization.

    Unlike :class:`Reloc3rPredictor` (which anchors on both the first *and*
    last frame of the full sequence), this predictor processes frames
    strictly causally: frame *k* is localised using only frames 0…k as
    context.  This mirrors a real-time deployment where the end of the
    sequence is unknown.

    Algorithm (per step *k*):
      * Frame 0  — assigned identity pose; sets the world-frame origin.
      * Frame 1  — bootstrapped via a single relpose call on (frame 0, frame 1);
                   the baseline is normalised to 1 m.
      * Frame k ≥ 2 — DB = [frame 0, frame k−1], query = frame k.
                       Absolute pose recovered via motion_averaging, identical
                       to the non-windowed predictor.

    Args:
        img_reso: Image resolution for reloc3r inference (224 or 512).
        device: Torch device string.
        use_amp: Use Automatic Mixed Precision (faster on Ampere+).
    """

    def __init__(
        self,
        img_reso: int = 512,
        device: str = "cuda",
        use_amp: bool = False,
    ):
        self.img_reso = img_reso
        self.device = torch.device(device)
        self.use_amp = use_amp
        self._relpose_model = None  # lazy-loaded on first predict_poses call

    def _load_model(self):
        return _load_reloc3r_model(self.img_reso, self.device)

    @torch.no_grad()
    def predict_poses(
        self,
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, None]:
        """Run causal windowed Reloc3r localization and return w2c extrinsics.

        Args:
            image_paths: Ordered list of image file paths.

        Returns:
            pred_w2c: [N, 4, 4] w2c SE(3) tensor (float64) on CPU.
            None: intrinsics are not predicted by reloc3r.
        """
        from reloc3r.reloc3r_relpose import inference_relpose
        from reloc3r.reloc3r_visloc import Reloc3rVisloc
        from reloc3r.utils.device import to_numpy
        from reloc3r.utils.image import load_images, check_images_shape_format

        if self._relpose_model is None:
            self._relpose_model = self._load_model()

        reloc3r_visloc = Reloc3rVisloc()

        with tempfile.TemporaryDirectory(prefix="reloc3r_win_") as tmpdir:
            img_dir = Path(tmpdir) / "images"
            img_dir.mkdir()
            for i, p in enumerate(image_paths):
                dst = img_dir / f"{i:04d}{Path(p).suffix}"
                os.symlink(os.path.abspath(p), dst)

            images = load_images(str(img_dir), size=self.img_reso)

        images = check_images_shape_format(images, self.device)
        N = len(images)
        logger.info(
            "Reloc3rWindow loaded %d images at resolution %d", N, self.img_reso
        )

        # --- Frame 0: world-frame origin ---
        pose_0 = np.eye(4)
        poses_c2w = [pose_0]

        if N == 1:
            w2c_np = np.stack([_invert_se3(p) for p in poses_c2w])
            return torch.from_numpy(w2c_np.astype(np.float64)), None

        # --- Frame 1: bootstrap from the (0, 1) pair ---
        batch = [images[0], images[1]]
        pose1to0 = to_numpy(
            inference_relpose(
                batch, self._relpose_model, self.device, use_amp=self.use_amp
            )[0]
        )
        norm = np.linalg.norm(pose1to0[:3, 3])
        pose1to0[:3, 3] /= norm if norm > 1e-6 else 1.0
        poses_c2w.append(pose_0 @ pose1to0)

        # --- Frames 2..N-1: causal localization ---
        for k in tqdm(range(2, N), desc="Reloc3rWindow localizing"):
            img_k = images[k]
            img_prev = images[k - 1]

            # DB batch: [frame 0, frame k-1]; query batch: [frame k, frame k]
            view1 = {
                "img":        torch.cat((images[0]["img"],    img_prev["img"]),    dim=0),
                "true_shape": torch.cat((images[0]["true_shape"], img_prev["true_shape"]), dim=0),
            }
            view2 = {
                "img":        torch.cat((img_k["img"],        img_k["img"]),        dim=0),
                "true_shape": torch.cat((img_k["true_shape"], img_k["true_shape"]), dim=0),
            }

            poses2to1 = to_numpy(
                inference_relpose([view1, view2], self._relpose_model, self.device)
            )
            poses_db  = [poses_c2w[0], poses_c2w[k - 1]]
            poses_q2d = [poses2to1[0], poses2to1[1]]

            pose = reloc3r_visloc.motion_averaging(poses_db, poses_q2d)
            poses_c2w.append(pose)

        # Convert c2w list → w2c [N, 4, 4] float64 tensor (CPU)
        w2c_np = np.stack([_invert_se3(p) for p in poses_c2w])
        pred_w2c = torch.from_numpy(w2c_np.astype(np.float64))

        logger.info("Reloc3rWindow predicted %d poses", pred_w2c.shape[0])
        return pred_w2c, None
