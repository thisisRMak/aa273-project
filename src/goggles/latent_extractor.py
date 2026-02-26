"""
Pre-decoder latent extraction from the StreamVGGT aggregator.

Runs images through the ViT backbone + alternating frame/global attention
and returns the aggregated token representations BEFORE any decoder head
(DPT, camera, track) processes them.

Each aggregator layer produces tokens of shape [B, S, P, 2*embed_dim] where:
  - B = batch size (always 1 here)
  - S = number of frames in the chunk
  - P = 1 (camera) + 4 (register) + n_patches
  - 2*embed_dim = 2048 (concat of frame + global attention outputs)

Patch tokens start at index 5 (patch_start_idx).
For 518-wide images with patch_size=14, n_patches = (H//14) * (W//14).

See notes/aggregator_latents.md for detailed explanation of the latent structure.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

# DPT head tap points — the 4 layers the decoder actually uses (FPN-style)
DPT_LAYER_INDICES = [4, 11, 17, 23]

# All 24 aggregator layers
ALL_LAYER_INDICES = list(range(24))


class LatentExtractor:
    """Extract pre-decoder latent tokens from the StreamVGGT aggregator.

    Args:
        checkpoint_path: Path to StreamVGGT checkpoint (.pth).
            If None, attempts /workspace/StreamVGGT/ckpt/checkpoints.pth
            then downloads from HuggingFace.
        device: torch device string.
        dtype: Autocast dtype for inference. Default bfloat16 on Ampere+,
            float16 otherwise.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(device)

        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str]):
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT()

        if checkpoint_path is None:
            checkpoint_path = "/workspace/StreamVGGT/ckpt/checkpoints.pth"

        if os.path.exists(checkpoint_path):
            logger.info("Loading checkpoint from %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        else:
            logger.info("Local checkpoint not found, downloading from HuggingFace")
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename="checkpoints.pth",
                revision="main",
            )
            ckpt = torch.load(path, map_location="cpu")

        model.load_state_dict(ckpt, strict=True)
        del ckpt
        model.eval()
        model.to(self.device)
        return model

    def _preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """Load and preprocess images to [S, 3, H, W] tensor in [0, 1]."""
        from streamvggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(image_paths, mode="crop")

    @torch.no_grad()
    def extract(
        self,
        image_paths: List[str],
        layer_indices: Optional[List[int]] = None,
        chunk_size: int = 16,
        include_special_tokens: bool = False,
    ) -> dict:
        """Extract aggregator tokens for a list of images (batch mode).

        Processes images in chunks. Frames within a chunk get full
        cross-frame context via global attention. Frames across chunks
        are independent.

        Args:
            image_paths: Paths to input images.
            layer_indices: Which aggregator layers to extract (0-23).
                Defaults to DPT tap points [4, 11, 17, 23].
            chunk_size: Number of frames per forward pass.
                Larger = more cross-frame context but more GPU memory.
            include_special_tokens: If True, keep camera + register
                tokens (indices 0-4). Default False (patch tokens only).

        Returns:
            dict with keys:
                tokens:          float16 [n_frames, n_layers, n_tokens, 2048]
                layer_indices:   list[int]
                image_paths:     list[str]
                patch_start_idx: int
                patch_grid:      (patch_h, patch_w)
                image_size:      (H, W) after preprocessing
        """
        if layer_indices is None:
            layer_indices = DPT_LAYER_INDICES

        n_frames = len(image_paths)
        all_tokens = []

        # Get image dimensions from first image
        sample = self._preprocess_images(image_paths[:1])
        _, _, H, W = sample.shape
        patch_h, patch_w = H // 14, W // 14
        del sample

        for chunk_start in range(0, n_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_paths = image_paths[chunk_start:chunk_end]

            logger.info(
                "Processing frames %d-%d / %d",
                chunk_start + 1, chunk_end, n_frames,
            )

            # [S, 3, H, W] -> [1, S, 3, H, W]
            images = self._preprocess_images(chunk_paths).to(self.device)
            images = images.unsqueeze(0)

            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)

            # Extract selected layers for each frame in the chunk
            for s in range(len(chunk_paths)):
                frame_layers = []
                for li in layer_indices:
                    tokens = aggregated_tokens_list[li]  # [1, S, P, 2048]
                    if include_special_tokens:
                        frame_tokens = tokens[0, s]  # [P, 2048]
                    else:
                        frame_tokens = tokens[0, s, patch_start_idx:]  # [n_patches, 2048]
                    frame_layers.append(frame_tokens.cpu().half())

                all_tokens.append(torch.stack(frame_layers))  # [n_layers, n_tokens, 2048]

            del images, aggregated_tokens_list
            torch.cuda.empty_cache()

        # [n_frames, n_layers, n_tokens, 2048]
        tokens_tensor = torch.stack(all_tokens).numpy()

        return {
            "tokens": tokens_tensor,
            "layer_indices": layer_indices,
            "image_paths": image_paths,
            "patch_start_idx": patch_start_idx,
            "patch_grid": (patch_h, patch_w),
            "image_size": (H, W),
        }

    @staticmethod
    def _truncate_kv_cache(
        past_key_values: list, max_frames: int
    ) -> list:
        """Evict oldest frames from KV cache, keeping the most recent max_frames.

        Each cache entry is (k, v) with shape [B, num_heads, num_frames, P, head_dim].
        Truncates along dim 2.
        """
        truncated = []
        for kv in past_key_values:
            if kv is None:
                truncated.append(None)
                continue
            k, v = kv
            if k.shape[2] > max_frames:
                k = k[:, :, -max_frames:, :, :].contiguous()
                v = v[:, :, -max_frames:, :, :].contiguous()
            truncated.append((k, v))
        return truncated

    @torch.no_grad()
    def extract_streaming(
        self,
        image_paths: List[str],
        layer_indices: Optional[List[int]] = None,
        include_special_tokens: bool = False,
        max_cache_frames: Optional[int] = None,
    ) -> dict:
        """Extract aggregator tokens using streaming inference (KV cache).

        Processes one frame at a time. Each frame attends to previous
        frames via cached keys/values, giving causal context.

        Args:
            image_paths: Paths to input images (order matters — causal).
            layer_indices: Which aggregator layers to extract (0-23).
                Defaults to DPT tap points [4, 11, 17, 23].
            include_special_tokens: If True, keep camera + register
                tokens (indices 0-4). Default False (patch tokens only).
            max_cache_frames: Maximum past frames in KV cache. None =
                unlimited (may OOM on long sequences). Oldest frames
                are evicted when the limit is reached.

        Returns:
            Same format as extract().
        """
        if layer_indices is None:
            layer_indices = DPT_LAYER_INDICES

        n_frames = len(image_paths)
        all_tokens = []
        past_key_values = [None] * self.model.aggregator.depth

        # Get image dimensions from first image
        sample = self._preprocess_images(image_paths[:1])
        _, _, H, W = sample.shape
        patch_h, patch_w = H // 14, W // 14
        del sample

        if max_cache_frames is not None:
            logger.info("KV cache sliding window: %d frames", max_cache_frames)

        for i, path in enumerate(image_paths):
            logger.info(
                "Streaming frame %d / %d: %s",
                i + 1, n_frames, Path(path).name,
            )

            # [1, 3, H, W] -> [1, 1, 3, H, W]
            images = self._preprocess_images([path]).to(self.device)
            images = images.unsqueeze(0)

            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregator_output = self.model.aggregator(
                    images,
                    past_key_values=past_key_values,
                    use_cache=True,
                    past_frame_idx=i,
                )

            if len(aggregator_output) == 3:
                aggregated_tokens_list, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens_list, patch_start_idx = aggregator_output

            # Evict oldest frames if cache exceeds limit
            if max_cache_frames is not None:
                past_key_values = self._truncate_kv_cache(
                    past_key_values, max_cache_frames
                )

            frame_layers = []
            for li in layer_indices:
                tokens = aggregated_tokens_list[li]  # [1, 1, P, 2048]
                if include_special_tokens:
                    frame_tokens = tokens[0, 0]  # [P, 2048]
                else:
                    frame_tokens = tokens[0, 0, patch_start_idx:]  # [n_patches, 2048]
                frame_layers.append(frame_tokens.cpu().half())

            all_tokens.append(torch.stack(frame_layers))  # [n_layers, n_tokens, 2048]

            del images, aggregated_tokens_list
            torch.cuda.empty_cache()

        # [n_frames, n_layers, n_tokens, 2048]
        tokens_tensor = torch.stack(all_tokens).numpy()

        return {
            "tokens": tokens_tensor,
            "layer_indices": layer_indices,
            "image_paths": image_paths,
            "patch_start_idx": patch_start_idx,
            "patch_grid": (patch_h, patch_w),
            "image_size": (H, W),
        }


def save_latents(result: dict, output_path: str) -> str:
    """Save extraction results to HDF5.

    File layout:
        tokens:           float16 [n_frames, n_layers, n_tokens, 2048]
        layer_indices:    int     [n_layers]
        image_paths:      str     [n_frames]  (variable-length UTF-8)
        patch_start_idx:  int     scalar
        patch_grid:       int     [2]  (patch_h, patch_w)
        image_size:       int     [2]  (H, W)

    Chunked by frame so individual frames can be read without loading
    the entire file into memory.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tokens = result["tokens"]
    n_frames, n_layers, n_tokens, feat_dim = tokens.shape

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "tokens",
            data=tokens,
            dtype="float16",
            chunks=(1, n_layers, n_tokens, feat_dim),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("layer_indices", data=result["layer_indices"])
        f.create_dataset("patch_start_idx", data=result["patch_start_idx"])
        f.create_dataset("patch_grid", data=result["patch_grid"])
        f.create_dataset("image_size", data=result["image_size"])

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_paths", data=result["image_paths"], dtype=dt)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        "Saved %d frames x %d layers to %s (%.1f MB)",
        n_frames, n_layers, output_path, size_mb,
    )
    return output_path


def load_latents(path: str) -> dict:
    """Load extraction results from HDF5.

    Returns dict with same keys as LatentExtractor.extract().
    """
    with h5py.File(path, "r") as f:
        return {
            "tokens": f["tokens"][:],
            "layer_indices": f["layer_indices"][:].tolist(),
            "image_paths": [
                p.decode() if isinstance(p, bytes) else p
                for p in f["image_paths"][:]
            ],
            "patch_start_idx": int(f["patch_start_idx"][()]),
            "patch_grid": tuple(f["patch_grid"][:]),
            "image_size": tuple(f["image_size"][:]),
        }
