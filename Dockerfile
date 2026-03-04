FROM figs:latest

# StreamVGGT dependencies not already in figs:latest
# Already present in base: torch, torchvision, numpy, pillow, roma, matplotlib,
#   tqdm, einops, open3d, scipy, gsplat, opencv-python, scikit-learn, trimesh
RUN python -m pip install --no-cache-dir \
    huggingface_hub \
    safetensors \
    "transformers>=4.37,<4.46" \
    accelerate \
    h5py

# DA3 (Depth Anything 3) dependencies not already in figs:latest or above
# Omitted (would replace torch 2.1.2+cu118, breaking FiGS/gsplat/torchvision):
#   xformers — DA3 falls back to pure PyTorch SwiGLU
#   e3nn — requires torch>=2.2.0; DA3 only uses it for SH rotation (not needed for poses)
RUN python -m pip install --no-cache-dir \
    "typer>=0.9.0" \
    "moviepy==1.0.3" \
    evo \
    pillow_heif \
    plyfile

# Re-pin numpy after all installs (downstream packages may pull in numpy 2.x)
RUN python -m pip install --no-cache-dir numpy==1.26.4
