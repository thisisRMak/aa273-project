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

# Re-pin numpy after all installs (downstream packages may pull in numpy 2.x)
RUN python -m pip install --no-cache-dir numpy==1.26.4
