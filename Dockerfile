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

# Docker CLI for cross-container orchestration (e.g., OpenVINS)
# Only the CLI — no daemon. Socket is mounted at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*


# --- UI & Shell Enhancements ---
# Set the terminal to support 256 colors
ENV TERM=xterm-256color

# Enable color output for common bash commands by adding aliases to the system-wide bashrc
RUN echo "alias ls='ls -al --color=auto'" >> /etc/bash.bashrc && \
    echo "alias grep='grep --color=auto'" >> /etc/bash.bashrc && \
    echo "alias diff='diff --color=auto'" >> /etc/bash.bashrc

RUN echo 'export PS1="\[\033[01;32m\]\u@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /etc/bash.bashrc