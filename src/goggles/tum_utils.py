"""TUM trajectory format utilities for OpenVINS integration.

TUM format: timestamp tx ty tz qx qy qz qw (Hamilton convention)
Represents the IMU/body pose in the global frame (c2w equivalent).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def load_tum_trajectory(path, device="cuda"):
    """Load a TUM-format trajectory file and convert to w2c matrices.

    Args:
        path: Path to TUM trajectory file.
        device: Torch device for output tensors.

    Returns:
        pred_w2c: [N, 4, 4] float64 tensor (world-to-camera SE(3)).
        timestamps: [N] numpy array of timestamps.
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            data.append([float(x) for x in parts[:8]])

    if not data:
        raise ValueError(f"No valid poses found in {path}")

    data = np.array(data)  # (N, 8)
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # tx, ty, tz
    quaternions = data[:, 4:8]  # qx, qy, qz, qw (Hamilton)

    N = len(data)
    w2c = np.zeros((N, 4, 4), dtype=np.float64)

    for i in range(N):
        # TUM gives pose in global frame (c2w): position + orientation
        R = Rotation.from_quat(quaternions[i]).as_matrix()  # scipy uses [x,y,z,w]
        t = positions[i]

        # Build c2w
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = t

        # Invert to w2c
        w2c[i] = np.linalg.inv(c2w)

    pred_w2c = torch.from_numpy(w2c).to(device)
    return pred_w2c, timestamps
