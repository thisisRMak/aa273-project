"""TUM trajectory format utilities for OpenVINS integration.

TUM format: timestamp tx ty tz qx qy qz qw (Hamilton convention)

OpenVINS outputs IMU/body poses (T_IinG). To get camera poses we apply
the camera-to-IMU extrinsic: T_CinG = T_IinG @ T_CtoI.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Camera-to-IMU extrinsic: OpenCV camera frame → FLU (z-up) body frame.
# Must match kalibr_imucam_chain.yaml so that pose conversion uses the
# same frame as OpenVINS internally.
#
# Derivation: carl.json (OpenGL cam → FRD body), then FRD→FLU, then
# OpenGL→OpenCV: T = diag(1,-1,-1) @ T_carl @ diag(1,-1,-1,1).
T_CAM_TO_IMU = np.array([
    [ 0.0,  0.0,  1.0,  0.10],
    [-1.0,  0.0,  0.0,  0.03],
    [ 0.0, -1.0,  0.0,  0.01],
    [ 0.0,  0.0,  0.0,  1.00],
], dtype=np.float64)


def load_tum_trajectory(path, device="cuda", T_cam_to_imu=None):
    """Load a TUM-format trajectory file and convert to camera w2c matrices.

    OpenVINS outputs IMU/body poses. This function applies the camera-to-IMU
    extrinsic to convert them to camera poses before inverting to w2c.

    Args:
        path: Path to TUM trajectory file.
        device: Torch device for output tensors.
        T_cam_to_imu: (4, 4) camera-to-IMU extrinsic. Default: carl.json.

    Returns:
        pred_w2c: [N, 4, 4] float64 tensor (world-to-camera SE(3)).
        timestamps: [N] numpy array of timestamps.
    """
    if T_cam_to_imu is None:
        T_cam_to_imu = T_CAM_TO_IMU

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
        # TUM gives IMU pose in global frame: T_IinG (IMU c2w)
        R = Rotation.from_quat(quaternions[i]).as_matrix()  # scipy uses [x,y,z,w]
        T_IinG = np.eye(4, dtype=np.float64)
        T_IinG[:3, :3] = R
        T_IinG[:3, 3] = positions[i]

        # Camera c2w = T_IinG @ T_CtoI (chain IMU-in-global with cam-to-IMU)
        T_CinG = T_IinG @ T_cam_to_imu

        # Invert to w2c
        w2c[i] = np.linalg.inv(T_CinG)

    pred_w2c = torch.from_numpy(w2c).to(device)
    return pred_w2c, timestamps
