"""
Error-State EKF fusing IMU propagation with foundation model pose measurements.

State vector (15-dim):
  position     (3)  — world frame (FLU)
  velocity     (3)  — world frame
  orientation  (4)  — quaternion [qx, qy, qz, qw], but error-state uses 3-dim
  gyro_bias    (3)
  accel_bias   (3)

Error state (15-dim):
  dp (3), dv (3), dtheta (3), dbg (3), dba (3)

IMU data in FLU body frame (from imu_synthesizer.py).
Foundation model poses are w2c SE(3) in OpenCV camera convention.

Usage:
    ekf = IMUVisionEKF(imu_csv_path)
    fused_w2c = ekf.fuse(vision_w2c, vision_timestamps)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Camera-to-IMU extrinsic (must match tum_utils.py / kalibr_imucam_chain.yaml)
T_CAM_TO_IMU = np.array([
    [ 0.0,  0.0,  1.0,  0.10],
    [-1.0,  0.0,  0.0,  0.03],
    [ 0.0, -1.0,  0.0,  0.01],
    [ 0.0,  0.0,  0.0,  1.00],
], dtype=np.float64)

# Gravity in FLU world frame (z-up): g = [0, 0, -9.81]
GRAVITY_FLU = np.array([0.0, 0.0, -9.81])


def skew(v):
    """3x3 skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)


def quat_multiply(q, r):
    """Hamilton quaternion product q * r. Convention: [x, y, z, w]."""
    qx, qy, qz, qw = q
    rx, ry, rz, rw = r
    return np.array([
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
        qw*rw - qx*rx - qy*ry - qz*rz,
    ])


def quat_to_rotmat(q):
    """Quaternion [x, y, z, w] to 3x3 rotation matrix (body-to-world)."""
    return Rotation.from_quat(q).as_matrix()


def rotmat_to_quat(R):
    """3x3 rotation matrix to quaternion [x, y, z, w]."""
    return Rotation.from_matrix(R).as_quat()


def normalize_quat(q):
    """Normalize quaternion to unit length."""
    return q / np.linalg.norm(q)


class IMUVisionEKF:
    """Error-state EKF fusing IMU + vision pose measurements.

    Args:
        imu_csv: Path to IMU CSV file (timestamp, ax, ay, az, wx, wy, wz).
        T_cam_to_imu: 4x4 camera-to-IMU extrinsic. Default: carl.json.
        sigma_accel: Accelerometer noise density (m/s^2/sqrt(Hz)).
        sigma_gyro: Gyroscope noise density (rad/s/sqrt(Hz)).
        sigma_ba: Accelerometer bias random walk (m/s^3/sqrt(Hz)).
        sigma_bg: Gyroscope bias random walk (rad/s^2/sqrt(Hz)).
        sigma_vision_pos: Vision position measurement noise (m).
        sigma_vision_rot: Vision rotation measurement noise (rad).
    """

    def __init__(
        self,
        imu_csv: Union[str, Path],
        T_cam_to_imu: Optional[np.ndarray] = None,
        sigma_accel: float = 2.0e-3,
        sigma_gyro: float = 1.6968e-4,
        sigma_ba: float = 3.0e-3,
        sigma_bg: float = 1.9393e-5,
        sigma_vision_pos: float = 0.05,
        sigma_vision_rot: float = 0.02,
    ):
        self.imu_data = self._load_imu(imu_csv)
        self.T_cam_to_imu = T_cam_to_imu if T_cam_to_imu is not None else T_CAM_TO_IMU
        self.T_imu_to_cam = np.linalg.inv(self.T_cam_to_imu)

        # Process noise densities
        self.sigma_accel = sigma_accel
        self.sigma_gyro = sigma_gyro
        self.sigma_ba = sigma_ba
        self.sigma_bg = sigma_bg

        # Measurement noise
        self.sigma_vision_pos = sigma_vision_pos
        self.sigma_vision_rot = sigma_vision_rot

        # State
        self.pos = np.zeros(3)       # world frame position
        self.vel = np.zeros(3)       # world frame velocity
        self.quat = np.array([0, 0, 0, 1.0])  # [x,y,z,w] identity
        self.bg = np.zeros(3)        # gyro bias
        self.ba = np.zeros(3)        # accel bias

        # Error-state covariance (15x15)
        self.P = np.eye(15) * 1e-4
        # Initial uncertainty: larger for biases
        self.P[9:12, 9:12] = np.eye(3) * 1e-6   # bg
        self.P[12:15, 12:15] = np.eye(3) * 1e-6  # ba

    @staticmethod
    def _load_imu(path):
        """Load IMU CSV: timestamp, ax, ay, az, wx, wy, wz."""
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 7:
                    data.append([float(x) for x in parts[:7]])
        return np.array(data)  # (M, 7)

    def _imu_propagate(self, accel, gyro, dt):
        """Propagate state and covariance with one IMU measurement.

        Uses first-order integration with error-state Jacobians.
        """
        R = quat_to_rotmat(self.quat)  # body-to-world

        # Corrected measurements
        accel_corr = accel - self.ba
        gyro_corr = gyro - self.bg

        # State propagation
        accel_world = R @ accel_corr + GRAVITY_FLU
        self.pos = self.pos + self.vel * dt + 0.5 * accel_world * dt**2
        self.vel = self.vel + accel_world * dt

        # Orientation: integrate angular velocity
        angle = np.linalg.norm(gyro_corr) * dt
        if angle > 1e-10:
            axis = gyro_corr / np.linalg.norm(gyro_corr)
            dq = np.append(axis * np.sin(angle / 2), np.cos(angle / 2))
            self.quat = normalize_quat(quat_multiply(self.quat, dq))
        # else: no rotation update needed

        # Error-state transition matrix F (15x15)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt                     # dp/dv
        F[3:6, 6:9] = -R @ skew(accel_corr) * dt          # dv/dtheta
        F[3:6, 12:15] = -R * dt                           # dv/dba
        F[6:9, 6:9] = np.eye(3) - skew(gyro_corr) * dt    # dtheta/dtheta
        F[6:9, 9:12] = -np.eye(3) * dt                    # dtheta/dbg

        # Process noise covariance Q
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * (self.sigma_accel ** 2) * dt  # velocity noise
        Q[6:9, 6:9] = np.eye(3) * (self.sigma_gyro ** 2) * dt   # orientation noise
        Q[9:12, 9:12] = np.eye(3) * (self.sigma_bg ** 2) * dt   # gyro bias walk
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_ba ** 2) * dt # accel bias walk

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    def _vision_update(self, pos_meas, R_meas):
        """Update state with a vision pose measurement (IMU frame, world coords).

        Args:
            pos_meas: (3,) measured IMU position in world frame.
            R_meas: (3, 3) measured IMU orientation (body-to-world).
        """
        R_est = quat_to_rotmat(self.quat)

        # Position residual
        dp = pos_meas - self.pos

        # Orientation residual: small-angle from R_meas @ R_est^T
        dR = R_meas @ R_est.T
        # Extract rotation vector (log map)
        dtheta = Rotation.from_matrix(dR).as_rotvec()

        # Measurement residual (6x1)
        z = np.concatenate([dp, dtheta])

        # Observation matrix H (6x15)
        # z = H @ dx + noise
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)   # position
        H[3:6, 6:9] = np.eye(3)   # orientation

        # Measurement noise R
        R_noise = np.diag([
            self.sigma_vision_pos**2, self.sigma_vision_pos**2, self.sigma_vision_pos**2,
            self.sigma_vision_rot**2, self.sigma_vision_rot**2, self.sigma_vision_rot**2,
        ])

        # Kalman gain
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error-state update
        dx = K @ z

        # Inject error state
        self.pos += dx[0:3]
        self.vel += dx[3:6]

        # Orientation correction
        dq_corr = np.append(dx[6:9] / 2, 1.0)
        dq_corr = normalize_quat(dq_corr)
        self.quat = normalize_quat(quat_multiply(self.quat, dq_corr))

        self.bg += dx[9:12]
        self.ba += dx[12:15]

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_noise @ K.T

    def _vision_w2c_to_imu_pose(self, w2c):
        """Convert a w2c camera pose to IMU position and rotation in world frame.

        Args:
            w2c: (4, 4) world-to-camera SE(3) matrix (OpenCV convention).

        Returns:
            pos: (3,) IMU position in world frame.
            R: (3, 3) IMU body-to-world rotation.
        """
        c2w = np.linalg.inv(w2c)
        # Camera c2w to IMU c2w: T_IinG = T_CinG @ T_ItoC = c2w @ inv(T_CtoI)
        T_IinG = c2w @ self.T_imu_to_cam
        return T_IinG[:3, 3], T_IinG[:3, :3]

    def _initialize_from_vision(self, w2c):
        """Set initial state from first vision measurement."""
        pos, R = self._vision_w2c_to_imu_pose(w2c)
        self.pos = pos.copy()
        self.vel = np.zeros(3)
        self.quat = rotmat_to_quat(R)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        logger.info(
            "EKF initialized: pos=[%.3f, %.3f, %.3f]",
            *self.pos,
        )

    def fuse(
        self,
        vision_w2c: np.ndarray,
        vision_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Run EKF fusion over the full trajectory.

        Args:
            vision_w2c: (N, 4, 4) w2c poses from foundation model.
            vision_timestamps: (N,) timestamps for each vision pose.

        Returns:
            fused_w2c: (N, 4, 4) fused w2c poses at vision timestamps.
        """
        N = len(vision_timestamps)
        imu_t = self.imu_data[:, 0]
        imu_accel = self.imu_data[:, 1:4]
        imu_gyro = self.imu_data[:, 4:7]

        # Initialize from first vision pose
        self._initialize_from_vision(vision_w2c[0])

        fused_w2c = np.zeros((N, 4, 4), dtype=np.float64)
        fused_w2c[0] = vision_w2c[0]

        imu_idx = 0  # current position in IMU stream

        for vi in range(1, N):
            t_target = vision_timestamps[vi]

            # Propagate IMU up to this vision timestamp
            while imu_idx < len(imu_t) - 1 and imu_t[imu_idx + 1] <= t_target:
                dt = imu_t[imu_idx + 1] - imu_t[imu_idx]
                if dt > 0:
                    self._imu_propagate(imu_accel[imu_idx], imu_gyro[imu_idx], dt)
                imu_idx += 1

            # Fractional step to exact vision timestamp
            if imu_idx < len(imu_t) and imu_t[imu_idx] < t_target:
                dt_frac = t_target - imu_t[imu_idx]
                if dt_frac > 0:
                    self._imu_propagate(imu_accel[imu_idx], imu_gyro[imu_idx], dt_frac)

            # Vision measurement update
            pos_meas, R_meas = self._vision_w2c_to_imu_pose(vision_w2c[vi])
            self._vision_update(pos_meas, R_meas)

            # Record fused pose as w2c
            R_imu = quat_to_rotmat(self.quat)
            T_IinG = np.eye(4)
            T_IinG[:3, :3] = R_imu
            T_IinG[:3, 3] = self.pos
            T_CinG = T_IinG @ self.T_cam_to_imu
            fused_w2c[vi] = np.linalg.inv(T_CinG)

            if (vi + 1) % 50 == 0 or vi == N - 1:
                logger.info(
                    "EKF frame %d/%d: pos=[%.3f, %.3f, %.3f] |bg|=%.5f |ba|=%.5f",
                    vi + 1, N, *self.pos,
                    np.linalg.norm(self.bg), np.linalg.norm(self.ba),
                )

        return fused_w2c

    def fuse_to_torch(
        self,
        vision_w2c: np.ndarray,
        vision_timestamps: np.ndarray,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, None]:
        """Run fusion and return in the standard predictor contract.

        Returns:
            pred_w2c: [N, 4, 4] float64 tensor.
            pred_intrinsics: None.
        """
        fused = self.fuse(vision_w2c, vision_timestamps)
        return torch.from_numpy(fused).to(device), None
