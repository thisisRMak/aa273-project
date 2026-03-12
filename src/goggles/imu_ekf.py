"""
Error-State EKF fusing IMU propagation with foundation model **relative** pose measurements.

State vector (16-dim error state):
  dp           (3)  — position error, world frame (FLU)
  dv           (3)  — velocity error, world frame
  dtheta       (3)  — orientation error (rotation vector)
  dbg          (3)  — gyro bias error
  dba          (3)  — accel bias error
  ds           (1)  — vision scale factor error

Nominal state:
  position     (3)  — world frame (FLU)
  velocity     (3)  — world frame
  orientation  (4)  — quaternion [qx, qy, qz, qw]
  gyro_bias    (3)
  accel_bias   (3)
  scale        (1)  — multiplier on vision translations to get metric

IMU data in FLU body frame (from imu_synthesizer.py).
Foundation model poses are w2c SE(3) in OpenCV camera convention.

Key design: vision measurements are **relative** (frame-to-frame), not absolute.
This eliminates world-frame alignment issues and makes scale observable over
short inter-frame intervals where IMU position drift is negligible.

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

N_ERR = 16  # error-state dimension


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
    """Error-state EKF fusing IMU + vision relative pose measurements with scale estimation.

    Vision measurements are used as **relative** transforms between consecutive
    frames, not absolute poses. This avoids world-frame alignment and makes
    scale observable over short inter-frame intervals.

    Args:
        imu_csv: Path to IMU CSV file (timestamp, ax, ay, az, wx, wy, wz).
        T_cam_to_imu: 4x4 camera-to-IMU extrinsic. Default: carl.json.
        sigma_accel: Accelerometer noise density (m/s^2/sqrt(Hz)).
        sigma_gyro: Gyroscope noise density (rad/s/sqrt(Hz)).
        sigma_ba: Accelerometer bias random walk (m/s^3/sqrt(Hz)).
        sigma_bg: Gyroscope bias random walk (rad/s^2/sqrt(Hz)).
        sigma_vision_pos: Vision relative position measurement noise (m).
        sigma_vision_rot: Vision relative rotation measurement noise (rad).
        sigma_scale: Scale factor random walk noise (1/sqrt(Hz)).
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
        sigma_vision_rot: float = 0.03,
        sigma_scale: float = 0.01,
    ):
        self.imu_data = self._load_imu(imu_csv)
        self.T_cam_to_imu = T_cam_to_imu if T_cam_to_imu is not None else T_CAM_TO_IMU
        self.T_imu_to_cam = np.linalg.inv(self.T_cam_to_imu)

        # Process noise densities
        self.sigma_accel = sigma_accel
        self.sigma_gyro = sigma_gyro
        self.sigma_ba = sigma_ba
        self.sigma_bg = sigma_bg
        self.sigma_scale = sigma_scale

        # Measurement noise
        self.sigma_vision_pos = sigma_vision_pos
        self.sigma_vision_rot = sigma_vision_rot

        # Nominal state
        self.pos = np.zeros(3)       # world frame position
        self.vel = np.zeros(3)       # world frame velocity
        self.quat = np.array([0, 0, 0, 1.0])  # [x,y,z,w] identity
        self.bg = np.zeros(3)        # gyro bias
        self.ba = np.zeros(3)        # accel bias
        self.scale = 1.0             # vision scale factor

        # Error-state covariance (16x16)
        self.P = np.eye(N_ERR) * 1e-4
        self.P[0:3, 0:3] = np.eye(3) * 1e-2    # position
        self.P[3:6, 3:6] = np.eye(3) * 1e-2    # velocity
        self.P[6:9, 6:9] = np.eye(3) * 1e-4    # orientation
        self.P[9:12, 9:12] = np.eye(3) * 1e-6  # bg
        self.P[12:15, 12:15] = np.eye(3) * 1e-6  # ba
        self.P[15, 15] = 10.0                   # scale: uncertain

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
        """Propagate state and covariance with one IMU measurement."""
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

        # Error-state transition matrix F (16x16)
        F = np.eye(N_ERR)
        F[0:3, 3:6] = np.eye(3) * dt                     # dp/dv
        F[3:6, 6:9] = -R @ skew(accel_corr) * dt          # dv/dtheta
        F[3:6, 12:15] = -R * dt                           # dv/dba
        F[6:9, 6:9] = np.eye(3) - skew(gyro_corr) * dt    # dtheta/dtheta
        F[6:9, 9:12] = -np.eye(3) * dt                    # dtheta/dbg

        # Process noise covariance Q
        Q = np.zeros((N_ERR, N_ERR))
        Q[3:6, 3:6] = np.eye(3) * (self.sigma_accel ** 2) * dt   # velocity
        Q[6:9, 6:9] = np.eye(3) * (self.sigma_gyro ** 2) * dt    # orientation
        Q[9:12, 9:12] = np.eye(3) * (self.sigma_bg ** 2) * dt    # gyro bias walk
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_ba ** 2) * dt  # accel bias walk
        Q[15, 15] = (self.sigma_scale ** 2) * dt                  # scale random walk

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    def _vision_update_relative(self, dp_vision_body, dR_vision_body, pos_prev, R_prev):
        """Update state with a relative vision pose measurement.

        Args:
            dp_vision_body: (3,) relative translation from vision, in previous IMU body frame.
            dR_vision_body: (3, 3) relative rotation from vision, in previous IMU body frame.
            pos_prev: (3,) saved EKF position at previous vision frame.
            R_prev: (3, 3) saved EKF rotation (body-to-world) at previous vision frame.
        """
        R_est = quat_to_rotmat(self.quat)

        # Vision-predicted absolute pose at current frame (via relative measurement)
        dp_world = R_prev @ dp_vision_body  # relative translation in world frame
        pos_meas = pos_prev + self.scale * dp_world
        R_meas = R_prev @ dR_vision_body

        # Position residual
        dp = pos_meas - self.pos

        # Orientation residual: small-angle from R_meas @ R_est^T
        dR = R_meas @ R_est.T
        dtheta = Rotation.from_matrix(dR).as_rotvec()

        # Measurement residual (6x1)
        z = np.concatenate([dp, dtheta])

        # Observation matrix H (6x16)
        H = np.zeros((6, N_ERR))
        H[0:3, 0:3] = np.eye(3)               # position
        H[3:6, 6:9] = np.eye(3)               # orientation
        H[0:3, 15] = dp_world                  # d(pos_meas)/ds = R_prev @ dp_vision_body

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
        self.scale += dx[15]

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(N_ERR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_noise @ K.T

    def _initialize_from_gravity(self, t0):
        """Initialize orientation from accelerometer (gravity alignment).

        At near-hover, the accelerometer reads specific force ≈ [0, 0, +9.81]
        in FLU body frame. This determines roll and pitch; yaw is set to zero.
        """
        imu_t = self.imu_data[:, 0]
        imu_accel = self.imu_data[:, 1:4]

        # Average accel near t0
        window = 0.1
        mask = np.abs(imu_t - t0) <= window
        if mask.sum() < 3:
            closest = np.argsort(np.abs(imu_t - t0))[:10]
            mask = np.zeros(len(imu_t), dtype=bool)
            mask[closest] = True

        accel_avg = imu_accel[mask].mean(axis=0)

        # In FLU body frame at hover, accel ≈ [0, 0, +9.81] (specific force opposing gravity).
        # Gravity direction in body frame: g_body = -accel / |accel|
        g_body = -accel_avg / np.linalg.norm(accel_avg)

        # We need R (body-to-world) such that R @ g_body = [0, 0, -1] (gravity in FLU world).
        # align_vectors finds R mapping g_body → [0, 0, -1]
        R_init, _ = Rotation.align_vectors([[0, 0, -1]], [g_body])

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = R_init.as_quat()  # [x, y, z, w]
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        logger.info(
            "EKF initialized from gravity: accel_avg=[%.3f, %.3f, %.3f], "
            "g_body=[%.3f, %.3f, %.3f]",
            *accel_avg, *g_body,
        )

    def _extract_relative_imu_pose(self, w2c_prev, w2c_curr):
        """Extract relative IMU pose from two consecutive vision w2c matrices.

        Returns the relative transform in the IMU body frame of the previous frame.

        Args:
            w2c_prev: (4, 4) w2c at frame i-1.
            w2c_curr: (4, 4) w2c at frame i.

        Returns:
            dp_body: (3,) relative translation in previous IMU body frame.
            dR_body: (3, 3) relative rotation in previous IMU body frame.
        """
        # Relative camera pose: camera i expressed in camera i-1's frame
        c2w_prev = np.linalg.inv(w2c_prev)
        T_rel_cam = w2c_prev @ np.linalg.inv(w2c_curr)
        # T_rel_cam maps: camera i world coords → camera i-1 world coords
        # But we want: pose of frame i IN frame i-1's local frame
        # That's: inv(c2w_prev) @ c2w_curr = w2c_prev @ c2w_curr
        # Actually: T_{C_{i-1}, C_i} = inv(T_{W,C_{i-1}}) @ T_{W,C_i}
        #         = w2c_prev @ c2w_curr ... wait that's not right either
        # w2c = T_{C,W}, c2w = T_{W,C}
        # T_{C_{i-1}, C_i} (pose of cam i in cam i-1 frame):
        #   = T_{C_{i-1},W} @ T_{W,C_i} = w2c[i-1] @ c2w[i]
        T_rel_cam = w2c_prev @ np.linalg.inv(w2c_curr)

        # Convert camera relative to IMU relative
        # T_{I_{i-1}, I_i} = T_{I,C} @ T_{C_{i-1}, C_i} @ T_{C,I}
        #                   = T_cam_to_imu @ T_rel_cam @ T_imu_to_cam
        T_rel_imu = self.T_cam_to_imu @ T_rel_cam @ self.T_imu_to_cam

        dp_body = T_rel_imu[:3, 3]
        dR_body = T_rel_imu[:3, :3]

        return dp_body, dR_body

    def fuse(
        self,
        vision_w2c: np.ndarray,
        vision_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Run EKF fusion over the full trajectory using relative vision measurements.

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

        # Initialize from gravity
        self._initialize_from_gravity(vision_timestamps[0])

        # Compute EKF camera pose at frame 0 (for output frame mapping)
        R_init = quat_to_rotmat(self.quat)
        T_IinEKF_0 = np.eye(4)
        T_IinEKF_0[:3, :3] = R_init
        T_IinEKF_0[:3, 3] = self.pos
        T_CinEKF_0 = T_IinEKF_0 @ self.T_cam_to_imu  # c2w in EKF frame
        w2c_ekf_0 = np.linalg.inv(T_CinEKF_0)

        # T_ekf_to_vision maps EKF c2w → vision c2w
        c2w_vision_0 = np.linalg.inv(vision_w2c[0])
        T_ekf_to_vision = c2w_vision_0 @ w2c_ekf_0  # c2w_vision = T_ekf_to_vision @ c2w_ekf

        fused_w2c = np.zeros((N, 4, 4), dtype=np.float64)
        fused_w2c[0] = vision_w2c[0]

        # Save state after each vision update
        pos_prev = self.pos.copy()
        R_prev = quat_to_rotmat(self.quat)

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

            # Extract relative vision pose (in IMU body frame of previous frame)
            dp_vision_body, dR_vision_body = self._extract_relative_imu_pose(
                vision_w2c[vi - 1], vision_w2c[vi]
            )

            # Relative vision measurement update
            self._vision_update_relative(dp_vision_body, dR_vision_body, pos_prev, R_prev)

            # Record fused pose as w2c (mapped to vision frame)
            R_imu = quat_to_rotmat(self.quat)
            T_IinEKF = np.eye(4)
            T_IinEKF[:3, :3] = R_imu
            T_IinEKF[:3, 3] = self.pos
            T_CinEKF = T_IinEKF @ self.T_cam_to_imu  # c2w in EKF frame
            c2w_vision = T_ekf_to_vision @ T_CinEKF   # c2w in vision frame
            fused_w2c[vi] = np.linalg.inv(c2w_vision)

            # Save state for next relative measurement
            pos_prev = self.pos.copy()
            R_prev = R_imu.copy()

            if (vi + 1) % 50 == 0 or vi == N - 1:
                logger.info(
                    "EKF frame %d/%d: pos=[%.3f, %.3f, %.3f] scale=%.3f "
                    "|bg|=%.5f |ba|=%.5f P_s=%.4f",
                    vi + 1, N, *self.pos, self.scale,
                    np.linalg.norm(self.bg), np.linalg.norm(self.ba),
                    self.P[15, 15],
                )

        logger.info("Final scale estimate: %.4f", self.scale)
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
