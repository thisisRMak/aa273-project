# OpenVINS Initialization Fix for Synthetic Data

## Problem

OpenVINS produced ~47-54 deg median rotation error and 74m of positional drift in a 7m room on synthetic GSplat-rendered images with synthetic IMU data. Visual MSCKF updates provided almost no correction (median 1-3 features per update vs 20-30+ needed).

## Root Cause

The **static initializer** was absorbing the drone's yaw rotation rate as gyro bias.

OpenVINS decides between static and dynamic initialization based on two checks:
1. `init_imu_thresh` — IMU excitation threshold (default 1.5)
2. `init_max_disparity` — pixel disparity threshold (default 200)

If both checks say "stationary," the static initializer runs. It assumes the platform is still and estimates gravity direction + gyro bias from the IMU data. Our drone was rotating at ~0.588 rad/s in yaw, and the static initializer fit that as gyro bias: `bg = [-0.006, 0.024, -0.588]`. With ~34 deg/s of yaw bias baked in, the filter thought there was almost no rotation, poisoning all triangulation and visual updates downstream.

The drone's actual pixel disparity was ~152px, which was below the 200px default threshold — so OpenVINS classified the moving drone as stationary.

## Fix

Two settings in `open_vins/config/flightroom/estimator_config.yaml`:

| Setting | Default | Fixed | Why |
|---------|---------|-------|-----|
| `init_imu_thresh` | 1.5 | **0.0** | Disables static init entirely (drone is always moving) |
| `init_max_disparity` | 200.0 | **10.0** | Detects motion early so dynamic init is triggered |

`init_dyn_use: true` was already set but never triggered because the static initializer kept claiming the platform was stationary.

Everything else (KLT features, chi-squared thresholds, noise parameters, extrinsics, intrinsics) stayed at defaults.

## Results

| Metric | Before (Static Init Bug) | After (Dynamic Init Fix) |
|--------|--------------------------|--------------------------|
| ATE RMSE | 74m drift | 0.063m |
| Rotation Median | 47 deg | 1.17 deg |
| Scale | meaningless | 0.966 |
| Gyro Bias | [-0.006, 0.024, -0.588] | [-0.0007, 0.0000, 0.0017] |

After the fix, OpenVINS results are comparable to published benchmarks on real datasets (EuRoC: ~0.06m ATE).

## What Was Tuned From Defaults

**Only two parameters** (both in `estimator_config.yaml`):

1. **`init_imu_thresh: 0.0`** (default `1.5`) — Disables the static initializer entirely. With the default, OpenVINS checks if IMU excitation is below this threshold to classify the platform as "stationary." Our drone is always moving, but the threshold was permissive enough that it passed, causing the static initializer to absorb the yaw rotation rate (−0.588 rad/s) as gyro bias.

2. **`init_max_disparity: 10.0`** (default `200.0`) — Controls the pixel disparity threshold below which the platform is considered stationary. Our drone had ~152px disparity which was below the 200px default, so it was misclassified as still. Setting it to 10 ensures any real motion triggers dynamic initialization instead.

Both changes force OpenVINS to use `init_dyn_use: true` (dynamic initialization), which was already enabled but never triggered because the static initializer kept claiming the platform was stationary.

Everything else — KLT features, chi-squared thresholds, IMU noise parameters, camera-to-IMU extrinsics, intrinsics — stayed at defaults or previously verified values. The entire problem was initialization, not tracking or filtering.

### Dead Ends (Things That Didn't Help)

| Change | Result | Why It Failed |
|--------|--------|---------------|
| ORB features (use_klt: false) | 57m drift, median 3 feats/update | Still poisoned by bad init |
| Relaxed chi-squared (sigma=2, chi2_mult=5) | 102m drift | Admitted bad observations on top of bad init |
| Near-zero IMU noise (1e-6 to 1e-9) | 53.7° (worse) | EKF became so confident in IMU propagation it ignored visual updates |

## Validation

The OpenVINS build was independently validated on UZH-FPV real drone data: 0.59m RMSE on a 124m trajectory (0.5% error), confirming the build and `run_from_files` driver are correct.
