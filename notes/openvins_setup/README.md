# OpenVINS ROS-Free Setup

Everything needed to reproduce the OpenVINS ROS-free build from a fresh `rpng/open_vins` clone.

## Files

| File | Goes where | Purpose |
|------|-----------|---------|
| `Dockerfile.rosfree` | `open_vins/` | Minimal Ubuntu 22.04 build (no ROS) |
| `docker-compose.yml` | `open_vins/` | Container orchestration with data mounts |
| `run_from_files.cpp` | `open_vins/ov_msckf/src/` | Standalone C++ driver (reads CSV, not ROS bags) |
| `ROS1.cmake.patch` | Apply to `open_vins/ov_msckf/cmake/ROS1.cmake` | Adds `run_from_files` build target |
| `config_flightroom/` | `open_vins/config/flightroom/` | Flightroom camera/IMU/filter config |
| `euroc_to_files.py` | `open_vins/scripts/` | Convert EuRoC ASL format to run_from_files input |
| `uzhfpv_to_files.py` | `open_vins/scripts/` | Convert UZH-FPV format to run_from_files input |

## Setup from scratch

```bash
# 1. Clone upstream OpenVINS
cd /path/to/StanfordMSL
git clone https://github.com/rpng/open_vins.git
cd open_vins

# 2. Copy our custom files
cp <GOGGLES>/notes/openvins_setup/Dockerfile.rosfree .
cp <GOGGLES>/notes/openvins_setup/docker-compose.yml .
cp <GOGGLES>/notes/openvins_setup/run_from_files.cpp ov_msckf/src/
cp -r <GOGGLES>/notes/openvins_setup/config_flightroom config/flightroom
cp <GOGGLES>/notes/openvins_setup/euroc_to_files.py scripts/
cp <GOGGLES>/notes/openvins_setup/uzhfpv_to_files.py scripts/

# 3. Apply the CMake patch (adds run_from_files build target)
git apply <GOGGLES>/notes/openvins_setup/ROS1.cmake.patch

# 4. Build the Docker image
docker compose build

# 5. Verify
docker run --rm openvins:rosfree /opt/open_vins/ov_msckf/build/run_from_files --help
```

## What we changed from upstream

1. **`run_from_files.cpp`** — New file. Standalone C++ driver that reads IMU CSV + image timestamps CSV and feeds them to the OpenVINS VioManager without any ROS dependency. Outputs TUM-format trajectory.

2. **`ROS1.cmake` patch** — Adds `run_from_files` as a build target (3 lines: `add_executable`, `target_link_libraries`, `install`).

3. **`Dockerfile.rosfree`** — New file. Builds OpenVINS with `-DENABLE_ROS=OFF` on Ubuntu 22.04 with only C++ dependencies (Eigen3, Ceres, OpenCV, Boost, glog/gflags, SuiteSparse).

4. **`config/flightroom/`** — New directory. Camera intrinsics and extrinsic derived from FiGS `carl.json`, IMU noise from EuRoC ADIS16448 defaults, filter tuned for synthetic drone flight (dynamic initialization forced).

## Key config decisions

See `openvins_initialization_fix.md` in the parent directory for the full debugging story. Summary:

- `init_imu_thresh: 0.0` — Disable static init (drone is always moving)
- `init_max_disparity: 10.0` — Force dynamic init trigger
- Camera-to-IMU extrinsic derived from carl.json through OpenGL→OpenCV and FRD→FLU conversions
- EuRoC noise parameters used as EKF process noise even with zero-noise synthetic IMU (filter needs nonzero process noise to weight visual updates properly)
