#!/usr/bin/env python3
"""Convert UZH-FPV dataset format to run_from_files input format.

UZH-FPV:
  imu.txt:          # id timestamp wx wy wz ax ay az
  left_images.txt:  # id timestamp img/image_0_N.png

run_from_files:
  imu.csv:             timestamp_s, ax, ay, az, wx, wy, wz
  image_timestamps.csv: timestamp_s, filename
"""

import argparse
from pathlib import Path


def convert(uzh_dir, output_dir, cam="left"):
    uzh_dir = Path(uzh_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMU: reorder from (id, ts, wx, wy, wz, ax, ay, az) → (ts, ax, ay, az, wx, wy, wz)
    imu_in = uzh_dir / "imu.txt"
    imu_out = output_dir / "imu.csv"
    n_imu = 0
    with open(imu_in) as fin, open(imu_out, "w") as fout:
        fout.write("# timestamp,ax,ay,az,wx,wy,wz\n")
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # id, timestamp, wx, wy, wz, ax, ay, az
            ts = parts[1]
            wx, wy, wz = parts[2], parts[3], parts[4]
            ax, ay, az = parts[5], parts[6], parts[7]
            fout.write(f"{ts},{ax},{ay},{az},{wx},{wy},{wz}\n")
            n_imu += 1
    print(f"Wrote {n_imu} IMU samples to {imu_out}")

    # Camera timestamps
    cam_file = uzh_dir / ("left_images.txt" if cam == "left" else "right_images.txt")
    cam_out = output_dir / "image_timestamps.csv"
    n_cam = 0
    with open(cam_file) as fin, open(cam_out, "w") as fout:
        fout.write("# timestamp,filename\n")
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # id, timestamp, img/image_X_N.png
            ts = parts[1]
            filename = parts[2]
            fout.write(f"{ts},{filename}\n")
            n_cam += 1
    print(f"Wrote {n_cam} image timestamps to {cam_out}")

    image_dir = uzh_dir / "img"
    print(f"Image directory: {image_dir}")
    print(f"\nRun OpenVINS with:")
    print(f"  run_from_files <config.yaml> {imu_out} {image_dir} {cam_out} <output.txt>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UZH-FPV → run_from_files format")
    parser.add_argument("uzh_dir", help="Path to UZH-FPV sequence directory")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: uzh_dir/run_from_files/)")
    parser.add_argument("--cam", default="left", choices=["left", "right"], help="Camera to use")
    args = parser.parse_args()

    output = args.output or str(Path(args.uzh_dir) / "run_from_files")
    convert(args.uzh_dir, output, args.cam)
