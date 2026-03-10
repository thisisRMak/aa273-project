#!/usr/bin/env python3
"""Convert EuRoC ASL format to run_from_files input format.

EuRoC ASL:
  mav0/imu0/data.csv:  timestamp_ns, wx, wy, wz, ax, ay, az
  mav0/cam0/data.csv:  timestamp_ns, filename

run_from_files:
  imu.csv:             timestamp_s, ax, ay, az, wx, wy, wz
  image_timestamps.csv: timestamp_s, filename
"""

import argparse
import csv
from pathlib import Path


def convert(euroc_dir, output_dir, cam="cam0"):
    euroc_dir = Path(euroc_dir) / "mav0"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMU: reorder columns and convert ns → s
    imu_in = euroc_dir / "imu0" / "data.csv"
    imu_out = output_dir / "imu.csv"
    with open(imu_in) as fin, open(imu_out, "w") as fout:
        fout.write("# timestamp,ax,ay,az,wx,wy,wz\n")
        reader = csv.reader(fin)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts = int(row[0].strip()) * 1e-9
            wx, wy, wz = float(row[1]), float(row[2]), float(row[3])
            ax, ay, az = float(row[4]), float(row[5]), float(row[6])
            fout.write(f"{ts:.9f},{ax:.9f},{ay:.9f},{az:.9f},{wx:.9f},{wy:.9f},{wz:.9f}\n")
    print(f"Wrote {imu_out}")

    # Camera timestamps: convert ns → s
    cam_in = euroc_dir / cam / "data.csv"
    cam_out = output_dir / "image_timestamps.csv"
    with open(cam_in) as fin, open(cam_out, "w") as fout:
        fout.write("# timestamp,filename\n")
        reader = csv.reader(fin)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts = int(row[0].strip()) * 1e-9
            filename = row[1].strip()
            fout.write(f"{ts:.9f},{filename}\n")
    print(f"Wrote {cam_out}")

    image_dir = euroc_dir / cam / "data"
    print(f"Image directory: {image_dir}")
    print(f"\nRun OpenVINS with:")
    print(f"  run_from_files <config.yaml> {imu_out} {image_dir} {cam_out} <output.txt>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EuRoC ASL → run_from_files format")
    parser.add_argument("euroc_dir", help="Path to EuRoC sequence (containing mav0/)")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: euroc_dir/run_from_files/)")
    parser.add_argument("--cam", default="cam0", help="Camera to use (cam0 or cam1)")
    args = parser.parse_args()

    output = args.output or str(Path(args.euroc_dir) / "run_from_files")
    convert(args.euroc_dir, output, args.cam)
