#!/usr/bin/env python3
"""End-to-end pose estimation experiment orchestrator.

Chains two stages:
  1. Simulate flight in GSplat scene → video + GT poses
  2. Evaluate pose predictions against GT

Supported models: streamvggt, da3, da3_chunked, da3_pairwise, openvins.

OpenVINS runs in a separate Docker container (openvins:rosfree) via the
Docker socket. Before evaluation, it synthesizes IMU data from the saved
tXUd trajectory and calls the OpenVINS binary to produce TUM-format poses.

Usage:
    # Full pipeline
    python scripts/eval_poses_experiment.py \
        --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
        --course circle_toward_center

    # Re-evaluate only (skip simulation)
    python scripts/eval_poses_experiment.py \
        --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
        --course circle_toward_center \
        --skip-to evaluate --num-frames 50

    # OpenVINS benchmark
    python scripts/eval_poses_experiment.py \
        --scene flightroom_ssv_exp/gemsplat/2026-02-27_025654 \
        --course circle_toward_center \
        --skip-to evaluate --model openvins
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

DEFAULT_EXPERIMENTS_DIR = "/workspace/GOGGLES/experiments"
DEFAULT_NUM_FRAMES = 20

STAGES = ["simulate", "evaluate"]


def find_figs_root():
    """Auto-detect FiGS-Standalone root."""
    candidates = [
        Path("/workspace/FiGS-Standalone"),  # Inside GOGGLES Docker container
        Path(__file__).resolve().parent.parent.parent / "FiGS-Standalone",  # Host
    ]
    for c in candidates:
        if (c / "notebooks").is_dir():
            return c
    return None


def run_stage(name, cmd):
    """Run a subprocess stage, raising on failure."""
    logger.info("=" * 60)
    logger.info("STAGE: %s", name)
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Stage '%s' failed with return code %d", name, result.returncode)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pose estimation experiment.",
    )
    parser.add_argument(
        "--scene", required=True,
        help="GSplat model path (relative to 3dgs/workspace/outputs/).",
    )
    parser.add_argument(
        "--course", required=True,
        help="Course config name (from FiGS configs/course/).",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Experiment directory name (default: course name).",
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames for StreamVGGT eval (default: {DEFAULT_NUM_FRAMES}).",
    )
    parser.add_argument(
        "--skip-to", choices=STAGES, default=None,
        help="Resume from a specific stage (skip earlier stages).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run of all stages (ignore checkpoints).",
    )
    parser.add_argument(
        "--figs-root", default=None,
        help="Path to FiGS-Standalone repo (auto-detected if omitted).",
    )
    parser.add_argument(
        "--experiments-dir", default=DEFAULT_EXPERIMENTS_DIR,
        help=f"Base experiments directory (default: {DEFAULT_EXPERIMENTS_DIR}).",
    )
    parser.add_argument(
        "--sparse-pc", default=None,
        help="Path to sparse_pc.ply for trajectory visualization.",
    )
    # Pass-through pose model args
    parser.add_argument(
        "--model",
        choices=["streamvggt", "da3", "da3_chunked", "da3_pairwise", "openvins", "reloc3r"],
        default="streamvggt",
        help="Pose prediction model (default: streamvggt).\n"\
             "reloc3r uses first+last frames as anchors (batch). "
    )
    parser.add_argument("--da3-model-name", default=None)
    parser.add_argument(
        "--img-reso", type=int, default=512, choices=[224, 512],
        help="Reloc3r image resolution (default: 512).",
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Sliding window size for KV-cache trimming (StreamVGGT only).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=60,
        help="Chunk size for da3_chunked model (default: 60).",
    )
    parser.add_argument(
        "--overlap", type=int, default=20,
        help="Overlap between chunks for da3_chunked model (default: 20).",
    )
    # OpenVINS-specific args
    parser.add_argument(
        "--openvins-config", default="/workspace/open_vins/config/flightroom",
        help="OpenVINS config directory (default: flightroom).",
    )
    parser.add_argument(
        "--imu-noise", default="euroc", choices=["euroc", "none"],
        help="IMU noise preset for synthesis (default: euroc).",
    )
    parser.add_argument(
        "--image-rate", type=float, default=None,
        help="Camera frame rate in Hz for image timestamps (default: auto from tXUd).",
    )
    # Pass-through simulation args
    parser.add_argument("--rollout", default="baseline")
    parser.add_argument("--frame", default="carl")
    parser.add_argument("--policy", default="vrmpc_rrt")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    scene_short = args.scene.split("/")[0]  # e.g. "flightroom_ssv_exp"
    experiment_name = args.experiment_name or f"{scene_short}_{args.course}_{timestamp}"
    experiments_dir = Path(args.experiments_dir)
    exp_dir = experiments_dir / experiment_name

    figs_root = Path(args.figs_root) if args.figs_root else find_figs_root()
    if figs_root is None:
        logger.error("Cannot find FiGS-Standalone. Use --figs-root.")
        sys.exit(1)
    logger.info("FiGS root: %s", figs_root)

    simulate_cli = figs_root / "notebooks" / "figs_simulate_flight_CLI.py"
    if not simulate_cli.exists():
        logger.error("Missing: %s", simulate_cli)
        sys.exit(1)

    goggles_root = Path(__file__).resolve().parent.parent
    eval_script = goggles_root / "scripts" / "eval_poses.py"

    # Ensure experiments base directory exists
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Create repo symlink if missing
    repo_link = goggles_root / "experiments"
    if not repo_link.exists():
        repo_link.symlink_to(experiments_dir)
        logger.info("Created symlink: %s -> %s", repo_link, experiments_dir)

    # Determine which stages to run
    skip_idx = STAGES.index(args.skip_to) if args.skip_to else 0

    # ------------------------------------------------------------------
    # Stage 1: SIMULATE
    # ------------------------------------------------------------------
    if skip_idx <= 0:
        transforms_exists = (exp_dir / "transforms.json").is_file()
        if transforms_exists and not args.force:
            logger.info("Simulation outputs exist, skipping (use --force to re-run)")
        else:
            run_stage("SIMULATE", [
                sys.executable, str(simulate_cli),
                "--scene", args.scene,
                "--course", args.course,
                "--output-dir", str(exp_dir),
                "--rollout", args.rollout,
                "--frame", args.frame,
                "--policy", args.policy,
            ])

    # Verify simulation outputs
    if not (exp_dir / "transforms.json").is_file():
        logger.error("No transforms.json in %s. Run simulation first.", exp_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 1.5: OPENVINS PREP (only for --model openvins)
    # ------------------------------------------------------------------
    openvins_tum_path = None
    if args.model == "openvins" and skip_idx <= 1:
        tXUd_path = exp_dir / "tXUd.npy"
        if not tXUd_path.is_file():
            logger.error("No tXUd.npy in %s. Re-run simulation (tXUd saving was added recently).", exp_dir)
            sys.exit(1)

        openvins_dir = exp_dir / "openvins"
        openvins_dir.mkdir(exist_ok=True)
        imu_csv = openvins_dir / "imu.csv"
        image_ts_csv = openvins_dir / "image_timestamps.csv"
        openvins_tum_path = openvins_dir / "poses_tum.txt"

        # 1. Synthesize IMU from tXUd (runs in this container, Python only)
        if not imu_csv.is_file() or args.force:
            run_stage("IMU_SYNTHESIS", [
                sys.executable, "-m", "figs.utilities.imu_synthesizer",
                str(tXUd_path),
                "-o", str(imu_csv),
                "--noise", args.imu_noise,
                "--seed", "42",
            ])
        else:
            logger.info("IMU data exists, skipping synthesis (use --force to re-run)")

        # 2. Generate image timestamps
        if not image_ts_csv.is_file() or args.force:
            import numpy as np
            tXUd = np.load(tXUd_path)
            num_images = len(list((exp_dir / "images").glob("*.png")))
            duration = tXUd[0, -1] - tXUd[0, 0]
            image_rate = args.image_rate or (num_images / duration if duration > 0 else 10.0)
            start_time = tXUd[0, 0]

            from figs.utilities.imu_synthesizer import generate_image_timestamps
            generate_image_timestamps(
                exp_dir / "images", image_ts_csv,
                image_rate=image_rate, start_time=start_time,
            )
            logger.info("Generated image timestamps: %s (%.1f Hz, %d images)",
                         image_ts_csv, image_rate, num_images)
        else:
            logger.info("Image timestamps exist, skipping (use --force to re-run)")

        # 3. Run OpenVINS via Docker (cross-container call)
        #    We use `docker run` with the pre-built openvins:rosfree image.
        #    Path challenge: `-v` source paths must be HOST paths, but we're
        #    inside the GOGGLES container. Solution: copy config into exp_dir
        #    (on the shared data mount, same path in both containers) and
        #    reference everything from there.
        if not openvins_tum_path.is_file() or args.force:
            import shutil

            openvins_binary = "/opt/open_vins/ov_msckf/build/run_from_files"

            # Copy OpenVINS config into experiment dir (shared mount)
            config_dest = openvins_dir / "config"
            if config_dest.exists():
                shutil.rmtree(config_dest)
            shutil.copytree(args.openvins_config, config_dest)
            config_yaml = str(config_dest / "estimator_config.yaml")
            logger.info("Copied OpenVINS config to %s", config_dest)

            # The data dir is mounted at the same host path in both containers
            data_root = os.environ.get("DATA_PATH", "/media/admin/data/StanfordMSL")

            run_stage("OPENVINS", [
                "docker", "run", "--rm",
                "-v", f"{data_root}:{data_root}",
                "openvins:rosfree",
                openvins_binary,
                config_yaml,
                str(imu_csv),
                str(exp_dir / "images"),
                str(image_ts_csv),
                str(openvins_tum_path),
            ])
        else:
            logger.info("OpenVINS poses exist, skipping (use --force to re-run)")

        if not openvins_tum_path.is_file():
            logger.error("OpenVINS did not produce poses at %s", openvins_tum_path)
            sys.exit(1)

        # Check that poses were actually written (not just an empty/header-only file)
        num_poses = sum(1 for line in open(openvins_tum_path)
                        if line.strip() and not line.startswith("#"))
        if num_poses == 0:
            logger.error("OpenVINS wrote 0 poses — initialization likely failed. "
                          "Check init_max_disparity in config.")
            sys.exit(1)
        logger.info("OpenVINS produced %d poses", num_poses)

    # ------------------------------------------------------------------
    # Stage 2: EVALUATE
    # ------------------------------------------------------------------
    # Streaming models process ALL frames; batch models use -n subsampling
    is_streaming = (
        args.model in ("da3_pairwise", "openvins", "reloc3r")
        or (args.model == "streamvggt" and args.window_size is not None)
    )
    if is_streaming:
        metrics_stem = f"metrics_{args.model}_all"
    else:
        metrics_stem = f"metrics_{args.model}_{args.num_frames}f"

    if skip_idx <= 1:
        eval_cmd = [
            sys.executable, str(eval_script),
            "--transforms", str(exp_dir / "transforms.json"),
            "--model", args.model,
            "-o", str(exp_dir / f"{metrics_stem}.json"),
            "--plot",
            "--visualize",
        ]
        if not is_streaming:
            eval_cmd += ["-n", str(args.num_frames)]
        if args.da3_model_name:
            eval_cmd += ["--da3-model-name", args.da3_model_name]
        if args.window_size is not None:
            eval_cmd += ["--window-size", str(args.window_size)]
        if args.model == "da3_chunked":
            eval_cmd += ["--chunk-size", str(args.chunk_size)]
            eval_cmd += ["--overlap", str(args.overlap)]
        if args.model == "openvins" and openvins_tum_path:
            eval_cmd += ["--openvins-trajectory", str(openvins_tum_path)]
        if args.model == "reloc3r":
            eval_cmd += ["--img-reso", str(args.img_reso)]

        # Point cloud: prefer NED-frame sparse_pc_ned.ply (saved by simulate CLI)
        # over COLMAP-frame sparse_pc.ply (wrong coordinate system)
        sparse_pc = args.sparse_pc
        if sparse_pc is None:
            ned_candidate = exp_dir / "sparse_pc_ned.ply"
            if ned_candidate.is_file():
                sparse_pc = str(ned_candidate)
                logger.info("Using NED point cloud: %s", sparse_pc)
        if sparse_pc:
            eval_cmd += ["--sparse-pc", sparse_pc]

        run_stage("EVALUATE", eval_cmd)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE: %s", experiment_name)
    logger.info("=" * 60)
    logger.info("  Directory:   %s", exp_dir)
    output_files = ["video.mp4", "transforms.json", "tXUd.npy", "sparse_pc_ned.ply",
                    f"{metrics_stem}.json", f"{metrics_stem}.png",
                    f"{metrics_stem}_trajectory.png"]
    if args.model == "openvins":
        output_files += ["openvins/imu.csv", "openvins/image_timestamps.csv",
                          "openvins/poses_tum.txt"]
    for name in output_files:
        path = exp_dir / name
        status = "OK" if path.exists() else "missing"
        logger.info("  %-25s %s", name, status)


if __name__ == "__main__":
    main()
