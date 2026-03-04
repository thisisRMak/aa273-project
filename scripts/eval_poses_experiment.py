#!/usr/bin/env python3
"""End-to-end pose estimation experiment orchestrator.

Chains two stages:
  1. Simulate flight in GSplat scene → video + GT poses
  2. Evaluate StreamVGGT pose predictions against GT

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
"""

import argparse
import logging
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

DEFAULT_EXPERIMENTS_DIR = "/media/admin/data/StanfordMSL/GOGGLES/experiments"
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
        choices=["streamvggt", "da3", "da3_chunked", "da3_pairwise"],
        default="streamvggt",
        help="Pose prediction model (default: streamvggt).",
    )
    parser.add_argument("--da3-model-name", default=None)
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
    # Stage 2: EVALUATE
    # ------------------------------------------------------------------
    # Include model + frame count in output files so runs coexist
    metrics_stem = f"metrics_{args.model}_{args.num_frames}f"
    if skip_idx <= 1:
        eval_cmd = [
            sys.executable, str(eval_script),
            "--transforms", str(exp_dir / "transforms.json"),
            "--model", args.model,
            "-n", str(args.num_frames),
            "-o", str(exp_dir / f"{metrics_stem}.json"),
            "--plot",
            "--visualize",
        ]
        if args.da3_model_name:
            eval_cmd += ["--da3-model-name", args.da3_model_name]
        if args.window_size is not None:
            eval_cmd += ["--window-size", str(args.window_size)]
        if args.model == "da3_chunked":
            eval_cmd += ["--chunk-size", str(args.chunk_size)]
            eval_cmd += ["--overlap", str(args.overlap)]

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
    for name in ["video.mp4", "transforms.json", "sparse_pc_ned.ply",
                  f"{metrics_stem}.json", f"{metrics_stem}.png",
                  f"{metrics_stem}_trajectory.png"]:
        path = exp_dir / name
        status = "OK" if path.exists() else "missing"
        logger.info("  %-25s %s", name, status)


if __name__ == "__main__":
    main()
