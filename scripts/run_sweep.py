#!/usr/bin/env python3
"""Run a grid sweep of pose estimation experiments.

Reads a YAML config defining the cross-product of
  scenes × courses × rollouts × models × num_frames
and calls eval_poses_experiment.py for each combination.

Simulation (scene × course) is run once and reused across models/frames.
Results are collected into a single CSV summary table.

Usage:
    python scripts/run_sweep.py sweeps/example_sweep.yaml
    python scripts/run_sweep.py sweeps/example_sweep.yaml --dry-run
    python scripts/run_sweep.py sweeps/example_sweep.yaml --summary-only
"""

import argparse
import csv
import itertools
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep")

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = SCRIPT_DIR / "eval_poses_experiment.py"

DEFAULT_EXPERIMENTS_DIR = "/workspace/GOGGLES/experiments"

# Metric columns for the summary table (order matters for CSV).
METRIC_COLS = [
    "auc_at_5",
    "auc_at_15",
    "auc_at_30",
    "rotation_error_median_deg",
    "translation_error_median_deg",
]


def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Validate required keys
    for key in ("scenes", "courses", "models", "num_frames"):
        if key not in cfg:
            raise ValueError(f"Sweep config missing required key: '{key}'")
        val = cfg[key]
        if not isinstance(val, list) or len(val) == 0:
            raise ValueError(f"'{key}' must be a non-empty list, got: {val}")

    # Optional keys with defaults
    if "rollouts" not in cfg:
        cfg["rollouts"] = ["baseline"]

    return cfg


def experiment_dir_name(scene: str, course: str, rollout: str) -> str:
    """Deterministic directory name for a (scene, course) pair."""
    scene_short = scene.split("/")[0]
    return f"{scene_short}_{course}_{rollout}"


def sweep_dir_name(config_path: str) -> str:
    """Create a timestamped sweep directory name from the config filename.

    e.g. sweeps/example_sweep.yaml → example_sweep_2026-03-10_143000
    """
    stem = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{stem}_{timestamp}"


def is_streaming_model(model: str, cfg: dict) -> bool:
    """Check if a model runs in streaming mode (processes all frames)."""
    base = model[4:] if model.startswith("ekf_") else model
    return (
        base in ("da3_pairwise", "openvins", "reloc3r", "depth_pnp", "da3_metric_pairwise")
        or (base == "streamvggt" and cfg.get("window_size") is not None)
    )


def metrics_filename(model: str, num_frames: int, cfg: dict) -> str:
    """Return the metrics JSON filename for a model configuration."""
    if is_streaming_model(model, cfg):
        return f"metrics_{model}_all.json"
    return f"metrics_{model}_{num_frames}f.json"


def run_single(
    scene: str,
    course: str,
    rollout: str,
    model: str,
    num_frames: int,
    cfg: dict,
    experiments_dir: str,
    dry_run: bool = False,
) -> dict | None:
    """Run one experiment combo. Returns metrics dict or None on failure."""
    exp_name = experiment_dir_name(scene, course, rollout)
    exp_dir = Path(experiments_dir) / exp_name
    metrics_file = exp_dir / metrics_filename(model, num_frames, cfg)

    # Skip if results already exist
    if metrics_file.is_file():
        logger.info("CACHED: %s / %s / %s / %s / %df", scene, course, rollout, model, num_frames)
        with open(metrics_file) as f:
            return json.load(f)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--scene", scene,
        "--course", course,
        "--rollout", rollout,
        "--model", model,
        "-n", str(num_frames),
        "--experiment-name", exp_name,
        "--experiments-dir", experiments_dir,
    ]

    # Optional pass-through args from sweep config
    base = model[4:] if model.startswith("ekf_") else model
    if cfg.get("chunk_size") is not None and base == "da3_chunked":
        cmd += ["--chunk-size", str(cfg["chunk_size"])]
    if cfg.get("overlap") is not None and base == "da3_chunked":
        cmd += ["--overlap", str(cfg["overlap"])]
    if cfg.get("window_size") is not None and base == "streamvggt":
        cmd += ["--window-size", str(cfg["window_size"])]
    if base == "openvins":
        if cfg.get("openvins_config") is not None:
            cmd += ["--openvins-config", str(cfg["openvins_config"])]
    if base == "openvins" or model.startswith("ekf_"):
        if cfg.get("imu_noise") is not None:
            cmd += ["--imu-noise", str(cfg["imu_noise"])]
    for key in ("frame", "policy"):
        if cfg.get(key) is not None:
            cmd += [f"--{key}", str(cfg[key])]

    if dry_run:
        logger.info("DRY-RUN: %s", " ".join(cmd))
        return None

    logger.info("=" * 70)
    logger.info("RUN: scene=%s  course=%s  rollout=%s  model=%s  n=%d", scene, course, rollout, model, num_frames)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 70)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("FAILED (%.0fs): %s / %s / %s / %s / %df",
                      elapsed, scene, course, rollout, model, num_frames)
        return None

    logger.info("DONE (%.0fs): %s / %s / %s / %s / %df",
                 elapsed, scene, course, rollout, model, num_frames)

    if metrics_file.is_file():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def collect_existing_results(cfg: dict, experiments_dir: str) -> list[dict]:
    """Scan experiment directories for existing metrics JSONs."""
    results = []
    for scene, course, rollout, model, n in itertools.product(
        cfg["scenes"], cfg["courses"], cfg["rollouts"], cfg["models"], cfg["num_frames"]
    ):
        exp_name = experiment_dir_name(scene, course, rollout)
        exp_dir = Path(experiments_dir) / exp_name
        metrics_file = exp_dir / metrics_filename(model, n, cfg)
        if metrics_file.is_file():
            with open(metrics_file) as f:
                metrics = json.load(f)
            results.append({
                "scene": scene,
                "course": course,
                "rollout": rollout,
                "model": model,
                "num_frames": n,
                **{k: metrics.get(k) for k in METRIC_COLS},
            })
    return results


def write_summary(results: list[dict], output_path: Path):
    """Write a CSV summary of all results."""
    if not results:
        logger.warning("No results to summarize.")
        return

    fieldnames = ["scene", "course", "rollout", "model", "num_frames"] + METRIC_COLS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("Summary written to %s (%d rows)", output_path, len(results))


def print_summary_table(results: list[dict]):
    """Print a readable summary table to stdout."""
    if not results:
        return

    # Header
    hdr = f"{'scene':>30s}  {'course':>30s}  {'rollout':>14s}  {'model':>14s}  {'n':>3s}"
    for col in METRIC_COLS:
        short = col.replace("_error_median_deg", "_med").replace("auc_at_", "AUC@")
        hdr += f"  {short:>10s}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for row in sorted(results, key=lambda r: (r["scene"], r["course"], r["model"], r["num_frames"])):
        line = f"{row['scene'].split('/')[0]:>30s}  {row['course']:>30s}  {row['rollout']:>14s}  {row['model']:>14s}  {row['num_frames']:>3d}"
        for col in METRIC_COLS:
            val = row.get(col)
            line += f"  {val:>10.4f}" if val is not None else f"  {'—':>10s}"
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description="Run sweep of pose estimation experiments.")
    parser.add_argument("config", help="Path to sweep YAML config.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Collect existing results and print summary (no new runs).")
    parser.add_argument("--experiments-dir", default=None,
                        help=f"Override experiments directory (default from config or {DEFAULT_EXPERIMENTS_DIR}).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the latest sweep matching this config (skip creating a new sweep directory).")
    args = parser.parse_args()

    cfg = load_sweep_config(args.config)
    base_experiments_dir = args.experiments_dir or cfg.get("experiments_dir", DEFAULT_EXPERIMENTS_DIR)

    combos = list(itertools.product(
        cfg["scenes"], cfg["courses"], cfg["rollouts"], cfg["models"], cfg["num_frames"]
    ))
    logger.info("Sweep: %d scenes × %d courses × %d rollouts × %d models × %d frame counts = %d runs",
                len(cfg["scenes"]), len(cfg["courses"]), len(cfg["rollouts"]), len(cfg["models"]),
                len(cfg["num_frames"]), len(combos))

    
    sweep_name = sweep_dir_name(args.config)
    if not args.resume:
        # Create new timestamped directory
        experiments_dir = str(Path(base_experiments_dir) / sweep_name)
        Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Find latest matching sweep directory
        stem = Path(args.config).stem
        candidates = sorted(Path(base_experiments_dir).glob(f"{stem}_*"), reverse=True)
        candidates = [d for d in candidates if d.is_dir() and (d / "sweep_config.yaml").is_file()]
        if candidates:
            experiments_dir = str(candidates[0])
            logger.info("Resuming sweep: %s", experiments_dir)
        else:
            logger.warning("No existing sweep found for '%s', creating new one", stem)
            experiments_dir = str(Path(base_experiments_dir) / sweep_name)
            Path(experiments_dir).mkdir(parents=True, exist_ok=True)

    

    # ---- Snapshot sweep config for reproducibility ----
    config_path = Path(args.config)
    shutil.copy2(config_path, Path(experiments_dir) / "sweep_config.yaml")
    for scene, course, rollout in itertools.product(cfg["scenes"], cfg["courses"], cfg["rollouts"]):
        exp_dir = Path(experiments_dir) / experiment_dir_name(scene, course, rollout)
        exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created %d experiment directories under %s",
                len(cfg["scenes"]) * len(cfg["courses"]) * len(cfg["rollouts"]), sweep_name)

    # ---- Run the sweep ----
    results = []
    n_done, n_fail, n_cached = 0, 0, 0

    # Order: iterate scenes × courses × rollouts first (simulation reuse), then models × frames
    for scene, course, rollout in itertools.product(cfg["scenes"], cfg["courses"], cfg["rollouts"] ):
        for model, num_frames in itertools.product(cfg["models"], cfg["num_frames"]):
            exp_name = experiment_dir_name(scene, course, rollout)
            exp_dir = Path(experiments_dir) / exp_name
            metrics_file = exp_dir / metrics_filename(model, num_frames, cfg)

            was_cached = metrics_file.is_file()
            metrics = run_single(scene, course, rollout, model, num_frames, cfg,
                                 experiments_dir, dry_run=args.dry_run)

            if metrics is not None:
                results.append({
                    "scene": scene,
                    "course": course,
                    "rollout": rollout,
                    "model": model,
                    "num_frames": num_frames,
                    **{k: metrics.get(k) for k in METRIC_COLS},
                })
                if was_cached:
                    n_cached += 1
                else:
                    n_done += 1
            else:
                if not args.dry_run:
                    n_fail += 1

    # ---- Summary ----
    if not args.dry_run:
        logger.info("Sweep complete: %d succeeded, %d cached, %d failed (of %d total)",
                     n_done, n_cached, n_fail, len(combos))
        print_summary_table(results)

        summary_path = Path(experiments_dir) / "sweep_summary.csv"
        write_summary(results, summary_path)

        logger.info("Run analyze_sweep.py for detailed tables and figures:")
        logger.info("  python scripts/analyze_sweep.py %s", experiments_dir)


if __name__ == "__main__":
    main()
