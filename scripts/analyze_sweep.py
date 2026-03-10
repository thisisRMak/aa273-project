#!/usr/bin/env python3
"""Generate comparison tables and figures from sweep experiment results.

Reads a sweep YAML config, collects metrics from experiment directories,
and produces markdown tables, LaTeX tables, and presentation-ready PNG figures.

All data products are written to a timestamped directory under the experiments
directory (e.g. experiments/sweep_example_sweep_2026-03-09_163000/) along with
a copy of the sweep config for reproducibility.

Usage:
    python scripts/analyze_sweep.py sweeps/example_sweep.yaml
    python scripts/analyze_sweep.py sweeps/example_sweep.yaml --latex
    python scripts/analyze_sweep.py sweeps/example_sweep.yaml -o /path/to/output_dir
"""

import argparse
import glob
import itertools
import json
import re
from pathlib import Path

import numpy as np
import yaml


DEFAULT_EXPERIMENTS_DIR = "/media/admin/data/StanfordMSL/GOGGLES/experiments"


def experiment_dir_name(scene: str, course: str) -> str:
    scene_short = scene.split("/")[0]
    return f"{scene_short}_{course}"


def is_streaming_model(model: str, cfg: dict) -> bool:
    """Check if a model runs in streaming mode (processes all frames)."""
    return (
        model in ("da3_pairwise", "openvins")
        or (model == "streamvggt" and cfg.get("window_size") is not None)
    )


def metrics_filename(model: str, num_frames: int, cfg: dict) -> str:
    """Return the metrics JSON filename for a model configuration."""
    if is_streaming_model(model, cfg):
        return f"metrics_{model}_all.json"
    return f"metrics_{model}_{num_frames}f.json"


def load_all_metrics(cfg: dict, experiments_dir: str) -> list[dict]:
    """Collect all metrics JSONs matching the sweep config."""
    results = []
    for scene, course, model, n in itertools.product(
        cfg["scenes"], cfg["courses"], cfg["models"], cfg["num_frames"]
    ):
        exp_name = experiment_dir_name(scene, course)
        metrics_file = Path(experiments_dir) / exp_name / metrics_filename(model, n, cfg)
        if metrics_file.is_file():
            with open(metrics_file) as f:
                m = json.load(f)
            results.append({
                "scene": scene.split("/")[0],
                "course": course,
                "model": model,
                "num_frames": n,
                "rot_med": m.get("rotation_error_median_deg"),
                "rot_mean": m.get("rotation_error_mean_deg"),
                "trans_med": m.get("translation_error_median_deg"),
                "trans_mean": m.get("translation_error_mean_deg"),
                "auc5": m.get("auc_at_5"),
                "auc15": m.get("auc_at_15"),
                "auc30": m.get("auc_at_30"),
            })
    return results


def fmt(val, bold=False):
    """Format a float for table display."""
    if val is None:
        return "—"
    s = f"{val:.2f}"
    if bold:
        s = f"**{s}**"
    return s


def fmt_latex(val, bold=False):
    if val is None:
        return "—"
    s = f"{val:.2f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

HIGHER_IS_BETTER = {"auc5", "auc15", "auc30"}


def pivot_by_scene_and_model(results: list[dict], metric_key: str,
                             metric_label: str, latex: bool = False) -> str:
    """Pivot table: rows = (scene, course), columns = models.

    Bold the best value per row (lowest for errors, highest for AUC).
    """
    if not results:
        return f"No results for {metric_label}.\n"

    models = sorted(set(r["model"] for r in results))
    # Build lookup: (scene, course, num_frames, model) -> value
    lookup = {}
    for r in results:
        key = (r["scene"], r["course"], r["num_frames"], r["model"])
        lookup[key] = r[metric_key]

    rows_keys = sorted(set((r["scene"], r["course"], r["num_frames"]) for r in results))

    pick_best = max if metric_key in HIGHER_IS_BETTER else min

    if latex:
        return _pivot_latex(rows_keys, models, lookup, metric_label, pick_best)
    else:
        return _pivot_markdown(rows_keys, models, lookup, metric_label, pick_best)


def _pivot_markdown(rows_keys, models, lookup, metric_label, pick_best) -> str:
    lines = []
    lines.append(f"### {metric_label}")
    lines.append("")

    # Pre-compute all cells to determine column widths
    headers = ["Scene", "Course", "N"] + list(models)
    all_rows = []
    for scene, course, n in rows_keys:
        vals = {m: lookup.get((scene, course, n, m)) for m in models}
        numeric = [v for v in vals.values() if v is not None]
        best = pick_best(numeric) if numeric else None
        cells = [scene, course, str(n)]
        for m in models:
            v = vals[m]
            is_best = (v is not None and best is not None and abs(v - best) < 1e-6)
            cells.append(fmt(v, bold=is_best))
        all_rows.append(cells)

    # Column widths
    widths = [len(h) for h in headers]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell, i):
        return cell.rjust(widths[i]) if i >= 2 else cell.ljust(widths[i])

    # Header + separator
    lines.append("| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |")
    seps = []
    for i, w in enumerate(widths):
        seps.append("-" * (w - 1) + ":" if i >= 2 else "-" * w)
    lines.append("| " + " | ".join(seps) + " |")

    # Data rows
    for row in all_rows:
        lines.append("| " + " | ".join(pad(c, i) for i, c in enumerate(row)) + " |")

    lines.append("")
    return "\n".join(lines)


def _pivot_latex(rows_keys, models, lookup, metric_label, pick_best) -> str:
    n_cols = 3 + len(models)
    lines = []
    lines.append(f"% {metric_label}")
    lines.append(f"\\begin{{tabular}}{{ll r{'r' * len(models)}}}")
    lines.append("\\toprule")
    hdr = "Scene & Course & N & " + " & ".join(models) + " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    for scene, course, n in rows_keys:
        vals = {m: lookup.get((scene, course, n, m)) for m in models}
        numeric = [v for v in vals.values() if v is not None]
        best = pick_best(numeric) if numeric else None

        cells = [scene.replace("_", "\\_"), course.replace("_", "\\_"), str(n)]
        for m in models:
            v = vals[m]
            is_best = (v is not None and best is not None and abs(v - best) < 1e-6)
            cells.append(fmt_latex(v, bold=is_best))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    return "\n".join(lines)


def _aggregate_summary(results: list[dict], group_key: str) -> tuple[list[str], list[tuple], dict]:
    """Aggregate metrics by a grouping key (model, scene, or course).

    Returns (groups, metrics, agg) where agg maps group → {metric: avg_value}.
    """
    groups = sorted(set(r[group_key] for r in results))
    metrics = [
        ("rot_med", "Rot Med (°)"),
        ("rot_mean", "Rot Mean (°)"),
        ("trans_med", "Trans Med (°)"),
        ("trans_mean", "Trans Mean (°)"),
        ("auc5", "AUC@5"),
        ("auc15", "AUC@15"),
    ]
    agg = {}
    for g in groups:
        group_results = [r for r in results if r[group_key] == g]
        row = {}
        for key, _ in metrics:
            vals = [r[key] for r in group_results if r[key] is not None]
            row[key] = sum(vals) / len(vals) if vals else None
            row[f"{key}_n"] = len(vals)
        agg[g] = row
    return groups, metrics, agg


def scene_summary_table(results: list[dict], latex: bool = False) -> str:
    """Aggregate table: one row per scene, averaged across all courses/models."""
    if not results:
        return ""
    groups, metrics, agg = _aggregate_summary(results, "scene")
    title = "Scene Summary (averaged across courses/models)"
    if latex:
        return _summary_latex(groups, metrics, agg, title=title, row_header="Scene")
    return _summary_markdown(groups, metrics, agg, title=title, row_header="Scene")


def course_summary_table(results: list[dict], latex: bool = False) -> str:
    """Aggregate table: one row per course, averaged across all scenes/models."""
    if not results:
        return ""
    groups, metrics, agg = _aggregate_summary(results, "course")
    title = "Course Summary (averaged across scenes/models)"
    if latex:
        return _summary_latex(groups, metrics, agg, title=title, row_header="Course")
    return _summary_markdown(groups, metrics, agg, title=title, row_header="Course")


def model_summary_table(results: list[dict], latex: bool = False) -> str:
    """Aggregate table: one row per model, averaged across all scenes/courses."""
    if not results:
        return ""
    groups, metrics, agg = _aggregate_summary(results, "model")
    title = "Model Summary (averaged across scenes/courses)"
    if latex:
        return _summary_latex(groups, metrics, agg, title=title, row_header="Model")
    return _summary_markdown(groups, metrics, agg, title=title, row_header="Model")


def _summary_markdown(groups, metrics, agg, title="Model Summary", row_header="Model") -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append("")

    # Find best value per metric column (lowest for errors, highest for AUC)
    best_per_metric = {}
    for key, _ in metrics:
        vals = [agg[g][key] for g in groups if agg[g][key] is not None]
        if vals:
            best_per_metric[key] = max(vals) if key in HIGHER_IS_BETTER else min(vals)

    # Pre-compute all cells
    headers = [row_header, "N"] + [label for _, label in metrics]
    all_rows = []
    for g in groups:
        row_data = agg[g]
        n = row_data.get(f"{metrics[0][0]}_n", 0)
        cells = [g, str(n)]
        for key, _ in metrics:
            v = row_data[key]
            is_best = (v is not None and key in best_per_metric
                       and abs(v - best_per_metric[key]) < 1e-6)
            cells.append(fmt(v, bold=is_best))
        all_rows.append(cells)

    # Column widths
    widths = [len(h) for h in headers]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell, i):
        return cell.rjust(widths[i]) if i >= 1 else cell.ljust(widths[i])

    lines.append("| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |")
    seps = []
    for i, w in enumerate(widths):
        seps.append("-" * (w - 1) + ":" if i >= 1 else "-" * w)
    lines.append("| " + " | ".join(seps) + " |")

    for row in all_rows:
        lines.append("| " + " | ".join(pad(c, i) for i, c in enumerate(row)) + " |")

    lines.append("")
    return "\n".join(lines)


def _summary_latex(groups, metrics, agg, title="Model Summary", row_header="Model") -> str:
    lines = []
    lines.append(f"% {title}")
    cols = "l r" + "r" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    hdr = f"{row_header} & N & " + " & ".join(label for _, label in metrics) + " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    for g in groups:
        row_data = agg[g]
        n = row_data.get(f"{metrics[0][0]}_n", 0)
        cells = [g.replace("_", "\\_"), str(n)]
        for key, _ in metrics:
            cells.append(fmt_latex(row_data[key]))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG figure generation
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "streamvggt": "#4A90D9",
    "da3": "#D9534F",
    "da3_chunked": "#5CB85C",
    "da3_pairwise": "#F0AD4E",
    "openvins": "#9B59B6",
}

MODEL_DISPLAY = {
    "streamvggt": "StreamVGGT",
    "da3": "DA3 (batch)",
    "da3_chunked": "DA3 (chunked)",
    "da3_pairwise": "DA3 (pairwise)",
    "openvins": "OpenVINS",
}


def generate_figure(results: list[dict], output_path: str):
    """Create a presentation-ready grouped bar chart PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F8F8",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
    })

    models = sorted(set(r["model"] for r in results))
    # Row labels: short scene name + course
    row_keys = sorted(set((r["scene"], r["course"]) for r in results))
    # Shorten common suffixes for readability
    def short_scene(s):
        return (s.replace("_ssv_exp", "")
                 .replace("_drywall_on", "")
                 .replace("_", " "))
    def short_course(c):
        return c.replace("_along_track", "").replace("_", " ")
    row_labels = [f"{short_scene(s)}\n{short_course(c)}" for s, c in row_keys]

    # For each (scene, course), pick the first num_frames available
    def get_val(scene, course, model, key):
        for r in results:
            if r["scene"] == scene and r["course"] == course and r["model"] == model:
                return r[key]
        return None

    # Two panels: rotation error (median) and translation error (median)
    panels = [
        ("rot_med", "Rotation Error - Median (deg)"),
        ("trans_med", "Translation Error - Median (deg)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.18, top=0.86, left=0.06, right=0.97, wspace=0.25)

    n_rows = len(row_keys)
    n_models = len(models)
    bar_width = 0.7 / n_models
    x = np.arange(n_rows)

    for ax, (metric_key, title) in zip(axes, panels):
        for j, model in enumerate(models):
            vals = []
            for scene, course in row_keys:
                v = get_val(scene, course, model, metric_key)
                vals.append(v if v is not None else 0)

            offset = (j - (n_models - 1) / 2) * bar_width
            color = MODEL_COLORS.get(model, "#888888")
            label = MODEL_DISPLAY.get(model, model)
            bars = ax.bar(x + offset, vals, bar_width * 0.9, label=label,
                          color=color, edgecolor="white", linewidth=0.5, zorder=3)

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                            fontweight="medium", color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(row_labels, ha="center")
        ax.set_title(title)
        ax.set_ylabel("Degrees")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_models,
               frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def _flat_table_markdown(results: list[dict], sort_keys: list[tuple], label: str) -> str:
    """One row per (scene, course, model) with configurable sort order.

    sort_keys: list of (field_name, ascending) tuples defining sort priority.
    """
    lines = []
    lines.append(f"### {label}")
    lines.append("")

    headers = ["Scene", "Course", "Model", "N", "Rot Med (°)", "Trans Med (°)",
               "AUC@5", "AUC@15"]

    def sort_val(r, field, ascending):
        v = r.get(field)
        if v is None:
            return float("inf")
        return v if ascending else -v

    sorted_results = sorted(results, key=lambda r: tuple(
        sort_val(r, field, asc) if field not in ("scene", "course", "model")
        else r.get(field, "")
        for field, asc in sort_keys
    ))

    all_rows = []
    for r in sorted_results:
        cells = [
            r["scene"], r["course"], r["model"], str(r["num_frames"]),
            fmt(r.get("rot_med")), fmt(r.get("trans_med")),
            fmt(r.get("auc5")), fmt(r.get("auc15")),
        ]
        all_rows.append(cells)

    # Column widths
    widths = [len(h) for h in headers]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell, i):
        return cell.rjust(widths[i]) if i >= 3 else cell.ljust(widths[i])

    lines.append("| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |")
    seps = []
    for i, w in enumerate(widths):
        seps.append("-" * (w - 1) + ":" if i >= 3 else "-" * w)
    lines.append("| " + " | ".join(seps) + " |")

    for row in all_rows:
        lines.append("| " + " | ".join(pad(c, i) for i, c in enumerate(row)) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_tables(results: list[dict], latex: bool = False) -> str:
    """Generate all tables as a single string."""
    sections = []

    if not latex:
        sections.append("# Sweep Results\n")

    # --- By Scene: which scenes are hardest? ---
    # scene alpha → course alpha → best rotation error first
    if not latex:
        sections.append("## By Scene\n")
        sections.append("_Which scenes are most challenging? Sorted by scene, then course, "
                        "then best rotation error._\n")
        sections.append(_flat_table_markdown(
            results,
            [("scene", True), ("course", True), ("rot_med", True)],
            "Scene → Course → Performance"))

    # --- By Course: which courses are hardest? ---
    # course alpha → scene alpha → best rotation error first
    if not latex:
        sections.append("## By Course\n")
        sections.append("_Which flight courses are most challenging? Sorted by course, "
                        "then scene, then best rotation error._\n")
        sections.append(_flat_table_markdown(
            results,
            [("course", True), ("scene", True), ("rot_med", True)],
            "Course → Scene → Performance"))

    # --- By Model: which model is best? ---
    # model alpha → best rotation error first
    if not latex:
        sections.append("## By Model\n")
        sections.append("_Which model performs best overall? Sorted by model, "
                        "then best rotation error._\n")
        sections.append(_flat_table_markdown(
            results,
            [("model", True), ("rot_med", True)],
            "Model → Performance"))

    # --- Pivot tables (compact comparison) ---
    sections.append("## Comparison Tables\n")
    sections.append(pivot_by_scene_and_model(
        results, "rot_med", "Rotation Error — Median (°)", latex))
    sections.append(pivot_by_scene_and_model(
        results, "trans_med", "Translation Error — Median (°)", latex))
    sections.append(pivot_by_scene_and_model(
        results, "auc5", "AUC @ 5°", latex))

    # --- Aggregate summaries ---
    sections.append("## Aggregate Summaries\n")
    sections.append(scene_summary_table(results, latex))
    sections.append(course_summary_table(results, latex))
    sections.append(model_summary_table(results, latex))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Per-experiment trajectory comparison plots
# ---------------------------------------------------------------------------

# Regex to extract model name from trajectory npz filename
# e.g. metrics_da3_chunked_70f_trajectory.npz → da3_chunked
#      metrics_streamvggt_70f_trajectory.npz  → streamvggt
#      metrics_openvins_all_trajectory.npz    → openvins
_TRAJ_NPZ_RE = re.compile(r"metrics_(.+?)_(?:\d+f|all)_trajectory\.npz$")


def generate_trajectory_comparisons(cfg: dict, experiments_dir: str, out_dir: Path):
    """For each (scene, course) experiment, overlay all methods on one plot."""
    from goggles.visualization import (
        discover_sparse_pc,
        load_sparse_pointcloud,
        plot_multi_method_trajectories,
    )

    for scene, course in itertools.product(cfg["scenes"], cfg["courses"]):
        exp_name = experiment_dir_name(scene, course)
        exp_dir = Path(experiments_dir) / exp_name

        # Discover all trajectory .npz files
        npz_files = sorted(glob.glob(str(exp_dir / "*_trajectory.npz")))
        if not npz_files:
            continue

        # Load GT and predicted positions from each model.
        # Models may have different GT subsets (e.g. OpenVINS drops early
        # frames before init). Use the longest GT as the canonical one.
        gt_pos = None
        method_positions = {}
        for npz_path in npz_files:
            m = _TRAJ_NPZ_RE.search(Path(npz_path).name)
            if not m:
                continue
            model_key = m.group(1)
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            data = np.load(npz_path)
            method_positions[display_name] = data["pred"]
            # Keep the longest GT (full trajectory from models that use all frames)
            if gt_pos is None or len(data["gt"]) > len(gt_pos):
                gt_pos = data["gt"]

        if not method_positions:
            continue

        # Load point cloud
        transforms_path = str(exp_dir / "transforms.json")
        pc_path = discover_sparse_pc(transforms_path)
        if pc_path:
            pcd_pts, pcd_colors = load_sparse_pointcloud(pc_path)
        else:
            pcd_pts = np.empty((3, 0))
            pcd_colors = None

        # Save into the experiment dir (which lives under the sweep dir)
        plot_path = exp_dir / "comparison_trajectory.png"
        plot_multi_method_trajectories(
            gt_pos, method_positions, pcd_pts, pcd_colors,
            title=f"{exp_name} — all methods",
            output_path=str(plot_path),
        )

    print(f"Trajectory comparisons saved to {experiments_dir}")




def resolve_sweep_dir(arg: str) -> tuple[Path, dict]:
    """Resolve a sweep directory from either a path or a YAML config.

    Accepts:
      - A sweep directory path (contains sweep_config.yaml + experiment subdirs)
      - A YAML config path (finds the latest matching sweep dir under experiments)

    Returns:
        (sweep_dir, cfg) tuple.
    """
    arg_path = Path(arg)

    # Case 1: arg is a sweep directory
    if arg_path.is_dir() and (arg_path / "sweep_config.yaml").is_file():
        with open(arg_path / "sweep_config.yaml") as f:
            cfg = yaml.safe_load(f)
        return arg_path, cfg

    # Case 2: arg is a YAML config — find the latest matching sweep dir
    if arg_path.is_file() and arg_path.suffix in (".yaml", ".yml"):
        with open(arg_path) as f:
            cfg = yaml.safe_load(f)
        base_dir = Path(cfg.get("experiments_dir", DEFAULT_EXPERIMENTS_DIR))
        stem = arg_path.stem
        candidates = sorted(base_dir.glob(f"{stem}_*"), reverse=True)
        # Filter to dirs that contain sweep_config.yaml
        candidates = [d for d in candidates if (d / "sweep_config.yaml").is_file()]
        if candidates:
            print(f"Found {len(candidates)} sweep(s) for '{stem}', using latest: {candidates[0].name}")
            return candidates[0], cfg
        raise FileNotFoundError(
            f"No sweep directories found matching '{stem}_*' under {base_dir}.\n"
            f"Run the sweep first: python scripts/run_sweep.py {arg}"
        )

    raise FileNotFoundError(f"Not a sweep directory or YAML config: {arg}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tables and figures from sweep results.",
        epilog="Accepts either a sweep directory or a YAML config (finds latest sweep).",
    )
    parser.add_argument("sweep", help="Sweep directory or YAML config path.")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX tabular instead of markdown.")
    parser.add_argument("--png", action="store_true",
                        help="Generate presentation-ready PNG figure.")
    args = parser.parse_args()

    sweep_dir, cfg = resolve_sweep_dir(args.sweep)
    experiments_dir = str(sweep_dir)
    results = load_all_metrics(cfg, experiments_dir)

    if not results:
        print("No results found. Run the sweep first:")
        print(f"  python scripts/run_sweep.py {args.sweep}")
        return

    print(f"Collected {len(results)} results from {len(set(r['scene'] for r in results))} scenes, "
          f"{len(set(r['model'] for r in results))} models.\n")

    # Output goes into the sweep directory itself
    out_dir = sweep_dir

    # Always generate markdown tables
    tables = generate_tables(results, latex=args.latex)
    print(tables)

    ext = ".tex" if args.latex else ".md"
    table_path = out_dir / f"results{ext}"
    with open(table_path, "w") as f:
        f.write(tables)
    print(f"Tables written to {table_path}")

    # Always generate PNG figure
    fig_path = out_dir / "results.png"
    generate_figure(results, str(fig_path))

    # Generate per-experiment multi-method trajectory overlays
    generate_trajectory_comparisons(cfg, experiments_dir, out_dir)

    print(f"\nAll data products in: {out_dir}")


if __name__ == "__main__":
    main()
