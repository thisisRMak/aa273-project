"""
Linear Probe: Can a linear model predict reconstruction metrics from the camera token?

Tests whether reconstruction quality (PSNR, SSIM, LPIPS, coverage) is linearly
accessible in the StreamVGGT camera token representation. Also tests delta-metric
prediction (marginal value of the next view).

Uses ridge regression with k-fold cross-validation. 260 samples with 2048 features
is highly overparameterized, so regularization is essential.

Probes tested:
1. Camera token (layer 23) -> metric          [is quality encoded?]
2. Camera token (layer 23) -> delta-metric    [is marginal value encoded?]
3. Frame-local half (dims 0:1024) -> metric   [is quality in per-image features?]
4. Global half (dims 1024:2048) -> metric     [is quality in cross-frame features?]
5. View index baseline -> metric              [can a line do this?]

Usage:
    python scripts/linear_probe.py [path_to_h5]
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ── Load data ───────────────────────────────────────────────────────────

def load_h5(path: str) -> dict:
    """Load streaming extraction HDF5 into a flat dict."""
    data = {}
    with h5py.File(path, "r") as f:
        data["tokens"] = f["tokens"][:]
        data["layer_indices"] = f["layer_indices"][:].tolist()
        data["view_order"] = f["view_order"][:]

        if "metrics" in f:
            mg = f["metrics"]
            data["metrics"] = {key: mg[key][:] for key in mg.keys()}
        if "converged" in f:
            cg = f["converged"]
            data["converged"] = {key: cg[key][:] for key in cg.keys()}

        data["streaming"] = f.attrs.get("streaming", False)
        data["scene_name"] = f.attrs.get("scene_name", "unknown")

    return data


def interpolate_metrics_to_views(metrics, n_views):
    """Map eval-step metrics to per-view indices using num_active_views."""
    if "num_active_views" not in metrics:
        return None

    nav = metrics["num_active_views"]
    result = {}
    for key in ["psnr", "ssim", "lpips", "coverage_mean"]:
        if key not in metrics:
            continue
        vals = metrics[key]
        view_indices = np.arange(n_views)
        interp_vals = np.interp(view_indices, nav, vals)
        result[key] = interp_vals
    return result


# ── Probe functions ─────────────────────────────────────────────────────

def run_ridge_probe(X, y, name, cv_folds=5):
    """Run RidgeCV with cross-validated predictions.

    Returns dict with R², MAE, predictions, and the fitted model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = np.logspace(-2, 6, 50)
    ridge = RidgeCV(alphas=alphas, cv=cv_folds)
    ridge.fit(X_scaled, y)

    # Cross-validated predictions (each sample predicted when held out)
    kf = KFold(n_splits=cv_folds, shuffle=False)  # no shuffle — temporal order matters
    y_pred_cv = cross_val_predict(
        RidgeCV(alphas=alphas, cv=3),  # inner CV for alpha selection
        X_scaled, y, cv=kf,
    )

    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)

    return {
        "name": name,
        "r2": r2,
        "mae": mae,
        "y_true": y,
        "y_pred": y_pred_cv,
        "alpha": ridge.alpha_,
        "model": ridge,
        "scaler": scaler,
    }


# ── Plotting ────────────────────────────────────────────────────────────

def plot_probe_results(results_by_metric, save_dir, scene_name):
    """Plot predicted vs actual for each metric, all probe types overlaid."""

    for metric_name, probes in results_by_metric.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left: predicted vs actual scatter
        ax = axes[0]
        colors = ["steelblue", "coral", "green", "purple", "gray"]
        for i, probe in enumerate(probes):
            ax.scatter(
                probe["y_true"], probe["y_pred"],
                alpha=0.4, s=15, color=colors[i % len(colors)],
                label=f'{probe["name"]} (R²={probe["r2"]:.3f})',
            )
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Actual {metric_name}")
        ax.set_ylabel(f"Predicted {metric_name}")
        ax.set_title(f"Linear Probe: {metric_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: predictions over time (view index)
        ax2 = axes[1]
        n = len(probes[0]["y_true"])
        views = np.arange(n)
        ax2.plot(views, probes[0]["y_true"], "k-", linewidth=2, label="Ground truth")
        for i, probe in enumerate(probes):
            ax2.plot(
                views, probe["y_pred"],
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.7,
                label=f'{probe["name"]} (R²={probe["r2"]:.3f})',
            )
        ax2.set_xlabel("View index")
        ax2.set_ylabel(metric_name)
        ax2.set_title(f"Predictions Over Time: {metric_name}")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = save_dir / f"probe_{metric_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def plot_delta_probe(delta_results, save_dir, scene_name):
    """Plot delta-metric probe results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (metric_name, probe) in enumerate(delta_results.items()):
        ax = axes[idx // 2, idx % 2]
        n = len(probe["y_true"])
        views = np.arange(1, n + 1)

        ax.plot(views, probe["y_true"], "k-", linewidth=1.5, alpha=0.5, label="Actual delta")
        ax.plot(views, probe["y_pred"], "coral", linewidth=1.5, alpha=0.7,
                label=f'Predicted (R²={probe["r2"]:.3f})')
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("View index")
        ax.set_ylabel(f"Delta {metric_name}")
        ax.set_title(f"Delta-{metric_name} Probe (MAE={probe['mae']:.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Marginal Value Prediction — {scene_name}", fontsize=14)
    fig.tight_layout()
    path = save_dir / "probe_delta_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = str(
            Path(__file__).resolve().parent.parent
            / "data" / "flightroom_ssv_exp"
            / "2026-02-20_051708_260views_full.h5"
        )

    print(f"Loading: {h5_path}")
    data = load_h5(h5_path)

    tokens = data["tokens"]
    n_frames, n_layers, n_tokens, feat_dim = tokens.shape
    print(f"Tokens: {n_frames} frames, {n_layers} layers, {n_tokens} tokens/frame, {feat_dim}-dim")

    # Output directory
    save_dir = Path(h5_path).parent / "linear_probe"
    save_dir.mkdir(exist_ok=True)

    scene_name = data["scene_name"]

    # Interpolate metrics to per-view
    metrics_per_view = interpolate_metrics_to_views(data["metrics"], n_frames)
    if metrics_per_view is None:
        print("ERROR: No metrics found in HDF5")
        return

    # ── Build feature matrices ──────────────────────────────────────
    # Camera token at layer 23 (last extracted layer)
    cam_token = tokens[:, -1, 0, :].astype(np.float32)     # [N, 2048]
    cam_local = cam_token[:, :1024]                          # [N, 1024] frame-local
    cam_global = cam_token[:, 1024:]                         # [N, 1024] global-temporal
    view_idx = np.arange(n_frames).reshape(-1, 1).astype(np.float32)  # [N, 1] baseline

    feature_sets = {
        "Camera token (full)": cam_token,
        "Frame-local (0:1024)": cam_local,
        "Global (1024:2048)": cam_global,
        "View index (baseline)": view_idx,
    }

    # ── Probe 1: Absolute metric prediction ─────────────────────────
    print("\n" + "=" * 60)
    print("ABSOLUTE METRIC PROBES")
    print("=" * 60)

    results_by_metric = {}

    for metric_name in ["psnr", "ssim", "lpips", "coverage_mean"]:
        y = metrics_per_view[metric_name]
        probes = []

        print(f"\n── {metric_name.upper()} ──")
        for feat_name, X in feature_sets.items():
            result = run_ridge_probe(X, y, feat_name)
            probes.append(result)
            print(f"  {feat_name:30s}  R²={result['r2']:.4f}  MAE={result['mae']:.4f}  alpha={result['alpha']:.1f}")

        results_by_metric[metric_name] = probes

    plot_probe_results(results_by_metric, save_dir, scene_name)

    # ── Probe 2: Delta-metric prediction ────────────────────────────
    print("\n" + "=" * 60)
    print("DELTA-METRIC PROBES (marginal value of next view)")
    print("=" * 60)

    delta_results = {}

    for metric_name in ["psnr", "ssim", "lpips", "coverage_mean"]:
        y = metrics_per_view[metric_name]
        delta_y = np.diff(y)  # [N-1]

        # Use camera token at frame t to predict delta from t to t+1
        X_delta = cam_token[:-1]  # [N-1, 2048]

        result = run_ridge_probe(X_delta, delta_y, f"cam_token -> delta_{metric_name}")
        delta_results[metric_name] = result
        print(f"  delta_{metric_name:20s}  R²={result['r2']:.4f}  MAE={result['mae']:.4f}")

    plot_delta_probe(delta_results, save_dir, scene_name)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nAbsolute metric prediction (R²):")
    print(f"{'':30s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'Coverage':>8s}")
    for feat_name in feature_sets:
        r2s = []
        for metric_name in ["psnr", "ssim", "lpips", "coverage_mean"]:
            for probe in results_by_metric[metric_name]:
                if probe["name"] == feat_name:
                    r2s.append(probe["r2"])
                    break
        print(f"  {feat_name:28s} {r2s[0]:8.4f} {r2s[1]:8.4f} {r2s[2]:8.4f} {r2s[3]:8.4f}")

    print("\nDelta-metric prediction (R²):")
    for metric_name, result in delta_results.items():
        print(f"  delta_{metric_name:20s}  R²={result['r2']:.4f}")

    # Interpret
    best_abs = max(
        (probe["r2"], probe["name"], metric)
        for metric, probes in results_by_metric.items()
        for probe in probes
    )
    print(f"\nBest absolute probe: {best_abs[1]} -> {best_abs[2]} (R²={best_abs[0]:.4f})")

    if best_abs[0] > 0.8:
        print("  -> Strong linear encoding of reconstruction quality")
    elif best_abs[0] > 0.5:
        print("  -> Moderate linear encoding (MLP may improve)")
    else:
        print("  -> Weak linear encoding (information may be nonlinearly encoded)")

    # Check if camera token beats view index baseline
    for metric_name in ["psnr", "coverage_mean"]:
        cam_r2 = next(p["r2"] for p in results_by_metric[metric_name]
                       if p["name"] == "Camera token (full)")
        base_r2 = next(p["r2"] for p in results_by_metric[metric_name]
                        if p["name"] == "View index (baseline)")
        diff = cam_r2 - base_r2
        if diff > 0.05:
            print(f"  Camera token beats view-index baseline on {metric_name} by {diff:.4f} R²")
        else:
            print(f"  Camera token does NOT beat view-index baseline on {metric_name} "
                  f"(diff={diff:.4f}) — metric may just track view count")

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
