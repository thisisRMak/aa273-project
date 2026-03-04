"""GOGGLES: Pre-decoder latent extraction from StreamVGGT."""

from goggles.latent_extractor import (
    LatentExtractor,
    save_latents,
    load_latents,
    DPT_LAYER_INDICES,
    ALL_LAYER_INDICES,
)

from goggles.pose_eval import (
    se3_to_relative_pose_error,
    compute_pose_metrics,
    calculate_auc_np,
)

from goggles.visualization import (
    align_poses_procrustes,
    plot_trajectory_on_pointcloud,
)

from goggles.da3_predictor import DA3PosePredictor
