"""
Precompute Depth Maps Using DepthCrafter for Pipeline v3.
Runs offline across video datasets (e.g. RealEstate10K) and saves compressed .npz files.
This decouples depth estimation from DoRA training, allowing maximum training throughput on Kaggle / RTX 6000.
"""

import os
import glob
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import torch

_current_dir = Path(__file__).resolve().parent
_pipeline_root = _current_dir.parent
for _p in [str(_pipeline_root), str(_pipeline_root.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ..models.depthcrafter_wrapper import DepthCrafterEstimator, resolve_depthcrafter_paths
except (ImportError, ValueError):
    from pipeline_v3.models.depthcrafter_wrapper import DepthCrafterEstimator, resolve_depthcrafter_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DepthCrafter 3D Depth Maps")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing mp4 videos")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/depth_cache", help="Directory to save .npz files")
    parser.add_argument("--unet_path", type=str, default=None, help="Path to DepthCrafter UNet weights")
    parser.add_argument("--svd_path", type=str, default=None, help="Path to SVD-XT backbone")
    parser.add_argument("--max_videos", type=int, default=-1, help="Max videos to process (-1 for all)")
    parser.add_argument("--sample_size", nargs=2, type=int, default=[384, 672], help="Height Width")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cpu_offload", type=str, default="model", choices=["model", "sequential", "none"])
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def precompute():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect videos
    if os.path.isdir(args.video_dir):
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.avi")))
    elif os.path.isfile(args.video_dir):
        video_paths = [args.video_dir]
    else:
        raise FileNotFoundError(f"Video path not found: {args.video_dir}")

    if len(video_paths) == 0:
        raise ValueError(f"No video files found in: {args.video_dir}")

    if args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]

    print("=" * 75)
    print("   PIPELINE V3: OFFLINE DEPTHCRAFTER 3D PRECOMPUTATION")
    print(f"--> Total videos to inspect: {len(video_paths)}")
    print(f"--> Output directory:        {args.output_dir}")
    print(f"--> Target resolution:       {args.sample_size[0]}x{args.sample_size[1]}")
    print("=" * 75)

    # Filter out already computed videos
    pending_videos = []
    for vp in video_paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        cache_path = os.path.join(args.output_dir, f"{stem}_depth.npz")
        if not os.path.exists(cache_path):
            pending_videos.append(vp)

    print(f"--> Already cached: {len(video_paths) - len(pending_videos)} videos.")
    print(f"--> Need computing: {len(pending_videos)} videos.")

    if len(pending_videos) == 0:
        print("--> All videos are already cached! Exiting.")
        return

    # 2. Initialize DepthCrafter
    offload = None if args.cpu_offload == "none" else args.cpu_offload
    estimator = DepthCrafterEstimator(
        unet_path=args.unet_path,
        svd_path=args.svd_path,
        cpu_offload=offload if offload else "model",
        device=args.device,
    )

    # 3. Process videos
    target_res = tuple(args.sample_size)
    success_count = 0

    progress = tqdm(pending_videos, desc="Estimating Depth")
    for vp in progress:
        try:
            cache_file = estimator.infer_and_cache(
                video_path=vp,
                cache_dir=args.output_dir,
                near=args.near,
                far=args.far,
                target_size=target_res,
            )
            success_count += 1
            progress.set_postfix({"saved": os.path.basename(cache_file)})
        except Exception as e:
            print(f"\n[Warning] Failed to estimate depth for {vp}: {e}")

    # 4. Free Memory
    estimator.free_memory()
    print("\n" + "=" * 75)
    print(f"--> [Done] Successfully precomputed {success_count}/{len(pending_videos)} depth maps!")
    print(f"--> Cached files saved to: {args.output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    precompute()
