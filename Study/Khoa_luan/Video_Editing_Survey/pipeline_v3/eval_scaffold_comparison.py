"""
Evaluation Script: Pairwise Comparison between Old 3D Scaffold vs New MFS-Scaffold.
Runs a structured spectrum of trajectories from Normal to Complex (Stress Case):
  - Tier 1 (Normal): Pan -3 deg, Dolly 0.0m (smooth, minor displacement)
  - Tier 2 (Moderate): Pan -15 deg, Dolly 0.5m (medium rotation, small translation)
  - Tier 3 (Hard): Pan -30 deg, Dolly 1.5m (wide rotation, noticeable forward motion)
  - Tier 4 (Stress - User Failure Case): Pan -15 deg, Dolly 5.0m (extreme forward dolly)
  - Tier 5 (Stress - Extreme Pan): Pan -45 deg, Dolly 2.5m (extreme wide-angle rotation)

For each trajectory, runs:
  1. Old 3D Scaffold (Single-frame 2.5D warping, no grazing culling)
  2. New MFS-Scaffold (Multi-frame priority splatting, grazing angle culling, depth scale alignment)
And creates a 5-column side-by-side comparison video:
  [Input Video | Old Scaffold Render | New MFS Render | Old Generated Video | New MFS Generated Video]
"""

import os
import sys
import argparse
import torch
import torchvision
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.abspath("."))
from pipeline_v3.inference_v3 import PipelineV3, save_video, auto_resolve_model_name, auto_resolve_transformer_path


TRAJECTORIES = [
    {
        "id": "tier1_normal_pan3",
        "name": "Level 1: Normal (Pan -3 deg)",
        "pose": [0.0, -3.0, 0.0, 0.0, 0.0],
    },
    {
        "id": "tier2_moderate_pan15",
        "name": "Level 2: Moderate (Pan -15 deg, Dolly 0.5m)",
        "pose": [0.0, -15.0, 0.5, 0.0, 0.0],
    },
    {
        "id": "tier3_hard_pan30",
        "name": "Level 3: Hard (Pan -30 deg, Dolly 1.5m)",
        "pose": [0.0, -30.0, 1.5, 0.0, 0.0],
    },
    {
        "id": "tier4_stress_dolly5m",
        "name": "Level 4: Stress User Failure Case (Pan -15 deg, Dolly 5.0m)",
        "pose": [0.0, -15.0, 5.0, 0.0, 0.0],
    },
    {
        "id": "tier5_stress_pan45",
        "name": "Level 5: Stress Extreme Pan (Pan -45 deg, Dolly 2.5m)",
        "pose": [0.0, -45.0, 2.5, 0.0, 0.0],
    },
]


def load_video_tensor(path: str) -> torch.Tensor:
    """Loads an MP4 video into a [F, H, W, 3] float tensor in [0, 1]."""
    try:
        vframes, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        if len(vframes) > 0:
            return vframes.float() / 255.0
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Could not read video frames from {path}")
    return torch.from_numpy(np.stack(frames)).float() / 255.0


def create_comparison_grid(
    input_path: str,
    old_render_path: str,
    new_render_path: str,
    old_gen_path: str,
    new_gen_path: str,
    out_grid_path: str,
    fps: int = 10,
):
    """
    Stitches a 5-column video comparison:
    [Input Video | Old Scaffold Render | New MFS Render | Old Gen Video | New MFS Gen Video]
    """
    t_in = load_video_tensor(input_path)
    t_old_ren = load_video_tensor(old_render_path)
    t_new_ren = load_video_tensor(new_render_path)
    t_old_gen = load_video_tensor(old_gen_path)
    t_new_gen = load_video_tensor(new_gen_path)

    f = min(len(t_in), len(t_old_ren), len(t_new_ren), len(t_old_gen), len(t_new_gen))
    t_in = t_in[:f]
    t_old_ren = t_old_ren[:f]
    t_new_ren = t_new_ren[:f]
    t_old_gen = t_old_gen[:f]
    t_new_gen = t_new_gen[:f]

    h, w, c = t_old_gen.shape[1:]

    # Resize input if needed
    if t_in.shape[1:3] != (h, w):
        t_in_resized = torch.nn.functional.interpolate(
            t_in.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
    else:
        t_in_resized = t_in

    divider = torch.ones(f, h, 8, 3, device=t_in.device) * 0.15  # dark grey divider
    grid = torch.cat([t_in_resized, divider, t_old_ren, divider, t_new_ren, divider, t_old_gen, divider, t_new_gen], dim=2)

    save_video(grid, out_grid_path, fps=fps, quiet=True)
    print(f"--> Saved 5-Column Comparison Grid: {out_grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Pairwise 3D Scaffold Comparison: Old vs New across Trajectory Spectrum")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/eval_scaffold_pairwise")
    parser.add_argument("--selected_tiers", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="List of tiers to run (1 to 5)")
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--sample_size", nargs=2, type=int, default=[384, 672])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--low_gpu_memory_mode", action="store_true", default=False)

    parser.add_argument("--model_name", type=str, default="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP")
    parser.add_argument("--transformer_path", type=str, default="TrajectoryCrafter/TrajectoryCrafter")
    parser.add_argument("--use_official_weights", action="store_true", default=False, help="Use official SOTA TrajectoryCrafter weights")
    parser.add_argument("--stage2_checkpoint", type=str, default=None)
    parser.add_argument("--dora_checkpoint", type=str, default=None)
    parser.add_argument("--dora_r", type=int, default=16)
    parser.add_argument("--dora_alpha", type=float, default=32.0)
    parser.add_argument("--depthcrafter_unet", type=str, default=None)
    parser.add_argument("--depthcrafter_svd", type=str, default=None)
    parser.add_argument("--depth_cache_dir", type=str, default="/kaggle/working/depth_cache")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="blur, distortion, jitter, flickering, low quality")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--clean_mask", action="store_true", default=True)
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--depth_path", type=str, default=None, help="Explicit depth npz path")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--mask_threshold", type=float, default=0.85)
    parser.add_argument("--heal_hole_size", type=int, default=25)
    parser.add_argument("--no_heal_holes", action="store_true", default=False)

    opts = parser.parse_args()
    opts.model_name = auto_resolve_model_name(opts.model_name)
    opts.transformer_path = auto_resolve_transformer_path(opts.transformer_path)
    os.makedirs(opts.out_dir, exist_ok=True)

    print("===========================================================================")
    print("🚀 BẮT ĐẦU SO SÁNH THEO CẶP: 3D SCAFFOLD CŨ vs MFS-SCAFFOLD MỚI")
    print(f"📁 Video đầu vào: {opts.video_path}")
    print(f"🎯 Các bậc quỹ đạo cần chạy: {opts.selected_tiers}")
    print(f"📦 Model CogVideoX-Fun InP: {opts.model_name}")
    print(f"📦 Transformer Baseline: {opts.transformer_path}")
    print("===========================================================================\n")

    # Initialize PipelineV3 once (weights stay in GPU memory)
    pipeline = PipelineV3(opts)

    active_trajs = [t for i, t in enumerate(TRAJECTORIES, 1) if i in opts.selected_tiers]

    for traj in active_trajs:
        traj_id = traj["id"]
        traj_name = traj["name"]
        pose = traj["pose"]

        print(f"\n---------------------------------------------------------------------------")
        print(f"🎬 Đang chạy quỹ đạo: {traj_name}")
        print(f"📐 Target Pose: {pose}")
        print(f"---------------------------------------------------------------------------")

        # 1. Run Old 3D Scaffold (Baseline)
        print(f"\n[1/2] Đang chạy với 3D Scaffold CŨ (Single-Frame 2.5D)...")
        opts.use_mfs_scaffold = False
        opts.use_ak_scaffold = False
        opts.tag = f"old_{traj_id}"
        pipeline.run(opts.video_path, target_pose=pose)

        old_render = os.path.join(opts.out_dir, f"render_old_{traj_id}.mp4")
        old_mask = os.path.join(opts.out_dir, f"mask_old_{traj_id}.mp4")
        old_gen = os.path.join(opts.out_dir, f"gen_old_{traj_id}.mp4")

        # 2. Run AK-Scaffold (Adaptive Keyframe Occlusion Infilling)
        print(f"\n[2/2] Đang chạy với AK-Scaffold MỚI (Adaptive Keyframe Occlusion Infilling)...")
        opts.use_mfs_scaffold = False
        opts.use_ak_scaffold = True
        opts.ak_hole_threshold = 0.05
        opts.ak_lambda_smooth = 0.35
        opts.ak_feather_radius = 5
        opts.tag = f"ak_{traj_id}"
        pipeline.run(opts.video_path, target_pose=pose)

        ak_render = os.path.join(opts.out_dir, f"render_ak_{traj_id}.mp4")
        ak_mask = os.path.join(opts.out_dir, f"mask_ak_{traj_id}.mp4")
        ak_gen = os.path.join(opts.out_dir, f"gen_ak_{traj_id}.mp4")

        # 3. Stitch 5-Column Comparison Grid Video
        grid_out = os.path.join(opts.out_dir, f"comparison_pair_{traj_id}.mp4")
        try:
            create_comparison_grid(
                input_path=opts.video_path,
                old_render_path=old_render,
                new_render_path=ak_render,
                old_gen_path=old_gen,
                new_gen_path=ak_gen,
                out_grid_path=grid_out,
                fps=opts.fps,
            )
        except Exception as e:
            print(f"[Warning] Không thể tự động ghép video 5 cột cho {traj_id}: {e}")

    print("\n===========================================================================")
    print(f"🎉 HOÀN TẤT TOÀN BỘ BÀI ĐÁNH GIÁ THEO CẶP!")
    print(f"📁 Thư mục kết quả: {opts.out_dir}")
    print("===========================================================================")


if __name__ == "__main__":
    main()
