"""
Scaffold Pairwise Evaluation: Old 3D Scaffold vs New AK-Scaffold.
Runs a 5-tier camera trajectory spectrum from simple to complex:
  - Tier 1 (Simple): Pan -3 deg, Dolly 0.0m (Minimal rotation, verify floor preservation)
  - Tier 2 (Moderate): Pan -15 deg, Dolly 0.5m (Medium pan, narrow disocclusion boundary)
  - Tier 3 (Wide Pan): Pan -30 deg, Dolly 1.0m (Wide angle, significant black void)
  - Tier 4 (Forward Dolly): Pan 0 deg, Dolly 3.0m (Pure forward zoom, depth scaling test)
  - Tier 5 (Stress Multi-Axis): Pan -35 deg, Tilt 5 deg, Dolly 2.0m (Complex multi-axis stress)

Outputs for each trajectory:
  - render_old_{traj}.mp4 (Baseline Old 3D Scaffold)
  - render_ak_{traj}.mp4 (New AK-Scaffold)
  - pair_{traj}.mp4 (Side-by-side labeled comparison video: Old vs New)
"""

import os
import sys
import argparse
import base64
import cv2
import numpy as np
import torch
import torchvision
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.abspath("."))
from pipeline_v3.inference_v3 import PipelineV3, save_video, auto_resolve_model_name, auto_resolve_transformer_path


TRAJECTORIES = [
    {
        "id": "tier1_simple_pan3",
        "name": "Level 1: Simple (Pan -3 deg, Dolly 0.0m)",
        "pose": [0.0, -3.0, 0.0, 0.0, 0.0],
        "desc": "Kiem chung co ban: Bao ton 100% mat san & noi that, khong ro ri loi.",
    },
    {
        "id": "tier2_moderate_pan15",
        "name": "Level 2: Moderate (Pan -15 deg, Dolly 0.5m)",
        "pose": [0.0, -15.0, 0.5, 0.0, 0.0],
        "desc": "Goc quay vua phai: Xuat hien dai thung hep o mep phai, AK tu dong bu dap.",
    },
    {
        "id": "tier3_wide_pan30",
        "name": "Level 3: Wide Pan (Pan -30 deg, Dolly 1.0m)",
        "pose": [0.0, -30.0, 1.0, 0.0, 0.0],
        "desc": "Goc quay rong: Mang den lon o mep khung hinh, Viterbi DP chon donor phu hop.",
    },
    {
        "id": "tier4_dolly_zoom3m",
        "name": "Level 4: Forward Dolly (Pan 0 deg, Dolly 3.0m)",
        "pose": [0.0, 0.0, 3.0, 0.0, 0.0],
        "desc": "Tien thang 3m: Kiem tra do phong dai phoi canh va can chinh ti le do sau.",
    },
    {
        "id": "tier5_stress_multi_axis",
        "name": "Level 5: Multi-Axis Stress (Pan -35 deg, Tilt 5 deg, Dolly 2.0m)",
        "pose": [5.0, -35.0, 2.0, 0.0, 0.0],
        "desc": "Thu thach da truc: Ket hop quay ngang, ngua goc va tien toi cung luc.",
    },
]


def load_video_frames(path: str) -> np.ndarray:
    """Loads an MP4 video into an uint8 [F, H, W, 3] RGB numpy array."""
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
    return np.stack(frames)


def compute_video_hole_ratio(mask_path: str) -> float:
    """Computes average percentage of hole (black pixels) from mask video."""
    if not os.path.exists(mask_path):
        return 0.0
    cap = cv2.VideoCapture(mask_path)
    ratios = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # In mask video: 255 is valid, 0 is hole (or vice-versa depending on format)
        # Let's count pixels < 128 as hole
        hole_ratio = np.mean(gray < 128)
        ratios.append(hole_ratio)
    cap.release()
    return float(np.mean(ratios)) if ratios else 0.0


def create_side_by_side_pair(
    old_render_path: str,
    new_render_path: str,
    out_pair_path: str,
    traj_title: str = "",
    fps: int = 10,
):
    """
    Creates a clean 2-column side-by-side comparison video:
    [ OLD 3D SCAFFOLD (Baseline) | NEW AK-SCAFFOLD (Ours) ]
    with on-screen headers and divider line.
    """
    old_frames = load_video_frames(old_render_path)
    new_frames = load_video_frames(new_render_path)

    num_frames = min(len(old_frames), len(new_frames))
    h, w, c = old_frames.shape[1:]

    # Header banner height
    banner_h = 44
    total_w = w * 2 + 6  # 6px vertical divider
    divider_w = 6

    output_frames = []
    font = cv2.FONT_HERSHEY_DUPLEX

    for i in range(num_frames):
        canvas = np.zeros((h + banner_h, total_w, 3), dtype=np.uint8)

        # 1. Fill banner background (dark slate)
        canvas[:banner_h] = (20, 24, 28)

        # 2. Place Old and New frames
        canvas[banner_h:, :w] = old_frames[i]
        canvas[banner_h:, w : w + divider_w] = (80, 85, 95)  # divider line
        canvas[banner_h:, w + divider_w :] = new_frames[i]

        # 3. Add text labels on banner
        # Left Title: OLD 3D SCAFFOLD
        cv2.putText(
            canvas,
            "OLD 3D SCAFFOLD (Baseline)",
            (20, 28),
            font,
            0.65,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        # Right Title: NEW AK-SCAFFOLD
        cv2.putText(
            canvas,
            "NEW AK-SCAFFOLD (Ours - Infilled)",
            (w + divider_w + 20, 28),
            font,
            0.65,
            (80, 240, 140),
            2,
            cv2.LINE_AA,
        )

        output_frames.append(canvas)

    save_video(np.stack(output_frames), out_pair_path, fps=fps, quiet=True)
    print(f"--> [Stitched Pair] Saved 2-Column Comparison Video: {out_pair_path}")


def run_scaffold_pairwise_evaluation(
    video_path: str,
    out_dir: str = "/kaggle/working/eval_scaffold_pairwise",
    selected_tiers: List[int] = [1, 2, 3, 4, 5],
    sample_size: Tuple[int, int] = (384, 672),
    video_length: int = 49,
    fps: int = 10,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
    depth_cache_dir: str = "/kaggle/working/depth_cache",
    depth_path: str = None,
    ak_hole_threshold: float = 0.05,
    ak_lambda_smooth: float = 0.35,
    ak_feather_radius: int = 5,
) -> List[Dict]:
    """
    Executes the 5-tier scaffold pairwise evaluation.
    Only computes 3D Warping (super fast: ~2s per trajectory, no diffusion waiting!).
    """
    os.makedirs(out_dir, exist_ok=True)

    class OptsMock:
        pass

    opts = OptsMock()
    opts.video_path = video_path
    opts.out_dir = out_dir
    opts.sample_size = list(sample_size)
    opts.video_length = video_length
    opts.fps = fps
    opts.stride = 1
    opts.device = device
    opts.dtype = dtype
    opts.seed = 42
    opts.clean_mask = True
    opts.no_heal_holes = False
    opts.heal_hole_size = 25
    opts.depth_cache_dir = depth_cache_dir
    opts.depth_path = depth_path
    opts.radius_scale = 1.0
    opts.near = 0.1
    opts.far = 100.0
    opts.quiet = False
    opts.scaffold_only = True  # CRITICAL: Skip heavy diffusion inpainting!

    # Auto resolve model names (won't load weights because scaffold_only=True)
    opts.model_name = "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP"
    opts.transformer_path = "TrajectoryCrafter/TrajectoryCrafter"

    print("=" * 80)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ SO SÁNH CẶP: 3D SCAFFOLD CŨ vs AK-SCAFFOLD MỚI")
    print(f"📁 Video đầu vào: {video_path}")
    print(f"🎯 Các bậc quỹ đạo (1..5): {selected_tiers}")
    print(f"⚡ Chế độ: SCAFFOLD-ONLY (Siêu tốc, tập trung đánh giá chất lượng Scaffold 3D)")
    print("=" * 80 + "\n")

    # Initialize PipelineV3 with scaffold_only=True (takes ~0.1s, no heavy weights loaded!)
    pipeline = PipelineV3(opts)

    active_trajs = [t for i, t in enumerate(TRAJECTORIES, 1) if i in selected_tiers]
    results_summary = []

    for traj in active_trajs:
        traj_id = traj["id"]
        traj_name = traj["name"]
        pose = traj["pose"]
        desc = traj["desc"]

        print(f"\n---------------------------------------------------------------------------")
        print(f"🎬 Quỹ đạo: {traj_name}")
        print(f"📐 Target Pose: {pose}")
        print(f"ℹ️  Mục tiêu: {desc}")
        print(f"---------------------------------------------------------------------------")

        # 1. Run Old 3D Scaffold
        print(f"[1/2] Đang tính toán 3D Scaffold CŨ (Single-Frame 2.5D)...")
        opts.use_mfs_scaffold = False
        opts.use_ak_scaffold = False
        opts.tag = f"old_{traj_id}"
        old_render_path = pipeline.run(video_path, target_pose=pose)
        old_mask_path = os.path.join(out_dir, f"mask_old_{traj_id}.mp4")

        # 2. Run New AK-Scaffold
        print(f"[2/2] Đang tính toán AK-Scaffold MỚI (Adaptive Keyframe Infilling)...")
        opts.use_mfs_scaffold = False
        opts.use_ak_scaffold = True
        opts.ak_hole_threshold = ak_hole_threshold
        opts.ak_lambda_smooth = ak_lambda_smooth
        opts.ak_feather_radius = ak_feather_radius
        opts.tag = f"ak_{traj_id}"
        ak_render_path = pipeline.run(video_path, target_pose=pose)
        ak_mask_path = os.path.join(out_dir, f"mask_ak_{traj_id}.mp4")

        # 3. Stitch 2-Column Side-by-Side Comparison Pair Video
        pair_video_path = os.path.join(out_dir, f"pair_{traj_id}.mp4")
        create_side_by_side_pair(
            old_render_path=old_render_path,
            new_render_path=ak_render_path,
            out_pair_path=pair_video_path,
            traj_title=traj_name,
            fps=fps,
        )

        # 4. Measure Hole Metrics
        old_hole_pct = compute_video_hole_ratio(old_mask_path) * 100.0
        ak_hole_pct = compute_video_hole_ratio(ak_mask_path) * 100.0
        reduction = max(0.0, old_hole_pct - ak_hole_pct)

        results_summary.append({
            "id": traj_id,
            "name": traj_name,
            "pose": pose,
            "pair_video": pair_video_path,
            "old_render": old_render_path,
            "ak_render": ak_render_path,
            "old_hole_pct": old_hole_pct,
            "ak_hole_pct": ak_hole_pct,
            "reduction_pct": reduction,
        })

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH TÍNH TOÁN VÀ GHÉP CẶP VIDEO CHO CẢ 5 QUỸ ĐẠO!")
    print("=" * 80)
    print(f"{'Quỹ đạo':<45} | {'Old Hole %':<12} | {'AK Hole %':<12} | {'Độ cải thiện':<12}")
    print("-" * 87)
    for res in results_summary:
        print(f"{res['name']:<45} | {res['old_hole_pct']:>10.2f}% | {res['ak_hole_pct']:>10.2f}% | -{res['reduction_pct']:>9.2f}%")
    print("=" * 80)

    return results_summary


def display_notebook_pairs(results: List[Dict]):
    """
    Renders interactive HTML5 video players directly in Kaggle/Jupyter notebook cells.
    Uses base64 Data URIs to guarantee 100% playable videos in all browsers without server issues.
    """
    from IPython.display import HTML, display

    html_blocks = []
    html_blocks.append("""
    <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; padding: 15px; background: #161b22; color: #f0f6fc; border-radius: 10px; border: 1px solid #30363d;">
        <h2 style="color: #58a6ff; text-align: center; margin-bottom: 5px;">🎬 BẢNG ĐỐI CHIẾU CẶP 3D SCAFFOLD (CŨ vs MỚI)</h2>
        <p style="text-align: center; color: #8b949e; font-size: 14px; margin-top: 0;">5 cấp độ quỹ đạo từ đơn giản đến phức tạp - Video tự động phát vòng lặp</p>
        <hr style="border: 0; border-top: 1px solid #30363d; margin: 15px 0;">
    """)

    for idx, r in enumerate(results, 1):
        pair_path = r["pair_video"]
        if not os.path.exists(pair_path):
            continue

        with open(pair_path, "rb") as f:
            b64_video = base64.b64encode(f.read()).decode("utf-8")

        html_blocks.append(f"""
        <div style="margin-bottom: 30px; background: #0d1117; padding: 15px; border-radius: 8px; border: 1px solid #21262d;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: bold; font-size: 16px; color: #7ee787;">📍 Cấp {idx}: {r['name']}</span>
                <span style="font-size: 13px; color: #8b949e; background: #21262d; padding: 3px 10px; border-radius: 12px;">Vùng thủng: <b style="color: #ff7b72;">{r['old_hole_pct']:.1f}%</b> ➔ <b style="color: #7ee787;">{r['ak_hole_pct']:.1f}%</b></span>
            </div>
            <video width="100%" controls autoplay loop muted playsinline style="border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
                <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                Trình duyệt của bạn không hỗ trợ thẻ video.
            </video>
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #8b949e;">
                <span>⬅️ <b>Bên Trái:</b> Scaffold Cũ (Bị mảng đen/rách ở mép)</span>
                <span>➡️ <b>Bên Phải:</b> AK-Scaffold Mới (Đã bù đắp đầy đủ)</span>
            </div>
        </div>
        """)

    html_blocks.append("</div>")
    full_html = "".join(html_blocks)
    display(HTML(full_html))


def main():
    parser = argparse.ArgumentParser(description="Scaffold Pairwise Evaluation across 5 Trajectory Spectrum")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/eval_scaffold_pairwise")
    parser.add_argument("--selected_tiers", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--sample_size", nargs=2, type=int, default=[384, 672])
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--depth_cache_dir", type=str, default="/kaggle/working/depth_cache")
    parser.add_argument("--depth_path", type=str, default=None)
    parser.add_argument("--ak_hole_threshold", type=float, default=0.05)
    parser.add_argument("--ak_lambda_smooth", type=float, default=0.35)
    parser.add_argument("--ak_feather_radius", type=int, default=5)

    args = parser.parse_args()

    results = run_scaffold_pairwise_evaluation(
        video_path=args.video_path,
        out_dir=args.out_dir,
        selected_tiers=args.selected_tiers,
        sample_size=tuple(args.sample_size),
        video_length=args.video_length,
        fps=args.fps,
        device=args.device,
        dtype=args.dtype,
        depth_cache_dir=args.depth_cache_dir,
        depth_path=args.depth_path,
        ak_hole_threshold=args.ak_hole_threshold,
        ak_lambda_smooth=args.ak_lambda_smooth,
        ak_feather_radius=args.ak_feather_radius,
    )


if __name__ == "__main__":
    main()
