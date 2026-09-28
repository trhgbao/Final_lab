"""
Pipeline v3: Complete Camera Trajectory Retargeting Inference Pipeline.
Supports both official SOTA TrajectoryCrafter weights and custom-trained Two-Stage DoRA adapters.
Renders 3D Point Cloud Proxy Scaffold + Inpaints via 33-Channel Dual-Stream Diffusion Transformer.
"""

import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2
import torchvision
import sys
from pathlib import Path

# Ensure pipeline root is in sys.path
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
for _p in [str(_current_dir), str(_parent_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import T5EncoderModel
from diffusers import (
    DDIMScheduler,
    CogVideoXDDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

try:
    from .geometry.warper3d import Warper3D
    from .geometry.trajectory import generate_camera_trajectory
    from .models.crosstransformer3d import CrossTransformer3DModel
    from .models.autoencoder_magvit import AutoencoderKLCogVideoX
    from .models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
    from .models.depthcrafter_wrapper import DepthCrafterEstimator, resolve_depthcrafter_paths
except (ImportError, ValueError):
    try:
        from pipeline_v3.geometry.warper3d import Warper3D
        from pipeline_v3.geometry.trajectory import generate_camera_trajectory
        from pipeline_v3.models.crosstransformer3d import CrossTransformer3DModel
        from pipeline_v3.models.autoencoder_magvit import AutoencoderKLCogVideoX
        from pipeline_v3.models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
        from pipeline_v3.models.depthcrafter_wrapper import DepthCrafterEstimator, resolve_depthcrafter_paths
    except ImportError:
        from geometry.warper3d import Warper3D
        from geometry.trajectory import generate_camera_trajectory
        from models.crosstransformer3d import CrossTransformer3DModel
        from models.autoencoder_magvit import AutoencoderKLCogVideoX
        from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
        try:
            from models.depthcrafter_wrapper import DepthCrafterEstimator, resolve_depthcrafter_paths
        except ImportError:
            DepthCrafterEstimator = None
            resolve_depthcrafter_paths = None



def read_video_frames(video_path: str, process_length: int = 49, stride: int = 1, sample_size=(384, 672), quiet: bool = False):
    if not quiet:
        print(f"==> Reading input video: {video_path}")
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = list(range(0, len(vr), stride))
        if process_length != -1 and process_length < len(indices):
            indices = indices[:process_length]
        elif len(indices) < process_length:
            repeat_factor = (process_length // len(indices)) + 1
            indices = (indices * repeat_factor)[:process_length]
        frames = vr.get_batch(indices).asnumpy().astype(np.float32) / 255.0  # (F, H, W, 3) in [0, 1]
        return frames
    except ImportError:
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
        cap.release()
        indices = list(range(0, len(all_frames), stride))
        if process_length != -1 and process_length < len(indices):
            indices = indices[:process_length]
        elif len(indices) < process_length:
            repeat_factor = (process_length // len(indices)) + 1
            indices = (indices * repeat_factor)[:process_length]
        frames = np.stack([all_frames[i] for i in indices]).astype(np.float32) / 255.0
        return frames


def save_video(tensor_data, save_path: str, fps: int = 10, quiet: bool = False):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if isinstance(tensor_data, torch.Tensor):
        if tensor_data.dtype != torch.uint8:
            tensor_data = torch.clamp(tensor_data, 0.0, 1.0) * 255.0
        arr = tensor_data.detach().cpu().numpy().astype(np.uint8)
    else:
        arr = np.asarray(tensor_data)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    num_frames, height, width, _ = arr.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, float(fps), (width, height))
    for i in range(num_frames):
        frame_bgr = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    # Re-encode to standard web-compatible H.264 (yuv420p) if ffmpeg is available
    import shutil
    import subprocess
    ffmpeg_bin = shutil.which("ffmpeg") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None)
    if ffmpeg_bin:
        temp_path = save_path.replace(".mp4", "_raw.mp4")
        try:
            if os.path.exists(save_path):
                os.replace(save_path, temp_path)
                cmd = [
                    ffmpeg_bin, "-y", "-loglevel", "error",
                    "-i", temp_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    save_path
                ]
                subprocess.run(cmd, check=True)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception:
            if os.path.exists(temp_path) and not os.path.exists(save_path):
                os.replace(temp_path, save_path)

    if not quiet:
        print(f"--> Saved video to {save_path}")



def resolve_dir_with_target(base_dir: str, target_sub: str, target_file: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    if os.path.exists(os.path.join(base_dir, target_sub, target_file)):
        return base_dir
    for root, dirs, files in os.walk(base_dir):
        if target_sub in dirs and os.path.exists(os.path.join(root, target_sub, target_file)):
            return root
    return base_dir


def resolve_transformer_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            return root
    return base_dir


def auto_resolve_model_name(base_dir: str) -> str:
    """Auto-detects CogVideoX-Fun InP weights across Kaggle datasets and local directories."""
    if os.path.exists(base_dir):
        resolved = resolve_dir_with_target(base_dir, "vae", "config.json")
        if os.path.exists(os.path.join(resolved, "vae", "config.json")):
            return resolved

    candidates = [
        "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
        "/kaggle/input/cogvideox-fun-inp",
        "/kaggle/input/cogvideox-fun-v1-1-5b-inp",
        "./checkpoints/CogVideoX-Fun-V1.1-5b-InP",
    ]
    for c in candidates:
        if os.path.exists(c):
            resolved = resolve_dir_with_target(c, "vae", "config.json")
            if os.path.exists(os.path.join(resolved, "vae", "config.json")):
                print(f"[*] [Auto-Detect] Found CogVideoX-Fun InP at: {resolved}")
                return resolved

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "vae" in dirs and os.path.exists(os.path.join(root, "vae", "config.json")):
                print(f"[*] [Auto-Detect Scan] Found CogVideoX-Fun InP at: {root}")
                return root

    return base_dir


def auto_resolve_transformer_path(base_dir: str) -> str:
    """Auto-detects TrajectoryCrafter weights across Kaggle datasets and local directories."""
    if os.path.exists(base_dir):
        resolved = resolve_transformer_dir(base_dir)
        if os.path.exists(os.path.join(resolved, "config.json")):
            return resolved

    candidates = [
        "/kaggle/input/datasets/tranbao0105/trajectorycrafter-weights/TrajectoryCrafter",
        "/kaggle/input/datasets/nguynngcnhntrng/trajectorycrafter",
        "/kaggle/input/trajectorycrafter",
        "./checkpoints/TrajectoryCrafter",
    ]
    for c in candidates:
        if os.path.exists(c):
            resolved = resolve_transformer_dir(c)
            if os.path.exists(os.path.join(resolved, "config.json")):
                print(f"[*] [Auto-Detect] Found TrajectoryCrafter weights at: {resolved}")
                return resolved

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "trajectorycrafter" in root.lower() and "config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
                print(f"[*] [Auto-Detect Scan] Found TrajectoryCrafter at: {root}")
                return root

    return base_dir


class PipelineV3:
    def __init__(self, opts):
        self.opts = opts
        self.quiet = getattr(opts, "quiet", False)
        self.device = torch.device(opts.device)
        self.weight_dtype = torch.bfloat16 if opts.dtype == "bf16" else torch.float16

        if self.quiet:
            import warnings
            warnings.filterwarnings("ignore")
            try:
                from diffusers.utils import logging as d_logging
                from transformers import logging as t_logging
                d_logging.set_verbosity_error()
                t_logging.set_verbosity_error()
            except Exception:
                pass
        else:
            print("=" * 80)
            print("         PIPELINE V3: CAMERA TRAJECTORY RETARGETING SYSTEM")
            print("=" * 80)

        # 1. 3D Warper
        self.warper = Warper3D(resolution=tuple(opts.sample_size), device=self.device)

        # 2. Setup Diffusion Pipeline (only if not scaffold_only)
        if not getattr(opts, "scaffold_only", False):
            self._setup_pipeline()

    def _setup_pipeline(self):
        opts = self.opts
        # Auto-resolve nested directory paths
        opts.model_name = auto_resolve_model_name(opts.model_name)
        opts.transformer_path = auto_resolve_transformer_path(opts.transformer_path)

        local_only = os.path.exists(opts.model_name)
        if not self.quiet:
            print(f"--> Loading VAE, Text Encoder & Scheduler from {opts.model_name} (local_files_only={local_only})...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae", local_files_only=local_only
        ).to(self.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=self.weight_dtype, local_files_only=local_only
        )

        scheduler = DDIMScheduler.from_pretrained(opts.model_name, subfolder="scheduler", local_files_only=local_only)

        # 3. Load Transformer (Official SOTA weights or Base + DoRA + Stage 2)
        local_trans_only = os.path.exists(opts.transformer_path)
        if getattr(opts, "use_official_weights", False):
            if not self.quiet:
                print(f"--> Loading CrossTransformer3D from official SOTA baseline: {opts.transformer_path}...")
            transformer = CrossTransformer3DModel.from_pretrained(
                opts.transformer_path, local_files_only=local_trans_only
            ).to(self.weight_dtype)
        else:
            base_trans_path = getattr(opts, "base_transformer_path", None)
            if not base_trans_path and opts.transformer_path and opts.transformer_path != "none" and "trajectorycrafter" not in opts.transformer_path.lower():
                base_trans_path = opts.transformer_path
            if not base_trans_path or not os.path.exists(base_trans_path):
                if os.path.exists(os.path.join(opts.model_name, "transformer")):
                    base_trans_path = os.path.join(opts.model_name, "transformer")
                else:
                    cand_trans = [
                        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer/transformer",
                        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer",
                        "/kaggle/input/cogvideox-fun-transformer/transformer",
                    ]
                    if os.path.exists("/kaggle/input"):
                        for root, dirs, files in os.walk("/kaggle/input"):
                            if "config.json" in files and any(f.endswith(".safetensors") for f in files) and "vae" not in root.lower() and "text_encoder" not in root.lower() and "trajectorycrafter" not in root.lower():
                                cand_trans.append(root)
                    for ct in cand_trans:
                        if os.path.exists(ct):
                            base_trans_path = ct
                            break
            if not self.quiet:
                print(f"--> Loading Base Transformer from {base_trans_path or opts.model_name} with Stage 2 Cross-Attention enabled...")
            transformer = CrossTransformer3DModel.from_pretrained_2d(
                base_trans_path or opts.model_name,
                subfolder=None if base_trans_path else "transformer",
                transformer_additional_kwargs={"is_train_cross": True},
            ).to(self.weight_dtype)

        # Load DoRA checkpoint if provided
        dora_ckpt = getattr(opts, "dora_checkpoint", None)
        if dora_ckpt and os.path.exists(dora_ckpt):
            if not self.quiet:
                print(f"--> Loading trained DoRA checkpoint from {dora_ckpt}...")
            dora_r = getattr(opts, "dora_r", 16)
            dora_alpha = getattr(opts, "dora_alpha", 32.0)
            transformer.enable_dora(r=dora_r, lora_alpha=dora_alpha)
            transformer.load_dora_checkpoint(opts.dora_checkpoint)
            transformer.merge_dora()

        # Load Stage 2 checkpoint if provided
        if getattr(opts, "stage2_checkpoint", None) and os.path.exists(opts.stage2_checkpoint):
            if not self.quiet:
                print(f"--> Loading trained Stage 2 Appearance checkpoint from {opts.stage2_checkpoint}...")
            transformer.load_stage2_checkpoint(opts.stage2_checkpoint)

        # Build Pipeline
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.pipeline = TrajCrafter_Pipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )

        if self.quiet:
            try:
                self.pipeline.set_progress_bar_config(leave=True)
            except Exception:
                pass

        if opts.low_gpu_memory_mode:
            if not self.quiet:
                print("--> Low GPU Memory Mode: Sequential CPU Offloading enabled (16GB VRAM friendly).")
            self.pipeline.enable_sequential_cpu_offload()
        else:
            if not self.quiet:
                print("--> Standard GPU Mode: Model offloaded to RTX 6000 VRAM.")
            try:
                self.pipeline.enable_model_cpu_offload()
            except Exception:
                self.pipeline.to(self.device)

    def run(self, video_path: str, target_pose=(0.0, -30.0, 0.3, 0.0, 0.0)):
        opts = self.opts
        theta, phi, r, x, y = target_pose
        if not self.quiet:
            print(f"\n[Execution] Retargeting Camera: Pitch={theta}°, Pan={phi}°, Distance={r}, dX={x}, dY={y}")

        # 1. Read input frames
        frames = read_video_frames(video_path, opts.video_length, getattr(opts, "stride", 1), tuple(opts.sample_size))
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device) * 2.0 - 1.0  # (F, 3, H, W) in [-1, 1]
        frames_tensor = F.interpolate(frames_tensor, size=opts.sample_size, mode="bilinear", align_corners=False)

        # 2. Get Depth (from file cache, DepthCrafter, or fallback)
        depths_tensor = None
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        cache_dir = getattr(opts, "depth_cache_dir", "/kaggle/working/depth_cache")
        cache_path = os.path.join(cache_dir, f"{video_stem}_depth.npz") if cache_dir else None

        # Check explicit path or cache
        found_cache = None
        depth_path = getattr(opts, "depth_path", None)
        if depth_path and os.path.exists(depth_path):
            found_cache = depth_path
        elif cache_path and os.path.exists(cache_path):
            found_cache = cache_path
        else:
            candidate_dirs = [
                "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
                "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
                "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
                "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache",
                "/kaggle/input/realestate10k-depthcrafter-cache/depth_cache",
                "/kaggle/input/realestate10k-depthcrafter-cache",
                "/kaggle/working/depth_cache",
            ]
            for cd in candidate_dirs:
                target = os.path.join(cd, f"{video_stem}_depth.npz")
                if os.path.exists(target):
                    found_cache = target
                    break
            if not found_cache and os.path.exists("/kaggle/input"):
                for root, dirs, files in os.walk("/kaggle/input"):
                    if f"{video_stem}_depth.npz" in files:
                        found_cache = os.path.join(root, f"{video_stem}_depth.npz")
                        break
                    elif f"{video_stem}.npz" in files:
                        found_cache = os.path.join(root, f"{video_stem}.npz")
                        break

        if found_cache:
            if not self.quiet:
                print(f"--> [DepthCache] Hit: Loading cached depth from {found_cache}...")
            depth_data = np.load(found_cache)
            key = "depths" if "depths" in depth_data else ("depth" if "depth" in depth_data else "arr_0")
            depths_np = depth_data[key][: opts.video_length]
            depths_tensor = torch.from_numpy(depths_np).unsqueeze(1).to(self.device).float()
        elif DepthCrafterEstimator is not None:
            # Run DepthCrafter on the fly
            try:
                if not self.quiet:
                    print(f"--> [DepthCrafter] Estimating real metric depth for {video_path}...")
                depth_estimator = DepthCrafterEstimator(
                    unet_path=getattr(opts, "depthcrafter_unet", None),
                    svd_path=getattr(opts, "depthcrafter_svd", None),
                    cpu_offload="model",
                    device=str(self.device),
                )
                depths_tensor = depth_estimator.estimate(
                    video_path,
                    near=getattr(opts, "near", 0.1),
                    far=getattr(opts, "far", 100.0),
                    target_size=tuple(opts.sample_size),
                )[: opts.video_length].to(self.device)

                # Save cache for future runs
                if cache_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    depth_np = depths_tensor.squeeze(1).cpu().numpy().astype(np.float32)
                    np.savez_compressed(cache_path, depths=depth_np, depth=depth_np)
                    if not self.quiet:
                        print(f"--> [DepthCache] Saved cache to {cache_path}")

                # Free DepthCrafter VRAM
                depth_estimator.free_memory()
            except Exception as e:
                if not self.quiet:
                    print(f"[Warning] DepthCrafter on-the-fly estimation failed: {e}")
                depths_tensor = None

        if depths_tensor is None:
            # Fallback: Normalized geometric depth scaffold
            if not self.quiet:
                print("--> [Notice] Using smooth geometric depth scaffold fallback...")
            h, w = opts.sample_size
            y_grad = np.linspace(1.5, 3.5, h)[:, None]
            base_d = np.tile(y_grad, (1, w))
            depths_tensor = torch.from_numpy(np.tile(base_d[None, None, :, :], (opts.video_length, 1, 1, 1))).to(self.device).float()

            # Cache fallback depth to prevent re-running DepthCrafter on step [2/2] or next tiers
            if cache_path:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    depth_np = depths_tensor.squeeze(1).cpu().numpy().astype(np.float32)
                    np.savez_compressed(cache_path, depths=depth_np, depth=depth_np)
                    if not self.quiet:
                        print(f"--> [DepthCache] Cached fallback depth to {cache_path}")
                except Exception:
                    pass

        depths_tensor = F.interpolate(depths_tensor, size=opts.sample_size, mode="bilinear", align_corners=False)

        # 3. Dynamic Scene Scale & Camera Poses
        center_h = opts.sample_size[0] // 2
        center_w = opts.sample_size[1] // 2
        center_depth = depths_tensor[0, 0, center_h, center_w].item()
        radius_scale = getattr(opts, "radius_scale", 1.0)
        scene_radius = float(min(max(center_depth * radius_scale, 0.5), 5.0))

        f = 500.0
        cx = opts.sample_size[1] / 2.0
        cy = opts.sample_size[0] / 2.0
        K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).repeat(opts.video_length, 1, 1)

        c2w_anchor = torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        pose_s, pose_t = generate_camera_trajectory(
            c2w_anchor, theta=theta, phi=phi, d_r=r * scene_radius, d_x=x, d_y=y, num_frames=opts.video_length, device=self.device
        )

        # 4. 3D Forward Warping Loop
        use_ak = getattr(opts, "use_ak_scaffold", False)
        use_mfs = getattr(opts, "use_mfs_scaffold", False)

        if use_ak:
            if not self.quiet:
                print("--> [AK-Scaffold] Warping with Adaptive Keyframe Occlusion Infilling...")
            try:
                from .geometry.adaptive_scaffold import AdaptiveKeyframeScaffold
            except (ImportError, ValueError):
                try:
                    from pipeline_v3.geometry.adaptive_scaffold import AdaptiveKeyframeScaffold
                except ImportError:
                    from geometry.adaptive_scaffold import AdaptiveKeyframeScaffold

            ak_engine = AdaptiveKeyframeScaffold(
                hole_threshold=getattr(opts, "ak_hole_threshold", 0.05),
                lambda_smooth=getattr(opts, "ak_lambda_smooth", 0.35),
                feather_radius=getattr(opts, "ak_feather_radius", 5),
                device=str(self.device),
                dtype=self.weight_dtype,
            )
            warped_imgs_tensor, masks_tensor = ak_engine.run(
                frames=frames_tensor,
                depths=depths_tensor,
                poses_src=pose_s,
                poses_tgt=pose_t,
                intrinsics=K,
                quiet=self.quiet,
            )
            cond_video = (warped_imgs_tensor + 1.0) / 2.0  # (F, 3, H, W) in [0, 1]
            cond_masks = masks_tensor                      # (F, 1, H, W)
        else:
            if not self.quiet:
                if use_mfs:
                    print("--> [Experimental] Warping with Multi-Frame Priority Scaffold (MFS-Scaffold)...")
                else:
                    print("--> Warping point cloud to target camera trajectory (Default Single-Frame 3D Scaffold)...")

            warped_images = []
            masks = []
            heal_holes = not getattr(opts, "no_heal_holes", False)
            heal_hole_size = getattr(opts, "heal_hole_size", 25)

            for i in tqdm(
                range(opts.video_length),
                desc="MFS 3D Splatting" if use_mfs else "Point Cloud Splatting",
                disable=self.quiet,
            ):
                if use_mfs:
                    w_img, w_mask, _ = self.warper.multi_frame_forward_warp(
                        frames=frames_tensor,
                        depths=depths_tensor,
                        transformation_s=pose_s,
                        transformation_t=pose_t,
                        intrinsic_s=K,
                        intrinsic_t=K,
                        target_idx=i,
                        neighbor_radius=getattr(opts, "neighbor_radius", 3),
                        use_all_frames=not getattr(opts, "no_use_all_frames", False),
                        cull_grazing_angles=not getattr(opts, "no_cull_grazing", False),
                        max_grazing_angle=getattr(opts, "max_grazing_angle", 55.0),
                        filter_depth_edges=not getattr(opts, "no_depth_edge_filter", False),
                        align_depth_scale=not getattr(opts, "no_align_depth_scale", False),
                        clean_mask=opts.clean_mask,
                        heal_holes=heal_holes,
                        max_hole_area=heal_hole_size,
                    )
                else:
                    # Default Legacy: Single-frame 2.5D splatting
                    w_img, w_mask, _ = self.warper.forward_warp(
                        frames_tensor[i : i + 1],
                        None,
                        depths_tensor[i : i + 1],
                        pose_s[i : i + 1],
                        pose_t[i : i + 1],
                        K[0:1],
                        K[i : i + 1],
                        clean_mask=opts.clean_mask,
                        heal_holes=heal_holes,
                        max_hole_area=heal_hole_size,
                        cull_grazing_angles=False,
                        filter_depth_edges=False,
                    )
                warped_images.append(w_img)
                masks.append(w_mask)

            cond_video = (torch.cat(warped_images) + 1.0) / 2.0  # (F, 3, H, W) in [0, 1]
            cond_masks = torch.cat(masks)                        # (F, 1, H, W)

        # Reference frames: Uniformly sampled across entire video (0 to 48) so future objects are visible to Stage 2!
        frames_clean = (frames_tensor.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0  # (1, 3, F, H, W)
        ref_count = 10
        if opts.video_length <= ref_count:
            frames_ref = frames_clean[:, :, :opts.video_length, :, :]
        else:
            ref_indices = torch.linspace(0, opts.video_length - 1, ref_count).long()
            frames_ref = frames_clean[:, :, ref_indices, :, :]

        cond_video_in = cond_video.permute(1, 0, 2, 3).unsqueeze(0)                   # (1, 3, F, H, W)
        cond_masks_in = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0  # 255 for holes

        # Determine output file tag
        tag = getattr(opts, "tag", None)
        if tag is None:
            file_suffix = f"pan_{int(phi)}"
        else:
            file_suffix = str(tag)

        # Save Warped scaffold & Mask
        os.makedirs(opts.out_dir, exist_ok=True)
        render_path = os.path.join(opts.out_dir, f"render_{file_suffix}.mp4")
        mask_path = os.path.join(opts.out_dir, f"mask_{file_suffix}.mp4")
        save_video(cond_video.permute(0, 2, 3, 1), render_path, fps=opts.fps, quiet=self.quiet)
        save_video(cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1), mask_path, fps=opts.fps, quiet=self.quiet)

        if getattr(opts, "scaffold_only", False):
            if not self.quiet:
                print(f"--> [Scaffold-Only] Saved scaffold render to: {render_path}")
            return render_path

        # 5. Diffusion Denoising Inpainting
        if not self.quiet:
            print("--> Generating high-fidelity novel view via Ref-DiT...")
        generator = torch.Generator(device=self.device).manual_seed(opts.seed)

        with torch.no_grad():
            sample = self.pipeline(
                prompt=opts.prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.guidance_scale,
                num_inference_steps=opts.inference_steps,
                video=cond_video_in,
                mask_video=cond_masks_in,
                reference=frames_ref,
                mask_threshold=getattr(opts, "mask_threshold", 0.85),
            ).videos

        # 6. Save final output
        final_video = sample[0].permute(1, 2, 3, 0)  # (F, H, W, 3) in [0, 1]
        output_path = os.path.join(opts.out_dir, f"gen_{file_suffix}.mp4")
        save_video(final_video, output_path, fps=opts.fps, quiet=self.quiet)

        # 7. Side-by-side visualization
        tensor_left = (frames_tensor.permute(0, 2, 3, 1) + 1.0) / 2.0
        tensor_mid = cond_video.permute(0, 2, 3, 1)
        tensor_right = final_video.to(tensor_left.device)

        interval = torch.ones(opts.video_length, opts.sample_size[0], 20, 3, device=tensor_left.device)
        triptych = torch.cat([tensor_left, interval, tensor_mid, interval, tensor_right], dim=2)
        viz_path = os.path.join(opts.out_dir, f"viz_{file_suffix}.mp4")
        save_video(triptych, viz_path, fps=opts.fps, quiet=self.quiet)
        if not self.quiet:
            print(f"\n[Success] Generated Video: {output_path}")
            print(f"[Success] Triptych Comparison: {viz_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Pipeline v3: Dual-Stream Camera Trajectory Retargeting")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out_dir", type=str, default="./outputs_v3", help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Custom tag suffix for output filenames (e.g. old_scaffold_pan_-15)")
    parser.add_argument("--target_pose", nargs=5, type=float, default=[0.0, -30.0, 0.3, 0.0, 0.0], help="<theta phi r x y>")
    parser.add_argument("--video_length", type=int, default=49, help="Number of frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--sample_size", nargs=2, type=int, default=[384, 672], help="Height Width")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--low_gpu_memory_mode", action="store_true", default=False)
    parser.add_argument("--clean_mask", action="store_true", default=True)
    parser.add_argument("--heal_hole_size", type=int, default=25, help="Max pixel area of splatting cracks to heal on proxy (default: 25)")
    parser.add_argument("--no_heal_holes", action="store_true", default=False, help="Disable healing of small splatting cracks")
    parser.add_argument("--mask_threshold", type=float, default=0.85, help="Threshold to binarize latent inpaint mask (default: 0.85)")
    parser.add_argument("--use_mfs_scaffold", action="store_true", default=False, help="Enable experimental Multi-Frame Priority Scaffold (MFS-Scaffold). Default is False (uses legacy 3D scaffold).")
    parser.add_argument("--use_ak_scaffold", action="store_true", default=False, help="Enable Adaptive Keyframe Occlusion Infilling Scaffold (AK-Scaffold).")
    parser.add_argument("--ak_hole_threshold", type=float, default=0.05, help="Hole percentage threshold to trigger global infill (default: 0.05)")
    parser.add_argument("--ak_lambda_smooth", type=float, default=0.35, help="Viterbi DP temporal smoothness penalty (default: 0.35)")
    parser.add_argument("--ak_feather_radius", type=int, default=5, help="Soft boundary feather radius (default: 5)")
    parser.add_argument("--neighbor_radius", type=int, default=3, help="Radius of local temporal window for Tier 1 infilling (default: 3)")
    parser.add_argument("--no_use_all_frames", action="store_true", default=False, help="Disable Tier 2 global video memory infilling from all frames")
    parser.add_argument("--no_cull_grazing", action="store_true", default=False, help="Disable grazing-angle culling of stretched surfaces")
    parser.add_argument("--max_grazing_angle", type=float, default=55.0, help="Max viewing angle in degrees before surface is culled (default: 55.0)")
    parser.add_argument("--no_depth_edge_filter", action="store_true", default=False, help="Disable silhouette edge filtering")
    parser.add_argument("--no_align_depth_scale", action="store_true", default=False, help="Disable disparity least-squares scale alignment")
    parser.add_argument("--scaffold_only", action="store_true", default=False, help="Only compute and save 3D point cloud proxy scaffolds, skipping diffusion inpainting")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress verbose logging, only show progress bar")

    parser.add_argument("--model_name", type=str, default="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP")
    parser.add_argument("--transformer_path", type=str, default="TrajectoryCrafter/TrajectoryCrafter")
    parser.add_argument("--use_official_weights", action="store_true", default=False)
    parser.add_argument("--dora_checkpoint", type=str, default=None)
    parser.add_argument("--dora_r", type=int, default=16)
    parser.add_argument("--dora_alpha", type=float, default=32.0)
    parser.add_argument("--stage2_checkpoint", type=str, default=None, help="Path to Stage 2 Appearance checkpoint")
    parser.add_argument("--depth_path", type=str, default=None)
    parser.add_argument("--depth_cache_dir", type=str, default="/kaggle/working/depth_cache")
    parser.add_argument("--depthcrafter_unet", type=str, default=None, help="Path to DepthCrafter UNet")
    parser.add_argument("--depthcrafter_svd", type=str, default=None, help="Path to SVD-XT backbone")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--radius_scale", type=float, default=1.0)

    parser.add_argument("--prompt", type=str, default="", help="Prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default="blur, distortion, jitter, flickering, low quality")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--inference_steps", type=int, default=30)

    opts = parser.parse_args()
    pipeline = PipelineV3(opts)
    pipeline.run(opts.video_path, target_pose=opts.target_pose)


if __name__ == "__main__":
    main()
