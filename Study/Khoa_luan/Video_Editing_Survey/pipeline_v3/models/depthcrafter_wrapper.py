"""
DepthCrafter 3D Video Depth Estimation Wrapper for Pipeline v3.
Based on DepthCrafter (Tencent, CVPR 2025 Highlight) and TrajectoryCrafter.
Produces temporally consistent, metric-scale 3D depth maps from monocular video.
Supports automatic Kaggle path discovery, CPU/Model offloading, and compressed NPZ caching.
"""

import os
import gc
import glob
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionOutput,
)
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline,
    StableVideoDiffusionPipelineOutput,
    _resize_with_antialiasing,
    retrieve_timesteps,
)
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.training_utils import set_seed

logger = logging.getLogger("DepthCrafterWrapper")


# ==============================================================================
# 1. Custom UNet Spatio-Temporal Condition Model for DepthCrafter
# ==============================================================================
class DiffusersUNetSpatioTemporalConditionModelDepthCrafter(
    UNetSpatioTemporalConditionModel
):
    """
    Spatio-Temporal Condition UNet fine-tuned specifically for DepthCrafter video depth.
    """

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:

        # 1. Time projection
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float32 if sample.device.type == "mps" else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.conv_in.weight.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1)).to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # 2. Reshape dimensions
        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 1).unsqueeze(1)

        # 3. Pre-process
        sample = sample.to(dtype=self.conv_in.weight.dtype)
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        # 4. Down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        # 5. Mid block
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 6. Up blocks
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

        # 7. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)


# ==============================================================================
# 2. DepthCrafter Pipeline with Window Slicing and Overlap Blending
# ==============================================================================
class DepthCrafterPipeline(StableVideoDiffusionPipeline):
    """
    Inference pipeline for temporally consistent video depth estimation.
    """

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor, chunk_size: int = 14) -> torch.Tensor:
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            tmp = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(tmp).image_embeds)
        return torch.cat(embeddings, dim=0)

    @torch.inference_mode()
    def encode_vae_video(self, video: torch.Tensor, chunk_size: int = 14) -> torch.Tensor:
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        return torch.cat(video_latents, dim=0)

    @torch.no_grad()
    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        window_size: int = 110,
        overlap: int = 25,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = 8,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8

        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = max(window_size - overlap, 1)

        batch_size = 1
        device = self._execution_device
        self._guidance_scale = guidance_scale

        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        video = video.to(device=device, dtype=self.dtype)
        if video.max() <= 1.0:
            video = video * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        video_embeddings = self.encode_video(video, chunk_size=decode_chunk_size).unsqueeze(0)
        torch.cuda.empty_cache()

        noise = randn_tensor(video.shape, generator=generator, device=device, dtype=video.dtype)
        video_perturbed = video + noise_aug_strength * noise

        needs_upcasting = (self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False))
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video_perturbed.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(0)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength, video_embeddings.dtype, batch_size, 1, False
        ).to(device)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )

        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            batch_size,
            window_size,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            generator,
            None,
        )
        latents_all = None

        idx_start = 0
        weights = torch.linspace(0, 1, overlap, device=device).view(1, overlap, 1, 1, 1) if overlap > 0 else None

        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            latents = latents_init[:, : idx_end - idx_start].clone()
            if overlap > 0:
                latents_init = torch.cat(
                    [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
                )

            video_latents_cur = video_latents[:, idx_start:idx_end]
            video_embeds_cur = video_embeddings[:, idx_start:idx_end]

            for i, t in enumerate(timesteps):
                if latents_all is not None and i == 0 and overlap > 0:
                    latents[:, :overlap] = (
                        latents_all[:, -overlap:]
                        + latents[:, :overlap] / self.scheduler.init_noise_sigma * self.scheduler.sigmas[i]
                    )

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([latent_model_input, video_latents_cur], dim=2)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeds_cur,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    latent_model_input_uncond = self.scheduler.scale_model_input(latents, t)
                    latent_model_input_uncond = torch.cat(
                        [latent_model_input_uncond, torch.zeros_like(latent_model_input_uncond)], dim=2
                    )
                    noise_pred_uncond = self.unet(
                        latent_model_input_uncond,
                        t,
                        encoder_hidden_states=torch.zeros_like(video_embeds_cur),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if latents_all is None:
                latents_all = latents.clone()
            else:
                if weights is not None:
                    latents_all[:, -overlap:] = (
                        latents[:, :overlap] * weights + latents_all[:, -overlap:] * (1 - weights)
                    )
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        if output_type != "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents_all

        self.maybe_free_model_hooks()
        return StableVideoDiffusionPipelineOutput(frames=frames)


# ==============================================================================
# 3. Path Discovery for Kaggle (`tranbao0105`) and Local Environments
# ==============================================================================
def resolve_depthcrafter_paths(
    unet_path: Optional[str] = None,
    svd_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Automatically discovers valid paths for DepthCrafter UNet and SVD-XT backbone.
    Checks user-specified paths, standard Kaggle datasets for 'tranbao0105', and local checkpoints.
    """
    resolved_unet = None
    resolved_svd = None

    # Search candidates for UNet
    unet_candidates = [
        unet_path,
        "/kaggle/input/datasets/tranbao0105/depthcrafter",
        "/kaggle/input/depthcrafter",
        "/kaggle/input/datasets/tranbao0105/depthcrafter-weights/DepthCrafter",
        "/kaggle/input/datasets/tranbao0105/depthcrafter/DepthCrafter",
        "./checkpoints/DepthCrafter",
        "checkpoints/DepthCrafter",
        "d:/Study/Khoa_luan/Video_Editing_Survey/checkpoints/DepthCrafter",
        "tencent/DepthCrafter",
    ]

    for c in unet_candidates:
        if not c:
            continue
        if os.path.exists(c):
            # Check if directory contains config.json
            if os.path.exists(os.path.join(c, "config.json")):
                resolved_unet = c
                break
            # Walk subdirectories
            for root, dirs, files in os.walk(c):
                if "config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
                    resolved_unet = root
                    break
            if resolved_unet:
                break
        elif c == "tencent/DepthCrafter":
            resolved_unet = c

    # Search candidates for SVD Backbone
    svd_candidates = [
        svd_path,
        "/kaggle/input/datasets/tranbao0105/stable-video-diffusion-img2vid-xt/stable-video-diffusion-img2vid-xt",
        "/kaggle/input/datasets/tranbao0105/stable-video-diffusion-img2vid-xt",
        "/kaggle/input/stable-video-diffusion-img2vid-xt",
        "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/stable-video-diffusion-img2vid-xt",
        "/kaggle/input/datasets/tranbao0105/depthcrafter/stable-video-diffusion-img2vid-xt",
        "./checkpoints/stable-video-diffusion-img2vid-xt",
        "checkpoints/stable-video-diffusion-img2vid-xt",
        "/kaggle/working/stable-video-diffusion-img2vid-xt",
    ]

    for s in svd_candidates:
        if not s:
            continue
        if os.path.exists(s):
            if os.path.exists(os.path.join(s, "model_index.json")):
                resolved_svd = s
                break
            for root, dirs, files in os.walk(s):
                if "model_index.json" in files:
                    resolved_svd = root
                    break
            if resolved_svd:
                break

    # Dynamic fallback scan across all of /kaggle/input if still not found
    if resolved_svd is None and os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "model_index.json" in files:
                resolved_svd = root
                break

    if resolved_svd is None:
        resolved_svd = "stabilityai/stable-video-diffusion-img2vid-xt"

    return resolved_unet, resolved_svd


# ==============================================================================
# 4. High-Level DepthCrafter Estimator with Caching and VRAM Management
# ==============================================================================
class DepthCrafterEstimator:
    """
    High-level DepthCrafter manager.
    Handles inference, converting disparity to metric distance Z, and caching to .npz.
    """

    def __init__(
        self,
        unet_path: Optional[str] = None,
        svd_path: Optional[str] = None,
        cpu_offload: str = "model",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.cpu_offload = cpu_offload

        resolved_unet, resolved_svd = resolve_depthcrafter_paths(unet_path, svd_path)
        if resolved_unet is None or resolved_svd is None:
            raise FileNotFoundError(
                f"DepthCrafter paths could not be located!\n"
                f"  UNet: {resolved_unet} (Searched: {unet_path})\n"
                f"  SVD Backbone: {resolved_svd} (Searched: {svd_path})\n"
                f"Please ensure /kaggle/input/datasets/tranbao0105/depthcrafter and "
                f"stable-video-diffusion-img2vid-xt are attached."
            )

        print(f"--> [DepthCrafter] Loading UNet from: {resolved_unet}")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            resolved_unet,
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
            local_files_only=os.path.exists(resolved_unet),
        )

        print(f"--> [DepthCrafter] Loading Pipeline Backbone from: {resolved_svd}")
        try:
            self.pipe = DepthCrafterPipeline.from_pretrained(
                resolved_svd,
                unet=unet,
                torch_dtype=self.torch_dtype,
                variant="fp16",
                local_files_only=os.path.exists(resolved_svd),
            )
        except Exception:
            self.pipe = DepthCrafterPipeline.from_pretrained(
                resolved_svd,
                unet=unet,
                torch_dtype=self.torch_dtype,
                local_files_only=os.path.exists(resolved_svd),
            )

        if cpu_offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif cpu_offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe.enable_attention_slicing()
        print("--> [DepthCrafter] Ready for high-fidelity 3D depth estimation.")

    def estimate(
        self,
        frames: Union[str, np.ndarray, torch.Tensor],
        near: float = 0.1,
        far: float = 100.0,
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        window_size: int = 110,
        overlap: int = 25,
        target_size: Optional[Tuple[int, int]] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Estimates metric depth tensor (F, 1, H, W) for input frames.
        """
        set_seed(seed)

        # 1. Parse input frames
        if isinstance(frames, str):
            frames_np = self._read_video_cv2(frames, target_size)
        elif isinstance(frames, torch.Tensor):
            if frames.dim() == 5:
                frames = frames.squeeze(0)
            if frames.shape[1] == 3:  # (F, 3, H, W) -> (F, H, W, 3)
                frames = frames.permute(0, 2, 3, 1)
            frames_np = frames.detach().cpu().numpy().astype(np.float32)
            if frames_np.min() < 0.0:
                frames_np = (frames_np + 1.0) / 2.0
            if frames_np.max() > 1.0:
                frames_np = frames_np / 255.0
        else:
            frames_np = np.asarray(frames).astype(np.float32)
            if frames_np.max() > 1.0:
                frames_np = frames_np / 255.0

        num_f, h, w, _ = frames_np.shape
        # Round height and width to multiples of 64
        proc_h = round(h / 64) * 64
        proc_w = round(w / 64) * 64

        # Resize frames for pipeline
        if proc_h != h or proc_w != w:
            resized_frames = []
            for i in range(num_f):
                resized_frames.append(cv2.resize(frames_np[i], (proc_w, proc_h), interpolation=cv2.INTER_LINEAR))
            pipeline_input = np.stack(resized_frames)
        else:
            pipeline_input = frames_np

        with torch.inference_mode():
            res = self.pipe(
                pipeline_input,
                height=proc_h,
                width=proc_w,
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size,
                overlap=overlap,
            ).frames[0]

        # Convert 3-channel depth to single channel
        res = res.sum(-1) / res.shape[-1]  # (F, proc_h, proc_w)
        # Normalize across video sequence to [0, 1]
        res_min = res.min()
        res_max = res.max()
        depths_norm = (res - res_min) / (res_max - res_min + 1e-8)

        # Convert disparity to metric distance Z using TrajectoryCrafter formula
        depths_t = torch.from_numpy(depths_norm).unsqueeze(1).float()  # (F, 1, proc_h, proc_w)
        depths_scaled = depths_t * 3900.0
        depths_scaled = torch.clamp(depths_scaled, min=1e-5)
        metric_depth = 10000.0 / depths_scaled
        metric_depth = torch.clamp(metric_depth, min=near, max=far)

        # Resize back to original or target size
        out_h, out_w = target_size if target_size else (h, w)
        if metric_depth.shape[-2:] != (out_h, out_w):
            metric_depth = F.interpolate(metric_depth, size=(out_h, out_w), mode="bilinear", align_corners=False)

        return metric_depth

    def infer_and_cache(
        self,
        video_path: str,
        cache_dir: str,
        near: float = 0.1,
        far: float = 100.0,
        target_size: Tuple[int, int] = (384, 672),
    ) -> str:
        """
        Infers depth for a video and saves to compressed NPZ if not already present.
        Returns the absolute path to the cached .npz file.
        """
        os.makedirs(cache_dir, exist_ok=True)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        cache_path = os.path.join(cache_dir, f"{video_stem}_depth.npz")

        if os.path.exists(cache_path):
            print(f"--> [DepthCache] Hit: {cache_path}")
            return cache_path

        print(f"--> [DepthCrafter] Computing depth for {video_path}...")
        depth_tensor = self.estimate(video_path, near=near, far=far, target_size=target_size)
        depth_np = depth_tensor.squeeze(1).cpu().numpy().astype(np.float16)  # fp16 reduces disk usage by 50%

        np.savez_compressed(cache_path, depths=depth_np, depth=depth_np)
        print(f"--> [DepthCache] Saved cache to: {cache_path}")
        return cache_path

    def _read_video_cv2(self, video_path: str, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size:
                frame_rgb = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from video: {video_path}")
        return np.stack(frames).astype(np.float32) / 255.0

    def free_memory(self):
        """Releases DepthCrafter from memory to prepare VRAM for CogVideoX-Fun 5B."""
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
        print("--> [DepthCrafter] VRAM freed successfully.")
