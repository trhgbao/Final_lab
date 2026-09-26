"""
Double-Reprojection Self-Supervised Dataset for Pipeline v3.
Generates training pairs (I_warp, M_warp, I_ref, I_target) from monocular videos without requiring multi-view ground truth.
"""

import os
import glob
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
for _p in [str(_current_dir), str(_parent_dir), str(_parent_dir.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ..geometry.warper3d import Warper3D
    from ..geometry.trajectory import generate_camera_trajectory
except (ImportError, ValueError):
    from pipeline_v3.geometry.warper3d import Warper3D
    from pipeline_v3.geometry.trajectory import generate_camera_trajectory


class DoubleReprojectionDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        depth_dir: Optional[str] = None,
        num_frames: int = 49,
        sample_size: Tuple[int, int] = (384, 672),  # (Height, Width)
        stride: int = 1,
        ref_frames_count: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        self.video_paths = []
        for p in video_paths:
            if os.path.isdir(p):
                vids = glob.glob(os.path.join(p, "*.mp4")) + glob.glob(os.path.join(p, "*.avi"))
                if not vids:
                    vids = glob.glob(os.path.join(p, "**", "*.mp4"), recursive=True) + glob.glob(os.path.join(p, "**", "*.avi"), recursive=True)
                self.video_paths.extend(vids)
            elif os.path.isfile(p):
                self.video_paths.append(p)

        self.video_paths = sorted(list(set(self.video_paths)))
        if len(self.video_paths) == 0:
            raise ValueError(f"No video files found in provided paths: {video_paths}")

        self.depth_dir = depth_dir
        self.num_frames = num_frames
        self.sample_size = sample_size
        self.stride = stride
        self.ref_frames_count = ref_frames_count
        self.device = device

        self.warper = Warper3D(resolution=sample_size, device=device)

        # Pre-scan and index all depth cache files recursively
        self.depth_cache_map = {}
        search_dirs = [depth_dir] if depth_dir else []
        search_dirs.extend([
            "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
            "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
            "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache",
            "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
            "/kaggle/input/realestate10k-depthcrafter-cache",
            "/kaggle/input/realestate10k-depthcrafter-cache/depth_cache",
            "/kaggle/working/depth_cache",
        ])
        # Auto-discover any depth_cache or depthcrafter directory in /kaggle/input
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "depth_cache" in dirs:
                    search_dirs.append(os.path.join(root, "depth_cache"))
                for d in dirs:
                    if "depthcrafter" in d.lower() or "depth_cache" in d.lower():
                        search_dirs.append(os.path.join(root, d))

        for s_dir in search_dirs:
            if s_dir and os.path.exists(s_dir):
                for root, dirs, files in os.walk(s_dir):
                    for f in files:
                        if f.endswith(".npz"):
                            clean_stem = f.replace("_depth.npz", "").replace(".npz", "")
                            if clean_stem not in self.depth_cache_map:
                                self.depth_cache_map[clean_stem] = os.path.join(root, f)

        if len(self.depth_cache_map) > 0:
            matched = [v for v in self.video_paths if os.path.splitext(os.path.basename(v))[0] in self.depth_cache_map]
            if len(matched) > 0:
                self.video_paths = sorted(matched)
                print(f"--> [DoubleReprojectionDataset] Found {len(self.depth_cache_map)} depth maps! Filtered training set to {len(self.video_paths)} videos with REAL DepthCrafter 3D depth.")
            else:
                print(f"--> [DoubleReprojectionDataset] Indexed {len(self.depth_cache_map)} depth maps in cache.")

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video_frames(self, video_path: str) -> Tuple[np.ndarray, list]:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_len = len(vr)
            indices = list(range(0, total_len, self.stride))
            if len(indices) < self.num_frames:
                repeat_factor = (self.num_frames // len(indices)) + 1
                indices = (indices * repeat_factor)[: self.num_frames]
            else:
                max_start = len(indices) - self.num_frames
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                indices = indices[start_idx : start_idx + self.num_frames]
            frames = vr.get_batch(indices).asnumpy()  # (F, H, W, 3) in [0, 255]
            return frames, indices
        except Exception:
            cap = cv2.VideoCapture(video_path)
            all_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            indices = list(range(0, len(all_frames), self.stride))
            if len(indices) < self.num_frames:
                repeat_factor = (self.num_frames // len(indices)) + 1
                indices = (indices * repeat_factor)[: self.num_frames]
            else:
                max_start = len(indices) - self.num_frames
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                indices = indices[start_idx : start_idx + self.num_frames]
            frames = np.stack([all_frames[i] for i in indices])
            return frames, indices


    def _get_depth_maps(
        self, video_path: str, frames_shape: Tuple[int, int, int], indices: Optional[list] = None
    ) -> np.ndarray:
        """Loads cached depth maps from DepthCrafter or creates smooth geometric fallback."""
        num_f, h, w = frames_shape
        stem = os.path.splitext(os.path.basename(video_path))[0]

        # Candidate cache search paths
        candidates = []
        if stem in self.depth_cache_map and os.path.exists(self.depth_cache_map[stem]):
            candidates.append(self.depth_cache_map[stem])
        if self.depth_dir:
            candidates.extend([
                os.path.join(self.depth_dir, f"{stem}_depth.npz"),
                os.path.join(self.depth_dir, f"{stem}.npz"),
            ])
        candidates.extend([
            os.path.join("/kaggle/working/depth_cache", f"{stem}_depth.npz"),
            os.path.join("/kaggle/working/depth_cache", f"{stem}.npz"),
            os.path.join("/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache", f"{stem}_depth.npz"),
            os.path.join("/kaggle/input/datasets/tranbao0105/depthcrafter-cache", f"{stem}_depth.npz"),
            os.path.join("/kaggle/input/datasets/tranbao0105/depthcrafter/depth_cache", f"{stem}_depth.npz"),
            os.path.join(os.path.dirname(video_path), f"{stem}_depth.npz"),
        ])

        for c_path in candidates:
            if os.path.exists(c_path):
                try:
                    data = np.load(c_path)
                    for key in ["depths", "depth", "arr_0"]:
                        if key in data:
                            depths = data[key]
                            if depths.ndim == 4 and depths.shape[1] == 1:
                                depths = depths.squeeze(1)
                            if indices is not None and len(depths) > max(indices):
                                depths = depths[indices]
                            elif depths.shape[0] < num_f:
                                repeat = (num_f // depths.shape[0]) + 1
                                depths = np.tile(depths, (repeat, 1, 1))[:num_f]
                            else:
                                depths = depths[:num_f]
                            return depths.astype(np.float32)
                except Exception:
                    pass

        # Fallback: Normalized synthetic depth gradient (foreground nearer, background farther)
        y_grad = np.linspace(1.5, 3.5, h)[:, None]
        base_depth = np.tile(y_grad, (1, w))
        depths = np.tile(base_depth[None, :, :], (num_f, 1, 1)).astype(np.float32)
        return depths

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        raw_frames, frame_indices = self._load_video_frames(video_path)  # (F, H, W, 3)
        depths_np = self._get_depth_maps(video_path, raw_frames.shape[:3], indices=frame_indices)  # (F, H, W)

        # Convert to Tensor: (F, 3, H, W) in [-1, 1]
        frames_tensor = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        depths_tensor = torch.from_numpy(depths_np).unsqueeze(1).float()  # (F, 1, H, W)

        # Resize to sample_size
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor, size=self.sample_size, mode="bilinear", align_corners=False
        )
        depths_tensor = torch.nn.functional.interpolate(
            depths_tensor, size=self.sample_size, mode="bilinear", align_corners=False
        )

        # Intrinsics
        f = 500.0
        cx = self.sample_size[1] / 2.0
        cy = self.sample_size[0] / 2.0
        K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)

        # Target camera poses (Identity anchor)
        c2w_anchor = torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).unsqueeze(0)

        # Random perturbation for Double-Reprojection
        rand_theta = random.uniform(-10.0, 10.0)  # pitch
        rand_phi = random.uniform(-30.0, 30.0)    # yaw / pan
        rand_r = random.uniform(-0.2, 0.2)        # translation

        pose_s, pose_pert = generate_camera_trajectory(
            c2w_anchor, theta=rand_theta, phi=rand_phi, d_r=rand_r, num_frames=self.num_frames, device=self.device
        )

        # Move inputs to self.device for accelerated warping
        frames_dev = frames_tensor.to(self.device)
        depths_dev = depths_tensor.to(self.device)
        K_dev = K.unsqueeze(0).to(self.device)

        warped_frames = []
        warped_masks = []

        # Run Double-Reprojection for each frame
        with torch.no_grad():
            for i in range(self.num_frames):
                frame_i = frames_dev[i : i + 1]
                depth_i = depths_dev[i : i + 1]
                p_s = pose_s[i : i + 1]
                p_pert = pose_pert[i : i + 1]

                proxy_f, proxy_m = self.warper.double_reprojection_warp(
                    frame_target=frame_i,
                    depth_target=depth_i,
                    transformation_target=p_s,
                    transformation_perturbed=p_pert,
                    intrinsic=K_dev,
                )
                warped_frames.append(proxy_f.cpu())
                warped_masks.append(proxy_m.cpu())

        warped_video = torch.cat(warped_frames, dim=0).cpu()  # (F, 3, H, W)
        mask_video = torch.cat(warped_masks, dim=0).cpu()      # (F, 1, H, W)

        # Reference frames: Uniformly sampled across entire video so future appearances are learned
        if self.num_frames <= self.ref_frames_count:
            ref_video = frames_tensor[: self.ref_frames_count].cpu()
        else:
            ref_idx = torch.linspace(0, self.num_frames - 1, self.ref_frames_count).long()
            ref_video = frames_tensor[ref_idx].cpu()

        # Permute to (C, F, H, W)
        target_video_cf = frames_tensor.permute(1, 0, 2, 3).cpu()
        warped_video_cf = warped_video.permute(1, 0, 2, 3).cpu()
        mask_video_cf = mask_video.permute(1, 0, 2, 3).cpu()
        ref_video_cf = ref_video.permute(1, 0, 2, 3).cpu()

        return {
            "target_video": target_video_cf,
            "warped_video": warped_video_cf,
            "mask_video": mask_video_cf,
            "ref_video": ref_video_cf,
            "video_path": video_path,
        }
