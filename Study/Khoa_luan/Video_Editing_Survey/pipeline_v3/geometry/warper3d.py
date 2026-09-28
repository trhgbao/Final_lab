"""
3D Point Cloud Warper and Forward Bilinear Splatting for Pipeline v3.
Provides deterministic 3D novel view rendering and Double-Reprojection for self-supervised training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import cv2
from PIL import Image


class Warper3D:
    def __init__(self, resolution: Optional[Tuple[int, int]] = None, device: Union[str, torch.device] = "cuda"):
        self.resolution = resolution
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.float32

    def create_grid(self, b: int, h: int, w: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        x_1d = torch.arange(0, w, device=device, dtype=dtype)[None]
        y_1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unprojects 2D pixels to 3D points using depth and intrinsic1,
        applies rigid camera transformation (T2 * inv(T1)),
        and projects 3D points back to image coordinates using intrinsic2.
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape

        device = transformation1.device
        dtype = depth1.dtype
        depth1 = depth1.to(device=device, dtype=dtype)
        transformation1 = transformation1.to(device=device, dtype=torch.float32)
        transformation2 = transformation2.to(device=device, dtype=torch.float32)
        intrinsic1 = intrinsic1.to(device=device, dtype=torch.float32)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        else:
            intrinsic2 = intrinsic2.to(device=device, dtype=torch.float32)

        # Relative camera transformation (always in float32 for geometric precision and linalg safety)
        transformation = torch.bmm(
            transformation2, torch.linalg.inv(transformation1)
        ).to(dtype)

        x1d = torch.arange(0, w, device=device, dtype=dtype)[None]
        y1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
        x2d = x1d.repeat([h, 1])
        y2d = y1d.repeat([1, w])
        ones_2d = torch.ones(size=(h, w), device=device, dtype=dtype)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

        intrinsic1_inv = torch.linalg.inv(intrinsic1.float()).to(dtype)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]
        intrinsic2_4d = intrinsic2[:, None, None]
        depth_4d = depth1[:, 0][:, :, :, None, None]
        trans_4d = transformation[:, None, None]

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :, :3]
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor] = None,
        is_image: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward splats frame1 according to optical flow12 using bilinear weights and depth weighting (z-buffering).
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape

        device = frame1.device
        dtype = frame1.dtype

        flow12 = flow12.to(device=device, dtype=dtype)
        depth1 = depth1.to(device=device, dtype=dtype)

        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)
        else:
            mask1 = mask1.to(device=device, dtype=dtype)

        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)
        else:
            flow12_mask = flow12_mask.to(device=device, dtype=dtype)

        grid = self.create_grid(b, h, w, device=device, dtype=dtype)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()

        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        max_log = log_depth1.max()
        if max_log > 0:
            depth_weights = torch.exp(log_depth1 / max_log * 50)
        else:
            depth_weights = torch.ones_like(log_depth1)

        d_w = depth_weights.unsqueeze(1) if depth_weights.dim() == 3 else depth_weights
        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / d_w, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / d_w, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / d_w, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / d_w, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32, device=frame1.device)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32, device=frame1.device)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)

        # Splatting with index_put
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1.0 if is_image else 0.0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1.0, max=1.0)
        return warped_frame2, mask2

    def heal_small_holes(
        self,
        warped_frame2: torch.Tensor,
        mask2: torch.Tensor,
        max_hole_area: int = 25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solution 2: Proxy Splatting Crack & Hole Healing.
        Heals small splatting cracks and pinholes (area <= max_hole_area) on the proxy frame
        using local neighbor colors via fast inpainting, marking them as valid (1.0).
        Large disocclusions (area > max_hole_area) are preserved as holes for the generative DiT.

        Args:
            warped_frame2: (b, 3, h, w) in range [-1, 1]
            mask2: (b, 1, h, w) 1 for valid, 0 for disoccluded hole
            max_hole_area: maximum pixel area of a hole to be considered a splatting crack (default: 25 px)
        Returns:
            healed_frame: (b, 3, h, w) in range [-1, 1]
            healed_mask: (b, 1, h, w) 1 for valid, 0 for remaining large holes
        """
        if max_hole_area <= 0:
            return warped_frame2, mask2

        b, c, h, w = warped_frame2.shape
        healed_frames = []
        healed_masks = []

        for bi in range(b):
            # Convert frame to [0, 255] uint8 RGB for cv2
            img_np = ((warped_frame2[bi].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            # Hole mask: 255 for hole, 0 for valid
            hole_mask_bin = ((1.0 - mask2[bi, 0].detach().cpu().numpy()) > 0.5).astype(np.uint8) * 255

            if not np.any(hole_mask_bin):
                healed_frames.append(warped_frame2[bi : bi + 1])
                healed_masks.append(mask2[bi : bi + 1])
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_mask_bin, connectivity=8)
            if num_labels <= 1:
                healed_frames.append(warped_frame2[bi : bi + 1])
                healed_masks.append(mask2[bi : bi + 1])
                continue

            # Identify small holes (label 0 is background / valid)
            areas = stats[1:, cv2.CC_STAT_AREA]
            small_indices = np.where(areas <= max_hole_area)[0] + 1

            if len(small_indices) == 0:
                healed_frames.append(warped_frame2[bi : bi + 1])
                healed_masks.append(mask2[bi : bi + 1])
                continue

            # Binary mask containing ONLY small holes to heal
            small_holes_mask = np.isin(labels, small_indices).astype(np.uint8) * 255

            # Fast inpaint on small holes using Navier-Stokes or Telea
            healed_img_np = cv2.inpaint(img_np, small_holes_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Convert healed image back to [-1, 1] float tensor
            healed_t = torch.from_numpy(healed_img_np).float().permute(2, 0, 1).unsqueeze(0).to(
                device=warped_frame2.device, dtype=warped_frame2.dtype
            ) / 127.5 - 1.0

            # Update mask: mark healed small holes as valid (1.0)
            small_holes_t = torch.from_numpy(small_holes_mask > 0).to(
                device=mask2.device, dtype=mask2.dtype
            ).unsqueeze(0).unsqueeze(0)
            healed_m = torch.clamp(mask2[bi : bi + 1] + small_holes_t, 0.0, 1.0)

            healed_frames.append(healed_t)
            healed_masks.append(healed_m)

        return torch.cat(healed_frames, dim=0), torch.cat(healed_masks, dim=0)

    def clean_points(self, warped_frame2: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Removes flying pixel noise by dilating the disocclusion mask."""
        warped_frame2_norm = (warped_frame2 + 1.0) / 2.0
        mask = 1.0 - mask2
        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0
        mask_np = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask_np = mask_np.cpu().numpy().astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
        mask_dilated = mask_dilated.astype(np.float32) / 255.0
        mask_dilated = torch.from_numpy(mask_dilated).permute(2, 0, 1).unsqueeze(0).to(warped_frame2.device)
        mask_clean = 1.0 - mask_dilated[:, 0:1, :, :]
        cleaned_frame = warped_frame2_norm * mask_clean
        cleaned_frame = cleaned_frame * 2.0 - 1.0
        return cleaned_frame, mask_clean

    def compute_surface_normals(self, depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
        """
        Computes 3D surface normals for each pixel in camera coordinate frame.
        Args:
            depth: (b, 1, h, w) depth map
            intrinsic: (b, 3, 3) camera intrinsics
        Returns:
            normals: (b, 3, h, w) unit surface normals facing camera
        """
        b, _, h, w = depth.shape
        device = depth.device
        dtype = depth.dtype

        x1d = torch.arange(0, w, device=device, dtype=dtype)[None]
        y1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
        x2d = x1d.repeat([h, 1])
        y2d = y1d.repeat([1, w])
        ones_2d = torch.ones(size=(h, w), device=device, dtype=dtype)
        pos_vectors = torch.stack([x2d, y2d, ones_2d], dim=0)[None]  # (1, 3, h, w)

        fx = intrinsic[:, 0:1, 0:1].unsqueeze(-1)
        fy = intrinsic[:, 1:2, 1:2].unsqueeze(-1)
        cx = intrinsic[:, 0:1, 2:3].unsqueeze(-1)
        cy = intrinsic[:, 1:2, 2:3].unsqueeze(-1)

        X = (pos_vectors[:, 0:1] - cx) / fx * depth
        Y = (pos_vectors[:, 1:2] - cy) / fy * depth
        Z = depth
        points_3d = torch.cat([X, Y, Z], dim=1)  # (b, 3, h, w)

        dx = torch.zeros_like(points_3d)
        dx[:, :, :, 1:-1] = (points_3d[:, :, :, 2:] - points_3d[:, :, :, :-2]) / 2.0
        dx[:, :, :, 0] = points_3d[:, :, :, 1] - points_3d[:, :, :, 0]
        dx[:, :, :, -1] = points_3d[:, :, :, -1] - points_3d[:, :, :, -2]

        dy = torch.zeros_like(points_3d)
        dy[:, :, 1:-1, :] = (points_3d[:, :, 2:, :] - points_3d[:, :, :-2, :]) / 2.0
        dy[:, :, 0, :] = points_3d[:, :, 1, :] - points_3d[:, :, 0, :]
        dy[:, :, -1, :] = points_3d[:, :, -1, :] - points_3d[:, :, -2, :]

        nx = dx[:, 1] * dy[:, 2] - dx[:, 2] * dy[:, 1]
        ny = dx[:, 2] * dy[:, 0] - dx[:, 0] * dy[:, 2]
        nz = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
        normals = torch.stack([nx, ny, nz], dim=1)

        norm = torch.norm(normals, dim=1, keepdim=True) + 1e-8
        normals = normals / norm

        ray = points_3d / (torch.norm(points_3d, dim=1, keepdim=True) + 1e-8)
        dot = (normals * ray).sum(dim=1, keepdim=True)
        normals = torch.where(dot > 0, -normals, normals)

        return normals

    def compute_reliable_depth_mask(
        self,
        depth: torch.Tensor,
        window_size: int = 5,
        ratio_thresh: float = 0.35,
    ) -> torch.Tensor:
        """
        Detects depth discontinuities/silhouette edges and returns a mask of reliable surfaces
        (inspired by NVIDIA GEN3C reliable_depth_mask_range_batch).
        """
        depth_pos = torch.clamp(depth, min=1e-4)
        local_max = F.max_pool2d(depth_pos, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_min = -F.max_pool2d(-depth_pos, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_mean = F.avg_pool2d(depth_pos, kernel_size=window_size, stride=1, padding=window_size // 2)

        ratio = (local_max - local_min) / (local_mean + 1e-6)
        reliable = ((ratio < ratio_thresh) & (depth > 0.05)).float()
        return reliable

    def compute_grazing_angle_mask(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        max_angle_deg: float = 55.0,
    ) -> torch.Tensor:
        """
        Computes a mask of pixels whose surface normal forms an acute angle with the target camera viewing ray.
        Surfaces viewed at grazing angles (> max_angle_deg) are culled (mask = 0).
        """
        b, _, h, w = depth1.shape
        device = depth1.device
        dtype = depth1.dtype

        normals1 = self.compute_surface_normals(depth1, intrinsic1)

        rel_transform = torch.bmm(
            transformation2.float(),
            torch.linalg.inv(transformation1.float()),
        ).to(dtype)
        R_rel = rel_transform[:, :3, :3]
        t_rel = rel_transform[:, :3, 3:]

        fx = intrinsic1[:, 0:1, 0:1].unsqueeze(-1)
        fy = intrinsic1[:, 1:2, 1:2].unsqueeze(-1)
        cx = intrinsic1[:, 0:1, 2:3].unsqueeze(-1)
        cy = intrinsic1[:, 1:2, 2:3].unsqueeze(-1)
        x1d = torch.arange(0, w, device=device, dtype=dtype)[None]
        y1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
        x2d = x1d.repeat([h, 1])
        y2d = y1d.repeat([1, w])
        X = (x2d[None, None] - cx) / fx * depth1
        Y = (y2d[None, None] - cy) / fy * depth1
        Z = depth1
        pts1 = torch.cat([X, Y, Z], dim=1)

        pts1_flat = pts1.view(b, 3, -1)
        pts2_flat = torch.bmm(R_rel, pts1_flat) + t_rel
        pts2 = pts2_flat.view(b, 3, h, w)

        normals1_flat = normals1.view(b, 3, -1)
        normals2_flat = torch.bmm(R_rel, normals1_flat)
        normals2 = normals2_flat.view(b, 3, h, w)
        normals2 = normals2 / (torch.norm(normals2, dim=1, keepdim=True) + 1e-8)

        ray2 = pts2 / (torch.norm(pts2, dim=1, keepdim=True) + 1e-8)
        cos_angle = -(ray2 * normals2).sum(dim=1, keepdim=True)

        cos_thresh = np.cos(np.deg2rad(max_angle_deg))
        valid_mask = (cos_angle > cos_thresh).float()
        valid_mask = valid_mask * (pts2[:, 2:3] > 0.05).float()

        return valid_mask

    def align_depth_scale(
        self,
        source_depth: torch.Tensor,
        target_depth: torch.Tensor,
        covisible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aligns source_depth scale and shift to target_depth in inverse depth (disparity) space
        via least squares (inspired by NVIDIA GEN3C camera_utils.py).
        """
        device = source_depth.device
        src_inv = 1.0 / torch.clamp(source_depth, min=0.01, max=100.0)
        tgt_inv = 1.0 / torch.clamp(target_depth, min=0.01, max=100.0)

        valid = (covisible_mask > 0.5) & (source_depth > 0.05) & (target_depth > 0.05)
        if valid.sum() < 100:
            return source_depth

        src_val = src_inv[valid].view(-1, 1)
        tgt_val = tgt_inv[valid].view(-1, 1)

        try:
            q_low, q_high = torch.quantile(src_val, torch.tensor([0.1, 0.9], device=device))
            inliers = (src_val >= q_low) & (src_val <= q_high)
            if inliers.sum() < 50:
                return source_depth

            src_sub = src_val[inliers].view(-1, 1)
            tgt_sub = tgt_val[inliers].view(-1, 1)
            ones = torch.ones_like(src_sub)
            A = torch.cat([src_sub, ones], dim=1)

            solution = torch.linalg.lstsq(A, tgt_sub).solution
            scale = torch.clamp(solution[0, 0], min=0.2, max=5.0)
            bias = torch.clamp(solution[1, 0], min=-5.0, max=5.0)
            aligned_inv = src_inv * scale + bias
            aligned_depth = 1.0 / torch.clamp(aligned_inv, min=0.01, max=100.0)
            return aligned_depth
        except Exception:
            return source_depth

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor] = None,
        clean_mask: bool = True,
        heal_holes: bool = True,
        max_hole_area: int = 25,
        cull_grazing_angles: bool = False,
        max_grazing_angle: float = 55.0,
        filter_depth_edges: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward warps frame1 from transformation1 to transformation2 with optional grazing-angle culling.
        """
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=frame1.device, dtype=frame1.dtype)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device, dtype=torch.float32)
        transformation2 = transformation2.to(self.device, dtype=torch.float32)
        intrinsic1 = intrinsic1.to(self.device, dtype=torch.float32)
        intrinsic2 = intrinsic2.to(self.device, dtype=torch.float32)

        # 1. Edge & Silhouette filtering
        if filter_depth_edges:
            rel_mask = self.compute_reliable_depth_mask(depth1)
            mask1 = mask1 * rel_mask

        # 2. Grazing-Angle Surface Culling (GenWarp / NVS-Solver / GEN3C)
        if cull_grazing_angles:
            grazing_mask = self.compute_grazing_angle_mask(
                depth1, transformation1, transformation2, intrinsic1, max_angle_deg=max_grazing_angle
            )
            mask1 = mask1 * grazing_mask

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        if heal_holes and max_hole_area > 0:
            warped_frame2, mask2 = self.heal_small_holes(warped_frame2, mask2, max_hole_area=max_hole_area)
        if clean_mask:
            warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)

        return warped_frame2, mask2, flow12

    def multi_frame_forward_warp(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        transformation_s: torch.Tensor,
        transformation_t: torch.Tensor,
        intrinsic_s: torch.Tensor,
        intrinsic_t: Optional[torch.Tensor] = None,
        target_idx: int = 0,
        neighbor_radius: int = 3,
        use_all_frames: bool = True,
        cull_grazing_angles: bool = True,
        max_grazing_angle: float = 55.0,
        filter_depth_edges: bool = True,
        align_depth_scale: bool = True,
        clean_mask: bool = True,
        heal_holes: bool = True,
        max_hole_area: int = 25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-Frame Priority-Based Splatting (MFS-Scaffold).
        Combines visual information from across the video into the novel view proxy:
        - Tier 0: Primary anchor frame target_idx splats first (locked, zero ghosting).
        - Tier 1: Temporal neighbors (target_idx +/- 1, 2, 3...) fill local disocclusions.
        - Tier 2: Remaining frames across the entire video fill wide-angle voids.
        - Tier 3: Hole healing and boundary cleanup.
        """
        num_frames = frames.shape[0]
        if intrinsic_t is None:
            intrinsic_t = intrinsic_s.clone()

        # Step 1 (Tier 0): Splat primary anchor frame
        canvas, canvas_mask, flow = self.forward_warp(
            frames[target_idx : target_idx + 1],
            None,
            depths[target_idx : target_idx + 1],
            transformation_s[target_idx : target_idx + 1],
            transformation_t[target_idx : target_idx + 1],
            intrinsic_s[target_idx : target_idx + 1] if intrinsic_s.shape[0] > 1 else intrinsic_s,
            intrinsic_t[target_idx : target_idx + 1] if intrinsic_t.shape[0] > 1 else intrinsic_t,
            clean_mask=False,
            heal_holes=False,
            cull_grazing_angles=cull_grazing_angles,
            max_grazing_angle=max_grazing_angle,
            filter_depth_edges=filter_depth_edges,
        )

        # Step 2: Build candidate list in order of priority
        neighbors = []
        for r in range(1, neighbor_radius + 1):
            if target_idx - r >= 0:
                neighbors.append(target_idx - r)
            if target_idx + r < num_frames:
                neighbors.append(target_idx + r)

        remaining = []
        if use_all_frames:
            for idx in range(num_frames):
                if idx != target_idx and idx not in neighbors:
                    remaining.append(idx)

        candidates = neighbors + remaining

        # Step 3: Progressive Infilling into EMPTY pixels only (canvas_mask == 0)
        target_k = intrinsic_t[target_idx : target_idx + 1] if intrinsic_t.shape[0] > 1 else intrinsic_t
        target_pose = transformation_t[target_idx : target_idx + 1]

        for cand_idx in candidates:
            # Early exit if canvas is practically full
            if canvas_mask.mean().item() >= 0.98:
                break

            cand_depth = depths[cand_idx : cand_idx + 1]
            if align_depth_scale:
                cand_depth = self.align_depth_scale(
                    cand_depth, depths[target_idx : target_idx + 1], canvas_mask
                )

            cand_k = intrinsic_s[cand_idx : cand_idx + 1] if intrinsic_s.shape[0] > 1 else intrinsic_s
            cand_pose_s = transformation_s[cand_idx : cand_idx + 1]

            cand_warp, cand_mask, _ = self.forward_warp(
                frames[cand_idx : cand_idx + 1],
                None,
                cand_depth,
                cand_pose_s,
                target_pose,
                cand_k,
                target_k,
                clean_mask=False,
                heal_holes=False,
                cull_grazing_angles=cull_grazing_angles,
                max_grazing_angle=max_grazing_angle,
                filter_depth_edges=filter_depth_edges,
            )

            # Infill rule: Only overwrite pixels that are currently EMPTY
            fill_region = (canvas_mask < 0.5) & (cand_mask > 0.5)
            if fill_region.any():
                fill_mask_3c = fill_region.repeat(1, 3, 1, 1)
                canvas = torch.where(fill_mask_3c, cand_warp, canvas)
                canvas_mask = torch.where(fill_region, torch.ones_like(canvas_mask), canvas_mask)

        # Step 4 (Tier 3): Discretization crack healing & boundary cleanup
        if heal_holes and max_hole_area > 0:
            canvas, canvas_mask = self.heal_small_holes(canvas, canvas_mask, max_hole_area=max_hole_area)
        if clean_mask:
            canvas, canvas_mask = self.clean_points(canvas, canvas_mask)

        return canvas, canvas_mask, flow

    def double_reprojection_warp(
        self,
        frame_target: torch.Tensor,
        depth_target: torch.Tensor,
        transformation_target: torch.Tensor,
        transformation_perturbed: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Double-Reprojection for self-supervised training without multi-view ground truth:
        1. Warps frame_target from transformation_target -> transformation_perturbed (creates virtual novel view)
        2. Warps back from transformation_perturbed -> transformation_target (creates proxy with real holes & tears)
        
        The resulting proxy frame aligns geometrically with frame_target but has realistic disocclusions!
        """
        frame_target = frame_target.to(device=self.device, dtype=self.dtype)
        depth_target = depth_target.to(device=self.device, dtype=self.dtype)
        transformation_target = transformation_target.to(device=self.device, dtype=self.dtype)
        transformation_perturbed = transformation_perturbed.to(device=self.device, dtype=self.dtype)
        intrinsic = intrinsic.to(device=self.device, dtype=self.dtype)

        # Step 1: Forward warp to perturbed pose
        warped_temp, mask_temp, flow_temp = self.forward_warp(
            frame_target, None, depth_target, transformation_target, transformation_perturbed, intrinsic, intrinsic, clean_mask=False
        )
        depth_2d = depth_target.squeeze(1) if depth_target.dim() == 4 else depth_target
        warped_flow, _ = self.bilinear_splatting(
            flow_temp, None, depth_2d, flow_temp, None, is_image=False
        )

        # Step 2: Backward warp to target pose using inverted flow
        proxy_frame, proxy_mask = self.bilinear_splatting(
            warped_temp, mask_temp, depth_2d, -warped_flow, None, is_image=True
        )
        return proxy_frame, proxy_mask
