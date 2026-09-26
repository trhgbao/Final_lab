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
        dtype = transformation1.dtype
        depth1 = depth1.to(device=device, dtype=dtype)
        transformation2 = transformation2.to(device=device, dtype=dtype)
        intrinsic1 = intrinsic1.to(device=device, dtype=dtype)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        else:
            intrinsic2 = intrinsic2.to(device=device, dtype=dtype)

        # Relative camera transformation
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w, device=device, dtype=dtype)[None]
        y1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
        x2d = x1d.repeat([h, 1])
        y2d = y1d.repeat([1, w])
        ones_2d = torch.ones(size=(h, w), device=device, dtype=dtype)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

        intrinsic1_inv = torch.linalg.inv(intrinsic1)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward warps frame1 from transformation1 to transformation2.
        
        Args:
            frame1: (b, 3, h, w) in range [-1, 1]
            mask1: (b, 1, h, w) 1 for valid, 0 for unknown
            depth1: (b, 1, h, w) metric/relative depth
            transformation1: (b, 4, 4) source camera pose
            transformation2: (b, 4, 4) target camera pose
            intrinsic1: (b, 3, 3) source camera intrinsics
            intrinsic2: (b, 3, 3) target camera intrinsics
        Returns:
            warped_frame: (b, 3, h, w) in range [-1, 1]
            mask: (b, 1, h, w) 1 for valid, 0 for disoccluded hole
            flow12: (b, 2, h, w) pixel displacement flow
        """
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=frame1.device, dtype=frame1.dtype)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        transformation2 = transformation2.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1, intrinsic2)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        if clean_mask:
            warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)

        return warped_frame2, mask2, flow12

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
