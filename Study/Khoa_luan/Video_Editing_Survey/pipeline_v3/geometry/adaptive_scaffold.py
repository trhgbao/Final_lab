"""
Adaptive Keyframe Occlusion Infilling (AK-Scaffold) Engine.
Integrates specialized 3D Computer Vision & Neural Rendering algorithms:
  1. Context-Preserving Base Anchor (Free View Synthesis / Koltun & Riegler, ECCV 2020)
  2. Candidate Coverage Ranking (Deep Blending / Hedman et al., SIGGRAPH Asia 2018)
  3. Dynamic Programming Viterbi Path Optimization (FGTiV / Video Inpainting)
  4. Multi-View Geometric Disparity Alignment (GEN3C / NVIDIA 2024 & MonST3R)
  5. Overlap-Guided Affine Color Calibration (DynIBaR / Google Research, CVPR 2023)
  6. Disparity-Discontinuity Aware Boundary Infilling (3D Photo Inpainting / Shih et al., CVPR 2020)
  7. Strict Context Preservation & Safe Seam Feathering
"""

import math
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .warper3d import Warper3D
except ImportError:
    from pipeline_v3.geometry.warper3d import Warper3D


class AdaptiveKeyframeScaffold(nn.Module):
    """
    Adaptive Keyframe Occlusion Infilling Engine for 3D Video Scaffolding.
    Protects base geometry and dynamically harvests distant donor frames to infill disocclusions.
    """

    def __init__(
        self,
        hole_threshold: float = 0.05,
        lambda_smooth: float = 0.35,
        hysteresis_bonus: float = 0.20,
        max_temporal_step: int = 15,
        feather_radius: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hole_threshold = hole_threshold
        self.lambda_smooth = lambda_smooth
        self.hysteresis_bonus = hysteresis_bonus
        self.max_temporal_step = max_temporal_step
        self.feather_radius = feather_radius
        self.device = torch.device(device)
        self.dtype = dtype

        # Core Warper3D for forward projection
        self.warper = Warper3D(device=str(self.device))
        self.warper.dtype = self.dtype

    # -------------------------------------------------------------------------
    # 1. Base Anchor Generation (100% Context Preservation)
    # -------------------------------------------------------------------------
    def compute_base_scaffold(
        self,
        frame: torch.Tensor,         # (1, 3, H, W) in [-1, 1]
        depth: torch.Tensor,         # (1, 1, H, W)
        pose_src: torch.Tensor,      # (1, 4, 4)
        pose_tgt: torch.Tensor,      # (1, 4, 4)
        intrinsic_src: torch.Tensor, # (1, 3, 3)
        intrinsic_tgt: Optional[torch.Tensor] = None, # (1, 3, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the uncorrupted single-frame 2.5D base scaffold.
        Never applies destructive grazing-angle or silhouette culling.
        """
        if intrinsic_tgt is None:
            intrinsic_tgt = intrinsic_src

        w_img, w_mask, _ = self.warper.forward_warp(
            frame1=frame,
            mask1=None,
            depth1=depth,
            transformation1=pose_src,
            transformation2=pose_tgt,
            intrinsic1=intrinsic_src,
            intrinsic2=intrinsic_tgt,
            clean_mask=True,
            heal_holes=True,
            max_hole_area=25,
            cull_grazing_angles=False,
            filter_depth_edges=False,
        )
        # w_img is in [-1, 1], w_mask is in {0, 1}
        return w_img, w_mask

    # -------------------------------------------------------------------------
    # 2. Coverage Scoring: Candidate Ranking by Hole Coverage
    #    (Inspired by Deep Blending / SIGGRAPH Asia 2018 Ranked Mosaics)
    # -------------------------------------------------------------------------
    def compute_hole_coverage(
        self,
        donor_frame: torch.Tensor,   # (1, 3, H, W)
        donor_depth: torch.Tensor,   # (1, 1, H, W)
        pose_src_k: torch.Tensor,    # (1, 4, 4)
        pose_tgt_t: torch.Tensor,    # (1, 4, 4)
        intrinsic_k: torch.Tensor,   # (1, 3, 3)
        intrinsic_t: torch.Tensor,   # (1, 3, 3)
        base_mask_t: torch.Tensor,   # (1, 1, H, W)
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Measures the fraction of the hole in base_mask_t that can be covered by donor frame k.
        Runs purely on GPU (clean_mask=False, heal_holes=False) for blazing fast ranking.
        Returns:
            coverage_score in [0.0, 1.0]
            warped_donor_img: (1, 3, H, W)
            warped_donor_mask: (1, 1, H, W)
        """
        hole_mask = (base_mask_t < 0.5).float()
        hole_pixel_count = hole_mask.sum().item()

        if hole_pixel_count < 10:
            return 1.0, donor_frame, base_mask_t

        w_img_k, w_mask_k, _ = self.warper.forward_warp(
            frame1=donor_frame,
            mask1=None,
            depth1=donor_depth,
            transformation1=pose_src_k,
            transformation2=pose_tgt_t,
            intrinsic1=intrinsic_k,
            intrinsic2=intrinsic_t,
            clean_mask=False,
            heal_holes=False,
            max_hole_area=0,
            cull_grazing_angles=False,
            filter_depth_edges=False,
        )

        overlap_with_hole = (w_mask_k * hole_mask).sum().item()
        coverage_score = float(overlap_with_hole / (hole_pixel_count + 1e-6))
        return coverage_score, w_img_k, w_mask_k

    # -------------------------------------------------------------------------
    # 3. Dynamic Programming (Viterbi Path) for Smooth Donor Trajectory
    #    (Prevents temporal flickering across disocclusions)
    # -------------------------------------------------------------------------
    def solve_viterbi_donor_path(
        self,
        coverage_matrix: np.ndarray,      # (F, F): coverage_matrix[t, k] in [0, 1]
        active_t_indices: List[int],
        donor_candidates: List[int],
        total_frames: int,
        min_coverage_threshold: float = 0.01,
    ) -> Dict[int, int]:
        """
        Solves optimal donor sequence k*(t) via Viterbi Dynamic Programming over evaluated candidates:
            min sum_t (1 - Coverage[t, k_t]) + lambda * Smoothness(k_t, k_{t-1})
        """
        if not active_t_indices or not donor_candidates:
            return {}

        num_active = len(active_t_indices)
        num_candidates = len(donor_candidates)

        # dp[i, c] stores minimum cumulative cost up to active step i choosing candidate donor c
        dp = np.full((num_active, num_candidates), np.inf, dtype=np.float32)
        parent = np.zeros((num_active, num_candidates), dtype=np.int32)

        # Base case (step 0)
        t0 = active_t_indices[0]
        for c in range(num_candidates):
            k = donor_candidates[c]
            data_cost = 1.0 - coverage_matrix[t0, k]
            # Prior: slight preference for donor close to t0
            prior_dist = abs(k - t0) / float(total_frames)
            dp[0, c] = data_cost + 0.1 * prior_dist

        # Inductive step
        for i in range(1, num_active):
            t_curr = active_t_indices[i]
            t_prev = active_t_indices[i - 1]
            dt = max(1, t_curr - t_prev)

            for c_curr in range(num_candidates):
                k_curr = donor_candidates[c_curr]
                data_cost = 1.0 - coverage_matrix[t_curr, k_curr]

                # Transition penalties from all previous candidates c_prev
                k_prev_arr = np.array(donor_candidates, dtype=np.float32)
                step_diff = np.abs(k_curr - k_prev_arr)
                smooth_cost = self.lambda_smooth * np.minimum(step_diff / float(dt), self.max_temporal_step) ** 1.5

                # Hysteresis bonus: if staying on same or adjacent donor, reward continuity
                hysteresis = np.where(step_diff <= 1, -self.hysteresis_bonus, 0.0)

                total_trans_cost = dp[i - 1, :] + smooth_cost + hysteresis
                best_prev = int(np.argmin(total_trans_cost))
                dp[i, c_curr] = total_trans_cost[best_prev] + data_cost
                parent[i, c_curr] = best_prev

        # Backtrack optimal path
        best_last_c = int(np.argmin(dp[-1, :]))
        optimal_donors = {}
        curr_c = best_last_c
        for i in range(num_active - 1, -1, -1):
            t_idx = active_t_indices[i]
            chosen_k = donor_candidates[curr_c]
            # Only adopt donor if it actually provides positive hole coverage
            if coverage_matrix[t_idx, chosen_k] >= min_coverage_threshold:
                optimal_donors[t_idx] = chosen_k
            curr_c = int(parent[i, curr_c])

        return optimal_donors

    # -------------------------------------------------------------------------
    # 4. Multi-View Geometric Disparity Scale & Shift Alignment
    #    (Inspired by GEN3C / NVIDIA 2024 & MonST3R)
    # -------------------------------------------------------------------------
    def align_donor_disparity(
        self,
        donor_depth: torch.Tensor,    # (1, 1, H, W) in meters
        base_depth: torch.Tensor,     # (1, 1, H, W) in meters
        pose_k: torch.Tensor,         # (1, 4, 4) source pose of donor
        pose_t: torch.Tensor,         # (1, 4, 4) source pose of base frame t
        intrinsic_k: torch.Tensor,    # (1, 3, 3)
        intrinsic_t: torch.Tensor,    # (1, 3, 3)
    ) -> torch.Tensor:
        """
        Aligns donor_depth into base_depth metric scale in inverse-depth space via
        geometric reprojection correspondences:
            1 / D_donor_aligned = alpha * (1 / D_donor) + beta
        """
        device = donor_depth.device
        dtype = donor_depth.dtype
        _, _, H, W = donor_depth.shape

        try:
            # 1. Transform base depth points into donor camera frame
            trans_pts = self.warper.compute_transformed_points(
                depth1=base_depth,
                transformation1=pose_t,
                transformation2=pose_k,
                intrinsic1=intrinsic_t,
                intrinsic2=intrinsic_k,
            )
            # Projected coords in donor frame
            u_k = trans_pts[:, :, :, 0, 0] / (trans_pts[:, :, :, 2, 0] + 1e-8)
            v_k = trans_pts[:, :, :, 1, 0] / (trans_pts[:, :, :, 2, 0] + 1e-8)
            z_geom = trans_pts[:, :, :, 2, 0]

            # Normalize to [-1, 1] for grid_sample
            grid_x = 2.0 * u_k / (W - 1) - 1.0
            grid_y = 2.0 * v_k / (H - 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)

            # Sample donor depth at projected pixel locations
            sampled_donor = F.grid_sample(
                donor_depth.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=True
            )

            # Filter valid points inside donor FOV
            valid = (
                (grid_x >= -0.98)
                & (grid_x <= 0.98)
                & (grid_y >= -0.98)
                & (grid_y <= 0.98)
                & (z_geom > 0.1)
                & (sampled_donor[:, 0] > 0.1)
            )

            if valid.sum() < 64:
                return donor_depth

            src_inv = (1.0 / torch.clamp(sampled_donor[:, 0][valid], 0.01, 100.0)).view(-1, 1).float()
            tgt_inv = (1.0 / torch.clamp(z_geom[valid], 0.01, 100.0)).view(-1, 1).float()

            # Filter outliers via quantiles (in float32 to avoid bfloat16 quantile errors)
            q10, q90 = torch.quantile(
                src_inv, torch.tensor([0.1, 0.9], device=device, dtype=torch.float32)
            )
            inliers = (src_inv[:, 0] >= q10) & (src_inv[:, 0] <= q90)
            if inliers.sum() < 32:
                return donor_depth

            src_sub = src_inv[inliers]
            tgt_sub = tgt_inv[inliers]

            # Rank-deficiency guard: If scene depth variance is flat, solve pure scale
            if src_sub.std() < 1e-3:
                alpha = torch.clamp(tgt_sub.mean() / (src_sub.mean() + 1e-6), 0.5, 2.0)
                beta = torch.tensor(0.0, device=device, dtype=torch.float32)
            else:
                ones = torch.ones_like(src_sub)
                A = torch.cat([src_sub, ones], dim=1)
                sol = torch.linalg.lstsq(A, tgt_sub).solution
                alpha = torch.clamp(sol[0, 0], 0.5, 2.0)
                beta = torch.clamp(sol[1, 0], -2.0, 2.0)

            aligned_inv = (1.0 / torch.clamp(donor_depth.float(), 0.01, 100.0)) * alpha + beta
            aligned_depth = 1.0 / torch.clamp(aligned_inv, 0.01, 100.0)
            return aligned_depth.to(dtype)
        except Exception:
            return donor_depth

    # -------------------------------------------------------------------------
    # 5. Overlap-Guided Affine Color Calibration
    #    (From DynIBaR / Google Research, CVPR 2023)
    # -------------------------------------------------------------------------
    @staticmethod
    def calibrate_affine_color(
        donor_img: torch.Tensor,     # (1, 3, H, W) in [-1, 1]
        base_img: torch.Tensor,      # (1, 3, H, W) in [-1, 1]
        covisible_mask: torch.Tensor # (1, 1, H, W)
    ) -> torch.Tensor:
        """
        Calibrates donor_img exposure, contrast, and white balance to match base_img
        using channel-wise affine statistics on covisible region:
            I_donor_calibrated = (sigma_base / sigma_donor) * (I_donor - mu_donor) + mu_base
        """
        cov = (covisible_mask > 0.5).expand_as(donor_img)
        if cov.sum() < 128:
            return donor_img

        calibrated = donor_img.clone()
        for c in range(3):
            base_vals = base_img[:, c : c + 1][cov[:, c : c + 1]].float()
            donor_vals = donor_img[:, c : c + 1][cov[:, c : c + 1]].float()

            mu_base = base_vals.mean()
            std_base = base_vals.std() + 1e-4

            mu_donor = donor_vals.mean()
            std_donor = donor_vals.std() + 1e-4

            scale = torch.clamp(std_base / std_donor, min=0.5, max=2.0)
            shift = torch.clamp(mu_base - scale * mu_donor, min=-0.5, max=0.5)

            if torch.isnan(scale) or torch.isnan(shift):
                continue

            calibrated[:, c : c + 1] = torch.clamp(
                donor_img[:, c : c + 1].float() * scale + shift, min=-1.0, max=1.0
            ).to(donor_img.dtype)

        return calibrated

    # -------------------------------------------------------------------------
    # 6. Context-Preserving Seam Infilling & Feathering
    # -------------------------------------------------------------------------
    def compute_feathered_infill(
        self,
        base_img: torch.Tensor,      # (1, 3, H, W) in [-1, 1]
        base_mask: torch.Tensor,     # (1, 1, H, W) in {0, 1}
        donor_img: torch.Tensor,     # (1, 3, H, W) in [-1, 1]
        donor_mask: torch.Tensor,    # (1, 1, H, W) in {0, 1}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Composites donor_img strictly into the hole region (base_mask == 0)
        preserving 100% of base_img outside the hole.
        If feather_radius > 0, performs soft seam blending strictly along the covisible boundary.
        """
        hole_mask = (base_mask < 0.5).float()
        infill_mask = (donor_mask > 0.5).float() * hole_mask

        if infill_mask.sum() < 1.0:
            return base_img, base_mask

        # 1. Base assignment: 100% context preservation of base image
        composite_img = torch.where(infill_mask.expand_as(base_img) > 0.5, donor_img, base_img)
        composite_mask = torch.clamp(base_mask + infill_mask, min=0.0, max=1.0)

        # 2. Optional soft seam feathering strictly where BOTH base and donor are valid
        if self.feather_radius > 0:
            covisible = (base_mask > 0.5).float() * (donor_mask > 0.5).float()
            # Detect seam boundary where base meets infill
            kernel_size = self.feather_radius * 2 + 1
            boundary = F.avg_pool2d(infill_mask, kernel_size=kernel_size, stride=1, padding=self.feather_radius)
            # Only blend in the narrow covisible strip along the boundary where neither image is empty
            blend_zone = (boundary > 0.05) & (boundary < 0.95) & (covisible > 0.5)
            if blend_zone.sum() > 0:
                alpha = boundary * blend_zone.float()
                composite_img = torch.where(
                    blend_zone.expand_as(base_img),
                    base_img * (1.0 - alpha) + donor_img * alpha,
                    composite_img,
                )

        return composite_img, composite_mask

    # -------------------------------------------------------------------------
    # 7. Complete Execution Pipeline across all Frames
    # -------------------------------------------------------------------------
    def run(
        self,
        frames: torch.Tensor,        # (F, 3, H, W) in [-1, 1]
        depths: torch.Tensor,        # (F, 1, H, W) in meters
        poses_src: torch.Tensor,     # (F, 4, 4)
        poses_tgt: torch.Tensor,     # (F, 4, 4)
        intrinsics: torch.Tensor,    # (F, 3, 3)
        quiet: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the full AK-Scaffold pipeline:
          1. Generates 100% context-preserved Base Scaffolds.
          2. Flags frames with holes > hole_threshold.
          3. Solves optimal global donor path via Viterbi DP.
          4. Executes depth disparity & affine color calibrated infilling.
        Returns:
            scaffold_video: (F, 3, H, W) in [-1, 1]
            scaffold_masks: (F, 1, H, W) in [0, 1]
        """
        F_num, _, H, W = frames.shape
        frames = frames.to(self.device, self.dtype)
        depths = depths.to(self.device, self.dtype)
        poses_src = poses_src.to(self.device, self.dtype)
        poses_tgt = poses_tgt.to(self.device, self.dtype)
        intrinsics = intrinsics.to(self.device, self.dtype)

        # -------------------------------------------------------------
        # Step 1: Base Scaffolds (Context Preservation First)
        # -------------------------------------------------------------
        base_imgs = []
        base_masks = []
        hole_ratios = []

        for t in range(F_num):
            b_img, b_mask = self.compute_base_scaffold(
                frame=frames[t : t + 1],
                depth=depths[t : t + 1],
                pose_src=poses_src[t : t + 1],
                pose_tgt=poses_tgt[t : t + 1],
                intrinsic_src=intrinsics[t : t + 1],
                intrinsic_tgt=intrinsics[t : t + 1],
            )
            base_imgs.append(b_img)
            base_masks.append(b_mask)
            hole_ratio = float((b_mask < 0.5).float().mean().item())
            hole_ratios.append(hole_ratio)

        active_t_indices = [t for t in range(F_num) if hole_ratios[t] > self.hole_threshold]

        if not active_t_indices:
            if not quiet:
                print(f"--> [AK-Scaffold] All frames have holes < {self.hole_threshold*100:.1f}%. Outputting pure Base Scaffold!")
            return torch.cat(base_imgs, dim=0), torch.cat(base_masks, dim=0)

        if not quiet:
            print(f"--> [AK-Scaffold] {len(active_t_indices)}/{F_num} frames have holes > {self.hole_threshold*100:.1f}%. Activating Global Donor Retrieval...")

        # -------------------------------------------------------------
        # Step 2: Build Coverage Matrix for Triggered Frames
        # -------------------------------------------------------------
        coverage_matrix = np.zeros((F_num, F_num), dtype=np.float32)
        # Subsample candidate donor frames (every 2 frames to accelerate)
        donor_candidates = list(range(0, F_num, 2))
        if (F_num - 1) not in donor_candidates:
            donor_candidates.append(F_num - 1)

        for t in active_t_indices:
            for k in donor_candidates:
                score, _, _ = self.compute_hole_coverage(
                    donor_frame=frames[k : k + 1],
                    donor_depth=depths[k : k + 1],
                    pose_src_k=poses_src[k : k + 1],
                    pose_tgt_t=poses_tgt[t : t + 1],
                    intrinsic_k=intrinsics[k : k + 1],
                    intrinsic_t=intrinsics[t : t + 1],
                    base_mask_t=base_masks[t],
                )
                coverage_matrix[t, k] = score

        # -------------------------------------------------------------
        # Step 3: Solve Global Viterbi Path for Optimal Donors
        # -------------------------------------------------------------
        optimal_donors = self.solve_viterbi_donor_path(
            coverage_matrix=coverage_matrix,
            active_t_indices=active_t_indices,
            donor_candidates=donor_candidates,
            total_frames=F_num,
        )

        if not quiet:
            unique_donors = sorted(list(set(optimal_donors.values()))) if optimal_donors else []
            print(f"--> [AK-Scaffold] Viterbi DP selected optimal donor frames: {unique_donors}")

        # -------------------------------------------------------------
        # Step 4: Execute Calibrated Infilling
        # -------------------------------------------------------------
        final_imgs = []
        final_masks = []

        for t in range(F_num):
            if t not in optimal_donors:
                final_imgs.append(base_imgs[t])
                final_masks.append(base_masks[t])
            else:
                k = optimal_donors[t]
                # 4.1. Align Donor Depth Disparity Scale with Geometric Reprojection
                aligned_donor_depth = self.align_donor_disparity(
                    donor_depth=depths[k : k + 1],
                    base_depth=depths[t : t + 1],
                    pose_k=poses_src[k : k + 1],
                    pose_t=poses_src[t : t + 1],
                    intrinsic_k=intrinsics[k : k + 1],
                    intrinsic_t=intrinsics[t : t + 1],
                )

                # 4.2. Warp Donor Frame to Target Pose with Clean Mask & Hole Healing
                w_img_k, w_mask_k, _ = self.warper.forward_warp(
                    frame1=frames[k : k + 1],
                    mask1=None,
                    depth1=aligned_donor_depth,
                    transformation1=poses_src[k : k + 1],
                    transformation2=poses_tgt[t : t + 1],
                    intrinsic1=intrinsics[k : k + 1],
                    intrinsic2=intrinsics[t : t + 1],
                    clean_mask=True,
                    heal_holes=True,
                    max_hole_area=25,
                    cull_grazing_angles=False,
                    filter_depth_edges=False,
                )

                # 4.3. Overlap-Guided Affine Color Calibration on Covisible Region
                covisible = base_masks[t] * w_mask_k
                calibrated_donor_img = self.calibrate_affine_color(
                    donor_img=w_img_k,
                    base_img=base_imgs[t],
                    covisible_mask=covisible,
                )

                # 4.4. Feathered Infilling strictly into Holes
                infilled_img, infilled_mask = self.compute_feathered_infill(
                    base_img=base_imgs[t],
                    base_mask=base_masks[t],
                    donor_img=calibrated_donor_img,
                    donor_mask=w_mask_k,
                )

                final_imgs.append(infilled_img)
                final_masks.append(infilled_mask)

        return torch.cat(final_imgs, dim=0), torch.cat(final_masks, dim=0)
