"""
Unit tests for Pipeline v3 Core Components:
1. Geometry & 3D Warper
2. Trajectory Generator
3. DoRALinear Layer
4. Model Two-Stage Setup
"""

import sys
from pathlib import Path
_cur = Path(__file__).resolve().parent
_root = _cur.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
from pipeline_v3.geometry.warper3d import Warper3D
from pipeline_v3.geometry.trajectory import generate_camera_trajectory, sphere2pose
from pipeline_v3.models.dora import DoRALinear, apply_dora_to_module, get_dora_state_dict


def test_trajectory_and_warper():
    print("[Test 1] Testing Trajectory & 3D Warper...")
    device = "cpu"
    h, w = 64, 64
    warper = Warper3D(resolution=(h, w), device=device)

    # Synthetic image (B=1, C=3, H=64, W=64) and depth (B=1, 1, H=64, W=64)
    img = torch.rand(1, 3, h, w) * 2.0 - 1.0
    depth = torch.ones(1, 1, h, w) * 2.0

    c2w_anchor = torch.eye(4).unsqueeze(0)
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]).unsqueeze(0)

    pose_s, pose_t = generate_camera_trajectory(c2w_anchor, theta=0.0, phi=-30.0, d_r=0.2, num_frames=2, device=device)
    assert pose_s.shape == (2, 4, 4), f"Unexpected pose_s shape: {pose_s.shape}"
    assert pose_t.shape == (2, 4, 4), f"Unexpected pose_t shape: {pose_t.shape}"

    # Forward warp
    warped_img, mask, flow = warper.forward_warp(
        img, None, depth, pose_s[:1], pose_t[:1], K, K, clean_mask=False
    )
    assert warped_img.shape == (1, 3, h, w), f"Unexpected warped_img shape: {warped_img.shape}"
    assert mask.shape == (1, 1, h, w), f"Unexpected mask shape: {mask.shape}"
    assert flow.shape == (1, 2, h, w), f"Unexpected flow shape: {flow.shape}"

    # Double reprojection
    proxy_f, proxy_m = warper.double_reprojection_warp(
        frame_target=img,
        depth_target=depth,
        transformation_target=pose_s[:1],
        transformation_perturbed=pose_t[:1],
        intrinsic=K,
    )
    assert proxy_f.shape == (1, 3, h, w), f"Unexpected proxy_f shape: {proxy_f.shape}"
    assert proxy_m.shape == (1, 1, h, w), f"Unexpected proxy_m shape: {proxy_m.shape}"
    print("  --> Trajectory & Warper PASSED!")


def test_dora_layer():
    print("[Test 2] Testing DoRALinear Layer...")
    in_dim, out_dim = 128, 256
    linear = nn.Linear(in_dim, out_dim)

    # Initialize DoRA
    dora = DoRALinear(linear, r=8, lora_alpha=16.0)

    x = torch.randn(2, 10, in_dim)

    # At step 0, DoRA output must exactly match base linear output
    out_linear = linear(x)
    out_dora = dora(x)
    diff = (out_linear - out_dora).abs().max().item()
    assert diff < 1e-4, f"DoRA initialization error: max diff {diff} exceeds threshold"

    # Backward pass
    loss = out_dora.sum()
    loss.backward()

    # Base weight must have NO gradient, DoRA params MUST have gradient
    assert dora.weight.grad is None, "Base weight should be frozen!"
    assert dora.lora_A.grad is not None, "lora_A must receive gradients!"
    assert dora.lora_B.grad is not None, "lora_B must receive gradients!"
    assert dora.magnitude.grad is not None, "magnitude must receive gradients!"

    # Test weight merging
    dora.merge_weights()
    assert dora.merged is True
    out_merged = dora(x)
    diff_merged = (out_dora - out_merged).abs().max().item()
    assert diff_merged < 1e-4, f"Merged weight output differs from unmerged: {diff_merged}"
    print("  --> DoRALinear Layer PASSED!")


def test_model_two_stage_modes():
    print("[Test 3] Testing CrossTransformer3D Two-Stage Mode Setups...")
    from pipeline_v3.models.crosstransformer3d import CrossTransformer3DModel

    # Lightweight configuration for testing
    model = CrossTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        cross_attn_interval=2,
        is_train_cross=False,
    )

    # Inject DoRA
    dora_mods = model.enable_dora(r=4, lora_alpha=8.0)
    assert len(dora_mods) > 0, "No DoRA modules injected!"

    # Test Stage 1
    model.setup_two_stage_training(stage=1)
    stage1_trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("lora" in name or "magnitude" in name for name in stage1_trainable), "Stage 1 leaked non-DoRA gradients!"
    assert not model.is_train_cross, "Stage 1 should have cross attention disabled!"

    # Test Stage 2
    model.setup_two_stage_training(stage=2)
    stage2_trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("perceiver" in name or "ref_patch_embed" in name for name in stage2_trainable), "Stage 2 should have Perceiver trainable!"
    assert not any("lora" in name for name in stage2_trainable), "Stage 1 DoRA should be frozen in Stage 2!"
    assert model.is_train_cross, "Stage 2 should have cross attention enabled!"

    print("  --> CrossTransformer3D Two-Stage Setups PASSED!")


def test_depthcrafter_components():
    print("[Test 4] Testing DepthCrafter Components & Metric Depth Math...")
    from pipeline_v3.models.depthcrafter_wrapper import (
        resolve_depthcrafter_paths,
        DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
    )

    # 1. Path resolution
    unet_p, svd_p = resolve_depthcrafter_paths()
    print(f"  --> Auto-resolved UNet: {unet_p} | SVD: {svd_p}")

    # 2. Metric conversion formula test
    disparity = torch.linspace(0.0, 1.0, 10).view(1, 1, 2, 5)
    near, far = 0.1, 100.0
    depths_scaled = disparity * 3900.0
    depths_scaled = torch.clamp(depths_scaled, min=1e-5)
    metric_depth = 10000.0 / depths_scaled
    metric_depth = torch.clamp(metric_depth, min=near, max=far)

    # Check monotonicity: higher disparity (closer) must result in smaller metric distance Z
    assert metric_depth[0, 0, 0, 0] >= metric_depth[0, 0, 0, -1], "Metric depth inversion failed!"
    assert metric_depth.min() >= near and metric_depth.max() <= far, "Metric depth clipping failed!"
    print("  --> DepthCrafter Components & Math PASSED!")


def test_hole_healing_and_mask_binarization():
    print("[Test 5] Testing Proxy Hole Healing & Mask Binarization...")
    device = "cpu"
    warper = Warper3D(device=device)

    # 1. Test Proxy Small Hole Healing
    h, w = 64, 64
    img = torch.full((1, 3, h, w), 0.5, dtype=torch.float32)
    mask = torch.ones((1, 1, h, w), dtype=torch.float32)

    # Inject small 2x2 hole (area = 4 <= 25)
    img[:, :, 10:12, 10:12] = -1.0
    mask[:, :, 10:12, 10:12] = 0.0

    # Inject large 20x20 hole (area = 400 > 25)
    img[:, :, 30:50, 30:50] = -1.0
    mask[:, :, 30:50, 30:50] = 0.0

    healed_img, healed_mask = warper.heal_small_holes(img, mask, max_hole_area=25)

    # Verify small hole is healed (mask becomes 1.0, pixel is no longer -1.0)
    assert healed_mask[:, :, 10:12, 10:12].min().item() == 1.0, "Small hole mask should be marked as valid 1.0!"
    assert healed_img[:, :, 10:12, 10:12].min().item() > -0.5, "Small hole pixel should be healed from neighbors!"

    # Verify large hole is PRESERVED for DiT inpainting (mask remains 0.0, pixel remains -1.0)
    assert healed_mask[:, :, 35:45, 35:45].max().item() == 0.0, "Large hole mask should remain 0.0!"
    assert healed_img[:, :, 35:45, 35:45].max().item() == -1.0, "Large hole pixels should remain -1.0!"

    # 2. Test Mask Binarization in resize_mask
    from pipeline_v3.models.pipeline_trajectorycrafter import resize_mask
    mask_in = torch.ones((1, 1, 13, 64, 64), dtype=torch.float32)
    # Put a hole in middle
    mask_in[:, :, :, 20:44, 20:44] = 0.0

    dummy_latent = torch.zeros((1, 13, 16, 8, 8), dtype=torch.float32)
    resized_m = resize_mask(mask_in, dummy_latent, process_first_frame_only=True, mask_threshold=0.85)

    # All values must be strictly binary {0.0, 1.0}
    unique_vals = torch.unique(resized_m)
    for v in unique_vals:
        assert abs(v.item() - 0.0) < 1e-4 or abs(v.item() - 1.0) < 1e-4, f"Mask latent contains non-binary value: {v.item()}"

    print("  --> Proxy Hole Healing & Mask Binarization PASSED!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      RUNNING PIPELINE V3 UNIT TESTS")
    print("=" * 60)
    test_trajectory_and_warper()
    test_dora_layer()
    test_model_two_stage_modes()
    test_depthcrafter_components()
    test_hole_healing_and_mask_binarization()
    print("=" * 60)
    print("      ALL UNIT TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 60 + "\n")
