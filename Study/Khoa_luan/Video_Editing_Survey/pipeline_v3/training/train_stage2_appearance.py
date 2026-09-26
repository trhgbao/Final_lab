"""
Stage 2 Training: Appearance Transfer with Perceiver Cross-Attention for Pipeline v3.
Freezes Stage 1 Geometry DoRA, unfreezes and trains PerceiverCrossAttention & reference patch embed.
Learns to query sharp high-frequency texture, lighting, and reflections from the clean source video.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDIMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed

import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_pipeline_dir = _current_dir.parent
for _p in ["/kaggle/working", "/kaggle/working/pipeline_v3", str(_current_dir), str(_pipeline_dir), str(_pipeline_dir.parent)]:
    if os.path.exists(_p) and _p in sys.path:
        sys.path.remove(_p)
    if os.path.exists(_p) or not _p.startswith("/kaggle"):
        sys.path.insert(0, _p)

try:
    from ..models.crosstransformer3d import CrossTransformer3DModel
    from ..models.autoencoder_magvit import AutoencoderKLCogVideoX
    from ..dataset.double_reprojection_dataset import DoubleReprojectionDataset
except (ImportError, ValueError):
    from pipeline_v3.models.crosstransformer3d import CrossTransformer3DModel
    from pipeline_v3.models.autoencoder_magvit import AutoencoderKLCogVideoX
    from pipeline_v3.dataset.double_reprojection_dataset import DoubleReprojectionDataset


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 Appearance Transfer for Pipeline v3")
    parser.add_argument("--video_dir", "--data_root", dest="video_dir", type=str, default=None, help="Directory containing training mp4 videos")
    parser.add_argument("--depth_dir", "--depth_cache_dir", dest="depth_dir", type=str, default=None, help="Directory containing cached depth npz files")
    parser.add_argument("--stage1_checkpoint", "--stage1_dora_checkpoint", dest="stage1_checkpoint", type=str, default=None, help="Path to Stage 1 checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP")
    parser.add_argument("--transformer_path", type=str, default=None, help="Path to TrajectoryCrafter weights")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage2")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--train_batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dora_r", type=int, default=16, help="Rank of Stage 1 DoRA to match checkpoint")
    parser.add_argument("--dora_alpha", type=float, default=32.0, help="Alpha of Stage 1 DoRA to match checkpoint")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=50, help="Print training progress every N steps")
    parser.add_argument("--max_train_steps", type=int, default=-1, help="Max training steps (-1 for unlimited by epochs)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to Stage 2 checkpoint to resume training from")
    parser.add_argument("--resume_step", type=int, default=0, help="Initial step to resume counting from")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train():
    args = parse_args()

    # Auto-discover video_dir if not specified
    if not args.video_dir:
        cand_dirs = [
            "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini/test_256",
            "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini",
            "/kaggle/input/datasets/nguynngcnhntrng/realestate10k",
            "/kaggle/input/realestate10k",
            "./examples",
        ]
        for cd in cand_dirs:
            if os.path.exists(cd):
                args.video_dir = cd
                print(f"--> [AutoDiscovery] Found video_dir: {args.video_dir}")
                break
    if not args.video_dir or not os.path.exists(args.video_dir):
        raise ValueError("Missing --video_dir or --data_root argument, and no candidate video directory was found!")

    # Force purge VRAM from previous notebook cells
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)

    # Auto-discover model_name (Base CogVideoX-Fun InP)
    if not args.model_name or not os.path.exists(args.model_name):
        model_candidates = [
            "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
            "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
            "/kaggle/input/cogvideox-fun-inp",
            "./checkpoints/CogVideoX-Fun-V1.1-5b-InP",
        ]
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "vae" in dirs and os.path.exists(os.path.join(root, "vae", "config.json")):
                    model_candidates.append(root)
                    break
        for mc in model_candidates:
            if os.path.exists(mc):
                args.model_name = mc
                print(f"--> [AutoDiscovery] Found Base Model Dir: {args.model_name}")
                break

    args.model_name = resolve_dir_with_target(args.model_name, "vae", "config.json")
    if args.transformer_path:
        args.transformer_path = resolve_transformer_dir(args.transformer_path)
    else:
        # Auto-discover transformer if separated from base model
        trans_candidates = [
            "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer/transformer",
            "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer",
            "/kaggle/input/cogvideox-fun-transformer/transformer",
        ]
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "config.json" in files and any(f.endswith(".safetensors") for f in files) and "vae" not in root.lower() and "text_encoder" not in root.lower():
                    trans_candidates.append(root)
        for tc in trans_candidates:
            if os.path.exists(tc):
                args.transformer_path = resolve_transformer_dir(tc)
                print(f"--> [AutoDiscovery] Found Base Transformer dir: {args.transformer_path}")
                break

    # Auto-discover depth_dir
    candidate_depth_dirs = []
    if args.depth_dir:
        candidate_depth_dirs.append(args.depth_dir)
        candidate_depth_dirs.append(os.path.join(args.depth_dir, "depth_cache"))
    candidate_depth_dirs.extend([
        "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
        "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
        "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
        "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache",
        "/kaggle/input/realestate10k-depthcrafter-cache/depth_cache",
        "/kaggle/input/realestate10k-depthcrafter-cache",
        "/kaggle/working/depth_cache",
        "./depth_cache",
    ])
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if any(f.endswith(".npz") for f in files):
                candidate_depth_dirs.append(root)

    found_depth_dir = None
    for c in candidate_depth_dirs:
        if os.path.exists(c):
            npz_count = len([f for f in os.listdir(c) if f.endswith(".npz")])
            if npz_count > 0:
                found_depth_dir = c
                print(f"--> [DepthCache] Auto-detected cached depth directory ({npz_count} files): {found_depth_dir}")
                break
    if found_depth_dir:
        args.depth_dir = found_depth_dir

    # Auto-discover stage1_checkpoint (Prioritize newly trained in /kaggle/working/checkpoints_stage1 or datasets)
    def parse_step_num(path):
        if not path:
            return 0
        base = os.path.basename(path).lower()
        m = re.search(r"step(\d+)", base)
        if m:
            return int(m.group(1))
        nums = re.findall(r"\d+", base)
        if len(nums) > 1:
            return int(nums[-1])
        elif len(nums) == 1 and not ("stage" in base and nums[0] in ["1", "2"]):
            return int(nums[0])
        return 999999 if "final" in base else 0

    if not args.stage1_checkpoint or not os.path.exists(args.stage1_checkpoint):
        cand_s1_dirs = [
            "/kaggle/working/checkpoints_stage1",
            "/kaggle/working/checkpoints_dora_stage1",
            "/kaggle/input/datasets/tranbao0105/pipeline",
            "/kaggle/input/pipeline",
            "./checkpoints_stage1",
        ]
        found_s1_ckpts = []
        for cd in cand_s1_dirs:
            if os.path.exists(cd):
                for f in glob.glob(os.path.join(cd, "dora_stage1_*")):
                    if os.path.isfile(f) and os.path.getsize(f) > 1024:
                        found_s1_ckpts.append(f)
                for f in glob.glob(os.path.join(cd, "dora_checkpoint_*")):
                    if os.path.isfile(f) and os.path.getsize(f) > 1024:
                        found_s1_ckpts.append(f)
        if found_s1_ckpts:
            sorted_s1 = sorted(set(found_s1_ckpts), key=lambda x: (parse_step_num(x), os.path.getmtime(x)))
            args.stage1_checkpoint = sorted_s1[-1]
            s1_step = parse_step_num(args.stage1_checkpoint)
            print(f"--> [Stage 1 Checkpoint] Tìm thấy {len(sorted_s1)} checkpoint Stage 1. Đã chọn checkpoint có step lớn nhất: {args.stage1_checkpoint} (Step: {s1_step})")

    print("=" * 70)
    print("   PIPELINE V3: STAGE 2 TRAINING (APPEARANCE CROSS-ATTENTION TRANSFER)")
    print("=" * 70)

    # 1. Load Models
    print(f"--> Loading VAE from {args.model_name}...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_name, subfolder="vae", local_files_only=True
    ).to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    scheduler = DDIMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler", local_files_only=True
    )

    if args.transformer_path and os.path.exists(args.transformer_path):
        print(f"--> Loading Base Transformer from: {args.transformer_path} with Stage 2 Cross-Attention enabled...")
        try:
            transformer = CrossTransformer3DModel.from_pretrained_2d(
                args.transformer_path,
                subfolder=None,
                transformer_additional_kwargs={"is_train_cross": True},
            ).to(device, dtype=weight_dtype)
        except Exception as e:
            print(f"--> [Fallback] from_pretrained_2d failed ({e}), trying from_pretrained...")
            transformer = CrossTransformer3DModel.from_pretrained(
                args.transformer_path,
                transformer_additional_kwargs={"is_train_cross": True},
                local_files_only=True,
            ).to(device, dtype=weight_dtype)
    else:
        print(f"--> Loading Base Transformer from {args.model_name} with Stage 2 Cross-Attention enabled...")
        transformer = CrossTransformer3DModel.from_pretrained_2d(
            args.model_name,
            subfolder="transformer",
            transformer_additional_kwargs={"is_train_cross": True},
        ).to(device, dtype=weight_dtype)

    # 2. Inject DoRA and Load Stage 1 Checkpoint
    print(f"--> Injecting DoRA (rank={args.dora_r}, alpha={args.dora_alpha}) and loading Stage 1 Geometry checkpoint from {args.stage1_checkpoint}...")
    transformer.enable_dora(r=args.dora_r, lora_alpha=args.dora_alpha)
    if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
        transformer.load_dora_checkpoint(args.stage1_checkpoint)
        transformer.merge_dora()
        print("✅ Đã nạp và hợp nhất (merge) thành công DoRA Stage 1 vào Base Transformer!")
    else:
        print("⚠️ [Cảnh báo] Không tìm thấy checkpoint Stage 1, bắt đầu huấn luyện từ trọng số gốc.")

    # 3. Setup Stage 2 Training (Freeze Stage 1 DoRA, unfreeze Perceiver Cross-Attention & Ref Patch Embed)
    transformer.setup_two_stage_training(stage=2)
    transformer.gradient_checkpointing = True

    # Auto-resume Stage 2 checkpoint if available
    start_step = 0
    resume_s2_ckpt = None
    if getattr(args, "resume_from_checkpoint", None) and os.path.exists(args.resume_from_checkpoint):
        resume_s2_ckpt = args.resume_from_checkpoint
    else:
        cand_s2_dirs = [
            "/kaggle/input/datasets/tranbao0105/pipeline",
            "/kaggle/input/pipeline",
            "/kaggle/working/checkpoints_stage2",
            "/kaggle/working/checkpoints_dora_stage2",
            args.output_dir,
        ]
        found_s2_ckpts = []
        for cd in cand_s2_dirs:
            if os.path.exists(cd):
                for f in glob.glob(os.path.join(cd, "dora_stage2_*")):
                    if os.path.isfile(f) and os.path.getsize(f) > 1024:
                        found_s2_ckpts.append(f)
                for f in glob.glob(os.path.join(cd, "stage2_*")):
                    if os.path.isfile(f) and os.path.getsize(f) > 1024:
                        found_s2_ckpts.append(f)

        if found_s2_ckpts:
            sorted_s2 = sorted(set(found_s2_ckpts), key=lambda x: (parse_step_num(x), os.path.getmtime(x)))
            resume_s2_ckpt = sorted_s2[-1]
            max_s2_step = parse_step_num(resume_s2_ckpt)
            print(f"--> [AutoDiscovery] Tìm thấy {len(sorted_s2)} checkpoint Stage 2. Đã chọn checkpoint có step lớn nhất: {resume_s2_ckpt} (Step: {max_s2_step})")

    if resume_s2_ckpt and os.path.exists(resume_s2_ckpt):
        start_step = parse_step_num(resume_s2_ckpt)
        if start_step >= 999999:
            start_step = 1000
        if getattr(args, "resume_step", 0) > 0:
            start_step = args.resume_step
        print(f"--> [Resume Training] Tìm thấy checkpoint Stage 2: {resume_s2_ckpt}")
        print(f"--> [Resume Training] Tiếp tục huấn luyện Appearance Cross-Attention từ Step {start_step}...")
        transformer.load_stage2_checkpoint(resume_s2_ckpt)
    else:
        print(f"--> [New Training] Không tìm thấy checkpoint Stage 2 cũ, bắt đầu huấn luyện từ Step 0.")


    # 4. Collect Trainable Parameters (Perceiver + Ref Embed)
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"--> Total parameters: {total_params / 1e6:.2f}M | Trainable Perceiver parameters: {total_trainable / 1e6:.2f}M ({total_trainable / total_params * 100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    # 5. Dataset & DataLoader
    dataset = DoubleReprojectionDataset(
        video_paths=[args.video_dir],
        depth_dir=args.depth_dir,
        num_frames=49,
        sample_size=(384, 672),
        device=args.device,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Precompute rotary pos embedding matching CogVideoX / TrajectoryCrafter inference
    grid_h = 384 // 16
    grid_w = 672 // 16
    base_size_w = 720 // 16
    base_size_h = 480 // 16

    r = grid_h / grid_w
    if r > (base_size_h / base_size_w):
        resize_h = base_size_h
        resize_w = int(round(base_size_h / grid_h * grid_w))
    else:
        resize_w = base_size_w
        resize_h = int(round(base_size_w / grid_w * grid_h))
    crop_t = int(round((base_size_h - resize_h) / 2.0))
    crop_l = int(round((base_size_w - resize_w) / 2.0))
    grid_crops_coords = ((crop_t, crop_l), (crop_t + resize_h, crop_l + resize_w))

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer.config.attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_h, grid_w),
        temporal_size=13,
        use_real=True,
    )
    image_rotary_emb = (
        freqs_cos.to(device, dtype=weight_dtype),
        freqs_sin.to(device, dtype=weight_dtype),
    )

    # 6. Training Loop
    global_step = 0
    last_checkpoint_path = None
    transformer.train()

    prompt_embeds = torch.zeros((args.batch_size, 226, 4096), device=device, dtype=weight_dtype)

    if args.max_train_steps > start_step:
        steps_to_train = args.max_train_steps - start_step
    elif args.max_train_steps > 0:
        steps_to_train = args.max_train_steps
    else:
        steps_to_train = -1

    total_epochs = (steps_to_train + len(dataloader) - 1) // max(len(dataloader), 1) if steps_to_train > 0 else args.num_epochs
    target_max_step = args.max_train_steps if args.max_train_steps > start_step else (start_step + args.max_train_steps if args.max_train_steps > 0 else -1)
    print(f"--> [Training Schedule] Kế hoạch huấn luyện Stage 2:")
    print(f"    • Step khởi điểm: {start_step}")
    print(f"    • Target Max Step: {target_max_step if target_max_step > 0 else 'Theo epochs'}")
    print(f"    • Số steps cần chạy trong phiên này: {steps_to_train if steps_to_train > 0 else total_epochs * len(dataloader)} ({total_epochs} epochs, {len(dataloader)} samples/epoch)")

    running_loss = 0.0
    log_count = 0

    for epoch in range(total_epochs):
        progress_bar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{total_epochs}")
        for batch in progress_bar:
            target_vid = batch["target_video"].to(device, dtype=weight_dtype)  # (B, 3, F, H, W)
            warped_vid = batch["warped_video"].to(device, dtype=weight_dtype)  # (B, 3, F, H, W)
            mask_vid = batch["mask_video"].to(device, dtype=weight_dtype)      # (B, 1, F, H, W)
            ref_vid = batch["ref_video"].to(device, dtype=weight_dtype)        # (B, 3, F_ref, H, W)

            with torch.no_grad():
                target_latents = vae.encode(target_vid)[0].sample() * vae.config.scaling_factor
                target_latents = target_latents.permute(0, 2, 1, 3, 4)  # (B, F_lat, 16, H_lat, W_lat)

                warped_latents = vae.encode(warped_vid)[0].sample() * vae.config.scaling_factor
                warped_latents = warped_latents.permute(0, 2, 1, 3, 4)

                f_lat, h_lat, w_lat = target_latents.shape[1], target_latents.shape[3], target_latents.shape[4]
                mask_resized = torch.nn.functional.interpolate(
                    mask_vid, size=(f_lat, h_lat, w_lat), mode="trilinear", align_corners=False
                )
                mask_latents = mask_resized.permute(0, 2, 1, 3, 4) * vae.config.scaling_factor

                inpaint_latents = torch.cat([mask_latents, warped_latents], dim=2)

                # Encode clean reference frames to reference latents for Appearance Attention
                ref_latents = vae.encode(ref_vid)[0].sample() * vae.config.scaling_factor
                ref_latents = ref_latents.permute(0, 2, 1, 3, 4)

                # Free raw video pixel tensors immediately from VRAM
                del target_vid, warped_vid, mask_vid, ref_vid

                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

                # Compute ground truth target according to prediction_type
                if getattr(scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    target = noise

            # Enable grad on noisy_latents so that PyTorch gradient checkpointing functions properly with frozen base & DoRA
            noisy_latents.requires_grad_(True)

            # Forward pass: Query (warped) attends to Key/Value (ref_latents) via PerceiverCrossAttention
            prompt_embeds_batch = prompt_embeds[:bsz] if prompt_embeds.shape[0] >= bsz else prompt_embeds.repeat(bsz, 1, 1)[:bsz]
            model_pred = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds_batch,
                timestep=timesteps,
                inpaint_latents=inpaint_latents,
                cross_latents=ref_latents,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            current_total_step = start_step + global_step
            loss_val = loss.item() * args.gradient_accumulation_steps
            running_loss += loss_val
            log_count += 1
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "step": current_total_step})

            # In thông tin huấn luyện định kỳ mỗi args.log_steps (mặc định 50 steps), hoặc ở step đầu tiên / cuối cùng
            if args.log_steps > 0 and (current_total_step % args.log_steps == 0 or global_step == 1 or (target_max_step > 0 and current_total_step >= target_max_step)):
                avg_loss = running_loss / max(log_count, 1)
                lr_curr = optimizer.param_groups[0]["lr"]
                print(f"🚀 [Stage 2 Train] Step {current_total_step:04d}/{target_max_step} | Loss: {loss_val:.4f} (Avg: {avg_loss:.4f}) | LR: {lr_curr:.2e} | Epoch {epoch+1}/{total_epochs}", flush=True)
                running_loss = 0.0
                log_count = 0

            # Save Checkpoint with rolling cleanup to save disk space
            if args.save_steps > 0 and (current_total_step % args.save_steps == 0 or global_step % args.save_steps == 0):
                save_path = os.path.join(args.output_dir, f"dora_stage2_step{current_total_step}.pt")
                transformer.save_stage2_checkpoint(save_path)
                print(f"\n💾 [Checkpoint Saved] Đã lưu checkpoint: {os.path.basename(save_path)} (Step: {current_total_step})", flush=True)
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    if last_checkpoint_path and os.path.exists(last_checkpoint_path) and last_checkpoint_path != save_path:
                        try:
                            os.remove(last_checkpoint_path)
                            print(f"--> [Disk Cleanup] Đã giải phóng bộ nhớ, xóa checkpoint cũ: {os.path.basename(last_checkpoint_path)}", flush=True)
                        except Exception:
                            pass
                    last_checkpoint_path = save_path

            if target_max_step > 0 and current_total_step >= target_max_step:
                print(f"\n--> [Target Step Reached] Đã chạm mốc target max step ({target_max_step}). Dừng huấn luyện Stage 2.")
                break

        if target_max_step > 0 and current_total_step >= target_max_step:
            break

    total_steps = start_step + global_step
    final_step_path = os.path.join(args.output_dir, f"dora_stage2_step{total_steps}.pt")
    transformer.save_stage2_checkpoint(final_step_path)

    # Dọn dẹp checkpoint phụ cuối cùng nếu khác final step
    if last_checkpoint_path and os.path.exists(last_checkpoint_path) and last_checkpoint_path != final_step_path:
        try:
            os.remove(last_checkpoint_path)
        except Exception:
            pass

    print("=" * 70)
    print(f"--> [Done] Stage 2 Appearance Training Completed! Đã lưu checkpoint: {final_step_path} (Tổng steps: {total_steps})")
    print("=" * 70)

    # Explicit memory cleanup
    del vae, transformer, optimizer, dataloader, dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    print("🧹 [Cleanup] Đã thu hồi toàn bộ VRAM của Stage 2.")


if __name__ == "__main__":
    train()
