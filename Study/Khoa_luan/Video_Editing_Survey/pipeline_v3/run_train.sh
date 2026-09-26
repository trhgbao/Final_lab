#!/bin/bash
# ==============================================================================
# Pipeline v3: Huấn luyện DoRA 2 Giai đoạn trên Single GPU (RTX 6000 48GB / Kaggle)
# ==============================================================================

# Bước 0: Tiền xử lý DepthCrafter 3D (Nếu chưa có depth cache)
echo "=== BƯỚC 0: TIỀN XỬ LÝ DEPTHCRAFTER 3D (CACHE) ==="
python -m pipeline_v3.tools.precompute_depth \
    --video_dir "./test/videos" \
    --output_dir "./depth_cache" \
    --near 0.1 \
    --far 100.0

# Giai đoạn 1: Huấn luyện DoRA Geometry (Vá hình học & Lấp mảng đen)
echo "=== BẮT ĐẦU GIAI ĐOẠN 1: HUẤN LUYỆN DoRA HÌNH HỌC ==="
python -m pipeline_v3.training.train_stage1_geometry \
    --video_dir "./test/videos" \
    --depth_dir "./depth_cache" \
    --output_dir "./checkpoints_v3/stage1" \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --dora_r 16 \
    --dora_alpha 32.0 \
    --mixed_precision "bf16"

# Giai đoạn 2: Huấn luyện Appearance Transfer (Perceiver Cross-Attention & Độ nét)
echo "=== BẮT ĐẦU GIAI ĐOẠN 2: HUẤN LUYỆN PERCEIVER CROSS-ATTENTION ==="
python -m pipeline_v3.training.train_stage2_appearance \
    --video_dir "./test/videos" \
    --depth_dir "./depth_cache" \
    --stage1_checkpoint "./checkpoints_v3/stage1/dora_stage1_final.pt" \
    --output_dir "./checkpoints_v3/stage2" \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --mixed_precision "bf16"
