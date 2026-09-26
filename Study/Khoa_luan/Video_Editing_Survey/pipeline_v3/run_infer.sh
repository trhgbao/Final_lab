#!/bin/bash
# ==============================================================================
# Pipeline v3: Suy luận Novel View Camera Pan -30° & -60°
# Chạy mượt mà trên RTX 6000 (48GB) hoặc Kaggle (bật --low_gpu_memory_mode)
# ==============================================================================

# 1. Chạy góc Pan -30 độ
python -m pipeline_v3.inference_v3 \
    --video_path "test/videos/p7.mp4" \
    --out_dir "./outputs_v3/pan_30" \
    --target_pose 0 -30 0.3 0 0 \
    --video_length 49 \
    --sample_size 384 672 \
    --use_official_weights \
    --guidance_scale 6.0 \
    --inference_steps 30

# 2. Chạy góc Pan -60 độ (SOTA Baseline)
python -m pipeline_v3.inference_v3 \
    --video_path "test/videos/p7.mp4" \
    --out_dir "./outputs_v3/pan_60" \
    --target_pose 0 -60 0.3 0 0 \
    --video_length 49 \
    --sample_size 384 672 \
    --use_official_weights \
    --guidance_scale 6.0 \
    --inference_steps 30

# 3. Chạy với Pipeline v3 Đề xuất (Base Model + DoRA Stage 1 + Stage 2 Perceiver)
# python -m pipeline_v3.inference_v3 \
#     --video_path "test/videos/p7.mp4" \
#     --out_dir "./outputs_v3/pipeline_v3_pan_30" \
#     --target_pose 0 -30 0.3 0 0 \
#     --video_length 49 \
#     --sample_size 384 672 \
#     --model_name "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP" \
#     --dora_checkpoint "./checkpoints_v3/stage1/dora_stage1_final.pt" \
#     --stage2_checkpoint "./checkpoints_v3/stage2/stage2_final.pt" \
#     --guidance_scale 6.0 \
#     --inference_steps 30
