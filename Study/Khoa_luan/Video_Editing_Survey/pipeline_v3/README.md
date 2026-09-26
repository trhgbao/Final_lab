# Pipeline v3: Two-Stage DoRA TrajectoryCrafter (Phiên bản v3.2.0 - Cập nhật 25/09/2026)

Hệ thống **Camera Trajectory Retargeting** thế hệ thứ 3 cho video đơn góc nhìn (Monocular Video), kế thừa kiến trúc SOTA của **TrajectoryCrafter (ICCV 2025 Oral)** kết hợp công nghệ thích ứng **DoRA (Weight-Decomposed Low-Rank Adaptation) 2 giai đoạn**.

### Các Cập Nhật Mới Nhất (v3.2.0):
- **Auto-Discovery Base Model**: Tự động phát hiện CogVideoX-Fun InP offline trên Kaggle, khắc phục lỗi mất kết nối mạng.
- **Auto-Resume & Rolling Checkpoint**: Tự động nhận diện checkpoint cũ, cộng dồn step và xóa checkpoint trung gian để không tràn disk.
- **Chuẩn hóa 2 Quỹ đạo**: Pan Left 30° và Forward (Đi thẳng).
- **Bảng so sánh 4 Cột Ablation**: Baseline SOTA | 3D Scaffold | Stage 1 DoRA | Stage 2 Our Full Model (khóa ID từng mẫu).
- **Hỗ trợ `--batch_size` & `--train_batch_size`**.

Hệ thống được tối ưu hóa chuyên biệt để:
1. **Chạy suy luận (Inference)** cực nhanh trên GPU máy trạm **NVIDIA RTX 6000 (48GB)** (native bfloat16, thời gian render ~30-60s) hoặc chạy trên GPU **16GB VRAM (Kaggle T4)** với cờ `--low_gpu_memory_mode`.
2. **Huấn luyện tự giám sát (Training)** hoàn chỉnh trên **1 GPU đơn lẻ** nhờ cơ chế đóng băng 99% backbone và chia 2 giai đoạn DoRA chuẩn mực học thuật.

---

## 1. Cấu Trúc Thư Mục

```
pipeline_v3/
├── __init__.py
├── README.md                                  # Hướng dẫn chi tiết
├── geometry/                                  # Module toán hình học 3D
│   ├── __init__.py
│   ├── warper3d.py                            # 3D Point Cloud Lifting, Forward Bilinear Splatting, Z-Buffer
│   └── trajectory.py                          # Sinh ma trận quỹ đạo camera (Pan, Tilt, Zoom, Dolly)
├── models/                                    # Kiến trúc mạng nơ-ron
│   ├── __init__.py
│   ├── dora.py                                # Module DoRALinear chuẩn ICLR 2024
│   ├── crosstransformer3d_v3.py               # DiT 33-kênh tích hợp DoRA + Perceiver Cross-Attention
│   ├── pipeline_trajectorycrafter.py          # Diffusers pipeline core
│   └── autoencoder_magvit.py                  # VAE 3D nén/giải nén video latent
├── dataset/                                   # Xử lý dữ liệu & Tự giám sát
│   ├── __init__.py
│   └── double_reprojection_dataset.py         # Bộ sinh dữ liệu Double-Reprojection (I -> I' -> I'' -> I_GT)
├── training/                                  # Kịch bản huấn luyện 2 giai đoạn
│   ├── __init__.py
│   ├── train_stage1_geometry.py               # Train DoRA Stage 1 (Hình học & Inpainting)
│   └── train_stage2_appearance.py             # Train Stage 2 (Perceiver Cross-Attention & Độ nét)
├── inference_v3.py                            # Pipeline suy luận thống nhất (Hỗ trợ cả SOTA weights & DoRA)
├── run_infer.sh                               # Script chạy mẫu góc pan -30° và -60°
└── run_train.sh                               # Script chạy train 2 giai đoạn
```

---

## 2. Hướng Dẫn Sử Dụng

### A. Chạy Suy Luận (Inference với SOTA Weights)
Để kiểm tra video với góc Pan -30° hoặc -60°:

```bash
# Pan -30 độ
python -m pipeline_v3.inference_v3 \
    --video_path "test/videos/p7.mp4" \
    --out_dir "./outputs_v3/pan_30" \
    --target_pose 0 -30 0.3 0 0 \
    --video_length 49 \
    --sample_size 384 672 \
    --use_official_weights

# Pan -60 độ
python -m pipeline_v3.inference_v3 \
    --video_path "test/videos/p7.mp4" \
    --out_dir "./outputs_v3/pan_60" \
    --target_pose 0 -60 0.3 0 0 \
    --video_length 49 \
    --sample_size 384 672 \
    --use_official_weights
```

*(Lưu ý: Nếu chạy trên GPU 16GB như Kaggle T4, thêm cờ `--low_gpu_memory_mode`)*

### B. Huấn Luyện DoRA 2 Giai Đoạn (Training)

```bash
# Giai đoạn 1: Huấn luyện DoRA Geometry (Vá hình học & Lấp mảng đen)
python -m pipeline_v3.training.train_stage1_geometry \
    --video_dir "./test/videos" \
    --output_dir "./checkpoints_v3/stage1" \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --dora_r 16 \
    --dora_alpha 32.0 \
    --mixed_precision "bf16"

# Giai đoạn 2: Huấn luyện Appearance Transfer (Perceiver Cross-Attention & Độ nét)
python -m pipeline_v3.training.train_stage2_appearance \
    --video_dir "./test/videos" \
    --stage1_checkpoint "./checkpoints_v3/stage1/dora_stage1_final.pt" \
    --output_dir "./checkpoints_v3/stage2" \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --mixed_precision "bf16"
```

---

## 3. Bản Đồ Toán Học & Cơ Chế Hoạt Động

1. **Khung xương hình học 3D (`geometry/warper3d.py`):**
   - Không bắt Diffusion phải đoán mò camera di chuyển đi đâu.
   - Nâng pixel $(u, v)$ lên điểm 3D: $P = d \cdot K^{-1} [u, v, 1]^T$.
   - Xoay theo camera mới: $P' = T_2 \cdot T_1^{-1} \cdot P$.
   - Chiếu về khung hình mới bằng Bilinear Splatting kết hợp Z-buffer $\rightarrow$ Cho ra ảnh đã xoay $I^r$ và Mask lỗ thủng $M^r$.

2. **Patch Embedding 33-Kênh:**
   - Ghép: `noisy_latents (16ch)` + `mask_latents (1ch)` + `masked_video_latents (16ch)` $= 33\text{ channels}$.
   - Neo giữ chính xác tọa độ vật thể ở góc nhìn mới.

3. **Perceiver Cross-Attention:**
   - Token đang khử nhiễu gửi Query $Q$ tới Key/Value $K, V$ của video gốc $I^s$.
   - Truyền tải chất liệu vải, vân tường và ánh sáng phản chiếu cực nét vào vùng bị thủng và vùng méo viền.

4. **DoRA (Weight-Decomposed Low-Rank Adaptation):**
   $$W = m \odot \frac{W_0 + \frac{\alpha}{r} (B A)}{\|W_0 + \frac{\alpha}{r} (B A)\|_c}$$
   - Tách rời hướng (Direction) và độ lớn (Magnitude).
   - Đạt 98% hiệu quả của Full Fine-Tuning nhưng chỉ tốn < 1% tham số, vừa vặn hoàn hảo trong 48GB VRAM của RTX 6000.
