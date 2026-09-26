# Hướng Dẫn Vận Hành Thực Nghiệm Pipeline v2 Trên Kaggle Notebook (8 Cells Chuẩn)

Tài liệu này chứa toàn bộ mã nguồn của các ô lệnh (Notebook Cells) được thiết kế riêng cho **Pipeline v2 (Wan2.1-1.3B DiT + Decoupled Dual-Stream Attention + 8D Plücker Ray Fields + 2S-PCL Curriculum)** trên môi trường Kaggle (GPU 1x T4 16GB, 2x T4 hoặc P100).

---

## 1. Bản Đồ Quy Trình 8 Cells

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       QUY TRÌNH THỰC THI 8 CELLS PIPELINE V2 TRÊN KAGGLE                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ [CELL 1] THIẾT LẬP OFFLINE, ĐỒNG BỘ MÃ NGUỒN PIPELINE V2 & BIẾN MÔI TRƯỜNG                  │
│          - Khóa mạng 100% (HF/Diffusers Offline).                                           │
│          - Tự động giải nén/sao chép pipeline_v2 từ /kaggle/input/.                         │
│          - Tự dò tìm RealEstate10K, Wan2.1 weights, Wan2.1_VAE.pth, Sample Video.           │
│                                                                                             │
│ [CELL 2] KIỂM ĐỊNH TIỀN KHỞI CHẠY & CHỐNG FALLBACK (PRE-FLIGHT ZERO-FALLBACK SUITE) [MỚI]  │
│          - Chạy check_system_integrity.py quét 6 thành phần cốt lõi.                       │
│          - Xác thực 100% Wan2.1_VAE.pth nạp khớp (Missing: 0, Unexpected: 0).               │
│          - Thử nghiệm Encode/Decode VAE, 8D Plücker rays, DDSA DiT forward, NU-AB2 ODE.     │
│          - Báo cáo bảng kiểm định trạng thái: 🟢 PASS / 🟡 WARN / 🔴 FAIL trước khi chạy.    │
│                                                                                             │
│ [CELL 3] HUẤN LUYỆN 2S-PCL CURRICULUM TỪ ĐẦU (Wan2.1-1.3B DiT + Stage 1 Geometry)         │
│          - Tự động lưu Checkpoint Step 0 (adapter_step_0.pt) làm baseline chuẩn.            │
│          - Chạy train_v2.py (GADA-BF16, MP-PGO, ACO-EMA, Stage 1 Geometry).                 │
│                                                                                             │
│ [CELL 4] VIDEO 1A: XOAY ĐƠN GIẢN (-30°) - TRƯỚC KHI TRAIN (Step 0 Baseline)                 │
│          - Chạy inference_pipeline_v2.py với adapter_step_0.pt.                             │
│          - Xuất video chuẩn đối chứng nguyên bản: outputs/video_1A_simple_pan_before.mp4    │
│                                                                                             │
│ [CELL 5] VIDEO 1B: XOAY ĐƠN GIẢN (-30°) - SAU KHI TRAIN (Max Step Checkpoint)               │
│          - Tự động tìm checkpoint có số step cao nhất vừa train xong.                       │
│          - Giải mã qua WanVideoVAE thật: outputs/video_1B_simple_pan_after.mp4              │
│                                                                                             │
│ [CELL 6] VIDEO 2A: QUỸ ĐẠO PHỨC TẠP (CRANE UP + PAN 90°) - TRƯỚC KHI TRAIN (Step 0)         │
│          - Chạy quỹ đạo phức tạp với adapter_step_0.pt: crane_left_90                       │
│          - Tạo video chuẩn đối chứng: outputs/video_2A_complex_traj_before.mp4              │
│                                                                                             │
│ [CELL 7] VIDEO 2B: QUỸ ĐẠO PHỨC TẠP (CRANE UP + PAN 90°) - SAU KHI TRAIN (Max Step)         │
│          - Chạy quỹ đạo phức tạp với checkpoint tối ưu nhất.                                │
│          - Giải mã qua WanVideoVAE thật: outputs/video_2B_complex_traj_after.mp4            │
│                                                                                             │
│ [CELL 8] ĐỐI CHỨNG TRƯỚC - SAU: ĐO LƯỜNG ĐỊNH LƯỢNG MỤC 7 (Pixel Diff, PSNR, SSIM) & HTML5 │
│          - Tính toán độ lệch điểm ảnh trung bình (Mean Pixel Diff).                         │
│          - Tính toán chỉ số PSNR và SSIM giữa 2 video đối chứng.                            │
│          - Phát trực tiếp 2 cặp video độc lập với HTML5 Video Player qua Base64.            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi Tiết Mã Nguồn 8 Cells

```python
# ==============================================================================
# [CELL 1] THIẾT LẬP OFFLINE, ĐỒNG BỘ MÃ NGUỒN PIPELINE V2 & THIẾT LẬP ĐƯỜNG DẪN
# ==============================================================================
import os
import sys
import glob

# 1. Khóa hoàn toàn kết nối mạng của HuggingFace / Diffusers (Bảo đảm Offline 100%)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

DEST_CODE = "/kaggle/working/pipeline_v2_workspace"
os.makedirs(DEST_CODE, exist_ok=True)
os.makedirs(os.path.join(DEST_CODE, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(DEST_CODE, "outputs"), exist_ok=True)

# 2. Tự động đồng bộ mã nguồn mới nhất (Hỗ trợ file .zip lẫn thư mục giải nén sẵn)
zip_candidates = glob.glob("/kaggle/input/**/pipeline_v2*.zip", recursive=True)
src_candidates = glob.glob("/kaggle/input/**/train_v2.py", recursive=True)

if zip_candidates:
    print(f"📦 Phát hiện file zip mã nguồn: {zip_candidates[0]}")
    print(">>> Đang giải nén mã nguồn mới nhất (Ghi đè -o)...")
    !unzip -o -q "{zip_candidates[0]}" -d "{DEST_CODE}/"
elif src_candidates:
    src_dir = os.path.dirname(src_candidates[0])
    print(f"📁 Phát hiện thư mục mã nguồn thực: {src_dir}")
    print(">>> Đang sao chép mã nguồn mới nhất (cp -rf)...")
    !cp -rf "{src_dir}"/* "{DEST_CODE}/"
else:
    print("⚠️ Không tìm thấy nguồn code trong /kaggle/input/, sử dụng code hiện có trong working.")

%cd $DEST_CODE

# 3. Tự động dò tìm đường dẫn Dataset, Trọng số Wan2.1, VAE và Video Mẫu
def auto_find_dir(keywords):
    for kw in keywords:
        matches = glob.glob(f"/kaggle/input/**/{kw}", recursive=True)
        for m in matches:
            if os.path.isdir(m):
                return m
    return ""

DATA_DIR = auto_find_dir(["RealEstate10K_Mini", "realestate10k", "real-estate-10k-mini", "cameractrl-sota-weights"])

# Dò tìm thư mục chứa trọng số Wan2.1
wan_files = glob.glob("/kaggle/input/**/diffusion_pytorch_model.safetensors", recursive=True)
if wan_files:
    WAN_PATH = os.path.dirname(wan_files[0])
else:
    WAN_PATH = auto_find_dir(["wan2.1_weights", "Wan2.1-T2V-1.3B", "wan2.1", "wan_weights"])

# Dò tìm trọng số VAE của Wan2.1 (Wan2.1_VAE.pth)
vae_files = glob.glob("/kaggle/input/**/Wan2.1_VAE.pth", recursive=True)
if not vae_files:
    vae_files = glob.glob("/kaggle/input/**/wan2.1_vae*.pth", recursive=True)
VAE_PATH = vae_files[0] if vae_files else ""

# Tìm video test mẫu cho bước inference
video_matches = glob.glob("/kaggle/input/**/*.mp4", recursive=True)
SAMPLE_VIDEO = ""
for v in video_matches:
    if "1910e79a60d57aa7.mp4" in v or "sample_video.mp4" in v:
        SAMPLE_VIDEO = v
        break
if not SAMPLE_VIDEO and video_matches:
    SAMPLE_VIDEO = video_matches[0]

# Xuất biến môi trường cho các cell inference phía sau sử dụng
os.environ["DATA_DIR"] = DATA_DIR
os.environ["WAN_PATH"] = WAN_PATH
os.environ["VAE_PATH"] = VAE_PATH
os.environ["SAMPLE_VIDEO"] = SAMPLE_VIDEO

print("=" * 75)
print(f"  * Pipeline Workspace : {DEST_CODE}")
print(f"  * Dataset Root       : {DATA_DIR if DATA_DIR else 'Tự động tạo mẫu hình học mô phỏng'}")
print(f"  * Wan2.1 Foundation  : {WAN_PATH if WAN_PATH else 'Khởi tạo Wan2.1-1.3B DiT từ đầu'}")
print(f"  * Wan2.1 VAE         : {VAE_PATH if VAE_PATH else 'Không tìm thấy (Tự động tìm kiếm)'}")
print(f"  * Sample Video       : {SAMPLE_VIDEO if SAMPLE_VIDEO else 'Sử dụng video demo hình học 3D'}")
print("=" * 75)
```

```python
# ==============================================================================
# [CELL 2] KIỂM ĐỊNH HỆ THỐNG TOÀN DIỆN & CHỐNG FALLBACK (PRE-FLIGHT DIAGNOSTICS)
# Chạy chẩn đoán nhanh (< 15 giây) để đảm bảo 100% các thành phần sẵn sàng,
# không bị thiếu file trọng số, không bị lỗi NaN và không bị fallback âm thầm.
# ==============================================================================
!python check_system_integrity.py
```

```python
# ==============================================================================
# [CELL 3] HUẤN LUYỆN 2S-PCL CURRICULUM TỪ ĐẦU (Wan2.1-1.3B DiT + Stage 1 Geometry)
# ==============================================================================
# - GADA-BF16: Tự động bfloat16 không tiêu tốn thêm VRAM
# - MP-PGO: 8-bit AdamW cho ma trận lớn + FP32 AdamW cho cổng & norm
# - ACO-EMA: Shadow weights trên CPU Pinned Memory (0 MB GPU VRAM)
# - Tự động lưu Checkpoint Step 0 (adapter_step_0.pt) trước khi huấn luyện
!python train_v2.py \
    --data_dir "$DATA_DIR" \
    --wan_weights "$WAN_PATH" \
    --output_dir "./checkpoints" \
    --stage 1 \
    --num_frames 8 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --save_steps 500 \
    --mixed_precision bf16 \
    --use_8bit_adam \
    --use_ema
```

```python
# ==============================================================================
# [CELL 4] VIDEO 1A: XOAY ĐƠN GIẢN (-30°) - TRƯỚC KHI TRAIN (Step 0 Baseline)
# ==============================================================================
!python inference_pipeline_v2.py \
    --source_video "$SAMPLE_VIDEO" \
    --wan_weights "$WAN_PATH" \
    --vae_path "$VAE_PATH" \
    --adapter_checkpoint "./checkpoints/adapter_step_0.pt" \
    --traj_type pan \
    --pan_angle -30.0 \
    --num_frames 25 \
    --num_inference_steps 25 \
    --fps 8 \
    --output_path "outputs/video_1A_simple_pan_before.mp4"
```

```python
# ==============================================================================
# [CELL 5] VIDEO 1B: XOAY ĐƠN GIẢN (-30°) - SAU KHI TRAIN (Latest Step Checkpoint)
# ==============================================================================
import glob
import os

# Tự động chọn checkpoint có số step lớn nhất vừa train xong (bỏ qua file hỏng/incomplete)
ckpts = [
    f for f in glob.glob("./checkpoints/adapter_step_*.pt")
    if not f.endswith("_0.pt") and os.path.exists(f) and os.path.getsize(f) > 1024 * 1024
]
latest_ckpt = max(ckpts, key=lambda f: int(f.split("_step_")[-1].replace(".pt", ""))) if ckpts else "./checkpoints/adapter_step_0.pt"

print(f">>> Checkpoint sau khi train được chọn (Max Step): {latest_ckpt}")

!python inference_pipeline_v2.py \
    --source_video "$SAMPLE_VIDEO" \
    --wan_weights "$WAN_PATH" \
    --vae_path "$VAE_PATH" \
    --adapter_checkpoint "$latest_ckpt" \
    --traj_type pan \
    --pan_angle -30.0 \
    --num_frames 25 \
    --num_inference_steps 25 \
    --fps 8 \
    --output_path "outputs/video_1B_simple_pan_after.mp4"
```

```python
# ==============================================================================
# [CELL 6] VIDEO 2A: QUỸ ĐẠO PHỨC TẠP (CRANE UP + PAN 90°) - TRƯỚC KHI TRAIN (Step 0)
# ==============================================================================
!python inference_pipeline_v2.py \
    --source_video "$SAMPLE_VIDEO" \
    --wan_weights "$WAN_PATH" \
    --vae_path "$VAE_PATH" \
    --adapter_checkpoint "./checkpoints/adapter_step_0.pt" \
    --traj_type crane_left_90 \
    --pan_angle -90.0 \
    --trans_y -0.35 \
    --num_frames 25 \
    --num_inference_steps 25 \
    --fps 8 \
    --output_path "outputs/video_2A_complex_traj_before.mp4"
```

```python
# ==============================================================================
# [CELL 7] VIDEO 2B: QUỸ ĐẠO PHỨC TẠP (CRANE UP + PAN 90°) - SAU KHI TRAIN (Latest Step)
# ==============================================================================
import glob
import os

ckpts = [
    f for f in glob.glob("./checkpoints/adapter_step_*.pt")
    if not f.endswith("_0.pt") and os.path.exists(f) and os.path.getsize(f) > 1024 * 1024
]
latest_ckpt = max(ckpts, key=lambda f: int(f.split("_step_")[-1].replace(".pt", ""))) if ckpts else "./checkpoints/adapter_step_0.pt"

print(f">>> Checkpoint sau khi train được chọn (Max Step): {latest_ckpt}")

!python inference_pipeline_v2.py \
    --source_video "$SAMPLE_VIDEO" \
    --wan_weights "$WAN_PATH" \
    --vae_path "$VAE_PATH" \
    --adapter_checkpoint "$latest_ckpt" \
    --traj_type crane_left_90 \
    --pan_angle -90.0 \
    --trans_y -0.35 \
    --num_frames 25 \
    --num_inference_steps 25 \
    --fps 8 \
    --output_path "outputs/video_2B_complex_traj_after.mp4"
```

```python
# ==============================================================================
# [CELL 8] ĐỐI CHỨNG TRƯỚC - SAU: ĐO LƯỜNG ĐỊNH LƯỢNG MỤC 7 (Pixel Diff, PSNR, SSIM) & HTML5
# ==============================================================================
import os
import cv2
import base64
import numpy as np
import torch
import imageio
from IPython.display import HTML, display

from pipeline_v2.evaluation.photometric_metrics import compute_psnr_and_ssim

def ensure_browser_playable_mp4(mp4_path):
    """Chuyển đổi MP4 sang chuẩn H.264 (yuv420p) để tương thích trình duyệt web."""
    if not os.path.exists(mp4_path):
        return mp4_path
    h264_path = mp4_path.replace(".mp4", "_h264.mp4")
    if not os.path.exists(h264_path):
        os.system(f"ffmpeg -y -i '{mp4_path}' -vcodec libx264 -pix_fmt yuv420p -movflags +faststart '{h264_path}' >/dev/null 2>&1")
    return h264_path if os.path.exists(h264_path) else mp4_path

def mp4_to_animated_gif_b64(mp4_path, max_frames=25, fps=8):
    """Chuyển đổi MP4 thành Animated GIF Base64 siêu nhẹ - đảm bảo 100% tự động chạy trên mọi trình duyệt Kaggle."""
    if not os.path.exists(mp4_path):
        return ""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize nhỏ gọn 320x184 để tải tức thì không giật lag
        small = cv2.resize(frame, (320, 184), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return ""
    try:
        gif_bytes = imageio.mimsave(imageio.RETURN_BYTES, frames, format="GIF", fps=fps)
        return base64.b64encode(gif_bytes).decode("utf-8")
    except Exception as e:
        return ""

def generate_filmstrip_b64(mp4_path, sample_indices=[0, 8, 16, 24]):
    """Tạo thanh cuộn ảnh (Filmstrip) 4 khung hình đại diện."""
    if not os.path.exists(mp4_path):
        return ""
    cap = cv2.VideoCapture(mp4_path)
    all_frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    cap.release()
    if not all_frames:
        return ""
    
    strips = []
    for idx in sample_indices:
        real_idx = min(idx, len(all_frames) - 1)
        f = cv2.resize(all_frames[real_idx], (160, 92), interpolation=cv2.INTER_AREA)
        cv2.putText(f, f"F{real_idx:02d}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        strips.append(f)
    combined = np.hstack(strips)
    _, buffer = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")

def analyze_and_render_comparison(video_left_path, video_right_path, title_left, title_right):
    has_l = os.path.exists(video_left_path)
    has_r = os.path.exists(video_right_path)

    if not has_l and not has_r:
        print(f"⚠️ Chưa tìm thấy cả 2 video: {video_left_path} và {video_right_path}")
        return

    # 1. Đo lường sai khác định lượng (Pixel Diff, PSNR, SSIM)
    avg_diff, psnr_val, ssim_val = 0.0, 0.0, 0.0
    if has_l and has_r:
        cap_l = cv2.VideoCapture(video_left_path)
        cap_r = cv2.VideoCapture(video_right_path)
        frames_l, frames_r, diffs = [], [], []
        while cap_l.isOpened() and cap_r.isOpened():
            ret_l, f_l = cap_l.read()
            ret_r, f_r = cap_r.read()
            if not ret_l or not ret_r:
                break
            frames_l.append(cv2.cvtColor(f_l, cv2.COLOR_BGR2RGB))
            frames_r.append(cv2.cvtColor(f_r, cv2.COLOR_BGR2RGB))
            abs_diff = np.mean(np.abs(f_l.astype(np.float32) - f_r.astype(np.float32)))
            diffs.append(abs_diff)
        cap_l.release()
        cap_r.release()
        avg_diff = np.mean(diffs) if diffs else 0.0

        if frames_l and frames_r:
            tensor_l = torch.from_numpy(np.array(frames_l)).float().permute(0, 3, 1, 2) / 255.0
            tensor_r = torch.from_numpy(np.array(frames_r)).float().permute(0, 3, 1, 2) / 255.0
            psnr_val, ssim_val = compute_psnr_and_ssim(tensor_r, tensor_l)

    status_badge = (
        f"<span style='background:#0f172a; border:1px solid #38bdf8; padding:4px 12px; border-radius:16px; margin:0 4px;'>"
        f"📊 <b>Pixel Diff</b>: <span style='color:#38bdf8;'>{avg_diff:.2f}/255</span></span> "
        f"<span style='background:#0f172a; border:1px solid #34d399; padding:4px 12px; border-radius:16px; margin:0 4px;'>"
        f"📐 <b>PSNR</b>: <span style='color:#34d399;'>{psnr_val:.2f} dB</span></span> "
        f"<span style='background:#0f172a; border:1px solid #fbbf24; padding:4px 12px; border-radius:16px; margin:0 4px;'>"
        f"🔍 <b>SSIM</b>: <span style='color:#fbbf24;'>{ssim_val:.4f}</span></span>"
    )

    # 2. Tạo Animated GIF Base64 (Chạy 100% trên mọi trình duyệt)
    gif_b64_l = mp4_to_animated_gif_b64(video_left_path) if has_l else ""
    gif_b64_r = mp4_to_animated_gif_b64(video_right_path) if has_r else ""

    # 3. Tạo Filmstrip Snapshot
    strip_b64_l = generate_filmstrip_b64(video_left_path) if has_l else ""
    strip_b64_r = generate_filmstrip_b64(video_right_path) if has_r else ""

    # 4. Tạo Video H.264 Base64
    play_l = ensure_browser_playable_mp4(video_left_path) if has_l else ""
    play_r = ensure_browser_playable_mp4(video_right_path) if has_r else ""
    vid_b64_l, vid_b64_r = "", ""
    if play_l and os.path.exists(play_l):
        with open(play_l, "rb") as f:
            vid_b64_l = base64.b64encode(f.read()).decode("utf-8")
    if play_r and os.path.exists(play_r):
        with open(play_r, "rb") as f:
            vid_b64_r = base64.b64encode(f.read()).decode("utf-8")

    html_code = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #111827; padding: 22px; border-radius: 14px; color: #fff; margin-bottom: 24px; border: 1px solid #374151; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
        <div style="text-align: center; margin-bottom: 18px;">
            <h3 style="margin: 0 0 10px 0; color: #38bdf8; font-size: 18px; letter-spacing: 0.5px;">🎬 ĐỐI CHỨNG QUỸ ĐẠO CAMERA PIPELINE V2</h3>
            <div style="font-size: 13px; line-height: 2.2;">{status_badge}</div>
        </div>

        <div style="display: flex; justify-content: space-between; gap: 20px;">
            <!-- CỘT TRÁI: BASELINE TRƯỚC KHI TRAIN -->
            <div style="flex: 1; background: #1f2937; padding: 14px; border-radius: 10px; border: 1px solid #ef4444;">
                <div style="background: #991b1b; color: #fecaca; font-weight: bold; font-size: 13px; padding: 4px 8px; border-radius: 6px; text-align: center; margin-bottom: 10px;">
                    🔴 {title_left}
                </div>
                {f'<img src="data:image/gif;base64,{gif_b64_l}" width="100%" style="border-radius: 6px; display: block;" alt="GIF Left" />' if gif_b64_l else '<p style="color:#ef4444;text-align:center;">Chưa có video</p>'}
                {f'<div style="margin-top: 10px;"><p style="font-size: 11px; color: #9ca3af; margin: 4px 0;">Tiến trình khung hình (F00 - F24):</p><img src="data:image/jpeg;base64,{strip_b64_l}" width="100%" style="border-radius: 4px;" /></div>' if strip_b64_l else ''}
                <div style="margin-top: 10px;">
                    {f'<video width="100%" controls loop muted playsinline style="border-radius: 6px;"><source src="data:video/mp4;base64,{vid_b64_l}" type="video/mp4"></video>' if vid_b64_l else ''}
                </div>
                <p style="font-size: 11px; color: #6b7280; margin: 6px 0 0 0; text-align: center;">📁 {video_left_path}</p>
            </div>

            <!-- CỘT PHẢI: KẾT QUẢ SAU KHI TRAIN -->
            <div style="flex: 1; background: #1f2937; padding: 14px; border-radius: 10px; border: 1px solid #10b981;">
                <div style="background: #065f46; color: #a7f3d0; font-weight: bold; font-size: 13px; padding: 4px 8px; border-radius: 6px; text-align: center; margin-bottom: 10px;">
                    🟢 {title_right}
                </div>
                {f'<img src="data:image/gif;base64,{gif_b64_r}" width="100%" style="border-radius: 6px; display: block;" alt="GIF Right" />' if gif_b64_r else '<p style="color:#ef4444;text-align:center;">Chưa có video</p>'}
                {f'<div style="margin-top: 10px;"><p style="font-size: 11px; color: #9ca3af; margin: 4px 0;">Tiến trình khung hình (F00 - F24):</p><img src="data:image/jpeg;base64,{strip_b64_r}" width="100%" style="border-radius: 4px;" /></div>' if strip_b64_r else ''}
                <div style="margin-top: 10px;">
                    {f'<video width="100%" controls loop muted playsinline style="border-radius: 6px;"><source src="data:video/mp4;base64,{vid_b64_r}" type="video/mp4"></video>' if vid_b64_r else ''}
                </div>
                <p style="font-size: 11px; color: #6b7280; margin: 6px 0 0 0; text-align: center;">📁 {video_right_path}</p>
            </div>
        </div>
    </div>
    """
    display(HTML(html_code))

print("=" * 75)
print("🎬 KẾT QUẢ ĐỐI CHỨNG QUỸ ĐẠO CAMERA PIPELINE V2:")
print("=" * 75)

print("\n>>> [1/2] CẶP 1: XOAY ĐƠN GIẢN (PAN -30°):")
analyze_and_render_comparison(
    "outputs/video_1A_simple_pan_before.mp4",
    "outputs/video_1B_simple_pan_after.mp4",
    "Trước Train (Baseline Step 0)",
    "Sau Train (Quỹ Đạo Pan -30°)"
)

print("\n>>> [2/2] CẶP 2: QUỸ ĐẠO PHỨC TẠP (CRANE UP + PAN 90°):")
analyze_and_render_comparison(
    "outputs/video_2A_complex_traj_before.mp4",
    "outputs/video_2B_complex_traj_after.mp4",
    "Trước Train (Baseline Step 0)",
    "Sau Train (Crane Up + Pan 90°)"
)
```
