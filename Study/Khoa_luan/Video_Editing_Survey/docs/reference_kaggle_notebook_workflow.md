# Tài Liệu Tham Khảo: Quy Trình Thực Thi Kaggle Notebook 6 Cells (Legacy Reference Workflow)

Tài liệu này lưu trữ quy trình 6 ô lệnh (Notebook Cells) mẫu chạy trên môi trường Kaggle (GPU T4/P100 kép hoặc đơn) nhằm phục vụ tham chiếu cho các phiên bản tiếp theo của đề tài **Video-to-Video Camera Trajectory Retargeting**.

---

## 1. Mục Tiêu & Kiến Trúc Quy Trình 6 Cells

Quy trình được thiết kế theo nguyên tắc:
1. **Cô lập ngoại tuyến 100% (`Offline Enforcement`)**: Không bao giờ kết nối ra internet/HuggingFace Hub trong phiên chạy, chống rớt mạng giữa chừng trên Kaggle.
2. **Tự động đồng bộ và thích ứng đường dẫn (`Auto-Discovery`)**: Tự dò tìm mã nguồn (.zip hoặc folder) và vị trí bộ trọng số, dataset trong `/kaggle/input/**` bất kể người dùng đặt tên thư mục dataset là gì.
3. **Đối chứng khoa học trước và sau huấn luyện (`Before vs. After Dual Evaluation`)**:
   - Chạy suy luận với Checkpoint ban đầu tại **Step 0** (trước huấn luyện) để thiết lập đường cơ sở (Baseline).
   - Chạy suy luận với Checkpoint **Max Step** (sau huấn luyện) trên cùng một video mẫu và cùng một quỹ đạo camera.
   - Khảo sát trên cả **2 cấp độ stress**: Quỹ đạo đơn giản (Pan xoay ngang $-30^\circ$) và Quỹ đạo phức tạp (Crane nâng cao $+$ Pan xoay ngang $-90^\circ$).
4. **Phân tích định lượng & Trình chiếu trực quan (`Quantitative Diff & HTML5 Inline Player`)**: Đo lường sai khác trung bình điểm ảnh (Mean Pixel Diff) và phát video HTML5 song song bằng Base64 trực tiếp trong giao diện Notebook.

---

## 2. Chi Tiết Các Ô Lệnh Nguyên Bản (Reference Source Code)

### [CELL 1] Thiết Lập Offline, Đồng Bộ Mã Nguồn & Huấn Luyện Adapter
```python
# ==============================================================================
# [CELL 1] THIẾT LẬP OFFLINE, ĐỒNG BỘ MÃ NGUỒN & HUẤN LUYỆN ADAPTER
# (Tích hợp DoRA ICML 2024 & LoRA+ 8x Learning Rate Multiplier)
# ==============================================================================
import os
import sys
import glob

# 1. Khóa hoàn toàn kết nối mạng của HuggingFace / Diffusers (Bảo đảm Offline 100%)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

DEST_CODE = "/kaggle/working/video_retargeting_adapter"
os.makedirs(DEST_CODE, exist_ok=True)
os.makedirs(os.path.join(DEST_CODE, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(DEST_CODE, "outputs"), exist_ok=True)

# 2. Tự động đồng bộ mã nguồn mới nhất (Hỗ trợ cả file .zip lẫn thư mục giải nén sẵn)
zip_candidates = glob.glob("/kaggle/input/**/video_retargeting_adapter*.zip", recursive=True)
src_candidates = glob.glob("/kaggle/input/**/train_adapter.py", recursive=True)
if zip_candidates:
    print(f"📦 Phát hiện file zip: {zip_candidates[0]}")
    print(">>> Đang giải nén mã nguồn mới nhất (Ghi đè -o)...")
    !unzip -o -q "{zip_candidates[0]}" -d /kaggle/working/
elif src_candidates:
    src_dir = os.path.dirname(src_candidates[0])
    print(f"📁 Phát hiện thư mục mã nguồn thực: {src_dir}")
    print(">>> Đang sao chép mã nguồn mới nhất (cp -rf)...")
    !cp -rf "{src_dir}"/* "{DEST_CODE}/"
else:
    print("⚠️ Không tìm thấy nguồn code trong /kaggle/input/, sử dụng code hiện có trong working.")

# Chuyển hẳn thư mục làm việc về DEST_CODE
actual_train = glob.glob(f"{DEST_CODE}/**/train_adapter.py", recursive=True)
if actual_train:
    DEST_CODE = os.path.dirname(actual_train[0])
%cd $DEST_CODE

# 3. Tự động dò tìm đường dẫn Dataset, SDXL và DINOv2 (Không lo lệch tên dataset)
def auto_find(target_name):
    matches = glob.glob(f"/kaggle/input/**/{target_name}", recursive=True)
    return matches[0] if matches else ""

SDXL_PATH = auto_find("sdxl_inpaint_upload")
DATA_DIR = auto_find("RealEstate10K_Mini")
if not DATA_DIR:
    DATA_DIR = auto_find("cameractrl-sota-weights")

DINO_PATH = auto_find("dinov2-large")
if not DINO_PATH:
    DINO_PATH = auto_find("clip-vit-large-patch14")

DEPTH_PATH = auto_find("Depth-Anything-V2-Small-hf")
if not DEPTH_PATH:
    DEPTH_PATH = auto_find("depth-anything")

# Tìm video test mẫu cho bước inference
video_matches = glob.glob("/kaggle/input/**/*.mp4", recursive=True)
SAMPLE_VIDEO = ""
for v in video_matches:
    if "sample_video.mp4" in v:
        SAMPLE_VIDEO = v
        break
if not SAMPLE_VIDEO and video_matches:
    SAMPLE_VIDEO = video_matches[0]

# Xuất biến môi trường cho các cell inference phía sau sử dụng
os.environ["SDXL_PATH"] = SDXL_PATH
os.environ["DATA_DIR"] = DATA_DIR
os.environ["DINO_PATH"] = DINO_PATH
os.environ["DEPTH_PATH"] = DEPTH_PATH
os.environ["SAMPLE_VIDEO"] = SAMPLE_VIDEO

print("=" * 70)
print(f"  * SDXL Weights   : {SDXL_PATH}")
print(f"  * Dataset Root   : {DATA_DIR}")
print(f"  * DINOv2/CLIP    : {DINO_PATH if DINO_PATH else 'None (Sử dụng internal patch encoder)'}")
print(f"  * Depth Model    : {DEPTH_PATH if DEPTH_PATH else 'None (Sử dụng Geometric Depth)'}")
print(f"  * Sample Video   : {SAMPLE_VIDEO}")
print("=" * 70)

# 4. Thực thi huấn luyện với LoRA+ và DoRA
!python train_adapter.py \
    --data_dir "$DATA_DIR" \
    --sdxl_path "$SDXL_PATH" \
    --clip_or_dino_path "$DINO_PATH" \
    --output_dir "./checkpoints" \
    --use_dora \
    --lora_plus_ratio 8.0 \
    --num_frames 8 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --save_steps 500 \
    --mixed_precision bf16 \
    --in_channels 14 \
    --lambda_diff 1.0 \
    --lambda_hole_weight 3.0 \
    --sample_stride 3
```

---

### [CELL 2] Video 1A: Xoay Đơn Giản (-30°) - Trước Khi Train (Step 0)
```python
import glob

DINO_PATH = glob.glob("/kaggle/input/**/dinov2-large", recursive=True)[0]
DEPTH_PATH = glob.glob("/kaggle/input/**/Depth-Anything-V2-Small-hf", recursive=True)[0]

!python inference_pipeline.py \
    --source_video "$SAMPLE_VIDEO" \
    --sdxl_path "$SDXL_PATH" \
    --clip_or_dino_path "$DINO_PATH" \
    --depth_model "$DEPTH_PATH" \
    --adapter_checkpoint "./checkpoints/adapter_step_0.pt" \
    --traj_type pan \
    --pan_angle -30.0 \
    --mode progressive \
    --num_frames 25 \
    --num_inference_steps 25 \
    --noise_rho 0.85 \
    --strength 0.85 \
    --fps 8 \
    --output_path "outputs/video_1A_simple_pan_before.mp4"
```

---

### [CELL 3] Video 1B: Xoay Đơn Giản (-30°) - Sau Khi Train (Max Step)
```python
import glob

DINO_PATH = glob.glob("/kaggle/input/**/dinov2-large", recursive=True)[0]
DEPTH_PATH = glob.glob("/kaggle/input/**/Depth-Anything-V2-Small-hf", recursive=True)[0]

# Tự động chọn checkpoint có số step lớn nhất vừa train xong
ckpts = [f for f in glob.glob("./checkpoints/adapter_step_*.pt") if not f.endswith("_0.pt")]
latest_ckpt = max(ckpts, key=lambda f: int(f.split("_step_")[-1].replace(".pt", ""))) if ckpts else "./checkpoints/adapter_step_0.pt"

print(f">>> Checkpoint sau khi train được chọn (Max Step): {latest_ckpt}")

!python inference_pipeline.py \
    --source_video "$SAMPLE_VIDEO" \
    --sdxl_path "$SDXL_PATH" \
    --clip_or_dino_path "$DINO_PATH" \
    --depth_model "$DEPTH_PATH" \
    --adapter_checkpoint "$latest_ckpt" \
    --traj_type pan \
    --pan_angle -30.0 \
    --mode progressive \
    --num_frames 25 \
    --num_inference_steps 25 \
    --strength 0.70 \
    --noise_rho 0.90 \
    --fps 8 \
    --output_path "outputs/video_1B_simple_pan_after.mp4"
```

---

### [CELL 4] Video 2A: Quỹ Đạo Phức Tạp (Crane Up + Pan 90°) - Trước Khi Train
```python
import glob

DINO_PATH = glob.glob("/kaggle/input/**/dinov2-large", recursive=True)[0]
DEPTH_PATH = glob.glob("/kaggle/input/**/Depth-Anything-V2-Small-hf", recursive=True)[0]

!python inference_pipeline.py \
    --source_video "$SAMPLE_VIDEO" \
    --sdxl_path "$SDXL_PATH" \
    --clip_or_dino_path "$DINO_PATH" \
    --depth_model "$DEPTH_PATH" \
    --adapter_checkpoint "./checkpoints/adapter_step_0.pt" \
    --traj_type crane_left_90 \
    --pan_angle -90.0 \
    --trans_y -0.35 \
    --mode progressive \
    --num_frames 25 \
    --num_inference_steps 25 \
    --noise_rho 0.85 \
    --strength 0.85 \
    --fps 8 \
    --output_path "outputs/video_2A_complex_traj_before.mp4"
```

---

### [CELL 5] Video 2B: Quỹ Đạo Phức Tạp (Crane Up + Pan 90°) - Sau Khi Train
```python
import glob

DINO_PATH = glob.glob("/kaggle/input/**/dinov2-large", recursive=True)[0]
DEPTH_PATH = glob.glob("/kaggle/input/**/Depth-Anything-V2-Small-hf", recursive=True)[0]

ckpts = [f for f in glob.glob("./checkpoints/adapter_step_*.pt") if not f.endswith("_0.pt")]
latest_ckpt = max(ckpts, key=lambda f: int(f.split("_step_")[-1].replace(".pt", ""))) if ckpts else "./checkpoints/adapter_step_0.pt"

print(f">>> Checkpoint sau khi train được chọn (Max Step): {latest_ckpt}")

!python inference_pipeline.py \
    --source_video "$SAMPLE_VIDEO" \
    --sdxl_path "$SDXL_PATH" \
    --clip_or_dino_path "$DINO_PATH" \
    --depth_model "$DEPTH_PATH" \
    --adapter_checkpoint "$latest_ckpt" \
    --traj_type crane_left_90 \
    --pan_angle -90.0 \
    --trans_y -0.35 \
    --mode progressive \
    --num_frames 25 \
    --num_inference_steps 25 \
    --noise_rho 0.85 \
    --strength 0.85 \
    --fps 8 \
    --output_path "outputs/video_2B_complex_traj_after.mp4"
```

---

### [CELL 6] Đối Chứng Trước - Sau & Đo Lường Sai Khác Điểm Ảnh
```python
import os
import cv2
import base64
import numpy as np
from IPython.display import HTML, display

def analyze_and_render_comparison(video_left_path, video_right_path, title_left, title_right):
    if not os.path.exists(video_left_path) or not os.path.exists(video_right_path):
        print(f"[CẢNH BÁO] Không tìm thấy file: {video_left_path} hoặc {video_right_path}")
        return

    # 1. Đo lường định lượng sai khác điểm ảnh giữa 2 video
    cap_l = cv2.VideoCapture(video_left_path)
    cap_r = cv2.VideoCapture(video_right_path)
    
    diffs = []
    while cap_l.isOpened() and cap_r.isOpened():
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r:
            break
        abs_diff = np.mean(np.abs(f_l.astype(np.float32) - f_r.astype(np.float32)))
        diffs.append(abs_diff)
    cap_l.release()
    cap_r.release()

    avg_diff = np.mean(diffs) if diffs else 0.0
    status_text = f"Độ lệch điểm ảnh trung bình (Mean Pixel Diff): {avg_diff:.2f} / 255"
    if avg_diff == 0.0:
        badge = "<span style='color: #ef4444; font-weight: bold;'>⚠️ Hai video GIỐNG NHAU 100% (Chưa có tác động từ checkpoint mới)</span>"
    else:
        badge = f"<span style='color: #22c55e; font-weight: bold;'>✅ Hai video CÓ SAI KHÁC RÕ RỆT (Độ lệch {avg_diff:.2f}/255)</span>"

    # 2. Mã hóa Base64 để phát video HTML5 trực tiếp trong Notebook
    with open(video_left_path, "rb") as f:
        b64_l = base64.b64encode(f.read()).decode("utf-8")
    with open(video_right_path, "rb") as f:
        b64_r = base64.b64encode(f.read()).decode("utf-8")

    html_code = f"""
    <div style="font-family: Arial, sans-serif; background: #18181b; padding: 20px; border-radius: 12px; color: #fff; margin-bottom: 24px; border: 1px solid #3f3f46;">
        <div style="text-align: center; margin-bottom: 16px;">
            <h3 style="margin: 0 0 6px 0; color: #38bdf8;">ĐỐI CHỨNG TRƯỚC VÀ SAU KHI HUẤN LUYỆN</h3>
            <div style="font-size: 14px; margin-bottom: 4px;">{badge}</div>
            <div style="font-size: 12px; color: #a1a1aa;">{status_text}</div>
        </div>
        <div style="display: flex; justify-content: space-around; gap: 20px;">
            <div style="flex: 1; text-align: center; background: #27272a; padding: 12px; border-radius: 8px;">
                <h4 style="color: #f87171; margin: 4px 0 10px 0;">{title_left}</h4>
                <video width="100%" controls autoplay loop muted style="border-radius: 6px;">
                    <source src="data:video/mp4;base64,{b64_l}" type="video/mp4">
                </video>
                <p style="font-size: 11px; color: #a1a1aa; margin-top: 6px;">{video_left_path}</p>
            </div>
            <div style="flex: 1; text-align: center; background: #27272a; padding: 12px; border-radius: 8px;">
                <h4 style="color: #4ade80; margin: 4px 0 10px 0;">{title_right}</h4>
                <video width="100%" controls autoplay loop muted style="border-radius: 6px;">
                    <source src="data:video/mp4;base64,{b64_r}" type="video/mp4">
                </video>
                <p style="font-size: 11px; color: #a1a1aa; margin-top: 6px;">{video_right_path}</p>
            </div>
        </div>
    </div>
    """
    display(HTML(html_code))

# 1. So sánh Cặp 1: Xoay đơn giản
analyze_and_render_comparison(
    "outputs/video_1A_simple_pan_before.mp4",
    "outputs/video_1B_simple_pan_after.mp4",
    "Cặp 1: Trước train (Step 0)",
    "Cặp 1: Sau train (Max Step Checkpoint)"
)

# 2. So sánh Cặp 2: Quỹ đạo phức tạp
analyze_and_render_comparison(
    "outputs/video_2A_complex_traj_before.mp4",
    "outputs/video_2B_complex_traj_after.mp4",
    "Cặp 2: Trước train (Step 0)",
    "Cặp 2: Sau train (Max Step Checkpoint)"
)
```
