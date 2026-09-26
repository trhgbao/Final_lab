# TÀI LIỆU ĐẶC TẢ MÃ NGUỒN PIPELINE V3 (CODEBASE SPECIFICATION)
## Video Camera Trajectory Retargeting với CogVideoX-Fun-5B-InP & DoRA

Tài liệu này tổng hợp toàn bộ cấu trúc mã nguồn trong thư mục `pipeline_v3/`. Với từng phần, tài liệu mô tả chi tiết:
- Danh sách các hàm / lớp (Class / Function)
- Các tham số đầu vào (Parameters)
- Ý nghĩa và chức năng hoạt động của từng hàm

---

## 1. PHÂN HỆ HÌNH HỌC 3D (`pipeline_v3/geometry/`)

### 1.1. `trajectory.py` (Bộ sinh quỹ đạo Camera)
Chịu trách nhiệm tính toán ma trận chuyển đổi tọa độ Camera Extrinsics (Camera-to-World / World-to-Camera) theo quỹ đạo hình cầu hoặc tịnh tiến.

* `sphere2pose(theta, phi, r, x=0.0, y=0.0)`
  * **Tham số**:
    * `theta` (*float*): Góc quay theo phương dọc (Pitch/Elevation, tính theo độ).
    * `phi` (*float*): Góc quay theo phương ngang (Yaw/Azimuth, tính theo độ).
    * `r` (*float*): Bán kính/khoảng cách từ camera tới tâm cảnh (Radius).
    * `x`, `y` (*float*): Độ tịnh tiến dịch chuyển theo trục X và Y.
  * **Chức năng**: Tạo ma trận Camera-to-World $4 \times 4$ nhìn về gốc tọa độ cảnh dựa trên tọa độ cầu.

* `generate_camera_trajectory(c2w_anchor, theta, phi, d_r, d_x=0.0, d_y=0.0, num_frames=49, device="cpu")`
  * **Tham số**:
    * `c2w_anchor` (*torch.Tensor, [1, 4, 4]*): Ma trận pose gốc của frame đầu tiên.
    * `theta`, `phi`, `d_r`, `d_x`, `d_y` (*float*): Độ lệch góc pitch, yaw, khoảng cách zoom và dịch chuyển tịnh tiến mục tiêu.
    * `num_frames` (*int*): Số lượng frame cần nội suy quỹ đạo (mặc định 49).
    * `device` (*str/torch.device*): Thiết bị tính toán.
  * **Chức năng**: Tạo ra 2 quỹ đạo: `pose_s` (quỹ đạo camera nguồn đứng yên hoặc theo video gốc) và `pose_t` (quỹ đạo camera mục tiêu được làm mịn mượt mà qua các frame).

* `get_relative_pose(pose_src, pose_tgt)`
  * **Tham số**: `pose_src`, `pose_tgt` (*torch.Tensor, [B, 4, 4]*).
  * **Chức năng**: Tính ma trận biến đổi tương đối giữa 2 camera: $T_{\text{rel}} = T_{\text{tgt}} \cdot T_{\text{src}}^{-1}$.

---

### 1.2. `warper3d.py` (Bộ biến dạng đám mây điểm 3D Forward Warper)
Chịu trách nhiệm unproject các pixel 2D thành điểm 3D trong không gian thực, xoay/dịch theo quỹ đạo camera mới, và chiếu ngược (splatting) lại lên mặt phẳng ảnh mục tiêu.

* `Warper3D(resolution=(480, 720), device="cuda")`
  * **Hàm khởi tạo**: Tạo lưới tọa độ đồng nhất (homogeneous grid $[u, v, 1]^T$) cố định để tăng tốc độ tính toán.

* `compute_transformed_points(depth, pose_src, pose_tgt, K_src, K_tgt)`
  * **Tham số**:
    * `depth` (*torch.Tensor, [B, 1, H, W]*): Bản đồ độ sâu mét.
    * `pose_src`, `pose_tgt` (*torch.Tensor, [B, 4, 4]*): Ma trận Extrinsics nguồn và đích.
    * `K_src`, `K_tgt` (*torch.Tensor, [B, 3, 3]*): Ma trận Intrinsics của camera.
  * **Chức năng**: Giải phương trình hình học chiếu: $P_{\text{tgt}} = K_{\text{tgt}} \cdot T_{\text{tgt}}^{-1} \cdot T_{\text{src}} \cdot K_{\text{src}}^{-1} \cdot (D \cdot [u, v, 1]^T)$. Trả về tọa độ điểm 3D đã biến dạng.

* `bilinear_splatting(transformed_pts, img_src)`
  * **Tham số**:
    * `transformed_pts` (*torch.Tensor, [B, H, W, 3]*): Tọa độ điểm 3D đã chiếu lên mặt phẳng đích.
    * `img_src` (*torch.Tensor, [B, 3, H, W]*): Ảnh màu nguồn.
  * **Chức năng**: Ánh xạ từng điểm ảnh vào lưới pixel đích bằng nội suy song tuyến tính có xét đến thứ tự che khuất độ sâu (Z-buffer), sinh ra ảnh màu đích và binary mask (1 = pixel hợp lệ, 0 = lỗ hổng bị che khuất).

* `forward_warp(img, flow, depth, pose_s, pose_t, K_s, K_t, clean_mask=True)`
  * **Tham số**: Tập hợp ảnh, depth và camera poses.
  * **Chức năng**: Đóng gói toàn bộ quá trình biến dạng một frame từ camera nguồn sang camera mục tiêu kèm bộ lọc làm sạch nhiễu biên `clean_points`.

* `double_reprojection_warp(frame_target, depth_target, transformation_target, transformation_perturbed, intrinsic)`
  * **Tham số**: Frame và depth đích thực tế, kèm ma trận camera thực và ma trận camera bị nhiễu loạn (perturbed).
  * **Chức năng**: Thực thi cơ chế Double Reprojection để huấn luyện tự giám sát (self-supervised): từ 1 video đơn lẻ, warp ngược rồi warp xuôi để tạo ra cặp (Proxy Scaffold bị lủng lỗ, Ground Truth gốc).

---

## 2. PHÂN HỆ ƯỚC LƯỢNG ĐỘ SÂU (`pipeline_v3/models/depthcrafter_wrapper.py`)

* `resolve_depthcrafter_paths()`
  * **Chức năng**: Tự động dò tìm đường dẫn checkpoint của `DepthCrafter` và backbone `SVD` trên Kaggle, máy local hoặc HuggingFace hub.

* `DepthCrafterEstimator(unet_path, svd_path, device, dtype)`
  * **Hàm khởi tạo**: Nạp mô hình UNet chuyên biệt của DepthCrafter kết hợp với bộ trích xuất đặc trưng của Stable Video Diffusion.

* `estimate_depth(frames, process_length, num_inference_steps, window_size)`
  * **Tham số**:
    * `frames` (*np.ndarray, [F, H, W, 3]*): Chuỗi các frame video đầu vào.
    * `process_length` (*int*): Độ dài video xử lý (mặc định 49).
    * `window_size` (*int*): Kích thước cửa sổ trượt (sliding window) chống tràn VRAM.
  * **Chức năng**: Dự đoán bản đồ nghịch đảo độ sâu (disparity map) liên tục và mượt mà theo thời gian, sau đó chuyển đổi sang thang độ sâu vật lý (Metric Depth) bằng công thức chuẩn: $Z = \frac{10000}{\text{clamp}(D \cdot 3900, 10^{-5})}$.

* `free_memory()`
  * **Chức năng**: Xóa bỏ UNet DepthCrafter khỏi VRAM GPU và dọn rác bộ nhớ (`torch.cuda.empty_cache()`) để nhường toàn bộ VRAM cho mô hình DiT 5.6B.

---

## 3. PHÂN HỆ THÍCH ỨNG THAM SỐ DORA (`pipeline_v3/models/dora.py`)

* `DoRALinear(base_layer, r=16, lora_alpha=32.0, lora_dropout=0.0)`
  * **Lớp mô-đun**: Bọc lớp `nn.Linear` gốc của Base Transformer để phân rã ma trận trọng số theo công thức Weight Normalization:
    $$W' = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
  * **Các thuộc tính chính**:
    * `magnitude` ($m$): Vector độ lớn kích thước $1 \times \text{out\_features}$.
    * `lora_A`, `lora_B`: Hai ma trận rank thấp thích ứng phương hướng.
    * `cached_base_norm`: Bộ nhớ đệm lưu chuẩn độ dài ma trận gốc $W_0$, giúp loại bỏ 180 lần tính norm thừa mỗi lượt forward.

* `DoRALinear.forward(x)`
  * **Chức năng**: Thực hiện nhân ma trận với kỹ thuật **Detach Norm trick** (giảm 24.4% VRAM) và tính norm trong không gian **`torch.float32`** để chống tràn số nửa độ chính xác.

* `DoRALinear.merge_weights()`
  * **Chức năng**: Hợp nhất vĩnh viễn DoRA vào ma trận $W_0$ gốc:
    $$W_{\text{merged}} = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
    Loại bỏ hoàn toàn chi phí tính toán (Zero Inference Latency Overhead) khi chạy suy luận hoặc khi chuyển giao sang Stage 2.

* `apply_dora_to_module(model, target_keywords, r, lora_alpha, lora_dropout)`
  * **Chức năng**: Quét đệ quy toàn bộ các khối Transformer, tự động tìm và thay thế các lớp Linear thỏa mãn từ khóa (`attn1.to_q`, `attn1.to_k`, `attn1.to_v`, `attn1.to_out.0`, `ff.net.0.proj`, `ff.net.2`) bằng `DoRALinear`.

* `get_dora_state_dict(model)` & `load_dora_state_dict(model, state_dict)`
  * **Chức năng**: Trích xuất và nạp riêng biệt bộ trọng số DoRA siêu nhẹ (~27.4MB), tách rời hoàn toàn khỏi Base Model 5.6B.

---

## 4. PHÂN HỆ KIẾN TRÚC DIT BIẾN ĐỔI (`pipeline_v3/models/crosstransformer3d.py`)

* `RefPatchEmbed(patch_size=2, in_channels=16, embed_dim=1920)`
  * **Chức năng**: Chiếu 10 frame tham chiếu $I_{\text{ref}}$ từ không gian VAE Latent thành chuỗi các visual token 1D để làm Key và Value cho Cross-Attention.

* `PerceiverCrossAttention(dim=3072, dim_head=128, heads=16, kv_dim=2048, zero_init=True)`
  * **Lớp mô-đun**: Tầng chú ý chéo đặt cách quãng giữa các khối DiT.
  * **Chức năng**: Truy vấn thông tin chi tiết sắc nét từ token tham chiếu để bơm vào các vùng bị thủng lỗ.
  * **Điểm đột phá**: Thuộc tính `zero_init=True` (`nn.init.zeros_(to_out.weight)`) đảm bảo tại Step 0 của quá trình huấn luyện, đầu ra của Perceiver bằng đúng 0 tuyệt đối, bảo toàn 100% hình học Stage 1 và ngăn chặn hiện tượng sốc latent (Latent Shock).

* `CrossTransformer3DModel`
  * **Lớp Backbone chính**: Kế thừa kiến trúc CogVideoX-Fun-InP (nhận đầu vào 33 kênh).
  * **Các phương thức quản trị**:
    * `inject_dora(...)`: Tiêm DoRA vào toàn bộ 30 Transformer blocks.
    * `freeze_base_for_stage1()`: Khóa Base Model, chỉ mở gradient cho DoRA.
    * `freeze_for_stage2()`: Khóa Base Model + DoRA, chỉ mở gradient cho Perceiver và RefPatchEmbed.
    * `merge_dora()`: Gộp trọng số DoRA Stage 1 vào Base Transformer.
    * `save_dora_checkpoint(...)` & `load_dora_checkpoint(...)`: Quản lý checkpoint Stage 1.
    * `save_stage2_checkpoint(...)` & `load_stage2_checkpoint(...)`: Quản lý checkpoint Stage 2.

---

## 5. PHÂN HỆ DỮ LIỆU & HUẤN LUYỆN (`pipeline_v3/dataset/` & `training/`)

### 5.1. `double_reprojection_dataset.py`
* `DoubleReprojectionDataset(data_root, depth_cache_dir, num_frames=49, resolution=(480, 720))`
  * **Chức năng**:
    * Nạp video và pose từ RealEstate10K.
    * Cắt lát đồng bộ tuyệt đối giữa RGB frames và Depth map theo cùng chỉ số thời gian (`indices`).
    * Thực hiện Double Reprojection để sinh video giàn giáo (`proxy_video`) và mặt nạ (`proxy_mask`).
    * Trả về bộ tensor hoàn chỉnh: `[target_video, proxy_video, proxy_mask, reference_frames]`.

### 5.2. `train_stage1_geometry.py` (Kịch bản huấn luyện Stage 1)
* **Quy trình thực thi**:
  1. Khởi tạo Base `CogVideoX-Fun-V1.1-5b-InP` ở chế độ inpaint 2D thuần túy (`is_train_cross=False`).
  2. Tiêm DoRA vào Self-Attention & FFN; khóa 100% Base Model.
  3. Tách biệt Optimizer: áp dụng `weight_decay = 0.0` cho vector $m$ và `weight_decay = 1e-2` cho $BA$.
  4. Huấn luyện với hàm mục tiêu vận tốc $v$-prediction:
     $$\mathcal{L} = \|v_{\theta}(z_t, t, c_{\text{inpaint}}) - \text{scheduler.get\_velocity}(z_0, \epsilon, t)\|^2$$
  5. Định kỳ lưu checkpoint DoRA siêu nhẹ (~27.4MB).

### 5.3. `train_stage2_appearance.py` (Kịch bản huấn luyện Stage 2)
* **Quy trình thực thi**:
  1. Nạp Base Model và nạp checkpoint DoRA từ Stage 1.
  2. Gọi `transformer.merge_dora()` để gộp vĩnh viễn hình học Stage 1 vào mạng gốc.
  3. Khởi tạo mới hoàn toàn `ref_patch_embed` và 15 tầng `PerceiverCrossAttention`.
  4. Đóng băng Base Model + DoRA, chỉ huấn luyện Perceiver và RefPatchEmbed.
  5. Lưu checkpoint Stage 2 chuyên biệt bằng `save_stage2_checkpoint()`.

---

## 6. PHÂN HỆ SUY LUẬN TOÀN DIỆN (`pipeline_v3/inference_v3.py`)

* `PipelineV3Inference(opts)`
  * **Khởi tạo**: Tự động giải quyết đường dẫn lồng nhau, nạp VAE, Text Encoder, Scheduler và Transformer.
  * **Chế độ nạp mô hình**:
    * Nạp Base + DoRA Stage 1 + Stage 2 Checkpoint của chúng ta.
    * HOẶC nạp Official SOTA Baseline (`TrajectoryCrafter`) để chạy đối chuẩn đối đầu.

* `PipelineV3Inference.run(video_path, target_pose)`
  * **Quy trình xử lý**:
    1. Đọc video nguồn và lấy bản đồ độ sâu (từ file cache hoặc DepthCrafter).
    2. Căn chỉnh bán kính cảnh động `scene_radius` theo độ sâu trung tâm.
    3. Sinh quỹ đạo camera mượt mà theo các góc chỉ định (`theta, phi, r, x, y`).
    4. Warp đám mây điểm 3D tạo giàn giáo `cond_video` và `cond_masks`.
    5. Đảo ngược giá trị mask sang thang $[0, 255]$ để khớp chuẩn Inpainting Diffusers.
    6. Chạy vòng lặp khử nhiễu DiT Denoising.
    7. Giải mã VAE và xuất video kết quả kèm video triptych so sánh 3 màn hình trực quan.

---

## 7. PHÂN HỆ ĐÁNH GIÁ & KIỂM THỬ TRỰC QUAN (`pipeline_v3/evaluation/`)

### 7.1. `eval_stage1_geometry.py` (Đánh giá sau Stage 1)
* **Mục tiêu**: Chạy ngay sau khi hoàn thành huấn luyện Stage 1 DoRA Geometry.
* **Cơ chế hoạt động**:
  * Tự động dò tìm checkpoint DoRA Stage 1, Base Model, Baseline TrajectoryCrafter, Depth Cache và Video tập test.
  * Đánh giá trên 2 mẫu (`000c3ab189999a83` In-Domain và `145da324f69d1c6b` Out-of-Domain) ở 2 góc xoay Pan 30° và Pan 60°.
  * Chuyển đổi toàn bộ video sang định dạng H.264 qua `ffmpeg` để tương thích 100% với trình duyệt web.
  * Xuất bảng HTML 4 cột trực quan (Góc xoay | 3D Scaffold | Baseline SOTA | Stage 1 DoRA Hình học) hiển thị trực tiếp trong Jupyter/Kaggle Notebook và lưu ra file `eval_stage1_report.html`.

### 7.2. `eval_stage2_appearance.py` (Đánh giá sau Stage 2)
* **Mục tiêu**: Chạy ngay sau khi hoàn thành huấn luyện Stage 2 Perceiver Cross-Attention.
* **Cơ chế hoạt động**:
  * Nạp Base Model + DoRA Stage 1 (đã merge) + Checkpoint Perceiver Stage 2.
  * Tái sử dụng kết quả Stage 1 từ phiên trước nếu có (tránh suy luận lại tốn GPU).
  * So sánh đối đầu giữa: 3D Scaffold ➔ Stage 1 DoRA (Hình học vững chắc nhưng thiếu vân bề mặt) ➔ Stage 2 Hoàn chỉnh (Khôi phục sắc nét texture từ 10 frame tham chiếu).
  * Xuất bảng HTML hiển thị trực tiếp trong Notebook và lưu ra file `eval_stage2_report.html`.

### 7.3. `eval_final_ablation.py` (Kiểm thử Hoàn chỉnh Toàn diện - Ablation Study)
* **Mục tiêu**: Tạo bảng so sánh tổng kết toàn bộ tiến trình tiến hóa khoa học của đề tài.
* **Cơ chế hoạt động**:
  * Tổng hợp toàn bộ 4 cột so sánh đối đầu:
    1. Cột 1: **Góc Xoay** (`Pan 30°`, `Pan 60°`).
    2. Cột 2: **3D Scaffold** (Giàn giáo đám mây điểm biến dạng bị lỗ hổng).
    3. Cột 3: **Baseline SOTA** (`TrajectoryCrafter` Official Paper Checkpoint).
    4. Cột 4: **Stage 1 (DoRA Hình học)** (Đề xuất của chúng ta ở Phase 1).
    5. Cột 5: **Stage 2 (Mô hình Hoàn chỉnh)** (Đề xuất hoàn thiện: Base + DoRA + Perceiver).
  * Nhúng toàn bộ video dưới dạng Base64 vào bảng HTML tối màu sang trọng (Dark Theme), hỗ trợ video `autoplay loop muted`.
  * Xuất file báo cáo `eval_final_ablation_report.html` và hiển thị trực tiếp bằng `display(HTML(...))`.
