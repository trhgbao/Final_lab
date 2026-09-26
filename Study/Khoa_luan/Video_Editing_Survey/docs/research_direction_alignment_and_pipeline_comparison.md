# TÀI LIỆU KẾ HOẠCH NGHIÊN CỨU & THIẾT KẾ KIẾN TRÚC HỆ THỐNG
## Video Camera Trajectory Retargeting với CogVideoX-Fun-5B-InP & DoRA (Pipeline v3)

**Đề tài**: Nghiên cứu và Xây dựng Hệ thống Tái Định Vị Quỹ Đạo Camera từ Monocular Video (Pipeline v3)  
**Phương pháp**: Two-Stage Parameter-Efficient Adaptation (DoRA Geometry Adaptation + Perceiver Cross-Attention Ref-DiT)  
**Mô hình Nền tảng (Foundation Base Model)**: `CogVideoX-Fun-V1.1-5b-InP` (Alibaba PAI, 5.6 tỷ tham số)  
**Đối chuẩn Đối đầu (External SOTA Baseline)**: `TrajectoryCrafter` (ICCV 2025 Oral, Tencent & Tsinghua)  
**Ngày cập nhật**: 25/09/2026  

---

## PHẦN 1: TUYÊN NGÔN ĐỊNH HÌNH HƯỚNG NGHIÊN CỨU & ĐỘC LẬP HỌC THUẬT

### 1.1. Bản chất khoa học của đề tài
Nghiên cứu của chúng ta tập trung vào bài toán: **Làm thế nào để chuyển hóa một mô hình Video Inpainting 2D nền tảng mở tổng quát (`CogVideoX-Fun-V1.1-5b-InP`) thành một hệ thống Tái định vị Camera 3D (3D Camera Retargeting) hoàn chỉnh**, với hiệu quả tham số cao (PEFT) mà không cần phải huấn luyện lại toàn bộ mô hình (Full Fine-Tuning 5.6 tỷ tham số như bài báo TrajectoryCrafter).

Hệ thống được thiết kế theo quy trình thích ứng 2 giai đoạn (Two-Stage Adaptation):
* **Stage 1 (DoRA Geometry Adaptation)**: Đóng băng Base Model 5.6B, tiêm DoRA vào toàn bộ các tầng 3D Spatio-Temporal Self-Attention và FFN (~73.6 triệu tham số, chiếm ~1.3%) để dạy mô hình hiểu được giàn giáo 3D Scaffold và quy luật bù đắp hình học thị sai (parallax disocclusion) khi camera chuyển động.
* **Stage 2 (Appearance Detail Restoration)**: Khóa cố định Base Model và DoRA đã huấn luyện ở Stage 1; khởi tạo cấu trúc `PerceiverCrossAttention` (15 tầng) và `RefPatchEmbed` từ đầu để học cách trích xuất và chuyển giao chi tiết vân bề mặt sắc nét từ video nguồn gốc vào các vùng khuyết.

### 1.2. Phân định rõ ràng giữa Base Model và External Baseline
* **Mô hình nền tảng (Base Model)**: Bắt buộc và duy nhất là **`CogVideoX-Fun-V1.1-5b-InP`** (trọng số open-source từ Alibaba PAI). Toàn bộ quá trình huấn luyện DoRA và Perceiver đều bắt đầu từ nền tảng này.
* **Mô hình bài báo (`TrajectoryCrafter`)**: **CHỈ đóng vai trò là External Benchmark Baseline** (thước đo SOTA thế giới) để so sánh đối đầu trong bảng thực nghiệm kết quả. Chúng ta tuyệt đối không lấy trọng số transformer của tác giả làm nền tảng huấn luyện, bảo đảm 100% tính liêm chính và độc lập học thuật cho khóa luận.

---

## PHẦN 2: BẢNG SO SÁNH TOÀN DIỆN PIPELINE CŨ VS HƯỚNG NGHIÊN CỨU MỚI

| Tiêu Chí So Sánh | Pipeline Trước Đó (Thử nghiệm cũ) | Hướng Nghiên Cứu Mới Chuẩn Mực (Đã thống nhất) | Ý Nghĩa & Tác Động Khoa Học |
| :--- | :--- | :--- | :--- |
| **1. Điểm xuất phát (Base Transformer)** | Nạp thẳng checkpoint đã train của `TrajectoryCrafter` (`transformer_path = .../TrajectoryCrafter`). | Nạp **100% từ `CogVideoX-Fun-V1.1-5b-InP/transformer`** (mô hình inpaint 2D gốc của Alibaba PAI). | **Tính Độc Lập Học Thuật**: Đề tài không bị đánh giá là fine-tune phụ thuộc trên bài báo khác, khẳng định năng lực tự nghiên cứu và phát triển. |
| **2. Bản chất thích ứng của DoRA** | Tiêm DoRA đè lên mô hình vốn đã được tác giả huấn luyện 3D. | Tiêm DoRA lên **CogVideoX-Fun inpaint 2D tổng quát**. | **Chứng minh năng lực của DoRA**: DoRA thực sự dạy cho một mô hình inpaint 2D hiểu phối cảnh và động học 3D. |
| **3. Hàm mục tiêu huấn luyện (Loss Target)** | **Lỗi toán học**: Tính loss theo nhiễu Gaussian $\epsilon$ (`mse_loss(pred, noise)`). | **Chuẩn hóa toán học**: Tính loss theo vector vận tốc $v$-prediction (`scheduler.get_velocity(...)`). | **Khắc phục sương mù xám (Grey Fog)**: Loại bỏ triệt để hiện tượng triệt tiêu năng lượng latent về 0, khôi phục ảnh rõ nét. |
| **4. Cấu hình Optimizer cho DoRA** | Đặt `weight_decay = 1e-2` chung cho toàn bộ tham số (bao gồm cả vector $m$). | Tách riêng $m$ với **`weight_decay = 0.0`** và cấu hình learning rate chuyên biệt. | **Bảo toàn chuẩn ma trận**: Giữ vững biên độ kích hoạt qua 30 block transformer, chống sụp đổ biểu diễn. |
| **5. Cơ chế chuẩn hóa DoRA** | Tính norm đầy đủ trên autograd graph, dễ tràn số trên nửa độ chính xác. | **Detach Norm trick** (giảm 24.4% VRAM) và tính norm trong không gian **`float32`**. | **Tiết kiệm tài nguyên & Chống NaN**: Vừa huấn luyện ổn định trên GPU Kaggle vừa chống tràn số FP16/BF16. |
| **6. Khởi tạo Stage 2 Perceiver** | Kế thừa 15 tầng Perceiver đã huấn luyện của TrajectoryCrafter rồi train tiếp. | Khởi tạo cấu trúc Perceiver theo chuẩn Ref-DiT nhưng **trọng số khởi tạo mới từ đầu**. | **Sự trong sạch của trọng số**: Mô hình Stage 2 hoàn toàn do chính chúng ta huấn luyện trên dữ liệu. |
| **7. Vai trò trọng số TrajectoryCrafter** | Bị biến thành "gốc rễ" để train đè lên. | Trở thành **Cột SOTA Baseline độc lập** trong bảng đánh giá Ablation Study. | **Giá trị khoa học**: Thuyết phục hội đồng bằng việc chứng minh giải pháp PEFT tiệm cận hoặc vượt qua SOTA thế giới. |

---

## PHẦN 3: CƠ CHẾ DÒNG CHẢY DỮ LIỆU (DATAFLOW COMPARISON)

### 3.1. Dòng chảy lỗi trong Pipeline cũ (Nguyên nhân thất bại)
```
[Trọng số TrajectoryCrafter tác giả]
          │
          ▼
    [Tiêm DoRA] ──► Train với Loss: MSE(pred, noise)  <--- SAI LỆCH TOÁN HỌC!
          │        (Trong khi scheduler yêu cầu v-prediction)
          ▼
[Checkpoint DoRA Stage 1 bị hỏng] ──► Denoising Scheduler triệt tiêu tín hiệu về 0
          │                            ==> SINH RA MÀN SƯƠNG MÙ XÁM (GREY FOG)!
          ▼
[Train tiếp Stage 2 Perceiver trên nền DoRA hỏng] 
          ==> MÀN SƯƠNG MÙ XANH ĐEN / RÁCH NÁT (DARK GREEN FOG)!
```

### 3.2. Dòng chảy chuẩn mực trong Kiến trúc Đề xuất Mới
```
[Base Model: CogVideoX-Fun-InP 2D thuần túy (Alibaba PAI)]
          │
          ▼
    [Tiêm DoRA Stage 1] ──► Train với Loss: MSE(pred, v_target)  <--- ĐÚNG CHUẨN V-PRED!
          │                (Detach Norm + weight_decay=0.0 cho m)
          ▼
[Checkpoint DoRA Stage 1 Chuẩn] ──► Khung cảnh 3D sắc nét, inpaint bám chuẩn quỹ đạo camera!
          │
          ▼ (Đóng băng 100% Base Model + DoRA Stage 1)
[Thêm Perceiver Cross-Attention khởi tạo mới]
          │
          ▼
    [Train Stage 2] ──► Dạy Perceiver trích xuất texture sắc nét từ 10 frame nguồn
          │
          ▼
[MÔ HÌNH HOÀN CHỈNH ĐỀ XUẤT] ──► So sánh trực tiếp với External SOTA (TrajectoryCrafter Paper)!
```

---

## PHẦN 4: CƠ SỞ LÝ THUYẾT DORA & CÁC KỸ THUẬT TỐI ƯU CỐT LÕI (ICML 2024 ORAL)

### 4.1. Tư duy bản chất: Phân rã Độ lớn (Magnitude) và Phương hướng (Direction)
Trong chuẩn LoRA ($W' = W_0 + \frac{\alpha}{r}BA$), cập nhật $\Delta W = BA$ làm biến thiên đồng thời cả độ lớn và phương hướng với một hệ số góc dương cố định ($\Delta M \propto \Delta D$). Điều này khiến LoRA bị "khóa cứng", không thể thực hiện các điều chỉnh tinh tế như xoay hướng mà giữ nguyên biên độ năng lượng.

DoRA giải quyết triệt để vấn đề này bằng cách phân rã hình học theo Weight Normalization:
$$W' = m \odot \frac{V}{\|V\|_c} = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
* $m = \|W_0\|_c \in \mathbb{R}^{1 \times k}$: Vector độ lớn khả huấn, đại diện cho năng lượng kích hoạt của từng kênh.
* $\frac{W_0 + BA}{\|W_0 + BA\|_c}$: Ma trận vector đơn vị, chỉ chuyên tâm thực hiện phép xoay hướng trích xuất đặc trưng.
* **Zero Inference Overhead**: Khi suy luận, $W'$ được merge tĩnh thành một ma trận duy nhất $W_{\text{merged}} \in \mathbb{R}^{d \times k}$, không phát sinh bất kỳ độ trễ nào.

### 4.2. Năm kỹ thuật tối ưu mang tính quyết định
1. **Kỹ thuật Detach Norm khi Backward (Mục 4.3 trong paper)**:
   $$C = \|W_0 + BA\|_c.\text{detach}() \implies W' = m \odot \frac{W_0 + BA}{C}$$
   Giảm ngay **24.4% VRAM** khi huấn luyện, đơn giản hóa gradient mà không làm suy giảm độ chính xác.
2. **Kỹ thuật tính Norm ở FP32 (FP32 Stable Norm)**:
   Phép tính $\|V\|_c = \sqrt{\sum v_i^2}$ được thực hiện ở `torch.float32` để chống tràn số (overflow/underflow) gây NaN trong môi trường mixed precision.
3. **Phân lập Optimizer Parameters (Optimizer Isolation)**:
   * **Tuyệt đối KHÔNG phạt trọng số lên $m$ (`weight_decay = 0.0`)**: Tránh làm suy giảm biên độ ma trận về 0.
   * Tham số $B, A$ áp dụng `weight_decay = 1e-2`.
4. **Khả năng chịu Rank thấp (Rank Robustness)**:
   DoRA ở rank $r=16, 32$ đạt hiệu năng tương đương hoặc vượt trội LoRA ở rank $r=64, 128$. Thiết lập chuẩn khuyến nghị: **$r=16$ hoặc $r=32$, $\alpha=2r$**.
5. **Độ hạt tinh chỉnh (Tuning Granularity)**:
   Gắn DoRA vào toàn bộ các lớp tuyến tính của 3D Spatio-Temporal Self-Attention (`to_q`, `to_k`, `to_v`, `to_out[0]`) và FFN (`ff.net[0].proj`, `ff.net[2]`) trong các block DiT.

---

## PHẦN 5: THIẾT KẾ CHI TIẾT ÁP DỤNG VÀO KIẾN TRÚC CỦA CHÚNG TA

### 5.1. Thiết kế Stage 1: DoRA Geometry Adaptation
* **Mô hình nền tảng**: `CogVideoX-Fun-V1.1-5b-InP` (33 kênh input: 16 noisy latents + 16 masked video latents + 1 mask).
* **Mô-đun huấn luyện**:
  * Các adapter DoRA được tiêm vào Self-Attention và FFN trên toàn bộ các block DiT (~73.6M tham số).
  * Base Transformer 5.6B đóng băng 100%.
* **Dữ liệu huấn luyện**:
  * Video nguồn và Quỹ đạo Camera từ RealEstate10K.
  * Quá trình Double Reprojection sinh video biến dạng 3D (warped video) và binary mask.
* **Mục tiêu tối ưu hóa**:
  $$\mathcal{L}_{\text{Stage 1}} = \|v_{\theta}(z_t, t, c_{\text{inpaint}}) - v_t\|^2$$
  với $v_t = \alpha_t \epsilon - \sigma_t z_0$ từ `scheduler.get_velocity(...)`.

### 5.2. Thiết kế Stage 2: Ref-DiT Perceiver Cross-Attention
* **Chính sách đóng băng**:
  * Base Transformer 5.6B: Đóng băng 100%.
  * Stage 1 DoRA parameters: Đóng băng 100% (hoặc merge tĩnh vào $W_0$).
* **Mô-đun huấn luyện**:
  * `ref_patch_embed`: Chiếu 10 reference frames từ video gốc thành chuỗi visual tokens.
  * `PerceiverCrossAttention`: 15 block cross-attention đặt cách quãng mỗi 2 block DiT, trích xuất thông tin sắc nét từ reference tokens bơm vào query tokens.
* **Mục tiêu tối ưu hóa**:
  $$\mathcal{L}_{\text{Stage 2}} = \|v_{\theta}(z_t, t, c_{\text{inpaint}}, f_{\text{ref}}) - v_t\|^2$$

---

## PHẦN 6: ĐÁNH GIÁ KHẢ NĂNG THÀNH CÔNG VỀ MẶT LOGIC HỌC & TOÁN HỌC

Đánh giá tổng quan: **Xác suất thành công rất cao (85% - 90%)** dựa trên 4 trụ cột:

1. **Tính phân rã bài toán tường minh (Problem Decoupling)**:
   * Stage 1 giải quyết riêng bài toán *Hình học 3D & Động học Camera* (Geometry & Kinematics).
   * Stage 2 giải quyết riêng bài toán *Bảo toàn đặc trưng vân bề mặt & Nhận dạng* (Texture & Photometric Identity).
   * Loại bỏ triệt để hiện tượng xung đột gradient (gradient interference) khi phải học cả hai mục tiêu cùng lúc.
2. **Khắc phục hoàn toàn các lỗi toán học cốt tử trước đây**:
   * Sửa triệt để bug mục tiêu $v$-prediction vs $\epsilon$-prediction.
   * Khử hiện tượng trôi chuẩn ma trận nhờ `weight_decay = 0.0` trên magnitude vector $m$.
   * Không còn phụ thuộc hay vay mượn checkpoint đã train của tác giả TrajectoryCrafter.
3. **Năng lực biểu diễn của DoRA đã được kiểm chứng**:
   * Báo cáo thực nghiệm từ ICML 2024 chứng minh DoRA tiệm cận 100% Full Fine-Tuning trên các tác vụ Diffusion (SDXL) và Vision-Language.
4. **Kế hoạch phòng ngừa rủi ro (Risk Mitigation)**:
   * Dùng Detach Norm và gradient checkpointing đảm bảo fit vừa VRAM trên GPU Kaggle/A100.
   * Cố định rank ở vùng tối ưu $r=16, 32$ để chống hiện tượng under-fitting hoặc over-fitting.

---

## PHẦN 7: LỘ TRÌNH THỰC HIỆN TỪNG BƯỚC (STEP-BY-STEP ROADMAP)

* **Bước 1 (Đang thực hiện)**: Thống nhất tài liệu kế hoạch nghiên cứu, thiết kế kiến trúc và chuẩn hóa mã nguồn gốc.
* **Bước 2 (Chuẩn bị Codebase Base)**: Đồng bộ mã nguồn sạch của bài báo TrajectoryCrafter vào `pipeline_v3` để làm chuẩn nền tảng gốc.
* **Bước 3 (Module DoRA Chuẩn)**: Viết module DoRA độc lập tích hợp Detach Norm, FP32 Stable Norm, và cơ chế cấu hình tham số tách biệt.
* **Bước 4 (Kịch bản Huấn luyện Stage 1)**: Xây dựng `train_stage1_dora.py` nạp base `CogVideoX-Fun-V1.1-5b-InP`, huấn luyện DoRA với $v$-prediction loss trên tập RealEstate10K.
* **Bước 5 (Kịch bản Huấn luyện Stage 2)**: Xây dựng `train_stage2_perceiver.py` khóa DoRA Stage 1, khởi tạo mới và huấn luyện Perceiver Cross-Attention.
* **Bước 6 (Pipeline Suy luận & Đối chuẩn Hoàn chỉnh)**: Tích hợp suy luận 2 Stage, xuất video và đánh giá định lượng trên các chỉ số chuẩn (FVD, PSNR, SSIM, RPE, Trajectory Error) đối đầu trực tiếp với TrajectoryCrafter Pretrained.
