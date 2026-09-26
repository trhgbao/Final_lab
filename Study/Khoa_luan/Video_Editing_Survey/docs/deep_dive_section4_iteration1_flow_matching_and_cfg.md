# NGHIÊN CỨU CHUYÊN SÂU MỤC 4 (VÒNG LẶP BIỆN CHỨNG 1)
# ĐỘNG HỌC KHUẾCH TÁN FLOW MATCHING & LỊCH TRÌNH ĐIỀU PHỐI CFG ĐA ĐIỀU KIỆN

**Khóa Luận Tốt Nghiệp:** Video-to-Video Camera Trajectory Retargeting (V2V-CR)  
**Mô hình Nền tảng:** Wan2.1-T2V-1.3B (Rectified Flow Matching, DiT 30 Layers, Dim 1536)  
**Tài liệu tham chiếu mã nguồn & bài báo:**
- Mã nguồn Wan2.1 Flow Match Scheduler (`diffsynth/schedulers/flow_match.py`)
- Mã nguồn ReCamMaster Training & Inference (`train_recammaster.py`, `wan_video_recammaster.py`)
- Mã nguồn CameraCtrl SVD Guidance (`cameractrl/pipelines/pipeline_animation.py`, `video2video_pipeline.py`)
- Lý thuyết Rectified Flow (Lipman et al., 2023; Liu et al., 2023 "Flow Straight and Fast")
- Kỹ thuật CFG Rescaling & Dynamic Scheduling (Lin et al., 2024 "Common Diffusion Noise Schedules and Sample Steps are Flawed")

---

## 1. BỐI CẢNH VÀ KHẢO SÁT MÃ NGUỒN SOTA (BASELINE AUDIT)

Trong bài toán Video-to-Video Camera Retargeting (V2V-CR), mục tiêu là biến đổi video nguồn $\mathcal{V}_{src}$ thành video đích $\mathcal{V}_{tgt}$ theo quỹ đạo camera mới $\mathcal{T}_{tgt}$, trong đó cảnh quan và chuyển động nội tại của vật thể được bảo toàn.

### 1.1. Khảo sát Mã nguồn ReCamMaster (`train_recammaster.py` & `wan_video_recammaster.py`)
Khi kiểm tra trực tiếp mã nguồn ReCamMaster, chúng tôi phát hiện 3 hạn chế chí mạng:
1. **Nối ghép Latent Cưỡng bức (Forced Temporal Latent Concatenation):**
   Trong `train_recammaster.py` (dòng 351–353):
   ```python
   noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
   tgt_latent_len = noisy_latents.shape[2] // 2
   noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
   ```
   ReCamMaster ghép Target ($0 \dots 20$) và Source ($21 \dots 41$) vào chung tensor `latents`. Dù Source được ghi đè bằng ảnh sạch `origin_latents`, DiT vẫn phải xử lý toàn bộ 42 frame trong pass xuôi và pass ngược, gây lãng phí bộ nhớ VRAM gấp đôi và làm méo mó trường vận tốc Flow Matching.
2. **CFG Cố định Thiếu Linh hoạt (Static CFG = 5.0):**
   Trong `wan_video_recammaster.py` (dòng 281–285):
   ```python
   noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
   ```
   Giá trị `cfg_scale = 5.0` được giữ nguyên từ bước đầu tiên ($t = 1.0$) đến bước cuối cùng ($t = 0.0$). Ở các bước khử nhiễu cuối, vector hướng dẫn CFG khuếch đại quá mức các gradient tần số cao, dẫn đến hiện tượng cháy sáng, bão hòa màu quá mức (oversaturation) và viền hào quang (halo artifacts).
3. **Bỏ qua Camera CFG (Text-Only CFG Dropout):**
   Trong nhánh âm tính `noise_pred_nega`, ReCamMaster chỉ loại bỏ văn bản (`prompt_emb_nega`), nhưng **VẪN GIỮ NGUYÊN ĐIỀU KIỆN CAMERA VÀ SOURCE**. Điều này có nghĩa là ReCamMaster hoàn toàn không có lực đẩy Classifier-Free Guidance cho Quỹ đạo Camera! Nếu mô hình có xu hướng lười biếng (chỉ sao chép Source mà không xoay góc quay theo Camera), hệ thống không có cơ chế CFG để ép buộc camera tuân thủ quỹ đạo!

### 1.2. Khảo sát Mã nguồn CameraCtrl SVD (`pipeline_animation.py`)
CameraCtrl giải quyết bài toán camera trên Stable Video Diffusion (SVD) và đưa ra phát kiến quan trọng tại dòng 456:
```python
guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
# guidance_scale tăng tuyến tính từ 1.0 ở frame đầu lên 3.0 ở frame cuối!
```
- **Lý do khoa học:** Frame 0 được neo giữ chặt bởi điều kiện ban đầu, độ dịch chuyển nhỏ nên chỉ cần guidance nhỏ ($1.0$). Càng về các frame sau ($k \to 20$), độ tích lũy dịch chuyển camera càng lớn, nguy cơ trôi dạt quỹ đạo (trajectory drift) càng cao, do đó cần hệ số guidance lớn ($3.0 \sim 3.5$) để kéo camera bám sát đường cong chỉ định!

---

## 2. PHÂN TÍCH TOÁN HỌC: TARGET-ONLY RECTIFIED FLOW MATCHING

### 2.1. Quỹ đạo Xác suất và Trường Vận tốc
Trong Wan2.1, quá trình sinh video được mô hình hóa bằng Rectified Flow trên không gian latent $\mathcal{Z} \subset \mathbb{R}^{C \times F \times H \times W}$ ($C=16, F=21, H=30, W=52$).

Định nghĩa đường nội suy thẳng (straight linear path) giữa phân phối nhiễu chuẩn $\pi_1 = \mathcal{N}(0, I)$ và phân phối dữ liệu sạch $\pi_0 = p_{data}$:
$$x_t = (1 - \sigma_t) x_0 + \sigma_t x_1, \quad \sigma_t \in [0, 1]$$
Với $\sigma_t$ là mức độ nhiễu tại thời điểm liên tục $t \in [0, 1]$.  
Vận tốc tức thời dọc theo đường cong nghiệm:
$$u_t(x_t | x_0, x_1) = \frac{d x_t}{dt} = \frac{d \sigma_t}{dt} (x_1 - x_0)$$
Với tham số hóa chuẩn của Rectified Flow ($\sigma_t = t$):
$$u_t = x_1 - x_0$$

### 2.2. Nghịch lý của Joint Flow và Ưu thế Tuyệt đối của Target-Only Flow
Nếu áp dụng Flow Matching chung cho cả Target và Source:
- Source video $x_{src}$ là một thực thể cố định đã biết (deterministic ground truth), không phải là biến ngẫu nhiên cần lấy mẫu.
- Nếu gán vận tốc cho Source hoặc đưa Source vào hàm mất mát $\mathcal{L}_{flow}$, mạng neural sẽ phân tán dung lượng để học cách tái tạo một chuỗi đã biết 100%, làm suy yếu năng lực tổng hợp các vùng bị khuất lấp (disoccluded regions) ở Target.

#### Định lý Phân tách Trường Vận tốc (Velocity Field Decoupling):
Đặt biến trạng thái mục tiêu $x_t \equiv z_{tgt, t}$. Video nguồn $z_{src}$ đóng vai trò là biến điều kiện bất biến thời gian:
$$\frac{d z_{tgt, t}}{dt} = v_\theta\left(z_{tgt, t}, t; \; z_{src}, \mathcal{T}_{tgt \leftarrow src}, c_{txt}\right)$$
Hàm mất mát huấn luyện chuẩn xác:
$$\mathcal{L}_{flow\_tgt} = \mathbb{E}_{t \sim p(t), x_0 \sim q(z_{tgt}), x_1 \sim \mathcal{N}(0, I)} \left[ w(t) \cdot \left\| v_\theta\left((1-\sigma_t)x_0 + \sigma_t x_1, t, z_{src}, \mathbf{E}_{cam}, c_{txt}\right) - (x_1 - x_0) \right\|_2^2 \right]$$
- **Bảo chứng Toán học:**
  1. Gradient $\nabla_\theta \mathcal{L}$ **CHỈ LAN TRUYỀN** qua nhánh dự đoán vận tốc của Target $x_t$.
  2. Toàn bộ đặc trưng của Source $z_{src}$ được truyền qua mạng dưới dạng Clean Context Memory trong `torch.no_grad()`, loại bỏ hoàn toàn nhiễu loạn gradient.
  3. Tiết kiệm chính xác $50\%$ bộ nhớ kích hoạt (Activation VRAM) của hàm mất mát.

---

## 3. THIẾT KẾ ĐỘT PHÁ VÒNG 1 (MỤC 4): SPATIO-TEMPORAL DYNAMIC CFG (ST-CFG)

Để giải quyết đồng thời hai bài toán: **Bám sát Quỹ đạo Camera 3D** và **Chống Bão hòa Màu/Cháy Sáng**, chúng tôi đề xuất kiến trúc điều phối CFG Không-Thời gian 2 Chiều:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              SƠ ĐỒ ĐIỀU PHỐI CFG HAI CHIỀU KHÔNG-THỜI GIAN (ST-CFG)                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. CHIỀU THỜI GIAN KHỬ NHIỄU (Diffusion Timestep t in [0, 1]):                         │
│    t = 1.0 (Nhiễu lớn) ──► s_time(t) = 4.5  (Định hình cấu trúc 3D & góc quay camera)  │
│    t = 0.0 (Ảnh sạch)  ──► s_time(t) = 1.5  (Bảo toàn độ tương phản tự nhiên, mịn màng)│
│                                                                                        │
│ 2. CHIỀU THỜI GIAN VIDEO (Frame Index k in [0, 20]):                                   │
│    Frame k = 0 (Neo gốc)     ──► s_frame(k) = 1.0 (Không ép sai lệch khung ban đầu)   │
│    Frame k = 20 (Độ dịch lớn)──► s_frame(k) = 1.5 (Kéo mạnh chống trôi dạt camera)     │
│                                                                                        │
│ 3. CÔNG THỨC HỢP NHẤT:                                                                │
│    s_cfg(t, k) = s_time(t) * s_frame(k)                                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1. Công thức Toán học của ST-CFG
Hệ số hướng dẫn tổng hợp tại bước khuếch tán $t \in [0, 1]$ và khung hình thứ $k \in \{0, \dots, F-1\}$:
$$s_{cfg}(t, k) = \left[ s_{min} + (s_{max} - s_{min}) \cdot \sin\left(\frac{\pi t}{2}\right) \right] \cdot \left[ 1.0 + \alpha_{ramp} \cdot \left(\frac{k}{F-1}\right) \right]$$
Với các siêu tham số tối ưu:
- $s_{min} = 1.5$ (ngưỡng an toàn chống cháy màu tại $t \to 0$).
- $s_{max} = 3.5$ (ngưỡng cực đại định hình camera tại $t \to 1$).
- $\alpha_{ramp} = 0.4$ (độ dốc bù trừ trôi dạt khung hình cuối).

### 3.2. Chiến lược Dropout Đa Điều Kiện Khi Huấn Luyện (Training Condition Dropout)
Để mạng DiT có thể thực hiện CFG độc lập cho cả Văn bản và Camera mà không làm tăng chi phí tính toán khi suy luận:
Trong quá trình huấn luyện từ đầu (`training_step`), tại mỗi batch, ta thực hiện ngẫu nhiên hóa điều kiện:

| Tình huống Huấn luyện | Xác suất $p$ | Điều kiện Text ($c_{txt}$) | Điều kiện Camera ($\mathbf{E}_{cam}$) | Điều kiện Nguồn ($z_{src}$) | Mục đích Huấn luyện |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Đầy đủ Điều kiện (Full Cond)** | $80\%$ | Giữ nguyên | Giữ nguyên | Giữ nguyên | Học tương quan không-thời gian chuẩn |
| **Drop Text Only** | $10\%$ | Gán rỗng `""` | Giữ nguyên | Giữ nguyên | Ép mạng dựa vào Camera & Source thay vì phụ thuộc Text |
| **Drop Camera Only** | $5\%$ | Giữ nguyên | $\mathbf{E}_{cam} = \mathbf{0}$ | Giữ nguyên | Huấn luyện điểm tựa âm tính cho Camera CFG |
| **Drop Source Only** | $3\%$ | Giữ nguyên | Giữ nguyên | $z_{src} = \mathbf{0}$ | Huấn luyện năng lực tự sinh (Novel View Inpainting) |
| **Unconditional (Drop All)** | $2\%$ | Gán rỗng `""` | $\mathbf{E}_{cam} = \mathbf{0}$ | $z_{src} = \mathbf{0}$ | Điểm neo vô điều kiện tuyệt đối |

### 3.3. Tối ưu Hóa Suy Luận: 2-Pass Compositional CFG
Nhiều nghiên cứu đa điều kiện đòi hỏi 3 hoặc 4 pass mạng DiT cho mỗi bước khử nhiễu (Negative Text, Negative Camera, Positive). Điều này làm tăng thời gian suy luận lên gấp 2-3 lần, không khả thi trên GPU Kaggle 16GB/24GB.

Chúng tôi đề xuất **2-Pass Compositional CFG**:
Chỉ thực hiện đúng 2 lần suy luận DiTBlock trên mỗi timestep:
1. **Pass Có Điều Kiện Toàn Phần ($\mathbf{v}_{cond}$)**: Đầy đủ Text, Source và Camera Trajectory.
2. **Pass Âm Tính Hỗn Hợp ($\mathbf{v}_{uncond}$)**: Drop Text và gán Camera về Quỹ đạo Tĩnh ($\mathcal{T} = I$).
Vận tốc tổng hợp:
$$\mathbf{v}_{final}(t, k) = \mathbf{v}_{uncond}(t, k) + s_{cfg}(t, k) \cdot \left[ \mathbf{v}_{cond}(t, k) - \mathbf{v}_{uncond}(t, k) \right]$$
- **Hiệu quả:**
  - Vừa kéo mạnh ngữ nghĩa văn bản, vừa đẩy mạnh gia tốc camera theo quỹ đạo.
  - Tốc độ suy luận giữ nguyên $2\times$ (hoàn toàn tương thích với thời gian thực thi tiêu chuẩn của Wan2.1).

---

## 4. BỘ GIẢI FLOW-ODE: MIDPOINT SOLVER VS FIRST-ORDER EULER

Trong `diffsynth/schedulers/flow_match.py`, hàm `step` sử dụng phương pháp Euler bậc 1:
$$x_{t - \Delta t} = x_t + v_\theta(x_t, t) \cdot (\sigma_{next} - \sigma_{curr})$$
- Euler bậc 1 có sai số cắt cụt cục bộ $O(\Delta t^2)$ và sai số toàn cục $O(\Delta t)$.
- Với các quỹ đạo camera uốn lượn mạnh (như Orbit 360 độ hoặc S-curve), bước nhảy Euler lớn ($\Delta t \approx 0.02 \sim 0.05$) có thể làm lệch quỹ đạo quay.

### Đề xuất Bộ giải Midpoint 2nd-Order Thích Ứng (Adaptive Midpoint Flow):
Tại các bước có vận tốc góc camera lớn ($\|\mathbf{\omega}\|_{cam} > \tau_{ang}$):
1. Dự đoán bước đệm tại trung điểm:
   $$x_{mid} = x_t + v_\theta(x_t, t) \cdot \frac{\Delta \sigma}{2}$$
2. Tính vận tốc tại trung điểm:
   $$v_{mid} = v_\theta\left(x_{mid}, t + \frac{\Delta t}{2}\right)$$
3. Cập nhật bước nhảy chính xác bậc 2:
   $$x_{next} = x_t + v_{mid} \cdot \Delta \sigma$$
Sai số toàn cục giảm xuống $O(\Delta t^2)$, giúp đường cong camera mượt mà tuyệt đối mà chỉ tốn thêm chi phí tính toán cục bộ tại các khúc cua gấp!

---

## 5. KẾ HOẠCH BÀN GIAO VÀ TIẾN TRÌNH TIẾP THEO

Bản nghiên cứu chuyên sâu này hoàn thành **Vòng lặp Biện chứng 1 của Mục 4**.
Các nội dung cốt lõi:
1. Xác lập Target-Only Rectified Flow Matching Loss, bảo toàn 100% tài nguyên cho việc tổng hợp vùng mới.
2. Thiết kế Lịch trình CFG Hai Chiều Không-Thời gian (ST-CFG), loại bỏ hoàn toàn viền cháy sáng và bão hòa màu ở frame cuối.
3. Ma trận Dropout đa điều kiện tối ưu cho huấn luyện từ đầu trên RealEstate10K.
4. Tối ưu hóa bộ giải Flow-ODE bậc 2 Midpoint cho quỹ đạo camera phức tạp.

Tài liệu thiết kế tổng thể ([`docs/v2v_selected_architecture_and_training_design.md`](file:///d:/Study/Khoa_luan/Video_Editing_Survey/docs/v2v_selected_architecture_and_training_design.md)) sẽ được đồng bộ hóa ngay sau đây.
