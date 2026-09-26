# BÀI HỌC KIẾN TRÚC & THỰC NGHIỆM RÚT RA
# VIDEO-TO-VIDEO CAMERA TRAJECTORY RETARGETING
## Tổng Hợp Các Phương Pháp Đã Thử, Lý Do Thất Bại & Hướng Đi Đúng

---

## 📑 MỤC LỤC
1. [Tổng Quan Bài Toán & Mục Tiêu Ban Đầu](#1-tổng-quan-bài-toán--mục-tiêu-ban-đầu)
2. [Phương Pháp 1: Hook MLP Cộng Vào Hidden States (Synthetic Loss)](#2-phương-pháp-1-hook-mlp-cộng-vào-hidden-states)
3. [Phương Pháp 2: True Diffusion Denoising Loss](#3-phương-pháp-2-true-diffusion-denoising-loss)
4. [Phương Pháp 3: Real Dataset Pose + Plücker Rays](#4-phương-pháp-3-real-dataset-pose--plücker-rays)
5. [Phương Pháp 4: Multi-Feature Training (DINOv2 + Flow + Plücker)](#5-phương-pháp-4-multi-feature-training)
6. [Phương Pháp 5: Dual-Branch V2V Adapter (Patch Embed src_tokens)](#6-phương-pháp-5-dual-branch-v2v-adapter)
7. [Bảng Tổng Kết So Sánh Tất Cả Phương Pháp](#7-bảng-tổng-kết-so-sánh)
8. [Phân Tích Gốc Rễ: Tại Sao Tất Cả Đều Thất Bại](#8-phân-tích-gốc-rễ)
9. [Kiến Trúc Đúng Theo Nghiên Cứu SOTA](#9-kiến-trúc-đúng-theo-nghiên-cứu-sota)
10. [Bài Học Tổng Quát Cho Nghiên Cứu Tương Lai](#10-bài-học-tổng-quát)
11. [Phương Pháp 6: Chained Incremental Warping & Thất Bại Của Khe Hẹp 14-pixel (Latent Slit Smear)](#11-phương-pháp-6-chained-incremental-warping--thất-bại-của-khe-hẹp-14-pixel)
12. [Đột Phá Toán Học: Latent Replacement Protocol (RePaint Blending) & Dual-Stage Pixel Clamping](#12-đột-phá-toán-học-latent-replacement-protocol-repaint-blending--dual-stage-pixel-clamping)
13. [Thuật Toán Dynamic Spatial Reference Selection Thay Thế Neo Cứng Frame 0](#13-thuật-toán-dynamic-spatial-reference-selection-thay-thế-neo-cứng-frame-0)
14. [Kiến Trúc Điều Kiện 3 Tầng: Spatial Base + Temporal Context (t-1, t-2) + Global Scene Tokens](#14-kiến-trúc-điều-kiện-3-tầng-spatial-base--temporal-context-t-1-t-2--global-scene-tokens)

---

## 1. TỔNG QUAN BÀI TOÁN & MỤC TIÊU BAN ĐẦU

### 1.1. Định nghĩa bài toán
Cho một video nguồn $V_{src}$ với quỹ đạo camera gốc $\mathbf{P}_{src}$, sinh ra video mới $V_{tgt}$ với quỹ đạo camera khác $\mathbf{P}_{tgt}$, đồng thời **bảo toàn toàn bộ nội dung động** (chuyển động người, vật thể, hiệu ứng vật lý) từ video gốc.

### 1.2. Ý tưởng ban đầu
- Sử dụng CogVideoX-5B-I2V (42 DiT Blocks) làm backbone đóng băng 100%.
- Thêm một Adapter nhỏ (MLP/Attention module) được huấn luyện để:
  - Nhận input: Plücker rays (biểu diễn quỹ đạo camera mục tiêu)
  - Output: Tín hiệu điều chỉnh được cộng vào hidden_states của DiT blocks
- Kỳ vọng: Adapter sẽ "dạy" model đóng băng hiểu cách xoay camera theo quỹ đạo mới.

### 1.3. Backbone model
- **CogVideoX-5B-I2V**: 42 CogVideoXBlock (DiT), 3D Causal VAE (49 frames → 13 latent slices), Text tokens: 226, Visual tokens: 17,550, Full sequence: 17,776.
- **Latent dims**: Spatial 480×720 → 60×90 (VAE) → 30×45 (patch embed). Feature dim: 3072.

---

## 2. PHƯƠNG PHÁP 1: HOOK MLP CỘNG VÀO HIDDEN STATES (Synthetic Loss)

### 🎯 Ý tưởng
Tạo một MLP adapter nhỏ, hook vào các DiT blocks, cộng output của adapter vào hidden_states. Huấn luyện bằng synthetic MSE loss so sánh predicted noise với target noise.

### ⚙️ Chi tiết kỹ thuật
- **Adapter**: MLP 3 lớp: `Linear(6 * LAT_H * LAT_W, 4096) → GELU → Linear(4096, 3072)`
- **Input**: Plücker rays flatten thành vector 6D
- **Injection**: `hidden_states += adapter_output` tại mỗi DiT block
- **Loss**: MSE giữa predicted noise và synthetic target noise
- **Training**: 100 steps, learning rate không rõ

### ❌ Kết quả
- Góc xoay đo được: **~10.2°** (pan right)
- Loss: Không hội tụ rõ ràng

### 🔬 Phân tích thất bại

> **Bài học 1: Synthetic loss ≠ Real diffusion training.**
> MSE giữa predicted noise và một target noise tự tạo KHÔNG phải là cách huấn luyện diffusion model đúng cách. Cần phải:
> 1. Thêm noise thực sự vào clean latents theo schedule
> 2. Dự đoán noise đã thêm
> 3. Tính loss giữa predicted noise và noise thật đã thêm

> **Bài học 2: ~10° là DEFAULT motion của CogVideoX.**
> Khi dùng text prompt "camera pan right", CogVideoX-5B tự sinh ~10° xoay MÀ KHÔNG CẦN bất kỳ adapter nào. Con số 10.2° chứng tỏ adapter **không có tác dụng gì** — model chỉ đang chạy theo text prompt prior.

---

## 3. PHƯƠNG PHÁP 2: TRUE DIFFUSION DENOISING LOSS

### 🎯 Ý tưởng
Sửa lại training loop cho đúng chuẩn diffusion: thêm noise thật, predict noise, tính MSE loss chuẩn.

### ⚙️ Chi tiết kỹ thuật
- **Noise schedule**: Sử dụng scheduler của CogVideoX
- **Forward process**: $z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$
- **Loss**: $\mathcal{L} = \|{\epsilon - \epsilon_\theta(z_t, t, c)}\|^2$
- **Training**: 60 steps (quá ít cho 5B model)

### ❌ Kết quả
- Góc xoay đo được: **~9.0°** (thậm chí GIẢM so với baseline)
- Loss bế tắc tại: **1.41**

### 🔬 Phân tích thất bại

> **Bài học 3: 60 steps hoàn toàn không đủ cho model 5B tham số.**
> Với hàng tỷ tham số đóng băng, adapter chỉ vài triệu tham số cần HÀNG NGHÌN bước training trên dataset đa dạng để có tác động đáng kể. 60 bước training gần như chỉ là noise.

> **Bài học 4: Loss 1.41 cho thấy adapter quá yếu.**
> Loss diffusion chuẩn cần giảm xuống dưới 0.5 để có tác động thực sự. 1.41 nghĩa là model gần như bỏ qua hoàn toàn tín hiệu từ adapter.

---

## 4. PHƯƠNG PHÁP 3: REAL DATASET POSE + PLÜCKER RAYS

### 🎯 Ý tưởng
Thay vì dùng pose tổng hợp, lấy pose thật từ dataset RealEstate10K (file `0f68374b76390082_svd.txt` với +41° pure pan right). Tính Plücker rays từ pose thật.

### ⚙️ Chi tiết kỹ thuật
- **Pose format**: 19 số/dòng: `timestamp fx fy cx cy k1 k2 r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz`
- **Plücker rays**: 6D per-pixel: $[\mathbf{r}_o \times \mathbf{r}_d, \mathbf{r}_d]$
- **Lỗi phát hiện**: Thiếu Relative Pose Normalization → ghosting artifacts

### ❌ Kết quả
- Góc xoay đo được: **~5.5°** (GIẢM mạnh)
- Hiện tượng: Ghosting nghiêm trọng, méo vật thể

### 🔬 Phân tích thất bại

> **Bài học 5: PHẢI chuẩn hóa pose tương đối (Relative Pose Normalization).**
> Pose tuyệt đối từ dataset khác mang tọa độ world khác. Tại frame 0, camera position/rotation bị giật đột ngột → phá vỡ latent space.
> $$R_{norm}(t) = R(t) \cdot R(0)^T, \quad T_{norm}(t) = R(0) \cdot (T(t) - T(0))$$
> Đảm bảo tại $t=0$: $R_{norm}(0) = I$, $T_{norm}(0) = \mathbf{0}$.

> **Bài học 6: Dấu ma trận W2C phải tuân thủ đúng quy ước.**
> RealEstate10K dùng World-to-Camera. Quay phải +θ: $R_{02} = -\sin\theta$, $R_{20} = +\sin\theta$. Viết sai dấu → quay ngược hoặc triệt tiêu chuyển động.

---

## 5. PHƯƠNG PHÁP 4: MULTI-FEATURE TRAINING (DINOv2 + Flow + Plücker)

### 🎯 Ý tưởng
Không chỉ dùng Plücker rays, mà kết hợp 3 loại features:
1. **DINOv2**: Trích xuất cấu trúc semantic từ mỗi frame
2. **Optical Flow**: Mã hóa chuyển động giữa các frame
3. **Plücker Rays**: Biểu diễn quỹ đạo camera

### ⚙️ Chi tiết kỹ thuật
- **DINOv2**: `dinov2-large`, extract patch tokens
- **Optical Flow**: Tính flow giữa frame liên tiếp
- **Combined loss**: Weighted sum of flow loss, structure loss, diffusion loss
- **Training**: 400 steps, ~8 phút trên Kaggle GPU

### ❌ Kết quả
- Góc xoay đo được: **~10.2°** (quay lại đúng baseline!)
- Loss bế tắc tại: **1.41** (y hệt phương pháp 2)

### 🔬 Phân tích thất bại

> **Bài học 7: Thêm loss phụ không giúp khi kiến trúc sai fundamentally.**
> Multi-feature loss (DINOv2 + Flow + Plücker) chỉ cung cấp thêm supervision signal, nhưng nếu adapter injection mechanism (cộng vào hidden_states) quá yếu so với 5B-parameter backbone, thì BẤT KỲ loại loss nào cũng không thể thắng.
> Giống như cố gắng lái tàu chở hàng bằng cách đẩy nhẹ bằng tay — cho dù bạn biết chính xác phải đẩy theo hướng nào (perfect loss), lực đẩy vẫn không đủ.

> **Bài học 8: Loss 1.41 là dấu hiệu "adapter bị model đóng băng nuốt chửng".**
> Khi loss không thay đổi qua nhiều loại training khác nhau, đó là bằng chứng rằng gradient từ adapter không đủ mạnh để ảnh hưởng đến output cuối cùng qua 42 layers đóng băng.

---

## 6. PHƯƠNG PHÁP 5: DUAL-BRANCH V2V ADAPTER (Patch Embed src_tokens)

### 🎯 Ý tưởng
Thay đổi triệt để: Thay vì chỉ inject Plücker rays, tạo một nhánh thứ hai (dual-branch) mã hóa toàn bộ video nguồn thành src_tokens qua patch_embed, rồi cross-modulate với Plücker rays.

### ⚙️ Chi tiết kỹ thuật
- **Source encoding**: Dùng `pipe.transformer.patch_embed(text_embeds, image_embeds)` để mã hóa video nguồn → 17,550 src_tokens
- **Cross-modulation**: `src_modulated = cross_attn(src_tokens, pose_features)`
- **Injection**: `hidden_states += 0.15 * src_modulated` tại 16 DiT blocks (blocks 10–25)
- **Training**: 150 steps

### ❌ Kết quả
- **MÀN HÌNH ĐEN HOÀN TOÀN** — Hình ảnh bị corrupted 100%

### 🔬 Phân tích thất bại

> **Bài học 9: TUYỆT ĐỐI KHÔNG cộng tín hiệu vào hidden_states qua nhiều layer liên tiếp.**
> Khi cộng `0.15 * src_modulated` vào 16 layers liên tiếp (blocks 10-25), tín hiệu bị tích lũy theo cấp số nhân:
> - Block 10: norm ≈ baseline + 0.15x
> - Block 15: norm ≈ baseline + ~0.15x × amplification_factor^5
> - Block 25: norm → **OVERFLOW** → softmax saturates → attention outputs = NaN/0
>
> Đây là hiện tượng **Numerical Cascade Overflow** — mỗi layer amplify tín hiệu inject từ layer trước.

> **Bài học 10: Additive injection ≠ Cross-Attention injection.**
> Các paper SOTA (TrajectoryCrafter, ReCamMaster) KHÔNG BAO GIỜ cộng trực tiếp vào hidden_states. Thay vào đó:
> - **TrajectoryCrafter**: Dùng Perceiver Cross-Attention riêng biệt (Query từ target, Key/Value từ reference)
> - **ReCamMaster**: Concat tokens theo chiều frame ($2F$ frames) rồi dùng 3D self-attention tự nhiên
> - **CameraCtrl II**: Chỉ cộng MỘT LẦN DUY NHẤT trước DiT block đầu tiên (single-layer patchify)

---

## 7. BẢNG TỔNG KẾT SO SÁNH

| # | Phương pháp | Steps | Loss cuối | Góc xoay | Vấn đề chính |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | **Baseline** (không adapter, chỉ text prompt) | — | — | ~10° | CogVideoX default motion |
| 1 | Hook MLP + Synthetic Loss | 100 | N/A | 10.2° | = Baseline, adapter vô dụng |
| 2 | True Diffusion Loss | 60 | 1.41 | 9.0° | Quá ít steps, loss stuck |
| 3 | Real Pose + Plücker | N/A | N/A | 5.5° | Ghosting (thiếu relative norm) |
| 4 | Multi-Feature (DINOv2+Flow+Plücker) | 400 | 1.41 | 10.2° | = Baseline, adapter vẫn quá yếu |
| 5 | Dual-Branch V2V (src_tokens inject) | 150 | N/A | **MÀN HÌNH ĐEN** | Numerical overflow qua 16 layers |

**Kết luận quan trọng**: Tất cả 5 phương pháp đều xoay quanh ý tưởng "cộng tín hiệu adapter nhỏ vào hidden_states của model 5B đóng băng". Ý tưởng này **SAI CƠ BẢN** — không phải sai ở implementation, mà sai ở architectural design.

---

## 8. PHÂN TÍCH GỐC RỄ: TẠI SAO TẤT CẢ ĐỀU THẤT BẠI

### 8.1. Lý do cốt lõi #1: Thiếu thông tin 3D geometry
Tất cả phương pháp của chúng ta đều giả định rằng Plücker rays + text prompt đủ để "dạy" model xoay camera. Nhưng thực tế:

- **Model KHÔNG BIẾT cảnh 3D trông như thế nào.** Nó chỉ thấy 2D pixels.
- Khi yêu cầu "xoay camera sang phải 40°", model cần biết:
  - Vật thể nào ở phía bên phải (ngoài khung hình hiện tại)?
  - Cấu trúc 3D của cảnh (depth) để biết vật nào che vật nào?
  - Khu vực nào bị "disoccluded" (lộ ra) khi camera xoay?

> **Insight từ user (trích nguyên văn):**
> "ta phải lấy quỹ đạo gốc cùng những thông tin nó lấy được với quỹ đạo đó sẽ làm thông tin để khi ta đổi quỹ đạo thì những vật thể ở các gốc nhìn khác lấy được từ video gốc sẽ được model hấp thụ"

### 8.2. Lý do cốt lõi #2: Adapter additive injection quá yếu
- CogVideoX-5B có 42 DiT blocks × ~120M params/block = ~5B params đóng băng
- Adapter chỉ có ~5-20M params
- Tỉ lệ sức mạnh: adapter chiếm **0.1-0.4%** tổng model
- Giống như gắn 1 mũi tên chỉ hướng nhỏ trên đầu một con tàu 50,000 tấn đang chạy — mũi tên không thể thay đổi hướng tàu.

### 8.3. Lý do cốt lõi #3: Thiếu cơ chế tách biệt "cái gì giữ" vs "cái gì đổi"
Tất cả phương pháp đều cộng/trộn tín hiệu camera vào hidden_states chung — model không có cách nào phân biệt:
- Thông tin nào là **nội dung cần bảo toàn** (người đang chạy, xe đang di chuyển)
- Thông tin nào là **góc nhìn cần thay đổi** (vị trí camera, phối cảnh)

Các paper SOTA giải quyết điều này bằng cách dùng **dual-stream architecture** hoặc **separate conditioning pathways**.

---

## 9. KIẾN TRÚC ĐÚNG THEO NGHIÊN CỨU SOTA

### 9.1. Kiến trúc TrajectoryCrafter (ICCV 2025 Oral)
Pipeline đúng gồm 4 bước:

```
Video gốc → DepthCrafter → Depth maps
                ↓
            3D Forward Warp (source pose → target pose)
                ↓
            Warped video + Disocclusion masks
                ↓
            [z_noisy | z_masked | z_mask] → 33-channel DiT input
                ↓
            Perceiver Cross-Attention (Query=target, KV=reference frames)
                ↓
            VAE Decode → Video mới
```

**Điểm mấu chốt:**
1. **Depth estimation** (DepthCrafter): Chuyển 2D → 3D point cloud
2. **3D Forward Warping**: Chiếu điểm 3D từ source pose → target pose, tạo ra warped frame + mask
3. **Diffusion chỉ inpaint vùng mask**: Model KHÔNG phải tạo lại toàn bộ video, chỉ điền vào vùng bị che khuất
4. **Cross-Attention**: 15 Perceiver modules cho phép target tokens "nhìn" vào reference frames gốc

### 9.2. Kiến trúc ReCamMaster (ICCV 2025 Best Paper Finalist)
Cách tiếp cận khác — **không cần depth, không cần warping**:

```
Source video → VAE Encode → Source tokens (F frames)
Target noise  → VAE Sample → Target tokens (F frames)
Camera poses  → Pose Encoder → Pose embeddings
                    ↓
Concat: [Source tokens, Target tokens] → 2F frames
                    ↓
3D Spatio-Temporal Self-Attention (tự nhiên cross-attend)
                    ↓
Lấy F frames cuối → VAE Decode → Video mới
```

**Điểm mấu chốt:**
1. Concat source + target theo chiều frame → attention tự nhiên cho phép target "nhìn" source
2. Camera pose inject qua Pose Encoder riêng, không cộng vào hidden_states
3. Train trên UE5 Multi-Cam Dataset (136K video đồng bộ multi-camera)

### 9.3. PostCam: Query-Shared Cross-Attention (arXiv 2025)
Kết hợp ưu điểm cả hai:
- Render proxy video ở **độ phân giải thấp 4x** từ depth (tránh artifact depth)
- Camera pose encoder riêng biệt
- **QSCA**: Dùng chung Query, concat Key/Value từ cả rendered proxy và camera pose
- Kết quả: Vượt TrajectoryCrafter và ReCamMaster (RotErr 0.0501 vs 0.0649)

---

## 10. BÀI HỌC TỔNG QUÁT CHO NGHIÊN CỨU TƯƠNG LAI

### 10.1. Nguyên tắc thiết kế kiến trúc

| # | Nguyên tắc | Giải thích |
|:---:|:---|:---|
| 1 | **Không bao giờ cộng trực tiếp vào hidden_states qua nhiều layer** | Gây numerical overflow. Dùng cross-attention hoặc frame concatenation. |
| 2 | **Camera conditioning nên inject ở early layers hoặc input** | AC3D chứng minh: chỉ cần 8/32 blocks đầu. CameraCtrl II: chỉ 1 lần trước block đầu. |
| 3 | **Cần mechanism phân tách content vs. viewpoint** | Dual-stream (TrajectoryCrafter), Frame-concat (ReCamMaster), Flow re-alignment (Vid-CamEdit). |
| 4 | **3D geometry (depth) giúp nhưng không bắt buộc** | ReCamMaster đạt SOTA không cần depth. PostCam dùng depth nhưng downsample 4x. |
| 5 | **Dataset quyết định — RealEstate10K gây "static scene bias"** | Cần augment với stationary-camera dynamic footage (AC3D) hoặc UE5 synthetic (ReCamMaster). |

### 10.2. Checklist trước khi thử kiến trúc mới

- [ ] Kiến trúc có mechanism tách biệt content và viewpoint không?
- [ ] Camera signal inject bao nhiêu lần, ở đâu? (≤1 lần ở input, hoặc cross-attention riêng)
- [ ] Có thể scale training data không? (cần >10K video clips, không phải 1 video)
- [ ] Loss có đúng chuẩn diffusion không? (noise prediction hoặc v-prediction)
- [ ] Đã kiểm tra baseline (không adapter) chưa? So sánh với baseline trước.

### 10.3. Con đường khả thi nhất để tiếp tục

Dựa trên phân tích SOTA, **3 hướng khả thi theo thứ tự ưu tiên**:

1. **Sử dụng TrajectoryCrafter trực tiếp** (code đã có trong workspace tại `sota_baselines/TrajectoryCrafter/`):
   - Ưu điểm: Code sẵn, kiến trúc đã chứng minh, ICCV 2025 Oral
   - Cần: Download weights DepthCrafter + CrossTransformer3DModel

2. **Implement phương pháp ReCamMaster** (frame-dimension concatenation):
   - Ưu điểm: Không cần depth estimation, elegant, ICCV 2025 Best Paper Finalist
   - Cần: UE5 Multi-Cam Dataset hoặc tương đương, fine-tune 3D attention layers

3. **Implement PostCam QSCA** (Query-Shared Cross-Attention):
   - Ưu điểm: SOTA performance, hybrid approach tránh depth artifacts
   - Cần: Complexity cao, two-stage training

---

## 11. PHƯƠNG PHÁP 6: CHAINED INCREMENTAL WARPING & THẤT BẠI CỦA KHE HẸP 14-PIXEL (Latent Slit Smear)

### 🎯 Ý tưởng thực nghiệm
Trong lần thử nghiệm retargeting video nhà bếp (`0bf152ef84195293_svdxt.gif`), mục tiêu là quay camera tĩnh sang phải $30^\circ$.
Phương pháp đã thử:
- Chia góc quay $30^\circ$ thành 24 bước nhỏ liên tiếp: $\Delta \theta = \frac{30^\circ}{24} = 1.25^\circ$/frame.
- Frame $t$ được warp tịnh tiến từ kết quả đã sinh của frame $t-1$: $I_t^{\text{base}} = \text{warp}(I_{t-1}^{\text{syn}}, \Delta \theta)$.
- Dùng SDXL Inpainting để inpaint vùng khuyết nhỏ mở ra ở mỗi frame.

### ❌ Kết quả thực nghiệm
- Video sinh ra (`pure_pan_30deg_kitchen.mp4`) bị suy biến nặng nề: hình ảnh bị nhòe nhoẹt (blurry mush), các đường sọc ngang kéo dài (horizontal streaking artifacts), đồ nội thất (bàn, tủ, ghế) bị biến dạng hoàn toàn và góc quay không đạt được $30^\circ$ tĩnh mà bị giật loạn xạ.

### 🔬 Phân tích nguyên nhân gốc rễ (Root Cause Analysis)

> **Bài học 6: Nghịch lý khe hẹp Latent (Latent Slit Smear)**
> - Ở độ phân giải $576 \times 320$, góc quay $1.25^\circ$ chỉ mở ra một khe khuyết rộng khoảng **14 pixel RGB** ($576 \times 0.0249$).
> - Trong không gian Latent của SDXL (hạ mẫu không gian $8\times$), khe 14 pixel này chỉ tương đương với **1.75 pixel latent** ($14 / 8 \approx 1.75$).
> - **Cơ chế gây lỗi**: Các kernel tích chập ($3 \times 3$) và patch attention của SDXL không có đủ trường cảm thụ (receptive field) để tổng hợp các vật thể ngữ nghĩa (như chân bàn, tay nắm tủ, vân gỗ). Vùng khuyết 1.75 pixel latent nằm lọt thỏm giữa 2 vùng ảnh đã biết. Khi đó, hàm tối ưu diffusion và self-attention chỉ đơn giản thực hiện phép **nội suy trung bình màu (color smearing)** từ các pixel biên lân cận sang ngang, tạo ra các vệt mờ sọc ngang thô thiển thay vì sinh ra vật thể mới!

> **Bài học 7: Thảm họa suy biến tích lũy do Chained Warping (Warp-of-Warp)**
> - Khi thực hiện chuỗi warp liên tiếp $I_t = \text{warp}(I_{t-1}) = \text{warp}(\text{warp}(...))$, mỗi lần nội suy song tuyến (bilinear resampling) sẽ làm suy giảm tần số cao (mất độ tương phản, mất nét viền).
> - Đồng thời, sai số inpainting ở frame $t-1$ bị warp tiếp vào frame $t$, rồi lại inpaint đè lên. Đến frame 15-20, toàn bộ khung hình tích lũy hàng chục tầng sai số và biến thành một khối màu bết dính.

> **Bài học 8: Trùng chập 2 chuyển động (Compounding Motion)**
> - Video nguồn (`0bf152ef84195293_svdxt.gif`) vốn là một chuyển động tịnh tiến tiến về phía trước (dolly forward).
> - Việc áp đặt thêm một homography xoay lên trên một video đang di chuyển tịnh tiến tạo ra sự mâu thuẫn hình học phối cảnh (chuyển động kép), khiến người xem cảm thấy camera giật cục và mất định hướng không gian.

---

## 12. ĐỘT PHÁ TOÁN HỌC: LATENT REPLACEMENT PROTOCOL (RePaint Blending) & DUAL-STAGE PIXEL CLAMPING

### 🎯 Đặt bài toán nghiên cứu
*"Có thể nào đưa cả bức ảnh kể cả phần không masked vào latent, nhưng khi diffusion thì ngoài phần masked được vẽ cho phù hợp thì phần ảnh gốc vẫn giữ nguyên vẹn 100% không?"*

### 💡 Lời giải toán học: Giao thức Latent Replacement (RePaint Protocol)
Trong mô hình Latent Diffusion, câu trả lời là **HOÀN TOÀN ĐƯỢC VÀ ĐÂY LÀ CHUẨN SOTA**.

#### A. Định nghĩa toán học
Cho ảnh gốc hoàn chỉnh đã xoay về góc nhìn mới $x_{\text{orig}} \in \mathbb{R}^{3 \times H \times W}$, và mặt nạ nhị phân $M \in \{0, 1\}^{1 \times H \times W}$ ($M=1$ là vùng khuyết cần vẽ, $M=0$ là vùng ảnh gốc cần bảo tồn).

1. **Mã hóa không gian tiềm ẩn (Latent Encoding)**:
   $$z_0^{\text{orig}} = \mathcal{E}_{\text{VAE}}(x_{\text{orig}}) \in \mathbb{R}^{4 \times \frac{H}{8} \times \frac{W}{8}}$$
   $$M_{\text{latent}} = \text{Downsample}(M, \text{scale}=\frac{1}{8}) \in [0, 1]^{1 \times \frac{H}{8} \times \frac{W}{8}}$$

2. **Quy trình khử nhiễu với Latent Clamping tại mỗi bước $t \in [T, T-1, \dots, 1]$**:
   - Model UNet nhận input $z_t$ (và các điều kiện conditioning) để dự đoán nhiễu $\epsilon_\theta(z_t, t, c)$ và bộ lập lịch (scheduler) tính ra latent dự đoán cho bước tiếp theo:
     $$z_{t-1}^{\text{denoised}} = \text{SchedulerStep}(z_t, \epsilon_\theta(z_t, t, c))$$
   - Đồng thời, ta tính **chính xác 100% trạng thái latent của vùng ảnh gốc** tại timestep $t-1$ theo công thức Forward Diffusion:
     $$z_{t-1}^{\text{known}} = \sqrt{\bar{\alpha}_{t-1}} z_0^{\text{orig}} + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
   - **Thực hiện khóa cứng (Latent Replacement)**:
     $$\mathbf{z_{t-1} = z_{t-1}^{\text{known}} \odot (1 - M_{\text{latent}}) + z_{t-1}^{\text{denoised}} \odot M_{\text{latent}}}$$

#### B. Tác động kiến trúc
1. **Bảo tồn nội dung gốc tuyệt đối**: Vùng $(1 - M_{\text{latent}})$ không bị drift, không bị thay đổi màu sắc, không sinh ra chi tiết ảo giác (như hoa tím hay vân lạ).
2. **Hài hòa ngữ nghĩa qua Self-Attention**: Trong các khối Transformer của UNet, các latent token tại vùng khuyết ($M_{\text{latent}}$) liên tục tính Cross-Token Attention với các latent token vùng gốc ($1 - M_{\text{latent}}$). Do đó, vật thể mới sinh ra ở vùng trống được "ràng buộc hình học và phong cách" trực tiếp từ các vật thể thực tế lân cận.

#### C. Khắc phục sai số nén VAE: Dual-Stage Pixel Clamping
Mặc dù $z$ được bảo toàn ở latent space, bước giải nén $\mathcal{D}_{\text{VAE}}(z_0)$ vẫn chịu tổn thất nén 8x (khoảng 0.5% sai số vi lượng).
Để bảo toàn **từng byte pixel gốc ban đầu (Bit-Exact Pixel Preservation)**:
- Sau khi decode ra ảnh RGB $x_{\text{decoded}} = \mathcal{D}_{\text{VAE}}(z_0) \in [0, 1]^{3 \times H \times W}$:
- Áp dụng **Feathered Alpha Blending ở mức pixel**:
  $$\mathbf{x_{\text{final}} = x_{\text{orig}} \odot (1 - M_{\text{feathered}}) + x_{\text{decoded}} \odot M_{\text{feathered}}}$$
  Trong đó $M_{\text{feathered}} = \text{GaussianBlur}(M, \text{kernel}=5, \sigma=1.5)$ tạo đường chuyển tiếp mềm mỏng 3-5 pixel, triệt tiêu hoàn toàn đường nối biên (seam-free).

---

## 13. THUẬT TOÁN DYNAMIC SPATIAL REFERENCE SELECTION THAY THẾ NEO CỨNG FRAME 0

### 🎯 Phản biện cơ chế neo cứng Frame 0 (Fixed Frame 0 Fallacy)
Trong các video thực tế (như clip nhà bếp), camera di chuyển theo thời gian (dolly, pan, tilt).
- Tại frame $t=20$, camera đã tiến sâu vào trong phòng bếp. Nếu cố chấp dùng Frame 0 (đứng ở cửa ra vào) làm mỏ neo không gian, góc nhìn sẽ chịu sai số thị sai (parallax) cực lớn, độ phóng đại vật thể sai lệch và phần lớn không gian bị che khuất (occlusion).
- **Nguyên lý đúng**: Không gian 3D của cảnh quay là một thể liên tục. Mỗi camera pose mục tiêu $P_{\text{tgt}}^t$ phải được ghép nối với **frame nguồn phù hợp nhất trong không gian 3D**, chứ không phải frame 0 ngây thơ.

### 📐 Thuật toán Dynamic Pose Distance Optimization
Cho tập hợp $N$ khung hình nguồn $\{I_k^{\text{src}}\}_{k=0}^{N-1}$ với camera pose tương ứng $P_k^{\text{src}} = [R_k^{\text{src}} \mid \mathbf{t}_k^{\text{src}}] \in SE(3)$.
Tại mỗi thời điểm $t$ của quỹ đạo mục tiêu $P_t^{\text{tgt}} = [R_t^{\text{tgt}} \mid \mathbf{t}_t^{\text{tgt}}]$:

1. **Tính ma trận khoảng cách hình học (Geometric Pose Metric)**:
   $$\mathcal{D}(P_t^{\text{tgt}}, P_k^{\text{src}}) = \lambda_R \cdot \theta_{\text{geodesic}}(R_t^{\text{tgt}}, R_k^{\text{src}}) + \lambda_t \cdot \|\mathbf{t}_t^{\text{tgt}} - \mathbf{t}_k^{\text{src}}\|_2 - \lambda_v \cdot \text{FrustumOverlap}(P_t^{\text{tgt}}, P_k^{\text{src}})$$
   Trong đó:
   - $\theta_{\text{geodesic}}(R_1, R_2) = \arccos\left(\frac{\text{Tr}(R_1 R_2^T) - 1}{2}\right)$ đo góc lệch trục quang học.
   - $\|\mathbf{t}_1 - \mathbf{t}_2\|_2$ đo khoảng cách dịch chuyển tâm camera.
   - $\text{FrustumOverlap}$ đo tỷ lệ diện tích nón nhìn giao thoa giữa 2 camera.

2. **Lựa chọn Frame nền tảng tối ưu (Optimal Spatial Anchor)**:
   $$k^*(t) = \arg\min_{k \in \{0, \dots, N-1\}} \mathcal{D}(P_t^{\text{tgt}}, P_k^{\text{src}})$$

3. **Áp dụng cho bài toán Pure Rotation (Ví dụ xoay $30^\circ$ tại chỗ)**:
   - Khi muốn camera tại thời điểm $t$ xoay $+30^\circ$ sang phải, vị trí tâm camera mục tiêu $\mathbf{t}_t^{\text{tgt}}$ trùng với $\mathbf{t}_t^{\text{src}}$.
   - Do đó, frame nguồn tối ưu nhất **chính là frame $t$ của video gốc ($k^*(t) = t$)**!
   - Ta chỉ cần thực hiện phép xoay trực tiếp 1 lần:
     $$I_t^{\text{base}} = \text{Warp}(I_t^{\text{src}}, \Delta R_t)$$
     với $\Delta R_t = R_{\text{pan}}(\theta(t))$.
   - **Ưu điểm vượt trội**: Loại bỏ hoàn toàn chuỗi warp tích lũy (warp-of-warp), loại bỏ hiện tượng mờ nhòe, giữ nguyên độ nét 100% của frame gốc.

---

## 14. KIẾN TRÚC ĐIỀU KIỆN 3 TẦNG: SPATIAL BASE + TEMPORAL CONTEXT (t-1, t-2) + GLOBAL SCENE TOKENS

Để đảm bảo video sau khi retargeting đạt được cả 3 tiêu chí: **Đúng hình học (Geometry)**, **Mượt mà theo thời gian (Temporal Consistency)**, và **Đồng nhất phong cách toàn cảnh (Global Context)**, kiến trúc điều kiện được phân tầng thành 3 cấp:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               KIẾN TRÚC ĐIỀU KIỆN 3 TẦNG                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẦNG 3: GLOBAL SCENE PRIORS (Toàn bộ Video Đầu Vào)                                    │
│ - Trích xuất 5-8 Keyframes phân bố đều xuyên suốt video gốc.                           │
│ - Mã hóa qua DINOv2-Large (cấu trúc) + CLIP-ViT-Large (ngữ nghĩa).                     │
│ - Nén qua Perceiver Resampler thành 16 Distilled Scene Tokens (dim 2048).              │
│ - Đưa vào Cross-Attention của UNet -> Quyết định style, ánh sáng, vật liệu chung.      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẦNG 2: SHORT-TERM TEMPORAL CONTINUITY (2 Khung Hình Liền Trước: t-1, t-2)             │
│ - Lấy các vùng đã sinh thành công ở frame t-1 và t-2.                                  │
│ - Bù trừ chuyển động camera (Camera Motion Compensation / Optical Flow Warping).       │
│ - Đưa vào Spatio-Temporal Conv Pyramid Adapter (kênh điều kiện) hoặc Temporal Attn.    │
│ - Đảm bảo vật thể mới vẽ ở t-1 (ví dụ mép tủ gỗ) tiếp tục nhất quán ở frame t.        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẦNG 1: SPATIAL FOUNDATION (Frame Nền Tảng Tối Ưu k*(t))                               │
│ - Frame nguồn k*(t) được warp trực tiếp 1 lần sang pose mục tiêu P_tgt^t.              │
│ - Vùng nhìn thấy (unmasked) được đưa vào Latent và khóa cứng 100% qua Latent           │
│   Replacement và Dual-Stage Pixel Clamping.                                            │
│ - Vùng khuyết (masked hole) có kích thước lớn (>80px), cho phép Diffusion inpaint     │
│   rõ nét toàn bộ các vật thể nội thất mới.                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Bảng tóm tắt chức năng 3 tầng điều kiện:

| Tầng | Nguồn dữ liệu | Cơ chế tiếp nhận | Vai trò quyết định |
|:---|:---|:---|:---|
| **Tầng 1 (Spatial Foundation)** | Frame nguồn tối ưu $k^*(t)$ | 9-Channel UNet Concat + Latent Replacement + Pixel Clamp | Bảo toàn 100% hình ảnh thực tế, định vị góc nhìn camera chuẩn xác. |
| **Tầng 2 (Temporal Continuity)** | 2 Frame liền trước ($t-1, t-2$) | Flow Warping + Temporal ConvNet Adapter Channels | Chống rung giật, chống flickering, giữ tính liên tục của vùng mới sinh. |
| **Tầng 3 (Global Scene Priors)** | Toàn bộ video (5-8 keyframes) | Perceiver Resampler $\to$ Distilled Tokens $\to$ Cross-Attn | Giữ ánh sáng, tông màu, phong cách kiến trúc đồng nhất với cả căn phòng. |

---

## 📝 GHI CHÚ BỔ SUNG

### Insight quan trọng từ CameraCtrl II (AC3D):
> "Camera motion là thông tin low-frequency, được xác định trong 10% đầu tiên của denoising schedule."

Điều này giải thích tại sao inject camera signal vào middle/late layers (blocks 10-25 như chúng ta đã thử) là **sai thời điểm** — camera trajectory đã bị "khóa" từ trước khi tín hiệu đến.

### Insight từ thực nghiệm của chúng ta:
> "~10° rotation là DEFAULT motion từ text prompt 'camera pan right' — KHÔNG PHẢI từ adapter."

Bất kỳ phương pháp nào cho ra ~10° cần bị coi là **bằng baseline** — adapter không có tác dụng.

---

*Tài liệu này ghi lại toàn bộ hành trình thử nghiệm, bài học rút ra, và hướng đi đúng cho bài toán V2V Camera Trajectory Retargeting. Đây là tài liệu tham khảo quan trọng để tránh lặp lại sai lầm trong các thí nghiệm tiếp theo.*
