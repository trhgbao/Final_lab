# TÀI LIỆU ĐẶC TẢ KHUNG ĐÁNH GIÁ ĐỊNH LƯỢNG CHẤT LƯỢNG VIDEO
# COMPREHENSIVE SOTA VIDEO-TO-VIDEO EVALUATION FRAMEWORK
## Công Thức Toán Học, Ngưỡng Đo Lường & Phương Pháp Chẩn Đoán Khuyết Điểm Mô Hình

---

## 1. TỔNG QUAN HỆ THỐNG ĐO LƯỜNG ĐỊNH LƯỢNG

Để đánh giá một cách toàn diện và khách quan theo tiêu chuẩn các hội nghị hàng đầu (*CVPR, ICCV, ECCV, AAAI*), hệ thống sử dụng **Bộ 10 chỉ số đo lường phân bố trên 5 Trụ cột cốt lõi**:

```text
                                [ KHUNG ĐÁNH GIÁ 5 CHIỀU ]
                                            │
        ┌──────────────┬────────────────────┼───────────────────┬──────────────┐
        ▼              ▼                    ▼                   ▼              ▼
  [1. Cấu Trúc]  [2. Ngữ Nghĩa]      [3. Thời Gian]      [4. Động Lực]   [5. Điểm Tổng]
    • PSNR         • CLIP-I Cosine     • E_warp            • Flow Cosine   • CBI Score
    • SSIM         • Perceptual L2     • Flicker Index     • Motion Ratio    (Thang 100)
    • Color ΔE                         • Temporal SSIM
```

---

## 2. CHI TIẾT TOÁN HỌC & CÔNG THỨC CÁC CHỈ SỐ

### 2.1. Nhóm 1: Độ Sắc Nét, Cấu Trúc & Màu Sắc (Reconstruction & Color Fidelity)

#### 1. Peak Signal-to-Noise Ratio (PSNR)
* **Ý nghĩa**: Đo tỷ số tín hiệu cực đại trên nhiễu ở mức độ từng điểm ảnh (Pixel-level Fidelity).
* **Công thức toán học**:
  $$\text{MSE} = \frac{1}{3HW} \sum_{c=1}^3 \sum_{x=1}^W \sum_{y=1}^H (I_{src}(x,y,c) - I_{tgt}(x,y,c))^2$$
  $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10}(255) - 10 \cdot \log_{10}(\text{MSE}) \quad (\text{dB})$$
* **Ngưỡng đánh giá**:
  * $\text{PSNR} > 22.0\text{ dB}$: Tái tạo chi tiết sắc nét, không bị nhòe.
  * $\text{PSNR} < 15.0\text{ dB}$: Khung hình bị mờ hoặc sai lệch lớn về cường độ sáng.

---

#### 2. Structural Similarity Index Measure (SSIM)
* **Ý nghĩa**: Đánh giá sự tương đồng về cấu trúc không gian, độ tương phản và độ sáng dựa trên thị giác con người.
* **Công thức toán học**:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  * Trong đó: $\mu_x, \mu_y$ là giá trị trung bình cục bộ; $\sigma_x^2, \sigma_y^2$ là phương sai; $\sigma_{xy}$ là hiệp phương sai; $C_1 = (0.01 \times 255)^2, C_2 = (0.03 \times 255)^2$.
* **Ngưỡng đánh giá**:
  * $\text{SSIM} > 0.65$: Bề mặt vật thể (texture) và đường biên (edge) được bảo toàn tốt.
  * $\text{SSIM} < 0.50$: Cấu trúc vật thể bị biến dạng hoặc mất chi tiết vi mô.

---

#### 3. Color Drift ($\Delta E_{\text{CIELAB}}$)
* **Ý nghĩa**: Đo độ biến đổi màu sắc thực tế trong không gian màu đồng đều nhận thức CIELAB ($L^*a^*b^*$).
* **Công thức toán học**:
  $$\Delta E = \frac{1}{HW} \sum_{x,y} \sqrt{(L_{src}^* - L_{tgt}^*)^2 + (a_{src}^* - a_{tgt}^*)^2 + (b_{src}^* - b_{tgt}^*)^2}$$
* **Ngưỡng đánh giá**:
  * $\Delta E < 15.0$: Giữ nguyên tông màu gốc (Color consistency cao).
  * $\Delta E > 40.0$: Hiện tượng trôi màu hoặc lệch tone nghiêm trọng.

---

### 2.2. Nhóm 2: Nhận Diện & Ngữ Nghĩa Chủ Thể (Semantic Preservation)

#### 4. CLIP-I Visual Cosine Similarity
* **Ý nghĩa**: Sử dụng không gian nhúng của `CLIP-ViT/14` để đo mức độ đồng nhất về mặt ngữ nghĩa giữa khung hình gốc và mới.
* **Công thức toán học**:
  $$\mathbf{e}_t^{src} = \frac{\mathcal{E}_{clip}(f_t^{src})}{\|\mathcal{E}_{clip}(f_t^{src})\|_2}, \quad \mathbf{e}_t^{tgt} = \frac{\mathcal{E}_{clip}(\hat{f}_t^{tgt})}{\|\mathcal{E}_{clip}(\hat{f}_t^{tgt})\|_2}$$
  $$S_{clip} = \frac{1}{F} \sum_{t=0}^{F-1} \langle \mathbf{e}_t^{src}, \mathbf{e}_t^{tgt} \rangle$$
* **Ngưỡng đánh giá**:
  * $S_{clip} > 0.85$: Đối tượng giữ nguyên hình dạng nhận diện (không bị biến thành vật thể khác).

---

#### 5. Perceptual Feature $L_2$ Distance
* **Công thức toán học**:
  $$D_{perceptual} = \frac{1}{F} \sum_{t=0}^{F-1} \|\mathbf{e}_t^{src} - \mathbf{e}_t^{tgt}\|_2$$
* **Ngưỡng đánh giá**: $D_{perceptual} < 0.35$ (càng nhỏ càng tốt).

---

### 2.3. Nhóm 3: Tính Nhất Quán & Ổn Định Thời Gian (Temporal Coherence)

#### 6. Optical Flow Warping Error ($E_{warp}$)
* **Ý nghĩa**: Đo mức độ nhấp nháy, giật cục (flickering) giữa các khung hình liên tiếp bằng cách chiếu ngược khung hình sau về khung hình trước theo luồng quang học $\mathcal{W}(\cdot)$.
* **Công thức toán học**:
  $$\mathbf{u}_t = \text{FarnebackFlow}(\hat{f}_t, \hat{f}_{t+1})$$
  $$E_{warp} = \frac{1}{F-1} \sum_{t=0}^{F-2} \frac{\sum_{x,y} M_t(x,y) \cdot |\hat{f}_{t+1}(x,y) - \text{Warp}(\hat{f}_t, \mathbf{u}_t)(x,y)|}{\sum_{x,y} M_t(x,y)}$$
  *(với $M_t$ là mặt nạ loại trừ vùng che khuất occlusion).*
* **Ngưỡng đánh giá**:
  * $E_{warp} < 7.0$: Chuyển động cực kỳ mượt, không giật.
  * $E_{warp} > 11.0$: Video xuất hiện hiện tượng nhấp nháy thấy rõ bằng mắt thường.

---

#### 7. Temporal Brightness Flicker Index
* **Công thức toán học**:
  $$\text{Flicker} = \frac{1}{F-1} \sum_{t=0}^{F-2} \text{StdDev}_{x,y}\left( \text{Gray}(\hat{f}_{t+1}) - \text{Gray}(\hat{f}_t) \right)$$
* **Ngưỡng đánh giá**: $\text{Flicker} < 8.0$ (Mục tiêu chuẩn).

---

#### 8. Temporal SSIM ($T_{SSIM}$)
* **Công thức toán học**:
  $$T_{SSIM} = \frac{1}{F-1} \sum_{t=0}^{F-2} \text{SSIM}(\hat{f}_t, \hat{f}_{t+1})$$
* **Ngưỡng đánh giá**: $T_{SSIM} > 0.85$.

---

### 2.4. Nhóm 4: Bảo Toàn Động Lực Học Chuyển Động (Motion Dynamics Fidelity)

#### 9. Optical Flow Direction Cosine Similarity ($r_{flow}$)
* **Ý nghĩa**: Xác nhận xem hướng di chuyển của các đối tượng trong video mới có trùng với video gốc không (tránh hiện tượng camera quay mới làm đảo lộn hướng đi của vật thể).
* **Công thức toán học**:
  $$r_{flow} = \frac{1}{F-1} \sum_{t=0}^{F-2} \frac{1}{HW} \sum_{x,y} \cos(\theta_{src}(x,y,t) - \theta_{tgt}(x,y,t))$$
  * Trong đó $\theta = \text{atan2}(v_y, v_x)$ là góc của vector luồng quang học.
* **Ngưỡng đánh giá**:
  * $r_{flow} \rightarrow 1.0$: Chuyển động cùng hướng 100%.
  * $r_{flow} < 0$: Chuyển động bị giật ngược hoặc ngược hướng hoàn toàn.

---

#### 10. Motion Magnitude Ratio ($R_{mag}$)
* **Công thức toán học**:
  $$R_{mag} = \frac{\text{Mean}(\|\mathbf{u}_{tgt}\|) + \epsilon}{\text{Mean}(\|\mathbf{u}_{src}\|) + \epsilon}$$
* **Ngưỡng đánh giá**: $R_{mag} \approx 0.85 - 1.15\text{x}$ (Tốc độ chuyển động đồng đều).

---

### 2.5. Nhóm 5: Chỉ Số Điểm Tổng Hợp (Composite Benchmark Index - CBI)

Chỉ số tổng hợp chuẩn hóa trên thang điểm $100$:

$$\text{CBI} = 20 \cdot \min\left(\frac{\text{PSNR}}{30}, 1.0\right) + 20 \cdot \text{SSIM} + 20 \cdot S_{clip} + 20 \cdot \max\left(0, 1 - \frac{E_{warp}}{15}\right) + 20 \cdot \max\left(0, \frac{r_{flow} + 1}{2}\right)$$

---

## 3. BẢNG TRA CỨU CHẨN ĐOÁN KHUYẾT ĐIỂM HỆ THỐNG (DIAGNOSTIC MATRIX)

| Triệu chứng Định lượng | Khuyết điểm Thuật toán | Vị trí Cần Tối ưu hóa |
| :--- | :--- | :--- |
| $\text{PSNR} < 15\text{ dB}$, $\Delta E > 40$ | Nhiễu ngẫu nhiên phá vỡ kênh màu Latent. | Điều chỉnh dải `strength` ($0.75 \rightarrow 0.50$) hoặc khóa VAE Normalization. |
| $r_{flow} < 0.0$ | Quá trình Denoising không bám theo Optical Flow cũ. | Chuyển sang dùng Deterministic Latent Inversion ($z_T$) thay vì Gaussian noise. |
| $E_{warp} > 10.0$, $\text{Flicker} > 20$ | Thiếu liên kết đặc trưng thời gian giữa các frame. | Bổ sung Temporal Cross-Frame Smoothing hoặc MidBlock Spatial Injection. |
| $S_{clip} < 0.75$ | Đối tượng bị biến dạng sang hình thái khác. | Tăng trọng số cho vector điều kiện $f_0$ (`min_guidance_scale`). |
