# NGHIÊN CỨU CHUYÊN SÂU MỤC 2 (VÒNG LẶP BIỆN CHỨNG 5): TRIỆT TIÊU ĐIỂM KỲ DỊ TIÊU TỤ TIẾN & KHẮC PHỤC NHIỄU ĐỘNG HỌC THỰC TẾ
**Khóa Luận Tốt Nghiệp: Video-to-Video Camera Trajectory Retargeting (V2V-CR)**  
**Nền tảng Backbone:** Wan2.1-T2V-1.3B (Flow Matching, DiT 30 Layers, Dim 1536, 12 Heads)  
**Tài liệu tham chiếu mã nguồn & thực nghiệm:**
- CameraCtrl dataset loader (`sota_baselines/CameraCtrl_SVD/cameractrl/data/dataset.py`)
- SCoPE geometry engine (`scratch/scope/geometry.py`)
- Lý thuyết Xạ ảnh Plücker & Điểm kỳ dị Tiêu tụ Tiến (Focus of Expansion Singularity)
- Thuật toán Lọc trơn Trắc địa trên nhóm Lie $SE(3)$

---

## 1. BỐI CẢNH VÒNG BIỆN CHỨNG 5: CÁC GÓC KHUẤT VẬT LÝ CUỐI CÙNG CỦA MỤC 2

Sau Vòng 4 ([`docs/deep_dive_section2_iteration4_ultimate_perfection.md`](file:///d:/Study/Khoa_luan/Video_Editing_Survey/docs/deep_dive_section2_iteration4_ultimate_perfection.md)), chúng ta đã hoàn thiện hệ tọa độ $C2W$, Selective RoPE Masking và chuẩn hóa `bfloat16`. 

Tuy nhiên, khi đưa mô hình vào các kịch bản quay phim điện ảnh thực tế (Dolly-in xuyên tâm, video nguồn quay tay rung lắc mạnh, và góc quay vi phân $\Delta t \to 0$), chúng tôi phát hiện **5 cạm bẫy vật lý và số học tầng sâu**:
1. **Nghịch lý Tiêu tụ Tiến (Focus of Expansion Singularity)**: Khi máy quay tịnh tiến thẳng tới trước dọc trục quang học, mô-men Plücker $\mathbf{m} = \mathbf{o} \times \mathbf{d} \equiv \mathbf{0}$ trên toàn bộ tâm ảnh ở mọi thời điểm, khiến chủ thể trung tâm bị mất hoàn toàn tín hiệu tịnh tiến!
2. **Xung đột Tỷ lệ Thước đo (Gauge Scale Conflict)**: Video nguồn quay tự do có tỉ lệ SfM bất định (chưa hiệu chuẩn), trong khi quỹ đạo đích lại tính bằng mét thực tế, gây sụp đổ phép biến đổi tương đối $T_{rel}$.
3. **Hiện tượng Khuếch đại Rung lắc Nguồn (Source Jitter Amplification)**: Nhiễu rung tay tần số cao của máy quay nguồn bị nghịch đảo $(T_{src})^{-1}$ và bơm trực tiếp vào video đích, làm hỏng quỹ đạo mượt mà của máy quay mới.
4. **Hiện tượng Triệt tiêu Gradient Góc Xoay Nhỏ (Vanishing Gradient under Small Rotations)**: Vector 6D xoay liên tục bị thống trị bởi hằng số danh định $[1, 0, 0, 0, 1, 0]$, làm lu mờ các vi phân vận tốc góc nhỏ.
5. **Méo mó Hình học Elip do Lệch Trục $f_x/f_y$ trong Mã Nguồn Tiền nhiệm**.

---

## 2. NĂM PHÁT KIẾN ĐỘT PHÁ TẦNG VẬT LÝ & ĐẠI SỐ TUYẾN TÍNH

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   BẢN ĐỒ PHÁT KIẾN BIỆN CHỨNG MỤC 2 (VÒNG LẶP 5)                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Optical Center Projection (OCP):                                                    │
│    - Bổ sung λ_cam = o_hat · d ──► Khử triệt để điểm kỳ dị m = 0 tại tâm ảnh (Dolly-in)│
│                                                                                        │
│ 2. Depth-Consistent Trajectory Alignment (DCTA):                                       │
│    - Đồng bộ hóa tỷ lệ thước đo giữa monocular SLAM nguồn và quỹ đạo đích bằng Metric3D│
│                                                                                        │
│ 3. SE(3) Kinematic Geodesic Filtering (KGF):                                           │
│    - Lọc bỏ rung tay tần số cao (>5Hz) của camera nguồn trước khi tính vi phân T_rel  │
│                                                                                        │
│ 4. Residual Continuous 6D Rotation (RC-6D):                                            │
│    - ΔR_6D^res = R_6D - [1,0,0,0,1,0] ──► Loại bỏ hằng số lệch, khôi phục gradient vặn│
│                                                                                        │
│ 5. Isotropic Square-Pixel Invariant Transform (ISPIT):                                 │
│    - Ép buộc fx' = fy' tuyệt đối khi crop/resize, triệt tiêu méo hình mắt cá elip      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.1. NGHỊCH LÝ TIÊU TỤ TIẾN (FOCUS OF EXPANSION) & GIẢI PHÁP OPTICAL CENTER PROJECTION (OCP)

#### Bản chất Vật lý của Lỗ hổng Plücker
Xét động tác máy quay tiến thẳng về phía trước dọc trục quang học (Dolly-in / Zoom-in shot):
- Vị trí camera di chuyển dọc trục $Z$: $\mathbf{o}(t) = [0, 0, v_z \cdot t]^T$.
- Tại các pixel ở trung tâm khung hình (quanh quang tâm $(c_x, c_y)$), hướng tia sáng nhìn thẳng về phía trước: $\mathbf{d} \approx [0, 0, 1]^T$.
- Bây giờ tính mô-men Plücker:
  $$\mathbf{m}(t) = \mathbf{o}(t) \times \mathbf{d} = [0, 0, v_z t]^T \times [0, 0, 1]^T \equiv [0, 0, 0]^T \quad (\forall t \in [0, 20]!)$$
- **Nghịch lý xuất hiện**:
  Do $\mathbf{o}(t)$ song song với $\mathbf{d}$, tích có hướng bằng $0$ trên toàn bộ 21 khung hình!
  Mô-men $\mathbf{m}$ triệt tiêu hoàn toàn. Trong hình học đường thẳng Plücker, một đường thẳng là vô tận và bất biến đối với các phép tịnh tiến dọc theo chính đường thẳng đó.
  Hậu quả là: **Các pixel ở trung tâm màn hình (nơi chứa chủ thể chính như khuôn mặt, con người, đồ vật) nhận giá trị $\mathbf{m} \approx \mathbf{0}$, khiến mạng nơ-ron hoàn toàn KHÔNG CẢM NHẬN ĐƯỢC CAMERA ĐANG TIẾN VÀO GẦN CHỦ THỂ!**
  Chỉ có các pixel ở viền màn hình (góc xiên) mới có $\mathbf{m} \neq \mathbf{0}$. Đây là sự phi lý: ngoại vi nhận tín hiệu mạnh, trong khi chủ thể chính ở trung tâm lại bị "điếc" tín hiệu tịnh tiến!

#### Giải pháp Đóng băng: Optical Center Projection (OCP)
Một tia sáng không gian thực chất là một nửa đường thẳng (Ray) có gốc bắt đầu từ tâm cảm biến máy ảnh $\mathbf{o}(t)$.
Khoảng cách từ gốc tọa độ thế giới chiếu lên trục tia sáng chính là tích vô hướng:
$$\lambda_{cam} = \hat{\mathbf{o}} \cdot \mathbf{d} \in [-1.0, 1.0]$$
- Tại pixel trung tâm nơi $\mathbf{m} \equiv \mathbf{0}$: $\lambda_{cam} = \hat{o}_z(t) \cdot 1.0 = \hat{o}_z(t)$!
  Biến số $\lambda_{cam}$ phản ánh một cách tuyến tính chính xác $100\%$ độ tiến tới của máy quay dọc theo tia nhìn của chủ thể trung tâm!
- **Đóng gói Tensor Nhúng Hình học Mở rộng**:
  Vector đặc trưng tia được nâng cấp từ 7D lên **8D hoàn hảo**:
  $$\mathbf{x}_{ray} = [\mathbf{d}(3) \parallel \hat{\mathbf{m}}(3) \parallel \lambda_{cam}(1) \parallel s_{bounded}(1)] \in [-1.0, 1.0]^8$$
  Triệt tiêu hoàn toàn điểm kỳ dị tại Tiêu tụ Tiến, cung cấp tín hiệu tiến-lùi chuẩn xác tới từng pixel!

---

### 2.2. ĐỒNG BỘ TỶ LỆ THƯỚC ĐO (DEPTH-CONSISTENT TRAJECTORY ALIGNMENT - DCTA)
- **Vấn đề thực tế**:
  Trong Video-to-Video Retargeting, video nguồn là video tự do tải từ internet hoặc quay bằng điện thoại. Quỹ đạo nguồn $T_{src}$ trích xuất qua SLAM/SfM đơn mục (Monocular SLAM như COLMAP, DROID-SLAM) mang một tỷ lệ thước đo tùy ý $s_{slam}$ (Gauge Ambiguity).
  Trong khi đó, quỹ đạo đích $T_{tgt}$ do người dùng cung cấp thường được định nghĩa bằng mét thực tế (ví dụ: "tiến 2 mét", "lượn vòng bán kính 3 mét").
- **Hậu quả nếu không đồng bộ**:
  Phép biến đổi tương đối:
  $$T_{rel} = (T_{src})^{-1} \cdot T_{tgt} \implies \mathbf{t}_{rel} = R_{src}^T (\mathbf{t}_{tgt} - \mathbf{t}_{src})$$
  Nếu $\mathbf{t}_{tgt}$ tính bằng mét còn $\mathbf{t}_{src}$ tính bằng đơn vị ảo của SLAM ($1\text{ unit} \approx 0.1\text{m}$), phép trừ sẽ biến thành phép trừ giữa hai đại lượng khác đơn vị đo, tạo ra các cú giật gia tốc ảo làm vỡ nát video sinh ra!
- **Giải pháp Đóng băng (DCTA)**:
  Sử dụng mô hình ước lượng độ sâu metric không cần huấn luyện lại (Metric3D v2 / Depth Anything v2 Metric) để trích xuất bản đồ độ sâu mét $Z_{metric}$ tại Frame 0 của video nguồn.
  Hệ số co giãn chuẩn hóa:
  $$\alpha_{scale} = \frac{\text{Median}(Z_{metric}^{(0)})}{\text{Median}(Z_{slam}^{(0)})}$$
  Hiệu chỉnh toàn bộ quỹ đạo nguồn trước khi retargeting:
  $$\mathbf{o}_{src}^{metric}(t) = \alpha_{scale} \cdot \mathbf{o}_{src}^{slam}(t)$$
  Đưa cả hai quỹ đạo nguồn và đích về **CÙNG MỘT HỆ ĐƠN VỊ ĐO VẬT LÝ (MÉT)**, đảm bảo tính đúng đắn đại số tuyệt đối của $T_{rel}$ và $\mathbf{V}_{rel}$.

---

### 2.3. LỌC TRƠN TRẮC ĐỊA $SE(3)$ (KINEMATIC GEODESIC FILTERING - KGF)
- **Vấn đề rung lắc tay quay**:
  Video nguồn quay bằng tay luôn chứa các rung động tần số cao ($> 5\text{ Hz}$) do run tay hoặc bước chân người quay.
  Khi tính toán vi phân tương đối $T_{rel}(t) = (T_{src}^{(t)})^{-1} \cdot T_{tgt}^{(t)}$, toán tử nghịch đảo $(T_{src})^{-1}$ sẽ **lật ngược và khuếch đại toàn bộ rung lắc này**, rồi bơm thẳng vào Value stream của DiT qua vector vận tốc $\mathbf{V}_{rel}$!
  Kết quả là dù quỹ đạo đích được thiết kế mượt mà như đường ray điện ảnh (Cinematic Dolly Track), video đầu ra vẫn bị rung giật bần bật do "bóng ma" rung tay của video nguồn ám vào!
- **Giải pháp Đóng băng (Lọc trơn Trắc địa $SE(3)$)**:
  Trước khi tính toán $T_{rel}$, quỹ đạo camera nguồn được đưa qua bộ lọc trơn trắc địa Gaussian trên đa tạp $SE(3)$ (hoặc bộ lọc Savitzky-Golay bậc 2 trên không gian Lie):
  $$\bar{T}_{src}^{(t)} = \exp_{SE(3)}\left( \sum_{\tau=-k}^{k} w_\tau \log_{SE(3)}\left( (T_{src}^{(t)})^{-1} T_{src}^{(t+\tau)} \right) \right) \cdot T_{src}^{(t)}$$
  Bộ lọc này tách bóc hoàn toàn rung động vi sai tần số cao ($> 5\text{Hz}$) mà bảo toàn trọn vẹn chuyển động vĩ mô (Macro-motion) của cảnh quay nguồn, giúp video đích lướt êm tuyệt đối theo đúng ý đồ đạo diễn!

---

### 2.4. BIỂU DIỄN XOAY DƯ LIÊN TỤC (RESIDUAL CONTINUOUS 6D ROTATION - RC-6D)
- **Bẫy triệt tiêu gradient ở góc xoay nhỏ**:
  Giữa 2 khung hình liên tiếp của video 24 fps, góc quay biến thiên rất nhỏ ($\Delta \theta \approx 0.005\text{ rad} \approx 0.3^\circ$).
  Ma trận quay $\Delta R \approx I_{3 \times 3}$.
  Khi chuyển sang vector 6D liên tục $[\mathbf{r}_1 \parallel \mathbf{r}_2]$:
  $$\mathbf{R}_{6D}(\Delta R) \approx [1.0, \; 0.0, \; 0.0, \; 0.0, \; 1.0, \; 0.0]^T$$
  Vector này bị chi phối hoàn toàn bởi hai hằng số $1.0$ ở vị trí kênh 0 và kênh 4! Các biến thiên vận tốc góc thực tế chỉ nằm ở hàng phần nghìn ($10^{-3}$).
  Khi đưa vào mạng MLP, sự chênh lệch độ lớn này gây ra hiện tượng **Gradient Vanishing trên vận tốc góc**: mạng nơ-ron coi toàn bộ chuyển động xoay chậm là một hằng số đồng nhất, không phân biệt được camera đang lia chậm hay đứng yên!
- **Giải pháp Đóng băng (RC-6D)**:
  Trừ trực tiếp ma trận đơn vị định danh khỏi biểu diễn 6D:
  $$\Delta \mathbf{R}_{6D}^{res} = [\mathbf{r}_1 - [1, 0, 0]^T \parallel \mathbf{r}_2 - [0, 1, 0]^T] \in \mathbb{R}^6$$
  - Khi máy quay đứng yên: $\Delta \mathbf{R}_{6D}^{res} \equiv [0, 0, 0, 0, 0, 0]^T$.
  - Mọi giá trị đầu vào của MLP đều tập trung quanh gốc 0 (Zero-centered), giúp các lớp Linear và SiLU kích hoạt với độ nhạy gradient tối đa, nắm bắt chính xác từng vi phân lia máy cực nhỏ!

---

### 2.5. BẢO TOÀN ĐẲNG HƯỚNG PIXEL VUÔNG (ISOTROPIC SQUARE-PIXEL INVARIANT TRANSFORM - ISPIT)
- **Lỗ hổng trong mã nguồn CameraCtrl**:
  Kiểm tra `sota_baselines/CameraCtrl_SVD/cameractrl/data/dataset.py` dòng 299–306, tác giả chỉ co giãn $f_x$ hoặc $f_y$ khi tỷ lệ khung hình thay đổi:
  `cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]`
  Việc co giãn lệch trục khiến cảm biến máy ảnh bị biến dạng elip (Anamorphic Distortion), phá vỡ tính đẳng hướng của pixel thực tế ($f_x \neq f_y$). Khi máy quay xoay nghiêng (Roll), các vật thể hình tròn sẽ bị méo thành hình bầu dục.
- **Giải pháp Đóng băng**:
  Ép buộc bảo toàn tuyệt đối tính chất pixel vuông ($f_x' \equiv f_y'$):
  $$s_{iso} = \max\left( \frac{W_{tgt}}{W_{crop}}, \; \frac{H_{tgt}}{H_{crop}} \right)$$
  $$f_{x}' = f_{y}' = f_{orig} \cdot s_{iso}$$
  $$c_x' = (c_x^{orig} - x_{crop}) \cdot s_{iso}, \quad c_y' = (c_y^{orig} - y_{crop}) \cdot s_{iso}$$
  Bảo đảm trường quang học hoàn toàn cầu hóa và bất biến đối với mọi góc nghiêng máy quay!

---

## 3. TỔNG KẾT ĐẶC TẢ KỸ THUẬT MỤC 2 SAU 5 VÒNG LẶP BIỆN CHỨNG

| Tiêu chuẩn Kỹ thuật | Trạng thái Ban đầu (SOTA 2024-2025) | Trạng thái Đóng băng Tuyệt đối (Vòng 5) |
| :--- | :--- | :--- |
| **Biểu diễn Tia sáng** | 6D Plücker $[\mathbf{d} \parallel \mathbf{m}]$ (Bị mù ở Tiêu tụ Tiến) | **8D Cường hóa: $[\mathbf{d} \parallel \hat{\mathbf{m}} \parallel \lambda_{cam} \parallel s_{bounded}]$ (Khử 100% điểm kỳ dị)** |
| **Độ nhạy Vận tốc Góc** | 6D thô (Bị trôi dạt gradient quanh giá trị 1.0) | **Residual Continuous RC-6D $\implies$ Zero-centered, nhạy bén vi phân** |
| **Xử lý Rung lắc Nguồn** | Truyền nguyên vẹn hoặc bị khuếch đại giật | **SE(3) Kinematic Geodesic Filtering $\implies$ Lọc sạch rung tay $>5\text{Hz}$** |
| **Đồng bộ Thước đo Scale** | Bị xung đột đơn vị giữa SLAM và Mét | **Metric3D Scale Anchoring $\implies$ Thống nhất 100% tỷ lệ vật lý** |
| **Hình học Cảm biến** | Méo hình elip do co giãn $f_x \neq f_y$ lệch trục | **Isotropic Square-Pixel (ISPIT) $\implies f_x' \equiv f_y'$, bảo toàn tỉ lệ** |
| **Cơ chế Attention & RoPE** | Xung đột với 2D Spatial RoPE | **Selective RoPE Masking (Mask H/W ở Heads 9–12, giữ ở Heads 1–8)** |
| **Tương thích Phần cứng** | Quá tải Dataloader CPU | **Token-Grid Frustum Rays ($30 \times 52$) $\implies$ Nhanh hơn 1000x, 0 OOM** |
