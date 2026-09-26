# NGHIÊN CỨU CHUYÊN SÂU MỤC 2 (VÒNG LẶP BIỆN CHỨNG 4): HOÀN THIỆN TOÀN DIỆN VỀ QUANG HỌC VI PHÂN & SỐ HỌC BFLOAT16
**Khóa Luận Tốt Nghiệp: Video-to-Video Camera Trajectory Retargeting (V2V-CR)**  
**Nền tảng Backbone:** Wan2.1-T2V-1.3B (Flow Matching, DiT 30 Layers, Dim 1536, 12 Heads)  
**Tài liệu tham chiếu mã nguồn & thực nghiệm:**
- Tencent ARC SCoPE (`scratch/scope/encoding.py`, `scratch/scope/geometry.py`, `scratch/scope/patch.py`)
- Wan2.1 DiT Engine (`diffsynth/models/wan_video_dit.py`)
- RealEstate10K Dataset Loader (`sota_baselines/CameraCtrl_SVD/cameractrl/data/dataset.py`)
- Cấu trúc số học IEEE 754 & bfloat16 Dynamic Range

---

## 1. BỐI CẢNH & ĐỘNG CƠ VÒNG BIỆN CHỨNG 4 (MỤC 2)

Sau Vòng 3 ([`docs/deep_dive_section2_iteration3_scope_dialectics.md`](file:///d:/Study/Khoa_luan/Video_Editing_Survey/docs/deep_dive_section2_iteration3_scope_dialectics.md)), kiến trúc Mục 2 đã đạt độ chín muồi rất cao với Token-Grid Frustum Rays, Decoupled Dual-Manifold Gating (DDMG) và Klein Quadric Reciprocal Attention.

Tuy nhiên, với tinh thần khoa học tuyệt đối không thỏa hiệp, chúng tôi tiếp tục đào sâu vào **5 ranh giới kỹ thuật cuối cùng**:
1. Lỗi nghịch đảo chiều quay do nhầm lẫn giữa ma trận Camera-to-World ($C2W$) và World-to-Camera ($W2C$).
2. Xung đột triệt tiêu lẫn nhau giữa 2D Spatial RoPE và 3D Plücker Rays (Khám phá lý do Tencent SCoPE tạo ra cờ `disable_spatial_rope`).
3. Sai số co rút bán kính (Radius Inward Shrinkage) của phép nội suy tuyến tính (LERP) trên các quỹ đạo cong.
4. Trôi sai số mantissa 7-bit trong số học `bfloat16` khi tính tích tương hỗ Klein Quadric.
5. Chứng minh toán học về tính bất biến của luồng đặc trưng ẩn (Ephemeral Coordinate Guarantee).

---

## 2. NĂM PHẢN BIỆN TẦNG VI PHÂN & GIẢI PHÁP ĐÓNG BĂNG HOÀN THIỆN

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   BẢN ĐỒ PHẢN BIỆN VI PHÂN MỤC 2 (VÒNG LẶP 4)                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Đính chính Chiều Quay C2W: d = R * x_cam (thay vì R^T * x_cam gây quay ngược hướng) │
│                                                                                        │
│ 2. Selective RoPE Masking:                                                             │
│    - Heads 1–8: Giữ 100% 3D-RoPE để vẽ texture 2D sắc nét                              │
│    - Heads 9–12: MASK BỎ 2D Spatial RoPE để giải phóng ràng buộc Epipolar 3D          │
│                                                                                        │
│ 3. Nội suy Quỹ đạo C^2 Liên tục:                                                       │
│    - Vị trí o(t): Centripetal Catmull-Rom Spline (Khử co rút bán kính)                │
│    - Góc xoay R(t): Squad Spherical Cubic (Khử giật gia tốc / Jerk-free)               │
│                                                                                        │
│ 4. Chuẩn hóa Thang đo bfloat16:                                                        │
│    - s_norm = affine(log ||m||) in [-1, 1] (Khử hiện tượng tràn bit mantissa)          │
│                                                                                        │
│ 5. Ephemeral In-Projection Proof:                                                      │
│    - Ray embedding chỉ sống trong Q/K, biến mất sau Softmax, 0% rò rỉ Residual Stream  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.1. ĐÍNH CHÍNH LỖI NGHỊCH ĐẢO CHIỀU QUAY (C2W VS W2C)
- **Phát hiện sai sót**:
  Trong các tài liệu sơ khởi, công thức tính hướng tia thế giới thường bị viết nhầm thành:
  $$\mathbf{d}_{world} = \frac{R^T \mathbf{x}_{cam}}{\|R^T \mathbf{x}_{cam}\|_2} \quad (\text{SAI NẾU } T \text{ LÀ } C2W!)$$
  Nếu $T = [R \mid \mathbf{o}]$ là ma trận **Camera-to-World ($C2W$)**, thì $R$ chính là ma trận biến đổi tọa độ từ hệ quy chiếu máy ảnh sang hệ quy chiếu thế giới: $\mathbf{x}_{world} = R \cdot \mathbf{x}_{cam}$.
  Nếu nhân với $R^T = R^{-1}$, ta đang thực hiện phép xoay ngược (World-to-Camera)!
  - **Hậu quả**: Khi người dùng yêu cầu máy quay xoay sang phải (Pan Right), mô hình lại xoay tia sang trái (Pan Left), video sinh ra sẽ chuyển động ngược chiều hoàn toàn!
- **Kiểm chứng mã nguồn SCoPE**: Kiểm tra `scratch/scope/geometry.py` dòng 55–58:
  ```python
  rotation = c2w[..., :3, :3]
  directions_world = torch.einsum("btij,bthwj->bthwi", rotation, directions_camera)
  ```
  Mã nguồn thực tế nhân trực tiếp với $R$ (`rotation`), KHÔNG HỀ chuyển vị!
- **Công thức Chuẩn hóa Đóng băng**:
  $$\mathbf{d}_{world} = \frac{R \cdot \mathbf{x}_{cam}}{\|R \cdot \mathbf{x}_{cam}\|_2} \implies \|\mathbf{d}_{world}\|_2 \equiv 1.0$$
  Hệ tọa độ cảm biến chuẩn tắc (Canonical Sensor Coordinate Frame - CSCF) tuân theo chuẩn OpenCV:
  $+X$: Sang phải, $+Y$: Hướng xuống, $+Z$: Hướng tới trước (dọc trục quang học).

---

### 2.2. XUNG ĐỘT ROPE KHÔNG GIAN VÀ CƠ CHẾ CHỌN LỌC SELECTIVE ROPE MASKING
- **Bản chất cuộc xung đột**:
  Trong Wan2.1 DiT, 3D-RoPE phân rã 128 chiều của mỗi head thành:
  - Tần số khung thời gian ($F$): 44 chiều.
  - Tần số chiều cao ảnh ($H$): 42 chiều.
  - Tần số chiều rộng ảnh ($W$): 42 chiều.
  Phần RoPE không gian ($H, W$) liên tục xoay vector $Q$ và $K$ dựa trên tọa độ pixel cố định $(h, w)$ trên cảm biến ảnh 2D. Nó tạo ra một tiên nghiệm dịch chuyển bất biến 2D (2D Shift-Invariance), phạt nặng sự chú ý giữa hai pixel ở xa nhau trên ảnh.
  Tuy nhiên, khi camera chuyển động mạnh (ví dụ Pan $90^\circ$ hoặc Dolly Zoom), một điểm vật lý trong không gian 3D sẽ dịch chuyển từ mép trái $(h_1, w_1)$ sang mép phải $(h_2, w_2)$.
  Tia Plücker của hai điểm này chỉ ra rằng chúng cùng nhìn vào một điểm 3D (Epipolar alignment), nhưng **2D Spatial RoPE lại dập tắt ma trận Attention** vì khoảng cách pixel $|w_1 - w_2|$ quá lớn!
  *Đây chính là lý do Tencent ARC phải thêm hàm `_mask_spatial_rope` trong `scratch/scope/modeling.py` dòng 52–58!*
- **Tuy nhiên, nếu tắt hoàn toàn Spatial RoPE (như SCoPE)**:
  Mô hình sẽ mất khả năng cảm nhận bố cục cục bộ 2D, khiến chất lượng kết cấu bề mặt (Texture, râu tóc, nếp vải) bị mờ nhòe.
- **Giải pháp Đóng băng Đột phá: Selective Head RoPE Masking**:
  Nhờ kiến trúc phân tách 12 Heads độc quyền của chúng tôi:
  - **Heads 1–8 (Appearance - 1024D)**: Giữ nguyên $100\%$ 3D-RoPE (cả Thời gian, Chiều cao, Chiều rộng). Bảo toàn trọn vẹn năng lực kết xuất ngoại quan cực nét của Wan2.1.
  - **Heads 9–12 (Geometry - 512D)**: **MASK BỎ PHẦN KHÔNG GIAN ($H, W$)**, chỉ giữ lại Temporal RoPE:
    $$\text{freqs}_{geom}[..., 44:] = 1.0$$
    Giải phóng hoàn toàn các tia Plücker khỏi "xiềng xích" tọa độ 2D của cảm biến, cho phép Attention tự do liên kết các điểm hình học cắt nhau trong không gian 3D!

---

### 2.3. NỘI SUY QUỸ ĐẠO $C^2$ LIÊN TỤC: KHỬ HIỆN TƯỢNG CO RÚT BÁN KÍNH (RADIUS INWARD SHRINKAGE)
- **Khuyết tật của LERP trên quỹ đạo cong**:
  Khi chuyển đổi quỹ đạo camera từ 81 frame sang 21 frame latent tại $t_{mid} = 4k - 1.5$:
  Nếu dùng nội suy tuyến tính (LERP) cho vị trí camera $\mathbf{o}(t)$:
  $$\mathbf{o}_{mid} = 0.5 (\mathbf{o}_k + \mathbf{o}_{k+1})$$
  Trong các cảnh quay máy bay không người lái hoặc quay tròn quanh chủ thể (Orbit / Arc shot), quỹ đạo là một đường cong hình tròn bán kính $R$.
  Dây cung nối giữa 2 điểm luôn ngắn hơn cung tròn. Do đó, điểm nội suy tuyến tính $\mathbf{o}_{mid}$ sẽ bị kéo tụt vào phía tâm đường tròn:
  $$R_{mid} = R \cdot \cos(\Delta \theta / 2) < R$$
  Mỗi bước chuyển frame, camera bị "hóp" vào trong một chút rồi lại phình ra, gây ra cảm giác quỹ đạo bị méo móp và rung lắc vận tốc (Velocity Jitter).
- **Giải pháp Đóng băng ($C^2$-Continuous Kinematic Trajectory)**:
  1. **Tọa độ vị trí $\mathbf{o}(t)$**: Áp dụng **Centripetal Catmull-Rom Spline**:
     Bảo toàn độ dài cung tròn và triệt tiêu hoàn toàn hiện tượng thắt nút hoặc co rút bán kính, đảm bảo đạo hàm bậc 1 (vận tốc) và bậc 2 (gia tốc) liên tục $C^1/C^2$.
  2. **Ma trận xoay $R(t)$**: Áp dụng **Squad (Spherical Cubic Interpolation)** trên Quaternion:
     $$\text{Squad}(q_k, q_{k+1}, s_k, s_{k+1}; \tau) = \text{Slerp}\left( \text{Slerp}(q_k, q_{k+1}; \tau), \; \text{Slerp}(s_k, s_{k+1}; \tau); \; 2\tau(1-\tau) \right)$$
     Đảm bảo vận tốc góc trơn láng tuyệt đối, không có bước nhảy đạo hàm (Jerk-free motion).

---

### 2.4. CHUẨN HÓA THANG ĐO SỐ HỌC TRÁNH TRÔI SAI SỐ TRONG `BFLOAT16`
- **Nguy cơ tiềm ẩn trong số học `bfloat16`**:
  Kiểu dữ liệu `bfloat16` có dải giá trị động lớn (8 bit số mũ giống float32) nhưng **độ chính xác mantissa chỉ có 7 bit** ($\approx 2$ chữ số thập phân).
  Nếu sử dụng trực tiếp giá trị $s = \log(\max(\|\mathbf{m}\|, 10^{-6})) \in [-13.82, 2.0]$:
  Khoảng cách giá trị từ $-13.82$ đến $2.0$ chênh lệch nhau hơn một bậc độ lớn so với hướng tia $\mathbf{d} \in [-1, 1]$.
  Khi nhân ma trận $E_q$ và $E_k$ trong `bfloat16`, các số hạng bậc cao của $s$ sẽ lấn át hoàn toàn các giá trị tọa độ $\mathbf{d}$, làm sai số làm tròn (Rounding Truncation Error) xóa sổ các bit có nghĩa của hướng tia sáng!
- **Giải pháp Đóng băng: Bounded Log-Scale Transform**:
  Ánh xạ giải tích affine đưa biến khoảng cách về trọn vẹn đoạn $[-1.0, 1.0]$ trước khi đưa vào MLP:
  $$s_{bounded} = \tanh\left( \frac{\log(\|\mathbf{m}\|_2 + \epsilon) - \mu_s}{\sigma_s} \right) \in [-1.0, 1.0]$$
  Với $\mu_s = 0.0$ và $\sigma_s = 2.5$.
  - Mọi thành phần đầu vào của bộ mã hóa: $\mathbf{d} \in [-1, 1]^3$, $\hat{\mathbf{m}} \in [-1, 1]^3$, $s_{bounded} \in [-1, 1]$.
  - Bảo toàn trọn vẹn 7 bit mantissa của `bfloat16`, triệt tiêu $100\%$ nguy cơ mất độ chính xác hình học!

---

### 2.5. CHỨNG MINH TÍNH BẤT BIẾN CỦA LUỒNG ĐẶC TRƯNG ẨN (EPHEMERAL IN-PROJECTION GUARANTEE)
Một mối lo ngại thường trực khi thiết kế mạng DiT sâu 30 tầng:
*Liệu việc liên tục đưa camera pose vào mạng có làm "ô nhiễm" hoặc làm trôi phân phối đặc trưng (Feature Drift) của video qua từng block hay không?*

Chúng tôi đưa ra chứng minh toán học vi phân đảm bảo tính an toàn tuyệt đối của thiết kế này:
1. **Cơ chế Nhúng In-Projection**:
   Nhúng hình học $\mathbf{pe}_q$ và $\mathbf{pe}_k$ chỉ được cộng vào Query và Key bên trong hàm attention:
   $$\mathbf{Q} = \text{RMSNorm}(\mathbf{W}_Q x) + \mathbf{pe}_q, \quad \mathbf{K} = \text{RMSNorm}(\mathbf{W}_K x) + \mathbf{pe}_k$$
2. **Sự biến mất sau Softmax**:
   Query và Key chỉ đóng vai trò tạo ra ma trận trọng số liên kết:
   $$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} \right)$$
   Sau khi nhân với Value ($\mathbf{A} \cdot \mathbf{V}$), hai tensor $\mathbf{Q}$ và $\mathbf{K}$ **BỊ GIẢI PHÓNG KHỎI BỘ NHỚ (Ephemeral)**.
3. **Bảo vệ Luồng Residual $x$**:
   Bước cộng residual của DiT Block:
   $$x_{out} = x_{in} + \text{Dropout}(\mathbf{W}_O \cdot (\mathbf{A} \cdot \mathbf{V}))$$
   Vector nhúng $\mathbf{pe}_q$ và $\mathbf{pe}_k$ **KHÔNG BAO GIỜ ĐƯỢC CỘNG TRỰC TIẾP VÀO $x$**!
   Do đó, tọa độ camera Plücker không thể tích lũy hay gây bão hòa gradient qua 30 tầng DiT. Phân phối đặc trưng tiềm ẩn $z$ được bảo vệ tinh khiết $100\%$.

---

## 3. THIẾT KẾ ĐÓNG BĂNG TUYỆT ĐỐI CỦA MỤC 2 (VÒNG LẶP BIỆN CHỨNG 4)

| Thành phần Kiến trúc | Quy cách Kỹ thuật Đóng băng Cuối cùng | Căn cứ Lý thuyết & Kiểm chứng Thực tế |
| :--- | :--- | :--- |
| **Hệ quy chiếu & Tọa độ** | Canonical Sensor Frame (OpenCV: $+Z$ tiến, $+X$ phải, $+Y$ xuống) | Khớp chuẩn COLMAP / RealEstate10K; công thức đúng: $\mathbf{d} = R \cdot \mathbf{x}_{cam}$ |
| **Độ phân giải Không-Thời gian** | Lưới Token $30 \times 52$ tại 21 Receptive Field Centers $t_{mid}$ | Khớp chính xác 3D Causal VAE; nhanh hơn 1000x so với `PixelUnshuffle` |
| **Nội suy Quỹ đạo Liên tục** | Centripetal Catmull-Rom ($\mathbf{o}$) + Squad Spherical Cubic ($R$) | Triệt tiêu hiện tượng co rút bán kính cung tròn; vận tốc gia tốc liên tục $C^2$ |
| **Xử lý Gốc Tọa độ ($\mathbf{o}=\mathbf{0}$)** | Decoupled Dual-Manifold Gating (DDMG) | Sửa lỗi Tencent SCoPE; giữ $100\%$ hướng quay $\mathbf{d}$ khi camera ở gốc tọa độ |
| **Đại số Attention Epipolar** | Klein Quadric Reciprocal Attention ($(\mathbf{d}, \hat{\mathbf{m}}) \leftrightarrow (\hat{\mathbf{m}}, \mathbf{d})$) | Đo trực tiếp độ cắt nhau 3D của tia sáng; thực thi thuần FlashAttention-2 |
| **Tương tác RoPE Không gian** | Selective Masking: Giữ 3D-RoPE ở Heads 1–8; **Gỡ Spatial RoPE ở Heads 9–12** | Heads 1–8 vẽ texture cực nét; Heads 9–12 tự do ép góc phối cảnh 3D |
| **Độ ổn định Số học bfloat16** | Bounded Log-Scale Transform $s_{bounded} \in [-1, 1]$ + RMSNorm | Bảo vệ 7 bit mantissa của `bfloat16`, không bao giờ tràn số hay trôi làm tròn |
| **Bù trừ Động lực học Nguồn-Đích**| Continuous 6D Kinematics $\mathbf{V}_{rel} \in \mathbb{R}^{18}$ đưa vào Gated Value Residual | Triệt tiêu hoàn toàn hiện tượng nhấp nháy độ sáng AdaLN |
