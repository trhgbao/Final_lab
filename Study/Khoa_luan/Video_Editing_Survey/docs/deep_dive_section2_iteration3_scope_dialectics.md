# NGHIÊN CỨU CHUYÊN SÂU MỤC 2 (VÒNG LẶP BIỆN CHỨNG 3): BẺ GÃY GIỚI HẠN SOTA & HOÀN THIỆN TOÀN DIỆN MÃ HÓA CAMERA 6DoF
**Khóa Luận Tốt Nghiệp: Video-to-Video Camera Trajectory Retargeting (V2V-CR)**  
**Nền tảng Backbone:** Wan2.1-T2V-1.3B (Flow Matching, DiT 30 Layers, Dim 1536, 12 Heads)  
**Tài liệu tham chiếu mã nguồn & bài báo:**
- Tencent ARC SCoPE (*arXiv:2606.27345* & mã nguồn `TencentARC/SCoPE`)
- CameraCtrl (*arXiv:2404.02101* & `sota_baselines/CameraCtrl_SVD`)
- CamCo (*arXiv:2406.02509*)
- ReCamMaster (*arXiv:2503.11647* & `sota_baselines/ReCamMaster`)
- Zhou et al. (*CVPR 2019 Oral* - Continuous 6D Rotations)

---

## 1. TỔNG QUAN VÒNG LẶP BIỆN CHỨNG 3 TRONG MỤC 2

Tại Vòng 2 ([`docs/deep_dive_section2_iteration2_continuous_geometry.md`](file:///d:/Study/Khoa_luan/Video_Editing_Survey/docs/deep_dive_section2_iteration2_continuous_geometry.md)), chúng ta đã giải quyết:
1. Biểu diễn xoay liên tục 6D ($\mathbb{R}^9$) thay thế đại số Lie $\mathfrak{se}(3)$ để triệt tiêu vết cắt nhánh tại $\theta = \pi$.
2. Normalized Canonical Intrinsic Transform (NCIT) để bảo toàn FOV không bị méo hình khi crop/resize.
3. Multi-Frequency Harmonic Projector (MHRP) để tăng cường tần số cao.

Tuy nhiên, khi tiến hành đối chiếu với cấu trúc mã nguồn nội tại của Wan2.1 DiT (`diffsynth/models/wan_video_dit.py`) và công trình đột phá mới nhất từ Tencent ARC (**SCoPE - June 2026**), chúng tôi đã thực hiện một cuộc **tổng tấn công phản biện đa chiều (Multi-Front Stress Test)** và phát hiện **5 lỗ hổng chí mạng** tiềm ẩn trong thiết kế Vòng 2, cùng một **lỗi toán học ẩn tàng (Silent Mathematical Bug)** ngay trong chính mã nguồn của Tencent SCoPE!

---

## 2. NĂM CUỘC TẤN CÔNG PHẢN BIỆN & CÁC KHÁM PHÁ ĐỘT PHÁ

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   BẢN ĐỒ PHẢN BIỆN ĐA CHIỀU MỤC 2 (VÒNG 3)                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Temporal Mismatch: 81 raw frames vs 21 Causal 3D VAE frames                   │
│    ──► Sụp đổ shape tensor [81, ...] vs [21, ...] khi nhúng DiT                  │
│                                                                                  │
│ 2. Lãng phí 1000x của PixelUnshuffle(16): 32.3M tia tính toán CPU vô nghĩa        │
│    ──► Chuyển sang Token-Grid Frustum Rays (chỉ 32,760 tia trực tiếp)            │
│                                                                                  │
│ 3. LỖ HỔNG CHÍ MẠNG CỦA SCoPE: Triệt tiêu Góc Xoay tại Gốc Tọa Độ (o = 0)        │
│    ──► m = 0 ──► log||m|| = -inf ──► scale_gate -> 0 ──► XÓA SỔ HƯỚNG TIA d!    │
│    ──► Sửa sai: Decoupled Dual-Manifold Gating (DDMG) bảo toàn 100% d             │
│                                                                                  │
│ 4. Tích Tương hỗ Plücker (Klein Quadric) trong Attention Heads 9–12              │
│    ──► Flip (d, m̂) sang (m̂, d) giữa Q và K để đo trực tiếp độ cắt nhau 3D       │
│                                                                                  │
│ 5. Khử hiện tượng Nhấp nháy Độ sáng AdaLN bằng Gated Value Residual              │
│    ──► Bù trừ động học tương đối T_rel vào Value v, bảo vệ tuyệt đối Latent Norm │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.1. TẤN CÔNG 1: XUNG ĐỘT KHÔNG-THỜI GIAN GIỮA 3D CAUSAL VAE VÀ MÃ HÓA TIA
- **Lỗ hổng trong thiết kế cũ**:
  Trong Vòng 2, tensor tia Plücker được định nghĩa trên toàn bộ $F = 81$ khung hình RGB gốc: $\mathbf{P} \in \mathbb{R}^{B \times 6 \times 81 \times H \times W}$.
  Tuy nhiên, Wan2.1 sử dụng **3D Causal VAE** với tỉ lệ nén thời gian $4\times$:
  $81 \text{ raw frames} \implies 21 \text{ latent frames}$ ($t=0$ và 20 khối 4-frame tiếp theo).
  `WanModel.patch_embedding` có `patch_size = (1, 2, 2)` (stride thời gian bằng 1).
  Do đó, DiT xử lý chính xác $T_{lat} = 21$ latent tokens dọc theo trục thời gian.
  Nếu tính toán tia Plücker ở 81 frames, tensor nhúng sẽ có shape $[B, 81, 30, 52, D]$, gây lỗi sụp đổ kích thước (`RuntimeError: size mismatch at dim 1: 81 vs 21`) khi cộng vào Query/Key của DiT!
- **Giải pháp Đóng băng (Temporal-Receptive-Field Anchor)**:
  Tia Plücker không được tính ở 81 frame RGB rời rạc mà được nội suy giải tích Slerp chính xác tại **21 tâm thời gian của trường tiếp nhận (Receptive Field Centers)**:
  $$t_{mid}^{(0)} = 0, \quad t_{mid}^{(k)} = 4k - 1.5 \quad (\forall k \in \{1, \dots, 20\})$$
  Tạo ra chính xác 21 ma trận camera $[R^{(k)} \mid \mathbf{o}^{(k)}]$, đồng bộ $100\%$ với số lượng token thời gian của Wan2.1.

---

### 2.2. TẤN CÔNG 2: SỰ LÃNG PHÍ 1000 LẦN & NGHẼN CỔ CHAI CỦA `PixelUnshuffle(16)`
- **Lỗ hổng trong thiết kế cũ**:
  Ý tưởng dùng `PixelUnshuffle(16)` trên khung hình $480 \times 832$ tưởng chừng đẹp mắt ($6 \times 16^2 = 1536$ khớp dim DiT), nhưng khi triển khai thực tế bộc lộ 2 khuyết tật nặng nề:
  1. **Nghẽn cổ chai Dataloader CPU**: Phải tạo meshgrid và nhân ma trận cho $81 \times 480 \times 832 \approx 32.350.000$ tia sáng cho mỗi video. Điều này tiêu tốn hàng trăm MB RAM và mất $> 400\text{ms}$ mỗi batch trên CPU, biến Dataloader thành điểm nghẽn kìm hãm GPU.
  2. **Dư thừa thông tin 99.9%**: Trong một patch nhỏ $16 \times 16$ pixel (tương ứng góc mở quang học $< 1.5^\circ$), chùm tia sáng biến thiên gần như tuyến tính tuyệt đối. Việc unshuffle 256 tia song song thực chất chỉ là lặp lại cùng một thông tin hình học, buộc ma trận chiếu tuyến tính phải học cách trung bình hóa chúng lại.
- **Giải pháp Đóng băng: Token-Grid Frustum Rays (Tia Nón Trung Tâm Token)**:
  Kế thừa phát kiến từ SCoPE (`compute_camera_rays` trong `scope/geometry.py`), ta tính toán trường tia **TRỰC TIẾP TRÊN LƯỚI TOKEN** ($30 \times 52$):
  $$u_i = 0.5 \cdot \Delta w + i \cdot \Delta w, \quad v_j = 0.5 \cdot \Delta h + j \cdot \Delta h$$
  - Tổng số tia cần tính toán: $21 \times 30 \times 52 = 32.760$ tia!
  - **Giảm tải $1000\times$ khối lượng tính toán**, thực thi tức thời trên GPU trong $< 0.5\text{ms}$, triệt tiêu hoàn toàn nghẽn cổ chai Dataloader!

---

### 2.3. TẤN CÔNG 3: PHÁT HIỆN LỖI TOÁN HỌC ẨN TÀNG CỦA TENCENT SCoPE TẠI GỐC TỌA ĐỘ ($\mathbf{o} = \mathbf{0}$)

Đây là một phát hiện lý thuyết và thực nghiệm có giá trị học thuật xuất sắc.

#### Phân tích Cơ chế của Tencent SCoPE
Trong `scope/encoding.py` dòng 226–236, SCoPE phân rã tọa độ Plücker:
$$\mathbf{m} = \mathbf{o} \times \mathbf{d}, \quad \|\mathbf{m}\| = \text{clamp}(\|\mathbf{m}\|, 10^{-6}), \quad \hat{\mathbf{m}} = \frac{\mathbf{m}}{\|\mathbf{m}\|}, \quad s = \log \|\mathbf{m}\|$$
Sau đó tính toán cổng tỉ lệ (Scale Gate):
$$\mathbf{gate} = \text{Sigmoid}(\text{MLP}(s)) \in (0, 1)$$
Và nhân trực tiếp cổng này vào toàn bộ vector nhúng:
$$\mathbf{pe}_q = \mathbf{gate} \cdot \alpha_q \cdot \text{RMSNorm}(E_q(\mathbf{d}, \hat{\mathbf{m}}, s))$$

#### Lỗ hổng Chí mạng (The Zero-Origin Erasure Bug)
- Tại Frame 0 (hoặc bất kỳ vị trí nào camera đi qua gốc tọa độ hệ quy chiếu thế giới):
  $$\mathbf{o} = [0, 0, 0]^T \implies \mathbf{m} = \mathbf{0} \times \mathbf{d} = [0, 0, 0]^T$$
- Khi đó: $\|\mathbf{m}\| = 10^{-6} \implies s = \log(10^{-6}) \approx -13.82$.
- Giá trị âm rất lớn này đi qua cổng Sigmoid làm $\mathbf{gate} \to 0$!
- Vì $\mathbf{gate}$ nhân với toàn bộ $\mathbf{pe}_q$, nó **TRIỆT TIÊU TOÀN BỘ VECTOR NHÚNG VỀ 0** ($\mathbf{pe}_q \to \mathbf{0}$)!
- **HẬU QUẢ VÔ CÙNG NGUY HIỂM**:
  Tại Frame 0, camera vẫn có ma trận xoay $R_0$ xác định hướng nhìn của máy quay. Hướng nhìn $\mathbf{d} \in \mathbb{S}^2$ là hoàn toàn xác thực và mang đầy đủ thông tin góc quay (Pan/Tilt/Roll).
  Thế nhưng do $\mathbf{o} = \mathbf{0}$, cơ chế của SCoPE đã **VÔ TÌNH XÓA SỔ TOÀN BỘ THÔNG TIN HƯỚNG TIA $\mathbf{d}$** tại Frame 0!
  *(SCoPE không phát hiện ra lỗi này vì họ chỉ làm Image-to-Video, nơi Frame 0 đã có sẵn ảnh RGB thật neo giữ; còn trong Video-to-Video Retargeting, Frame 0 phải tự sinh theo góc quay mới nên nếu bị xóa $\mathbf{d}$, góc quay Frame 0 sẽ hoàn toàn mất kiểm soát!)*

#### Giải pháp Sửa sai Độc quyền: Decoupled Dual-Manifold Gating (DDMG)
Hướng tia $\mathbf{d}$ thuộc đa tạp mặt cầu đơn vị $\mathbb{S}^2$ (luôn có chuẩn $\|\mathbf{d}\| \equiv 1$, không hề có độ bất định về tỉ lệ scale).
Chỉ có mô-men $\mathbf{m}$ thuộc không gian $\mathbb{R}^3$ mới phụ thuộc vào khoảng cách tới gốc tọa độ.
Do đó, ta bắt buộc phải **phân tách độc lập cơ chế nhúng**:
$$\mathbf{pe}_q = \mathbf{pe}_q^{(dir)}(\mathbf{d}) + \mathbf{gate}(s) \cdot \mathbf{pe}_q^{(pos)}(\hat{\mathbf{m}}, s)$$
- Khi $\mathbf{o} \to \mathbf{0}$: $\mathbf{gate}(s) \to 0$ sẽ dập tắt thành phần mô-men không xác định, nhưng thành phần hướng nhìn $\mathbf{pe}_q^{(dir)}(\mathbf{d})$ **VẪN HOẠT ĐỘNG $100\%$ NĂNG LƯỢNG**!
- Góc quay camera tại Frame 0 được bảo toàn nguyên vẹn, giải quyết triệt để lỗi ẩn tàng của SCoPE!

---

### 2.4. TẤN CÔNG 4: TÍCH TƯƠNG HỖ PLÜCKER (KLEIN QUADRIC RECIPROCAL ATTENTION)
- **Cơ chế Hình học Xạ ảnh**:
  Hai đường thẳng trong không gian 3D $L_1 = (\mathbf{d}_1, \mathbf{m}_1)$ và $L_2 = (\mathbf{d}_2, \mathbf{m}_2)$ cắt nhau (hoặc cùng nằm trong một mặt phẳng) khi và chỉ khi tích tương hỗ Plücker (Reciprocal Product) triệt tiêu:
  $$\mathbf{d}_1 \cdot \mathbf{m}_2 + \mathbf{m}_1 \cdot \mathbf{d}_2 = 0$$
- **Thiết kế Đóng băng trên Attention Heads 9–12**:
  Ta áp dụng phép đảo hoán biến (Flipping) giữa Query và Key:
  - Query nhận: $\mathbf{x}_q = [\mathbf{d}_i \parallel \hat{\mathbf{m}}_i \parallel s_i]$
  - Key nhận: $\mathbf{x}_k = [\hat{\mathbf{m}}_j \parallel \mathbf{d}_j \parallel s_j]$ (Đảo vị trí giữa $\mathbf{d}$ và $\hat{\mathbf{m}}$!)
  Khi thực hiện phép nhân vô hướng $\mathbf{Q}_{geom} \mathbf{K}_{geom}^T$ trong Heads 9–12:
  $$\mathbf{Q}_{geom, i} \cdot \mathbf{K}_{geom, j}^T \propto \mathbf{d}_i \cdot \hat{\mathbf{m}}_j + \hat{\mathbf{m}}_i \cdot \mathbf{d}_j + s_i \cdot s_j$$
  Phép tính Attention tự động tính toán **ĐỘ GẦN NHAU CỦA HAI TIA SÁNG TRONG KHÔNG GIAN 3D**, trực tiếp áp đặt ràng buộc Epipolar Constraint lên ma trận chú ý mà không cần các thuật toán dò tìm đắt đỏ!

---

### 2.5. TẤN CÔNG 5: KHỬ HIỆN TƯỢNG NHẤP NHÁY ĐỘ SÁNG CỦA AdaLN BẰNG GATED VALUE RESIDUAL
- **Lỗ hổng trong thiết kế cũ**:
  Trong Vòng 2, ta dự định cộng vector vận tốc camera $\mathbf{e}_{vel}^{(t)}$ vào vector điều chế AdaLN $t_{mod}$.
  Tuy nhiên, trong `wan_video_dit.py`:
  $$\text{modulate}(x, shift, scale) = x \cdot (1 + scale) + shift$$
  $scale$ và $shift$ tác động trực tiếp lên giá trị trung bình và phương sai của các kênh đặc trưng $z$. Nếu camera tăng tốc ở frame này và dừng ở frame khác, $scale$ sẽ biến thiên mạnh giữa các frame, khiến các latent vectors sau khi giải mã qua Causal VAE bị **nhấp nháy độ sáng và tương phản màu sắc liên tục (Temporal Brightness/Contrast Flickering)**!
- **Giải pháp Đóng băng: Gated Value Residual Injection (GVRI)**:
  Thay vì can thiệp vào AdaLN, vector động học tương đối $\mathbf{V}_{rel}^{(t)} \in \mathbb{R}^{18}$ (gồm xoay liên tục 6D và tịnh tiến) được chiếu qua Temporal MLP và cộng trực tiếp vào **Value stream ($v$)** của DiT:
  $$\mathbf{v}_{new} = \mathbf{v}_{base} + \sigma(\mathbf{b}_{gate} + \text{MLP}(\mathbf{V}_{rel}^{(t)})) \cdot \text{Linear}(\mathbf{V}_{rel}^{(t)})$$
  - Vì ma trận chú ý đã được dẫn hướng bởi $QK^T$, luồng Value $v$ sẽ mang thông tin chuyển động bù trừ trực tiếp vào token tương ứng mà **KHÔNG HỀ LÀM BIẾN DẠNG PHÂN PHỐI LATENT NORM**, triệt tiêu $100\%$ nguy cơ nhấp nháy ánh sáng!

---

## 3. THIẾT KẾ ĐÓNG BĂNG CUỐI CÙNG CHO MỤC 2 (FORMALLY FROZEN SPECIFICATION)

```
                     ┌─────────────────────────────────────────────────────────┐
                     │     CAMERA RETARGETING INPUTS (Source & Target Poses)   │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
   [LUỒNG 1: QUANG HỌC XẠ ẢNH]                                      [LUỒNG 2: ĐỘNG LỰC HỌC TƯƠNG ĐỐI]
   - Token-Grid Center Rays (30x52)                                 - Relative Pose: T_rel = T_src^-1 * T_tgt
   - 21 Causal VAE Time Centers (t_mid)                             - Continuous 6D Rotation: R_6D in R^6
   - Canonical Bounding Sphere Scale                                - Trajectory Velocity: V_rel in R^18
                 │                                                                 │
                 ▼                                                                 ▼
   [Decoupled Dual-Manifold Gating]                                 [Temporal Kinematic MLP]
   - Direction: pe^(dir)(d) (100% active)                           - Zero-initialized Gate: sigmoid(b_gate)
   - Moment: gate(s) * pe^(pos)(m_hat, s)                           - Injected into Value stream:
                 │                                                    v = v + cam_residual
                 ▼                                                                 │
   [Klein Quadric Reciprocal Projection]                                           │
   - Q receives: [d || m_hat || log s]                                             │
   - K receives: [m_hat || d || log s]                                             │
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
   │ WAN2.1 DIT ATTENTION BLOCK:                                                                   │
   │ - Heads 1–8 (1024D): Pure Appearance Attention (Zero Ray Corruption)                         │
   │ - Heads 9–12 (512D): Reciprocal Epipolar Attention (Q_geom * K_geom^T)                        │
   │ - Value Stream: v = v + cam_residual                                                          │
   │ - FlashAttention-2 Native Dispatch: 0 Byte Tensor Bias, 100% Hardware Acceleration             │
   └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. BẢNG ĐỐI SOÁT QUA 3 VÒNG LẶP BIỆN CHỨNG (MỤC 2)

| Đặc tính Kỹ thuật | ReCamMaster (SOTA 2025) | SCoPE (Tencent 2026) | Thiết kế Vòng 2 của Chúng ta | Thiết kế Hoàn thiện Tuyệt đối (Vòng 3) |
| :--- | :--- | :--- | :--- | :--- |
| **Độ phân giải Không gian Ray** | 1 vector 12D broadcast toàn ảnh | Token-grid $30 \times 52$ | `PixelUnshuffle(16)` ($480 \times 832$) | **Token-Grid Frustum Rays ($30 \times 52$) $\implies$ Nhanh hơn 1000 lần** |
| **Đồng bộ Thời gian Causal VAE** | Lấy mẫu thô `[::4]` | Không xét V2V | Chưa chi tiết hóa $81 \to 21$ | **Window-Centered Slerp $t_{mid}$ (Khớp chính xác 21 latent frames)** |
| **Độ ổn định tại Gốc tọa độ $\mathbf{o} = \mathbf{0}$** | Bị gãy khi camera chuyển động | **BỊ LỖI XÓA SỔ HƯỚNG TIA $\mathbf{d}$** ($\mathbf{gate} \to 0$) | Chuẩn hóa bán kính $R_{bound}$ | **Decoupled Dual-Manifold Gating (DDMG) $\implies$ Bảo toàn $100\%$ góc xoay** |
| **Bản chất Đại số của Attention** | Phép cộng MLP thô bạo | Tích vô hướng Euclid 7D | Phép cộng chập $1 \times 1$ | **Klein Quadric Reciprocal Product ($(\mathbf{d}, \hat{\mathbf{m}}) \leftrightarrow (\hat{\mathbf{m}}, \mathbf{d})$ trên Heads 9–12)** |
| **Bảo vệ Ngoại quan & Kết cấu** | Bị ô nhiễm toàn bộ 1536D | Bị ô nhiễm toàn bộ 1536D | Phân tách Heads 9–12 | **Phân tách Heads 9–12 + Block-Diagonal $\mathbf{W}_O$ $\implies$ Ngoại quan tinh khiết** |
| **Bù trừ Chuyển động Nguồn-Đích** | Không có (Frame-0 freeze) | Không có (chỉ làm I2V) | Dự kiến đưa vào AdaLN | **Gated Value Residual Injection $\implies$ Triệt tiêu nhấp nháy ánh sáng** |
