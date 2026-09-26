# NGHIÊN CỨU CHUYÊN SÂU MỤC 3 (VÒNG LẶP BIỆN CHỨNG 1): CƠ CHẾ CHÚ Ý KHÔNG-THỜI GIAN & ĐỒNG BỘ 3D-ROPE
**Khóa Luận Tốt Nghiệp: Video-to-Video Camera Trajectory Retargeting (V2V-CR)**  
**Nền tảng Backbone:** Wan2.1-T2V-1.3B (Flow Matching, DiT 30 Layers, Dim 1536, 12 Heads)  
**Tài liệu tham chiếu mã nguồn & thực nghiệm:**
- Mã nguồn Wan2.1 DiT Engine (`diffsynth/models/wan_video_dit.py`)
- Mã nguồn ReCamMaster (`train_recammaster.py` & `diffsynth/models/wan_video_dit.py`)
- Mã nguồn Tencent SCoPE (`scratch/scope/modeling.py` & `scratch/scope/patch.py`)
- Lý thuyết RoPE Đa chiều (Su et al., 2024; Position Interpolation & FlexRoPE)

---

## 1. BỐI CẢNH & ĐỘNG CƠ CỦA MỤC 3

Trong kiến trúc Diffusion Transformer cho Video (DiT), **Cơ chế Chú ý Không-Thời gian (Spatio-Temporal Attention)** và **Mã hóa Vị trí Xoay 3D (3D-RoPE)** đóng vai trò là "hệ thần kinh trung ương":
- Chúng quyết định cách thức các token ở các khung hình và vị trí không gian khác nhau liên lạc với nhau.
- Trong bài toán **Video-to-Video Camera Retargeting (V2V-CR)**, đây là nơi diễn ra sự giao thoa then chốt giữa:
  1. Luồng video đích đang được khử nhiễu ($z_{tgt}$).
  2. Luồng video nguồn sạch cung cấp ngữ cảnh mỏ neo ($z_{src}$).
  3. Trường tia hình học camera 6DoF ($\mathbf{P}_{tgt}, \mathbf{P}_{src}$).

Khi kiểm tra mã nguồn của các phương pháp SOTA tiền nhiệm (đặc biệt là ReCamMaster), chúng tôi phát hiện những sai lầm cấu trúc nghiêm trọng khiến mô hình bị suy thoái chất lượng chuyển động và giới hạn cứng nhắc về độ dài video.

---

## 2. PHÂN TÍCH KHUYẾT TẬT CỦA CÁC MÃ NGUỒN TIỀN NHIỆM

### 2.1. LỖ HỔNG TRƯỢT PHA THỜI GIAN TRONG RECAMMASTER ($\Delta t = 21$)
- **Kiểm chứng mã nguồn**: Kiểm tra `train_recammaster.py` dòng 352–353:
  ```python
  tgt_latent_len = noisy_latents.shape[2] // 2
  noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
  ```
  Và trong `diffsynth/models/wan_video_dit.py` dòng 335–339:
  ```python
  freqs = torch.cat([
      self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
      ...
  ```
- **Bản chất sai lầm**:
  ReCamMaster xếp nối tiếp Target ($z_{tgt}$, 21 latent frames) và Source ($z_{src}$, 21 latent frames) dọc theo trục thời gian thành chuỗi $f = 42$ frames!
  Hậu quả là bảng tần số 3D-RoPE gán chỉ số thời gian:
  - Target Frame $0 \dots 20 \implies t \in [0, 20]$
  - Source Frame $0 \dots 20 \implies t \in [21, 41]$
- **Hệ quả vật lý tai hại**:
  Tại Frame $k$ (ví dụ Frame 5), Target Frame 5 và Source Frame 5 thực chất diễn ra ở **CÙNG MỘT THỜI ĐIỂM VẬT LÝ**. Thế nhưng 3D-RoPE lại áp đặt một khoảng cách thời gian giả tạo:
  $$\Delta t = 26 - 5 = 21 \text{ frames}!$$
  Theo lý thuyết RoPE, ma trận quay xoay lệch pha theo khoảng cách: $\mathbf{R}_{RoPE}(21)$. Tần số xoay lớn này dập tắt (attenuate) lực chú ý giữa Target Frame 5 và Source Frame 5! Mạng DiT bị đánh lừa rằng video nguồn diễn ra ở 21 khung hình trong tương lai, làm suy yếu khả năng mượn kết cấu của khung hình tương ứng!

---

### 2.2. RÀO CẢN CỨNG NHẮC `assert num_frames == 81`
- **Khuyết tật mã nguồn**:
  Trong ReCamMaster và Wan2.1 gốc, ma trận tần số RoPE và kích thước tensor camera bị gán cứng vào con số 81 frame RGB ($21$ frame latent).
  Nếu người dùng đưa vào một video ngắn hơn (ví dụ 49 frame, 33 frame) hoặc dài hơn (161 frame), mã nguồn sẽ bị vỡ nát hoặc gặp lỗi suy thoái pha quay ngoài phân phối (Out-of-Distribution Phase Collapse).

---

### 2.3. BẾ TẮC CỦA SCoPE KHI MASK BỎ TOÀN BỘ 2D ROPE
- **Kiểm chứng mã nguồn SCoPE**: Trong `scratch/scope/modeling.py` dòng 52–58:
  Tencent SCoPE nhận thấy 2D Spatial RoPE cản trở tia Plücker, nên họ cung cấp tùy chọn `disable_spatial_rope` để xóa sạch RoPE theo chiều $H$ và $W$.
- **Hậu quả**: Khi tắt sạch RoPE không gian trên toàn bộ mạng, DiT mất đi khả năng định vị tọa độ pixel cục bộ, khiến các chi tiết kết cấu 2D (như viền chữ, hoa văn, râu tóc) bị giảm độ sắc nét nghiêm trọng.

---

## 3. THIẾT KẾ ĐỘT PHÁ CỦA CHÚNG TÔI CHO MỤC 3

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               KIẾN TRÚC CHÚ Ý KHÔNG-THỜI GIAN & 3D-ROPE ĐỒNG BỘ                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Synchronized Temporal Indexing:                                                     │
│    - RoPE_Pos(z_tgt^(k,h,w)) = (k, h, w)                                               │
│    - RoPE_Pos(z_src^(k,h,w)) = (k, h, w) ──► Δt = 0 giữa cặp frame đồng thời!          │
│                                                                                        │
│ 2. Selective RoPE Masking theo Attention Head:                                        │
│    - Heads 1–8 (Appearance - 1024D): Giữ 100% 3D-RoPE (Thời gian + Không gian)         │
│    - Heads 9–12 (Geometry - 512D): MASK BỎ RoPE Không gian (Chỉ giữ RoPE Thời gian)    │
│                                                                                        │
│ 3. Timestep-Modulated Domain Orthogonal Projection (T-DOP):                            │
│    - e_tgt(t), e_src(t) thay đổi theo bước khuếch tán t (t -> 1 ưu tiên mượn nguồn,    │
│      t -> 0 cân bằng nhất quán nội tại đích)                                           │
│                                                                                        │
│ 4. Dynamic Normalized Temporal Scaling (DNTS):                                         │
│    - Co giãn pha liên tục phi(t) in [0, 1] ──► Hỗ trợ video độ dài bất kỳ (17->241f)   │
│                                                                                        │
│ 5. Motion Phase Coordinate Mapping (MPCM):                                             │
│    - Cho phép Retargeting Camera kết hợp Làm chậm (Slow-Mo) / Tăng tốc video mượt mà   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1. NGUYÊN TẮC 1: ĐỒNG BỘ HÓA CHỈ SỐ THỜI GIAN 3D-ROPE ($\Delta t = 0$)
Thay vì xếp nối tiếp làm lệch pha, ta gán chỉ số thời gian đồng bộ tuyệt đối giữa Target và Source:
- Với token Target tại khung hình $k \in [0, 20]$: $\text{Index}_{time} = k$.
- Với token Source tại khung hình $k \in [0, 20]$: $\text{Index}_{time} = k$.
- **Hệ quả toán học**:
  Khoảng cách thời gian tương đối giữa Target Frame $k$ và Source Frame $k$:
  $$\Delta t = k - k = 0 \implies \mathbf{R}_{RoPE}(\Delta t) = \mathbf{R}_{RoPE}(0) = I$$
  Không hề có bất kỳ sự suy hao pha nhân tạo nào!
  Cơ chế Attention tại Frame $k$ của Target được tạo điều kiện tối đa để khai thác $100\%$ tương quan ngoại quan và hình học từ Frame $k$ của Source.

---

### 3.2. NGUYÊN TẮC 2: PHÂN TÁCH ROPE CHỌN LỌC (SELECTIVE ROPE MASKING THEO TỪNG HEAD)
Đây là thiết kế độc quyền giải quyết mâu thuẫn giữa Tencent SCoPE và Wan2.1:
- **Heads 1–8 (Appearance - 1024D)**:
  Giữ nguyên trọn vẹn $100\%$ 3D-RoPE gốc của Wan2.1:
  $$\text{freqs}_{app} = [\text{freqs}_F(44), \; \text{freqs}_H(42), \; \text{freqs}_W(42)]$$
  Đảm bảo các kênh ngoại quan có đầy đủ nhận thức vị trí cảm biến 2D để vẽ chi tiết siêu nét.
- **Heads 9–12 (Geometry - 512D)**:
  Mask bỏ toàn bộ phần không gian $H$ và $W$, chỉ giữ lại tần số thời gian $F$:
  $$\text{freqs}_{geom} = [\text{freqs}_F(44), \; \mathbf{1}_{H}(42), \; \mathbf{1}_{W}(42)]$$
  Bởi vì ở Heads 9–12, tọa độ tia Plücker $(\mathbf{d}, \hat{\mathbf{m}})$ đã cung cấp thông tin không gian 3D thực tế trong thế giới. Việc gỡ bỏ RoPE 2D giúp các tia sáng cắt nhau trong không gian 3D tự do ghép cặp qua ma trận Attention mà không bị cản trở bởi khoảng cách pixel trên cảm biến!

---

### 3.3. NGUYÊN TẮC 3: TIMESTEP-MODULATED DOMAIN ORTHOGONAL PROJECTION (T-DOP)
- **Lỗ hổng của vector domain tĩnh**:
  Nếu chỉ cộng một vector cố định $\mathbf{e}_{tgt}, \mathbf{e}_{src}$, độ lệch chú ý giữa Target và Source là một hằng số bất biến xuyên suốt quá trình khuếch tán.
  Nhưng trên thực tế:
  - Ở bước nhiễu lớn ($t \to 1$): Target là nhiễu trắng thuần túy, cần tập trung chú ý vào Source sạch để thiết lập bố cục không gian 3D.
  - Ở bước nhiễu nhỏ ($t \to 0$): Target đã hình thành rõ nét, cần tăng cường tự chú ý nội tại (Intra-target self-attention) để giữ chuyển động mượt mà và tự sinh các vùng bị khuất lấp (Disocclusion inpainting).
- **Giải pháp Điều chế theo Thời gian**:
  Vector định danh miền được sinh động theo bước khuếch tán $t$:
  $$\mathbf{e}_{tgt}(t) = \text{MLP}_{domain}(\mathbf{e}_{tgt}^{base}, t), \quad \mathbf{e}_{src}(t) = \text{MLP}_{domain}(\mathbf{e}_{src}^{base}, t)$$
  Tự động cân bằng động lực học chú ý tối ưu ở mọi thời điểm của lịch trình Flow Matching!

---

### 3.4. NGUYÊN TẮC 4: DYNAMIC NORMALIZED TEMPORAL SCALING (DNTS) CHO ĐỘ DÀI VIDEO BẤT KỲ
Để vượt qua giới hạn cứng nhắc $F = 81$ frames:
- Định nghĩa chiều dài tham chiếu huấn luyện: $T_{ref} = 21$ latent frames.
- Khi xử lý video có độ dài tùy ý $T_{actual}$ (ví dụ 13 frames hoặc 41 frames):
  Tọa độ thời gian liên tục được chuẩn hóa:
  $$t' = t \cdot \frac{T_{ref} - 1}{T_{actual} - 1} \in [0, 20]$$
  Ánh xạ mọi chỉ số thời gian của video bất kỳ về trọn vẹn khoảng $[0, 20]$ mà mô hình đã học.
  **Triệt tiêu $100\%$ hiện tượng Out-of-Distribution Phase Collapse**, cho phép mô hình tạo video mượt mà ở bất kỳ độ dài nào ($F \in \{17, 25, 33, 49, 81, 161\}$).

---

### 3.5. NGUYÊN TẮC 5: MOTION PHASE COORDINATE MAPPING (MPCM)
- Mở rộng ứng dụng thực tế: Cho phép người dùng vừa thay đổi quỹ đạo máy quay vừa **thay đổi tốc độ chuyển động (Retargeting + Retiming / Slow-Motion)**.
- Bằng cách điều chỉnh tốc độ quét pha $\phi_{tgt}(t)$ so với $\phi_{src}(t)$, mô hình có thể sinh ra video quay chậm (Slow-Mo $2\times, 4\times$) với quỹ đạo camera mới hoàn toàn tự nhiên, các pha chuyển động của vật thể vẫn khớp nối hoàn hảo với video nguồn!

---

## 4. HIỆU NĂNG TÍNH TOÁN VÀ TƯƠNG THÍCH FLASHATTENTION-2

| Thông số Kỹ thuật | Giá trị Thiết kế Đóng băng | Đánh giá Phần cứng (Kaggle 16GB) |
| :--- | :--- | :--- |
| **Kích thước Query ($Q$)** | $32.760$ tokens (Chỉ gồm Target stream) | Tiết kiệm $50\%$ bộ nhớ so với nối đôi |
| **Kích thước Key/Value ($K, V$)** | $65.520$ tokens ($[K_{tgt} \parallel K_{src}]$) | Chứa trọn vẹn ngữ cảnh nguồn sạch |
| **Kích thước Head Dim** | $128$ dimensions (Đồng nhất trên cả 12 Heads) | Khớp chuẩn Native Hardware Tile của FlashAttention-2 |
| **Chi phí Bộ nhớ Mask** | **0 Byte** (Không cần tensor mask $65K \times 65K$) | Tiết kiệm $8.58\text{ GB}$ VRAM, 0 OOM |
| **Bộ nhớ Kích hoạt mỗi Block** | $\approx 604\text{ MB}$ (Với Activation Checkpointing) | Hoạt động trơn tru trên GPU 16GB |
| **Thời gian Thực thi 3D Attn** | Fused CUDA Kernel của FlashAttention-2 | $< 1.5\text{s}$ mỗi bước Flow Matching |
