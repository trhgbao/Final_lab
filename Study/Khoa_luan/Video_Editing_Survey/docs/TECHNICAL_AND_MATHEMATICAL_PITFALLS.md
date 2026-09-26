# SỔ TAY KỸ THUẬT & QUY TẮC BẤT BIẾN TOÁN HỌC
## (Technical & Mathematical Pitfalls Registry - Video Retargeting)

> [!IMPORTANT]
> **QUY TẮC HOẠT ĐỘNG BẮT BUỘC (MANDATORY OPERATIONAL RULES)**:
> 1. **ĐỌC TRƯỚC KHI LÀM (Pre-Action Reading)**: Tài liệu này **BẮT BUỘC PHẢI ĐƯỢC ĐỌC LẠI** trước khi đề xuất hoặc thực hiện bất kỳ cải tiến/chỉnh sửa nào trong phương pháp để đảm bảo không lặp lại bất kỳ sai lầm nào trong danh mục.
> 2. **ĐỐI CHIẾU SAU KHI LÀM (Post-Action Verification)**: Sau khi viết code xong, phải rà soát lại toàn bộ tiêu chí kiểm tra (Pre-flight Checklist) trong tài liệu này trước khi đóng gói hoặc chạy thử.
> 3. **BẮT BUỘC CẬP NHẬT KHI CÓ LỖI (Mandatory Update on Failure)**: Mỗi khi xuất hiện bất kỳ lỗi mới nào, **TRƯỚC HẾT (trước khi trả lời câu hỏi tiếp theo của người dùng)**, Agent PHẢI bổ sung phân tích lỗi đó vào tài liệu này theo đúng cấu trúc chuẩn.
> 4. **BẢO ĐẢM TÍNH CHUẨN MỰC KHOA HỌC (Scientific Rigor - No Patchwork / Ad-hoc Hacks)**: Đây là đề tài nghiên cứu khoa học / khóa luận tốt nghiệp, **TUYỆT ĐỐI KHÔNG** đề xuất hay chấp nhận các giải pháp chắp vá (patchwork), thủ thuật tình thế (workaround), gán cứng (hardcoding prompt, hardcoding giá trị cụ thể của một video riêng lẻ), hay các kỹ thuật "che mắt" không thể tổng quát hóa. Mọi giải pháp bắt buộc phải dựa trên nguyên lý cơ bản (First-Principles Thinking) của Hình học Chiếu 3D (3D Multi-View Projective Geometry), Mô hình Khuếch tán Xác suất (Denoising Diffusion Probabilistic Models), và Thị giác Máy tính (Computer Vision), có công thức toán học minh bạch, tổng quát cho mọi video đầu vào bất kỳ, và đủ chuẩn mực để bảo vệ trước hội đồng khoa học quốc tế.

---

## BẢNG TỔNG HỢP CÁC LỖI KỸ THUẬT & TOÁN HỌC CẦN TRÁNH

| Mã lỗi | Tên lỗi | Triệu chứng trực quan | Nguyên nhân toán học cốt lõi |
| :--- | :--- | :--- | :--- |
| **PIT-01** | Border Inset & Keyframe Slit Accumulation | Sọc đen dọc như nan quạt (accordion blinds) cách đều 12px + 2 tam giác đen ở mép trên/dưới | `ones_inset` gọt mất 1px biên, vòng lặp quét ngược 24 keyframe làm tích tụ 24 vạch đen |
| **PIT-02** | Latent Noise Advection / Warping | Tường phẳng bệt, xám/tím, mất 100% khối 3D và vân gỗ | Phép warp bilinear nội suy nhiễu $\epsilon$ liên tục làm triệt tiêu 50 lần tần số cao Fourier |
| **PIT-03** | Flow Loss on Independent Gaussian Noise | Video bị sọc quét ngang chu kỳ (scanlines) | Ép loss quang học lên nhiễu Gaussian $\epsilon_t \sim \mathcal{N}(0, I)$ độc lập từng frame |
| **PIT-04** | Recursive VAE Encoding / Color Collapse | Màu sắc nhạt dần, mờ sương, mất tương phản sau 25 frames | Tích lũy sai số nén 1% của VAE qua $(0.99)^{25} \approx 0.77$ khi encode/decode tuần tự |
| **PIT-05** | Accumulated Inpaint Hole Without Memory | Đột biến ngữ nghĩa (tủ gỗ biến thành cửa sổ ở Frame 17) | Hố inpaint tích lũy quá lớn (207px), $z_{masked}$ bị che đen toàn bộ khiến SDXL vẽ tự do |
| **PIT-06** | Prompt Hardcoding / Cheating | Mất tính tổng quát của nghiên cứu khoa học | Gán cứng text prompt `"wooden kitchen"` để chữa cháy thay vì giải quyết bằng thị giác máy tính |
| **PIT-07** | Autoregressive Slit-Scan Extrapolation / Sub-Latent Border Smearing | Chuỗi ~24 lát cắt thẳng đứng bậc thang lặp lại mép cửa (hiệu ứng đàn accordion / rèm xếp) | Inpaint tuần tự khe hẹp 12px ($1.5$ latents < kích thước kernel $3\times 3$) làm UNet chỉ sao chép mép lặp lại 24 lần |
| **PIT-08** | Independent Noise Mode Switching & 2D UNet Inter-Frame Morphing | Hố inpaint biến dạng ngữ nghĩa giữa các frame (frame 8 vẽ gương, frame 16 vẽ cửa, frame 24 vẽ gạch men) | Nhiễu $\epsilon_t$ độc lập rơi vào các mode khác nhau + UNet 2D thiếu Cross-Frame Attention trong khử nhiễu |

---

## CHI TIẾT TỪNG LỖI & QUY TẮC BẤT BIẾN (INVARIANTS)

---

### [PIT-01] Border Inset & Keyframe Slit Accumulation (Lỗi Sọc Nan Quạt & Tam Giác Đen Mép)
- **Thời điểm ghi nhận**: 2026-09-19 (Video `sota_pan_right_30deg_akr.mp4`).
- **Triệu chứng trực quan**:
  - Nửa bên phải khung hình xuất hiện một loạt vạch sọc đen thẳng đứng song song cách đều nhau đúng 12 – 13 pixel (hiệu ứng nan quạt / rèm xếp).
  - Góc trên-phải và dưới-phải có hai mảng tam giác đen kịt.
  - Frame 0 bị viền đen bao quanh thay vì giữ nguyên video gốc.

#### Phân tích Nguyên nhân Toán học & Thuật toán:
1. **Lỗi `ones_inset` tạo khe hở biên**:
   Để tránh răng cưa ở biên, code đã gán 0 cho các hàng/cột biên:
   ```python
   ones_inset = np.ones((height, width), dtype=np.uint8) * 255
   ones_inset[:, [0, -1]] = 0    # Xóa cột biên trái và phải
   ones_inset[[0, -1], :] = 0    # Xóa hàng biên trên và dưới
   ```
2. **Lỗi tích tụ khe hở qua vòng lặp quét ngược keyframe**:
   Vòng lặp `for theta_k, out_k in reversed(keyframe_outputs):` chiếu lại 24 frame đã sinh trước đó. Vì mỗi frame đều bị mất 1 cột pixel ở mép phải do `ones_inset`, khi chiếu về frame hiện tại qua các góc lệch $k \times \Delta\theta$, **24 khe hở đen 1-pixel này bị dịch chuyển sang trái lần lượt cách nhau đúng 12 pixel** ($\approx 1.25^\circ$). Kết quả tạo thành một "hàng rào" gồm 24 vạch đen.
3. **Biến dạng hình học Perspective Keystone**:
   Khi camera xoay quanh trục Y ($H = K R_y(\theta) K^{-1}$), mặt phẳng ảnh nghiêng trong không gian 3D, biến hình chữ nhật thành hình thang (keystone). Việc xóa hàng trên/dưới của `ones_inset` khiến các khoảng trống hình thang ở góc trên và góc dưới không được lấp đầy, biến thành 2 tam giác đen kịt.
4. **Vi phạm điều kiện Frame 0**:
   `ones_inset` làm Frame 0 có 3568 pixel viền bị coi là "hố inpaint", khiến Frame 0 bị đưa qua VAE/UNet một cách vô lý thay vì giữ nguyên gốc.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** dùng `ones_inset` hoặc tự ý xóa/gọt bớt pixel viền của mặt nạ warp. Luôn dùng mặt nạ nhị phân đầy đủ: `np.ones((h, w), dtype=np.uint8) * 255`.
- **TUYỆT ĐỐI KHÔNG** dùng vòng lặp quét ngược nhiều keyframe cũ (`reversed(keyframe_outputs)`) để ghép ảnh nền vì sai số biên sẽ bị nhân lên $N$ lần.
- **TUYỆT ĐỐI KHÔNG** để Frame 0 ($\theta = 0^\circ$) chạy qua pipeline khuếch tán hay VAE. Frame 0 phải là bản sao nguyên gốc 100%.

#### GIẢI PHÁP CHUẨN XÁC ĐÃ KIỂM CHỨNG:
Áp dụng **Two-Source Direct Compositing** (Chỉ 2 nguồn duy nhất, không lặp tích tụ):
1. **Nguồn A (Video gốc)**: Chiếu Frame 0 trực tiếp qua $H_{0 \to t}$ (1 bước chiếu duy nhất từ gốc $\implies$ 96.4% tần số cao, phủ kín cột $0 \to 364$).
2. **Nguồn B (Frame liền trước)**: Chiếu `output_frames[t-1]` qua đúng 1 bước nhảy tương đối $\Delta\theta = 1.25^\circ$ bằng $H_\Delta$ (phủ kín cột $364 \to 563$).
3. **Ảnh nền**: `base = np.where(mask_0 > 0, warped_0, warped_prev)`. (Số pixel đen trong vùng hợp lệ = 0).
4. **Hố Inpaint**: `hole_mask = (mask_prev == 0) * 255` (chính xác dải 12 pixel ở mép ngoài cùng bên phải).

---

### [PIT-02] Latent Noise Advection / Warping (Lỗi Mờ Bệt & Phẳng Lì Do Warp Nhiễu)
- **Thời điểm ghi nhận**: 2026-09-19 (Bản vá `perfect_pan_right_30deg_v8_crisp.mp4`).
- **Triệu chứng trực quan**: Khối tủ gỗ 3D và vân gỗ biến mất hoàn toàn, thay bằng một mảng tường phẳng lì, mờ mịt, mang sắc thái màu tím/xám nhân tạo.

#### Phân tích Nguyên nhân Toán học:
- Phép nội suy song tuyến tính (bilinear interpolation) khi warp ma trận nhiễu $\epsilon_{warped} = \text{warp}(\epsilon_{t-1})$ đóng vai trò như một bộ lọc thông thấp (low-pass filter) cực mạnh.
- Phép đo biến đổi Fourier 2D (2D-FFT) chứng minh: Tỉ số năng lượng tần số cao / tần số thấp (High/Low frequency ratio) bị sụt giảm từ **3.01 xuống 0.059** (mất hơn **98% năng lượng tần số cao**).
- Mô hình SDXL UNet cần nhiễu Gaussian chuẩn trắng $\mathcal{N}(0, I)$ với phổ tần số phẳng (white spectrum) để lấy "hạt giống" sinh ra các cạnh sắc nét và vi chi tiết. Khi bị cấp một ma trận nhiễu đã bị làm mờ, UNet không thể giải mã ra chi tiết tần số cao, dẫn đến bề mặt phẳng lì và xám bệt.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** warp, advect, trượt hay trộn (blend) nhiễu Gaussian trong không gian latent.
- Khởi tạo nhiễu khuếch tán ở mỗi frame $t$ **BẮT BUỘC PHẢI LÀ** nhiễu i.i.d. ngẫu nhiên độc lập chuẩn:
  $$\text{latents} = \text{torch.randn\_like}(z_0) \sim \mathcal{N}(0, I)$$

---

### [PIT-03] Flow Loss on Independent Gaussian Noise (Lỗi Sọc Quét Ngang Chu Kỳ)
- **Thời điểm ghi nhận**: 2026-09-18 (Bản train `v7` và `v8`).
- **Triệu chứng trực quan**: Video xuất hiện các đường sọc ngang quét qua khung hình theo chu kỳ (periodic horizontal scanlines), ảnh bị nhòe và biến dạng có gợn sóng.

#### Phân tích Nguyên nhân Toán học:
- Hàm mất mát thời gian được cài đặt sai lầm:
  $$\mathcal{L}_{temp} = \|\text{warp}(\hat{\epsilon}_{t-1}) - \hat{\epsilon}_t\|_1$$
- Trong lý thuyết Diffusion Models, nhiễu ground-truth $\epsilon_t \sim \mathcal{N}(0, I)$ và $\epsilon_{t-1} \sim \mathcal{N}(0, I)$ được lấy mẫu hoàn toàn độc lập tại mỗi frame. Giữa hai ma trận nhiễu ngẫu nhiên độc lập **KHÔNG TỒN TẠI** mối tương quan không gian-thời gian (spatial-temporal correlation).
- Việc ép đạo hàm gradient bắt nhiễu của frame $t$ phải khớp với nhiễu bị warp từ frame $t-1$ tạo ra xung đột nghiệm toán học (gradient fighting) giữa $\mathcal{L}_{diff}$ và $\mathcal{L}_{temp}$, làm hỏng các trọng số của Adapter và gây ra hiện tượng sọc quét ngang chu kỳ.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** tính hàm mất mát quang học (optical flow loss) hay bất kỳ hàm mất mát thời gian nào trên ma trận nhiễu $\epsilon$ hoặc dự đoán nhiễu $\hat{\epsilon}$.
- Mất mát thời gian (nếu có) CHỈ ĐƯỢC PHÉP tính trên biểu diễn ảnh sạch ước tính $\hat{z}_0$ hoặc decoded RGB, không bao giờ trên nhiễu.

---

### [PIT-04] Recursive VAE Encoding / Color Collapse (Lỗi Suy Thoái Màu Sắc Lũy Kế)
- **Thời điểm ghi nhận**: 2026-09-17 (Bản `v2` và `v3`).
- **Triệu chứng trực quan**: Video bị suy giảm độ tương phản, màu sắc nhợt nhạt dần và mờ sương (foggy/washed out) theo thời gian.

#### Phân tích Nguyên nhân Kỹ thuật:
- Quá trình tự hồi quy tuần tự: Frame $t-1$ được decode ra RGB, sau đó lại encode ngược lại vào VAE latent để làm điều kiện cho frame $t$.
- Bộ VAE của SDXL là bộ nén mất mát (lossy compression) với sai số tái tạo khoảng 1% mỗi chu trình:
  $$\text{Signal}(T) \approx (0.99)^T \times \text{Signal}(0) \implies (0.99)^{25} \approx 0.77$$
  Sau 25 frames, thông tin màu sắc và độ tương phản bị suy hao đến 23%.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** tuần hoàn decode $\to$ encode qua VAE qua từng frame.
- Luôn giữ pixel video gốc ở không gian RGB nguyên bản và dùng **Feathered Alpha Blending** ở không gian pixel để bảo vệ vùng đã biết khỏi sai số nén của VAE.

---

### [PIT-05] Accumulated Inpaint Hole Without Memory (Lỗi Đột Biến Ngữ Nghĩa V1)
- **Thời điểm ghi nhận**: 2026-09-18 (Bản `v1 baseline` 3000 steps).
- **Triệu chứng trực quan**: Đến Frame 17 ($\theta \approx -21^\circ$), tủ gỗ tự nhiên đột biến thành một cái cửa sổ.

#### Phân tích Nguyên nhân Thuật toán:
- Tại mỗi frame $t$, hố inpaint được tính tích lũy từ góc $0^\circ \to \theta_t$, độ rộng hố mở rộng lên tới 207 pixel (chiếm hơn 36% màn hình).
- Vì toàn bộ 207 pixel này bị che đen trong $z_{masked}$ và $M_t = 1$, SDXL không có bất kỳ thông tin nào về phần tủ gỗ mà nó đã sinh thành công ở Frame 1 đến 16.
- Đối mặt với một khoảng trống khổng lồ 207 pixel và nhiễu ngẫu nhiên, mô hình tự do tưởng tượng ra một cấu trúc mới (cửa sổ) phù hợp với ngữ cảnh phòng.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** cho phép hố inpaint tích lũy nở rộng vượt quá bước nhảy tương đối $\Delta\theta$ của một frame.
- Luôn chiếu lại phần đã sinh ở frame trước thành vùng đã biết ($M_t = 0$) và khóa cứng bằng **Hard Latent Clamping**.
- Độ rộng hố inpaint ở mọi frame $t \ge 1$ **BẮT BUỘC PHẢI ĐƯỢC GIỚI HẠN Ở MỨC $\approx 12 - 13$ PIXEL**.

---

### [PIT-06] Prompt Hardcoding / Cheating (Lỗi Chữa Cháy Phi Khoa Học)
- **Thời điểm ghi nhận**: 2026-09-18.
- **Triệu chứng**: Gán cứng chuỗi văn bản `"wooden kitchen cabinet"` vào text conditioning để ép mô hình vẽ tủ gỗ.

#### Phân tích Nguyên nhân Phương pháp luận:
- Đây là thủ thuật đánh lừa (heuristic cheat), hoàn toàn phá vỡ tính tổng quát của một nghiên cứu khoa học thị giác máy tính.
- Khi người dùng đưa một video phòng khách, phòng giặt, ngoài trời hay hành lang vào, prompt cứng sẽ gây xung đột ngữ nghĩa nghiêm trọng.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** sửa chữa hay hardcode prompt văn bản để giải quyết vấn đề nhất quán hình học.
- Mọi giải pháp về tính nhất quán video **PHẢI HOÀN TOÀN DỰA TRÊN THỊ GIÁC MÁY TÍNH VÀ HÌNH HỌC CHIẾU 3D (Computer Vision & 3D Projective Geometry)**.

---

### [PIT-07] Autoregressive Slit-Scan Extrapolation / Sub-Latent Border Smearing (Lỗi Kéo Giãn Biên Nan Quạt Do Inpaint Khe Hẹp Tuần Tự)
- **Thời điểm ghi nhận**: 2026-09-19 (Video `sota_3d_before_training.mp4` và `sota_3d_after_training.mp4`, ảnh người dùng tải lên `media_1789796780245.png`).
- **Triệu chứng trực quan**:
  - Tại phần biên lộ diện (bên trái khung hình khi camera pan sang phải): xuất hiện một chuỗi bậc thang gồm đúng 24 dải/lát cắt thẳng đứng (stepped concertina / slit-scan mirror) lặp lại chính xác khung cửa, giá để đồ của frame trước đó, trải dài hơn 150 pixel.
  - Bản trước khi train (`sota_3d_before_training.mp4`): các lát cắt cửa và kệ đồ bị nhân bản lặp lại nguyên xi 24 lần.
  - Bản sau khi train (`sota_3d_after_training.mp4`): các lát cắt bị làm mờ thành các vạch sọc dọc màu nâu/xám đậm.

#### Phân tích Nguyên nhân Toán học & Bản chất Thuật toán:
1. **Nghịch lý Trường Thụ Nhận (Receptive Field) & Khe Hở Sub-Latent**:
   - SDXL hoạt động trong không gian latent bị nén $8\times$ ($f=8$).
   - Một bước quay góc nhỏ $\Delta\theta \approx 1.25^\circ$ tạo ra khe hở disocclusion chỉ rộng $\Delta W \approx 12$ pixel.
   - Trong không gian latent: $\Delta W_{latent} = 12 / 8 = 1.5$ latent pixels!
   - Kích thước kernel tích chập cơ bản của UNet là $3 \times 3$ latents ($24 \times 24$ RGB pixels).
   - Khi hố inpaint chỉ hẹp $1.5$ latent nằm sát mép ngoài cùng của tensor:
     - Phía ngoài mép là zero-padding.
     - Phía trong là 98% diện tích ảnh đã bị khóa cứng (known ground-truth).
     - Trong điều kiện biên cực hẹp này, nghiệm khử nhiễu có năng lượng cực tiểu (minimum-energy solution) của mô hình diffusion bắt buộc phải là **Texture Continuation / Boundary Clamping** (sao chép tiếp tục màu sắc và đường nét của pixel biên liền kề). Mô hình hoàn toàn **KHÔNG CÓ ĐỦ KHÔNG GIAN HÌNH HỌC (receptive field)** để khởi tạo và kiến tạo nên một thực thể 3D mới (như bức tường mới, phòng mới, hay cửa mới).
2. **Sai số Nhân dồn Tự Hồi Quy (Autoregressive Compounding Error)**:
   - Frame 1: Mô hình ngoại suy mép của Frame 0 thêm 12px.
   - Frame 2: Mép 12px vừa ngoại suy này bị warp sang tọa độ mới, và mô hình lại tiếp tục ngoại suy từ chính mép đó thêm 12px nữa.
   - Qua 24 frames tuần tự: Mép ban đầu bị nhân bản và tịnh tiến 24 lần, tạo thành hiệu ứng máy quét rãnh hẹp (**Slit-Scan Camera / Concertina Accordion Blinds**).
   - Công thức suy thoái:
     $$\text{Extrapolate}_{12px}(\text{Extrapolate}_{12px}(\dots(\text{Extrapolate}_{12px}(I_0)))) \equiv \text{Accordion Slice Artifact}$$
3. **Vô hiệu hóa Temporal Attention đã Huấn luyện**:
   - Adapter được huấn luyện trên chuỗi clip $T=16$ frames với `InterleavedTemporalAttention` để duy trì liên kết không gian - thời gian xuyên suốt các frames.
   - Khi chạy inference tuần tự từng frame với `num_frames=1`, toàn bộ các tầng temporal attention bị bypass (`if num_frames > 1`), ép từng frame phải giải quyết bài toán inpaint 2D tĩnh cô lập.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** ép mô hình diffusion inpaint tuần tự trên một khe hẹp sub-latent ($12$ pixel) ở biên khung hình qua từng frame đơn lẻ.
- **TUYỆT ĐỐI KHÔNG** khóa cứng (clamp) từng dải 12 pixel một cách mù quáng vào keyframe trước đó rồi lại dùng nó làm gốc để ngoại suy tiếp sang frame sau.

#### GIẢI PHÁP CHUẨN MỰC KHOA HỌC (SCIENTIFIC SOTA SOLUTION):
Áp dụng **Full-Disocclusion Batch Spatio-Temporal Diffusion** (như TrajectoryCrafter / ViewCrafter):
1. **Chiếu 3D Trực tiếp Toàn bộ Chuỗi từ Frame 0 (Full Horizon Warping)**:
   Toàn bộ $T$ frames ($T=25$) được chiếu 3D trực tiếp từ Frame 0:
   $$I_{warped}^{(t)}, M_{0 \to t} = \text{Warp3D}(I_0, D_0, c2w_0 \to c2w_t, K)$$
   Hố inpaint $M_{0 \to t}$ tại Frame $t$ là vùng disocclusion thực sự (rộng tự nhiên từ $0 \to 150+$ pixel), cung cấp đầy đủ không gian receptive field để SDXL kiến tạo cấu trúc 3D hoàn chỉnh thay vì chỉ sao chép mép.
2. **Khử nhiễu Đồng thời Toàn bộ $T$ Frames (Batch Video Diffusion)**:
   Tensor `latents` có kích thước $[T, 4, H/8, W/8]$ được khử nhiễu đồng thời trong một chu trình khuếch tán duy nhất.
3. **Kích hoạt Toàn bộ Sức mạnh của `InterleavedTemporalAttention`**:
   Truyền `num_frames=T` vào Adapter. Các tầng Temporal Attention sẽ kết nối các token của Frame $t$ (với hố lớn) với Frame 1 và Frame 0, đảm bảo tính nhất quán thời gian tuyệt đối mà không có bất kỳ lát cắt nào!

---

### [PIT-08] Independent Noise Mode Switching & 2D UNet Inter-Frame Morphing (Lỗi Biến Dạng Ngữ Nghĩa Từng Frame)
- **Thời điểm ghi nhận**: 2026-09-19 (Video `1.mp4` và `2.mp4` sau khi train 5000 steps).
- **Triệu chứng trực quan**:
  - `1.mp4` (chưa train / Step 0): Hố inpaint disocclusion lớn ở bên trái hallucinate ra vật thể lạ hoàn toàn không thuộc miền dữ liệu (xe robot màu vàng viễn tưởng ở Frame 24).
  - `2.mp4` (đã train 5000 steps trên RealEstate10K): Đã học tốt miền dữ liệu nội thất (indoor priors), các vật thể sinh ra đều rất khớp với không gian phòng (tường, viền gỗ, gương, cửa, gạch ốp).
  - **VẤN ĐỀ CỐT LÕI**: Giữa các frame, phần inpaint liên tục bị biến dạng kiểu vẽ (semantic morphing): Frame 8 vẽ gương soi phòng tắm, Frame 12 vẽ tường viền gỗ, Frame 16 vẽ ô cửa tối, Frame 20 vẽ tường phẳng, Frame 24 vẽ gạch men phòng tắm. Không có sự nhất quán về kiểu vẽ hay cấu trúc giữa các frame.

#### Phân tích Nguyên nhân Toán học & Kiến trúc:
1. **Khởi tạo Nhiễu Gaussian Độc lập ($\epsilon_t \sim \mathcal{N}(0, I)$ i.i.d.)**:
   - Tại mỗi frame $t$, tensor nhiễu được lấy mẫu độc lập: $\mathbb{E}[\epsilon_t \epsilon_{t'}^T] = 0$ khi $t \neq t'$.
   - Không gian latent của mô hình khuếch tán đa phương thức (multimodal landscape). Khi đối mặt với một khoảng trống lớn (50-140px) mà không có ràng buộc pha tần số thấp chung, mỗi hạt giống nhiễu ngẫu nhiên $\epsilon_t$ độc lập sẽ rơi vào một cực tiểu cục bộ (attractor basin) khác nhau của mô hình (Frame 8 rơi vào mode "gương", Frame 24 rơi vào mode "gạch men").
2. **Xương sống UNet là Mô hình 2D Tĩnh (Thiếu Cross-Frame Attention trong Denoising)**:
   - Trọng tâm khử nhiễu là `UNet2DConditionModel`. Trong suốt 25 bước khuếch tán, các lớp Self-Attention (`attn1`) ở Down-blocks, Mid-block và Up-blocks chỉ tính tích vô hướng $Q_t K_t^T$ giữa các token trong CÙNG MỘT FRAME $t$.
   - Token ở Frame 24 hoàn toàn KHÔNG THỂ nhìn thấy (attend) token của Frame 0 (nơi có chất liệu gỗ và ánh sáng thật) hay Frame $t-1$.
3. **Lỗ hổng Điều kiện Hố Lộ diện (Disocclusion Conditioning Gap)**:
   - Do `base_tensor` được warp từ Frame 0, vùng hố lộ diện ở Frame $t-1$ và $t-2$ đều là pixel đen (zeros). Mạng STCondNet trong Adapter không nhận được thông tin vân bề mặt nào trong hố từ các frame trước.

#### ĐIỀU CẤM KỴ (NEVER-DO INVARIANTS):
- **TUYỆT ĐỐI KHÔNG** khởi tạo nhiễu Gaussian i.i.d. hoàn toàn độc lập cho các frame trong bài toán sinh video novel view / inpaint hố lớn khi UNet không có temporal cross-attention.
- **TUYỆT ĐỐI KHÔNG** để các tầng Self-Attention 2D của UNet chạy cô lập từng frame mà không cho truy vấn (attend) sang Frame mỏ neo (Frame 0).

#### GIẢI PHÁP CHUẨN MỰC KHOA HỌC (SCIENTIFIC SOTA SOLUTION - ZERO-SHOT / TRAINING-FREE):
1. **Temporally Coupled Noise Initialization (FreeInit-inspired, bảo toàn 100% PIT-02)**:
   Khởi tạo nhiễu liên kết thời gian thông qua thành phần không gian tần số chung:
   $$\epsilon_t = \sqrt{1 - \rho^2} \epsilon_t^{indep} + \rho \epsilon^{base}, \quad \rho \in [0.75, 0.85]$$
   với $\epsilon^{base} \sim \mathcal{N}(0, I)$ và $\epsilon_t^{indep} \sim \mathcal{N}(0, I)$.
   Về mặt toán học: $\text{Var}(\epsilon_t) = (1 - \rho^2) + \rho^2 = 1.0$. Phổ năng lượng Fourier vẫn là 100% nhiễu trắng (Flat Spectrum), tuân thủ triệt để PIT-02 (không mờ bệt), nhưng buộc toàn bộ chuỗi video rơi vào đúng MỘT cực tiểu ngữ nghĩa duy nhất.
2. **Cross-Frame Extended Self-Attention Hook (TokenFlow / Tune-A-Video / FateZero style)**:
   Cài đặt bộ xử lý `ExtendedSelfAttentionProcessor` vào toàn bộ các tầng `attn1` của SDXL UNet:
   $$K'_{ext} = [K_0, K_{t-1}, K_t], \quad V'_{ext} = [V_0, V_{t-1}, V_t]$$
   $$\text{Attn}(Q_t, K'_{ext}, V'_{ext}) = \text{Softmax}\left(\frac{Q_t (K'_{ext})^T}{\sqrt{d}}\right) V'_{ext}$$
   - **Frame 0**: Đóng vai trò Mỏ neo Toàn cục (Global Anchor), cung cấp trực tiếp Key/Value về màu sắc, vân gỗ, chất liệu và ánh sáng thực của căn phòng.
   - **Frame $t-1$**: Đóng vai trò Ràng buộc Cục bộ (Local Temporal Continuity), triệt tiêu rung giật vi mô giữa 2 frame kề nhau.

---

## TIÊU CHÍ KIỂM TRA BẮT BUỘC TRƯỚC KHI CHẠY (PRE-FLIGHT CHECKLIST)

Trước khi đóng gói mã nguồn hoặc hướng dẫn người dùng chạy bất kỳ phiên bản nào, Agent **BẮT BUỘC PHẢI CHẠY SCRIPT MÔ PHỎNG VÀ ĐẠT TẤT CẢ CÁC ĐIỀU KIỆN SAU**:

1. [ ] **Mặt nạ đầy đủ (Full Binary Mask)**: Tuyệt đối không có `ones_inset` hay gọt biên 1px.
2. [ ] **Không có sọc đen trong vùng hợp lệ**: Script mô phỏng kiểm tra `np.sum(base[h//2, :valid_w] == 0) == 0`.
3. [ ] **Độ rộng hố inpaint chuẩn xác**: Độ rộng hố $M_t$ tại đường giữa màn hình phải nằm trong khoảng $11 \le W_{hole} \le 14$ pixel trên mọi frame từ 1 đến 24.
4. [ ] **Bảo vệ Frame 0**: Tại $t=0$, số pixel hố $M_0$ phải bằng **0 tuyệt đối**, xuất thẳng frame gốc không qua VAE/UNet.
5. [ ] **Khởi tạo nhiễu trắng**: `latents = torch.randn_like(z_0)`, không có bất kỳ thao tác warp nhiễu nào.
6. [ ] **Không tính loss thời gian trên nhiễu**: File `train_adapter.py` không chứa bất kỳ loss nào tính `warp(noise)`.
7. [ ] **Không hardcode prompt**: Conditioning văn bản giữ nguyên token rỗng hoặc trích xuất tự động từ video.
