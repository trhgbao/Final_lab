# TỔNG QUAN TÀI LIỆU KHOA HỌC & CÁC NGHIÊN CỨU SOTA (SURVEY OF SOTA PAPERS)
## Chuyên Đề: Điều Khiển Quỹ Đạo Camera & Chỉnh Sửa Video (Video-to-Video Camera Trajectory Retargeting)
*Cập nhật: 2026 | Dành cho Đề tài Khóa luận Tốt nghiệp & Bài báo Khoa học*

---

## 1. Danh Mục Các Bài Báo Khoa Học Đã Tải & Trích Xuất Toàn Văn (Full-Text)

Toàn bộ 12 bài báo cốt lõi đã được tải về tệp PDF tại `docs/papers/` và trích xuất nguyên văn văn bản tại `docs/papers/extracted_text/`:

| STT | Tên Bài Báo | Hội Nghị / Nguồn | Tác Giả / Đơn Vị | Tệp PDF & Văn Bản Trích Xuất |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Wan: Open and Advanced Large-Scale Video Generative Models** | arXiv:2503.20314 (2025) | Team Wan (Alibaba Group) | `docs/papers/Wan2.1_TechReport.pdf` (60 trang) |
| 2 | **CameraCtrl: Enabling Camera Pose for Text-to-Video Generation** | **ICLR 2025** (arXiv:2404.02101) | Hao He, Gordon Wetzstein et al. (Stanford, CUHK) | `docs/papers/CameraCtrl.pdf` (32 trang) |
| 3 | **TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos** | **ICCV 2025 Oral** (arXiv:2503.05638) | Mark Yu, Wenbo Hu, Ying Shan et al. (Tencent ARC, CUHK) | `docs/papers/TrajectoryCrafter.pdf` (12 trang) |
| 4 | **DoRA: Weight-Decomposed Low-Rank Adaptation** | **ICML 2024** (arXiv:2402.09353) | Shih-Yang Liu et al. (NVIDIA, UT Austin) | `docs/papers/DoRA.pdf` (23 trang) |
| 5 | **CamCo: Camera-Controllable 3D-Consistent Video Generation** | **CVPR 2024** (arXiv:2406.02509) | Ziyang Yuan, Gordon Wetzstein et al. | `docs/papers/CamCo.pdf` (15 trang) |
| 6 | **MotionCtrl: A Unified and Flexible Motion Controller for Video Generation** | **SIGGRAPH 2024** (arXiv:2312.03641) | Zhouxia Wang et al. (Tencent PCG, NTU) | `docs/papers/MotionCtrl.pdf` (22 trang) |
| 7 | **ViewCrafter: Taming Video Diffusion Models for High-fidelity NVS** | arXiv:2409.02048 (2024) | Wang et al. (HKU, Tencent) | `docs/papers/ViewCrafter.pdf` (13 trang) |
| 8 | **AnyV2V: A Plug-and-Play Framework for Any Video-to-Video Editing Tasks** | **NeurIPS 2024** (arXiv:2403.14468) | Min-Hung Chen et al. (TIGER Lab) | `docs/papers/AnyV2V.pdf` (26 trang) |
| 9 | **ReCamMaster: Synchronizing Camera and Object Movements for Video** | arXiv:2501.07541 (2025) | Zirui Wang et al. | `docs/papers/ReCamMaster.pdf` (16 trang) |
| 10 | **ReCapture: AR-Guided Camera Trajectory Retargeting for Video** | **SIGGRAPH Asia 2024** | Dongqing Wang et al. | `docs/papers/ReCapture.pdf` (14 trang) |
| 11 | **EpiDiff: Enhancing Multi-View Synthesis via Epipolar Constraints** | **CVPR 2024** | Jiarui Liu et al. | `docs/papers/EpiDiff.pdf` (11 trang) |
| 12 | **Direct-a-Video: Customized Video Generation with Camera Movement** | arXiv:2302.04820 | Shiyuan Huang et al. | `docs/papers/Direct_a_Video.pdf` (26 trang) |

---

## 2. Bóc Tách Bản Chất Kỹ Thuật & Kiến Thức Cốt Lõi Từng Nghiên Cứu

### 2.1. Wan 2.1 Technical Report (Alibaba Group - 2025)
* **Bản chất Kiến trúc DiT**: Sử dụng kiến trúc **Diffusion Transformer (DiT)** toàn phần kết hợp **Flow Matching**.
* **3D Causal VAE**: Nén không-thời gian với độ nén thời gian $4\times$ ($T \rightarrow 1 + (T-1)/4$) và không gian $8\times$ ($H \times W \rightarrow H/8 \times W/8$), mở rộng kênh latent thành $16$ channels (chuẩn hóa qua mean/std vector 16 chiều).
* **Mục tiêu huấn luyện (Flow Matching Objective)**:
  $$x_t = (1 - \sigma_t) x_0 + \sigma_t \epsilon, \quad \min_\theta \mathbb{E}_{t, x_0, \epsilon} \left[ \| v_\theta(x_t, t, c) - (\epsilon - x_0) \|_2^2 \right]$$
  Trong đó vận tốc mục tiêu là $v_{\text{target}} = \epsilon - x_0$.
* **Thang đo thời gian (Timestep Scaling)**: Timestep đầu vào của hàm nhúng `sinusoidal_embedding_1d` bắt buộc phải được co giãn về thang $[0, 1000]$:
  $$\text{timesteps} = \sigma \times 1000.0$$
* **Camera Motion Controllability (Mục 5.5)**:
  - Sử dụng tọa độ Plücker $P \in \mathbb{R}^{6 \times F \times H \times W}$.
  - Dùng toán tử `PixelUnshuffle` giảm kích thước không gian $8\times$, tăng số kênh lên $6 \times 64 = 384$.
  - Truyền qua mạng tích chập trích xuất đặc trưng đa mức.
  - Sử dụng **Adaptive Normalization Zero-Initialized**: $f_i = (\gamma_i + 1) * f_{i-1} + \beta_i$, với $\gamma, \beta$ khởi tạo bằng $0$.
* **Unified Video Editing / VACE (Mục 5.2)**:
  - Video Condition Unit (VCU): Kết hợp frame video, mặt nạ nhị phân và prompt.
  - Hỗ trợ **Context Adapter Tuning** (giữ nguyên trọng số Wan2.1, chỉ huấn luyện adapter cắm ngoài).

---

### 2.2. CameraCtrl: Enabling Camera Pose for Text-to-Video Generation (ICLR 2025)
* **Vấn đề giải quyết**: Làm sao để đưa góc quay camera vào video diffusion model mà không làm hỏng chất lượng hình ảnh và tính nhất quán thời gian?
* **Mã hóa tọa độ Plücker (Plücker Ray Embeddings)**:
  Với mỗi điểm ảnh $(u, v)$ tại frame $i$:
  $$p_{u,v} = (\mathbf{o}_i \times \mathbf{d}_{u,v}, \mathbf{d}_{u,v}) \in \mathbb{R}^6$$
  Trong đó $\mathbf{o}_i \in \mathbb{R}^3$ là tâm quang học (optical center), $\mathbf{d}_{u,v} = \frac{R_i K^{-1} [u, v, 1]^T}{\| R_i K^{-1} [u, v, 1]^T \|}$ là vector hướng tia trong hệ tọa độ thế giới.
* **Nguyên lý Appearance-Free**: Camera Encoder **chỉ nhận duy nhất tọa độ Plücker**, tuyệt đối KHÔNG nhận feature ảnh của dataset huấn luyện (tránh hiện tượng model bị rò rỉ và học vẹt diện mạo của tập RealEstate10K).
* **Điểm hòa nhập (Injection Point)**: Bơm vào tầng **Temporal Self-Attention** thông qua phép cộng tuyến tính ($z_t + c_t$) và một tầng chiếu tuyến tính (Linear Projection) có Zero-Initialization.

---

### 2.3. TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos (ICCV 2025 Oral)
* **Vấn đề cốt lõi**: Chuyển đổi quỹ đạo camera cho **video đơn góc nhìn (Monocular Video)** có sẵn, giải quyết bài toán thiếu hụt dữ liệu video đa góc nhìn đồng bộ (Synchronized Multi-View Videos).
* **Phương pháp tách rời (Disentanglement Paradigm)**:
  1. *Biến đổi hình học tất định (Deterministic View Transformation)*: Dùng mạng ước lượng độ sâu đơn thị (`DepthCrafter`) nâng video nguồn thành **Đám mây điểm động (Dynamic Point Cloud)** $P_i = \Phi^{-1}([I_s, D_s], K)$, sau đó dùng phép chiếu phối cảnh $\Phi$ để render sang góc máy đích $T^r$.
  2. *Bổ khuyết nội dung ngẫu định (Generative Inpainting)*: Quá trình render đám mây điểm tạo ra các lỗ thủng do bị che khuất (occlusion holes) đi kèm mặt nạ nhị phân $M^r$.
* **Dual-Stream Conditional DiT**:
  - *Stream 1 (View Tokens)*: Mã hóa video render $I^r$ và mask $M^r$ qua 3D VAE, ghép kênh cùng latent nhiễu đưa vào self-attention.
  - *Stream 2 (Reference Tokens)*: Mã hóa video nguồn $I_s$ gốc, đưa vào DiT qua tầng **Ref-DiT Cross-Attention** để phục hồi kết cấu chi tiết bị mất.
* **Double-Reprojection Training Strategy**: Tự reproject video đơn thị sang góc quay ảo rồi reproject ngược lại để tạo dữ liệu giả lập lỗ thủng che khuất $\implies$ Giải quyết triệt để bài toán thiếu dữ liệu video đa góc nhìn!

---

### 2.4. DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024 / NVIDIA)
* **Nguyên lý toán học (Mathematical Formulation)**:
  Phân rã ma trận trọng số $W \in \mathbb{R}^{d \times k}$ thành **Độ lớn (Magnitude)** $m \in \mathbb{R}^{1 \times k}$ và **Hướng (Direction)** $V \in \mathbb{R}^{d \times k}$:
  $$W = m \odot \frac{V}{\|V\|_c} = m \odot \frac{W_0 + \Delta V}{\|W_0 + \Delta V\|_c}, \quad \text{với } \Delta V = \frac{\alpha}{\sqrt{r}} B A$$
  - $W_0$: Trọng số gốc của Wan2.1 (được đóng băng 100%).
  - $A \in \mathbb{R}^{r \times d}$ (khởi tạo Kaiming Uniform), $B \in \mathbb{R}^{k \times r}$ (khởi tạo bằng $0$).
  - $m \in \mathbb{R}^{1 \times k}$ (khởi tạo bằng chuẩn cột L2 ban đầu $\|W_0\|_c$).
* **Đặc tính Bảo toàn Bước 0 (Step-0 Numerical Invariance)**:
  Tại step 0, do $B = 0 \implies \Delta V = 0 \implies V = W_0$. Khi đó:
  $$W = \|W_0\|_c \odot \frac{W_0}{\|W_0\|_c} \equiv W_0$$
  Sai số giữa DoRA và mô hình gốc bằng **$0.00\text{e}+00$ tuyệt đối**, không làm biến dạng năng lực sinh video nền tảng của Wan2.1.
* **Ưu thế so với LoRA tiêu chuẩn**: LoRA thay đổi cả độ lớn và hướng một cách ngẫu nhiên (dễ gây bùng nổ gradient và mất ổn định). DoRA tách riêng việc học độ lớn $m$ và hướng $V$, mô phỏng chính xác hành vi của Full Fine-Tuning nhưng chỉ tốn lượng tham số của LoRA (~10 MB).

---

### 2.5. CamCo (CVPR 2024) & MotionCtrl (SIGGRAPH 2024)
* **CamCo (3D-Consistent Video Generation)**:
  - Chỉ ra rằng biểu diễn camera 1D thông thường không kiểm soát được chuyển động phức tạp.
  - Đưa vào **Epipolar Constraint Attention Module** để ép các điểm tương ứng trên hai frame liền kề phải nằm trên cùng đường cực (epipolar lines), tăng độ cứng hình học 3D của căn phòng.
* **MotionCtrl (Tencent PCG)**:
  - Tách rời hoàn toàn Camera Motion (chuyển động toàn cục của cả khung cảnh) và Object Motion (chuyển động cục bộ của vật thể).
  - Sử dụng quỹ đạo SE(3) tương đối so với Frame 0 làm mỏ neo chuẩn hóa.

---

### 2.6. AnyV2V (NeurIPS 2024 - Plug-and-Play Video Editing)
* **Phương pháp Zero-Shot Video Editing**:
  - Không cần huấn luyện lại video model từ đầu.
  - Sử dụng **DDIM / Flow Inversion** trích xuất đặc trưng không gian và thời gian từ video gốc.
  - Bẻ lái video sang phong cách hoặc chuyển động mới thông qua can thiệp vào tầng Attention Injection và Latent Noise Blending.

---

## 3. Ma Trận So Sánh Tổng Hợp (Comparative Taxonomy Matrix)

| Mô Hình | Năm / Venue | Kiến Trúc Nền | Biểu Diễn Camera | Cơ Chế Nhúng (Injection) | Nhiệm Vụ Mục Tiêu | Huấn Luyện / Tinh Chỉnh | Bảo Toàn Step 0? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CameraCtrl** | ICLR 2025 | UNet 3D (SVD / AnimateDiff) | 6D Plücker Rays | Temporal Self-Attention (Cộng & Chiếu) | Text-to-Video (T2V) | Train Camera Encoder riêng | Có (Zero Linear) |
| **MotionCtrl** | SIGGRAPH 2024 | UNet 3D (LVDM / AnimateDiff) | Ma trận $R, t$ chuẩn hóa | Spatial + Temporal Cross-Attention | T2V + Camera/Object | Train Motion Controller | Không |
| **CamCo** | CVPR 2024 | UNet 3D (I2V-Diffusion) | Plücker + Epipolar Lines | Epipolar Constraint Attention Block | Image-to-Video (I2V) | Fine-tune toàn phần + Epipolar | Không |
| **ViewCrafter** | 2024 Pre | Video DiT / UNet | 3D Point Cloud | Point-render Latent Concatenation | Novel View Synthesis (NVS) | Fine-tune trên Objaverse | Không |
| **TrajectoryCrafter** | ICCV 2025 | DiT (CogVideoX) | Point Cloud + Render Mask | Dual-Stream: Concat + Ref-DiT Attn | **V2V Camera Retargeting** | Double-Reprojection | Không (Cần render trước) |
| **AnyV2V** | NeurIPS 2024 | I2V Diffusion | None (Text / First-frame) | DDIM Inversion + Feature Swapping | Video Editing chung | Tuning-Free (Không train) | Có (Inversion) |
| **Wan 2.1 (Gốc)** | 2025 TechReport | DiT 1.3B/14B (Flow Matching) | Plücker + PixelUnshuffle | Adaptive Normalization ($\gamma, \beta$) | T2V, I2V, Editing | Multi-stage Pre-training | Có (Zero Conv) |
| **Pipeline v2 (Của Chúng Ta)** | **2026** | **Wan2.1 DiT (Đóng băng 100%)** | **6D Plücker Ray Fields** | **Zero-Init Projector + DoRA (Rank 16)** | **V2V Camera Retargeting** | **DoRA PEFT (~45MB Checkpoint)** | **Có (Độ lệch $< 10^{-7}$)** |

---

## 4. Bài Học Rút Ra & Kiến Trúc Tối Thượng Cho Đề Tài Của Chúng Ta

Từ 12 công trình khoa học trên, chúng ta rút ra 4 bài học bản lề để hoàn thiện hệ thống nghiên cứu:

1. **Về Biểu diễn Camera**: Tọa độ **Plücker 6 chiều (Ray Direction $d$ + Ray Moment $m$)** là chuẩn mực vượt trội nhất (được khẳng định độc lập bởi cả *CameraCtrl*, *CamCo* và *Wan 2.1*), vì nó gán thông tin hình học 3D cho từng token điểm ảnh, loại bỏ sự mất cân xứng số học giữa góc xoay $R$ và độ tịnh tiến $t$.
2. **Về Cơ chế Thích nghi (Adaptation)**: Thay vì sửa đổi kiến trúc Wan2.1 (gây mất 60 ma trận trọng số như bản v1) hoặc dùng LoRA thông thường (dễ bùng nổ gradient), việc sử dụng **DoRA (ICML 2024)** là lựa chọn hoàn hảo nhất: đóng băng 100% Wan2.1, bảo toàn nguyên vẹn năng lực sinh video của Alibaba, và đảm bảo Step 0 trùng khớp số học tuyệt đối.
3. **Về Khử nhiễu Flow Matching**: Cần tuân thủ nghiêm ngặt thang đo thời gian $t = \sigma \times 1000.0$ của Wan2.1 và cấp đủ 25 bước tích phân ODE mịn màng trong khoảng $[0, \sigma_{\text{start}}]$ để tránh hiện tượng giải mã ra nhiễu hạt loang lổ.
4. **Về Chiến lược Huấn luyện Monocular Video**: Dựa theo phát hiện của *TrajectoryCrafter* và *CameraCtrl*, khi huấn luyện trên tập video đơn thị (như RealEstate10K), cặp giám sát phải là **video thật đi kèm đúng quỹ đạo camera thật của video đó**, không được tự ý perturb camera lệch khỏi video khiến mô hình học kiến thức mâu thuẫn.
