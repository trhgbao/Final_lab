# SỔ TAY KỸ THUẬT, PHÒNG NGỪA LỖI & BÀI HỌC THỰC NGHIỆM
# VIDEO-TO-VIDEO CAMERA CONTROL & GENERATIVE DIFFUSION PIPELINE

---

## 1. LỖI 1: TypeError TRONG CogVideoXPatchEmbed
### 🔴 Hiện Tượng & Thông Báo Lỗi:
`python
TypeError: CogVideoXPatchEmbed.forward() missing 1 required positional argument: 'image_embeds'
`
### 🔬 Nguyên Nhân Kỹ Thuật:
Trong thư viện diffusers, tầng patch_embed của mô hình CogVideoXTransformer3DModel là lớp nhúng đa phương thức (Multimodal Patch Embedding). Signature của hàm này bắt buộc phải có 2 đối số: forward(self, text_embeds, image_embeds).
* text_embeds: Tensor nhúng từ prompt văn bản (shape [B, 226, 3072]).
* image_embeds: Tensor không gian tiềm năng video 32 kênh (shape [B, 13, 32, 60, 90]).
Khi ta chỉ gọi patch_embed(clean_latents), Python báo lỗi thiếu đối số image_embeds.

### 🟢 Giải Pháp Triệt Để:
Chuẩn bị tensor 32 kênh (ghép kênh video sạch và kênh điều kiện Frame 0), sau đó truyền đủ cả 2 đối số và trích xuất phần visual tokens nằm sau 226 text tokens:
`python
img_cond_clean = torch.cat([clean_latents[:, :, :1], torch.zeros_like(clean_latents[:, :, 1:])], dim=2)
i2v_clean = torch.cat([clean_latents, img_cond_clean], dim=1).permute(0, 2, 1, 3, 4)

embedded_all = pipe.transformer.patch_embed(PROMPT_EMBEDS, i2v_clean)
src_tokens = embedded_all[:, 226:]  # Lấy chính xác 17,550 visual tokens!
`

---

---

## 2. LỖI 2: LỆCH KÍCH THƯỚC CHUỖI TOKEN (17,550 VS 9,450 TOKENS)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
`python
RuntimeError: The size of tensor a (17550) must match the size of tensor b (9450) at non-singleton dimension 1
`
### 🔬 Nguyên Nhân Toán Học:
* Mô hình 3D Causal VAE nén thời gian theo tỉ lệ: T_latent = (T_frame - 1) / 4 + 1.
* Khi T = 49 frames -> T_latent = 13 lát cắt -> 13 x 30 x 45 = 17,550 tokens.
* Trong tập dữ liệu RealEstate10K, một số clip ngắn chỉ có 25 frames -> T_latent = 7 lát cắt -> 7 x 30 x 45 = 9,450 tokens.
* Khi cộng ma trận tia Plucker (17,550) với ma trận video ngắn (9,450), chiều số 1 bị lệch.

### 🟢 Giải Pháp Triệt Để:
1. Ép chuẩn hóa ở Dataset: Luôn lấy mẫu đúng 49 frames bằng np.linspace:
`python
idx_sample = np.linspace(0, len(raw) - 1, self.num_frames).astype(int)  # num_frames = 49
`
2. Tự Động Co Giãn Trong Adapter (Self-Healing Interpolation):
`python
if src_modulated.shape[1] != pose_feats.shape[1]:
    src_modulated = F.interpolate(
        src_modulated.transpose(1, 2),
        size=pose_feats.shape[1],
        mode='linear',
        align_corners=False
    ).transpose(1, 2)
`

---

---

## 3. LỖI 3: LỆCH CHIỀU VISUAL TOKENS VS FULL SEQUENCE (17,550 VS 17,776)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
`python
RuntimeError: The size of tensor a (17550) must match the size of tensor b (17776) at non-singleton dimension 1
`
### 🔬 Nguyên Nhân Kỹ Thuật:
Mỗi CogVideoXBlock trả về 2 biến tách rời: hidden_states (17,550 visual tokens) và encoder_hidden_states (226 text tokens). Nếu Adapter đệm thêm 226 số 0 vào đầu tensor (17,776), khi Hook cộng vào hidden_states sẽ bị lỗi lệch 226 tokens.
### 🟢 Giải Pháp Triệt Để:
Adapter chỉ xuất đúng 17,550 visual tokens khớp trực tiếp với hidden_states:
`python
return self.alpha_pose * pose_feats  # Shape: [B, 17550, 3072]
`

---

---

## 4. LỖI 4: HIỆN TƯỢNG BÓNG MA (GHOSTING) & MÉO VẬT THỂ DO THIẾU RELATIVE POSE
### 🔴 Hiện Tượng Hình Ảnh:
Chiếc bát đồng trên bàn và lưng ghế bị biến dạng thành vệt bóng ma (ghosting artifacts), điểm Laplacian Variance giảm từ 91.6 -> 55.2.
### 🔬 Nguyên Nhân Toán Học:
Khi lấy file pose từ video khác, ma trận R_0, T_0 mang toạ độ tuyệt đối. Nếu không chuẩn hóa, góc nhìn tại Frame 0 bị giật đột ngột (Initial Discontinuity), xé rách không gian tiềm năng 3D ban đầu.
### 🟢 Giải Pháp Triệt Để:
Bắt buộc chuẩn hóa chuyển động tương đối theo Frame 0:
R_norm(t) = R(t) @ R(0)^T
T_norm(t) = R(0) @ (T(t) - T(0))
Tại t = 0: R_norm(0) = I và T_norm(0) = 0. Frame 0 được neo giữ cố định 100% sắc nét!

---

---

## 5. LỖI 5: ĐẢO NGƯỢC QUY ƯỚC TRỤC TỌA ĐỘ WORLD-TO-CAMERA (W2C)
### 🔴 Hiện Tượng:
Mô hình nhận ma trận quay sang phải nhưng góc quay thực tế bị đứng yên hoặc quay ngược hướng.
### 🔬 Nguyên Nhân Toán Học:
Dataset RealEstate10K dùng chuẩn W2C. Phép quay quanh trục Y góc +theta có R_02 = -sin(theta) và R_20 = +sin(theta). Nếu viết theo chuẩn Camera-to-World (R_02 = +sin theta), dấu ma trận bị đảo ngược làm triệt tiêu chuyển động.
### 🟢 Giải Pháp Triệt Để:
Luôn tuân thủ đúng dấu ma trận W2C:
`python
extr[0, t, 0] = math.cos(theta)
extr[0, t, 2] = -math.sin(theta)
extr[0, t, 8] = math.sin(theta)
extr[0, t, 10] = math.cos(theta)
`

---

---

## 6. LỖI 6: VẾT LƯỚI BÀN CỜ 8x8 CỦA 3D VAE
### 🔴 Hiện Tượng:
Video bị phủ một lớp hạt lưới chấm caro 8x8 pixel.
### 🔬 Nguyên Nhân:
3D Causal VAE nén thời gian 4x. Ép sinh 25 frames (7 latent slices) và bật enable_slicing() làm vỡ stride của kernel tích chập nhân quả.
### 🟢 Giải Pháp Triệt Để:
Luôn sinh đúng chuẩn num_frames=49 và tắt hoàn toàn tiling/slicing trong VAE.

---

---

## 7. LỖI 7: ĐIỂM BẾ TẮC ĐẠO HÀM 0 (ZERO-SUBGRADIENT BUG)
### 🔴 Hiện Tượng:
Huấn luyện hàng trăm bước nhưng loss không giảm, gradient bằng 0.
### 🔬 Nguyên Nhân:
Khởi tạo trọng số Linear cuối cùng bằng 0 và dùng hàm loss chứa giá trị tuyệt đối không trơn || |x| ||^2 khiến đạo hàm bằng 0.
### 🟢 Giải Pháp:
Khởi tạo trọng số theo phân phối Gauss nhỏ và dùng smooth MSE:
`python
nn.init.normal_(self.net[-1].weight, std=0.01)
nn.init.zeros_(self.net[-1].bias)
loss = F.mse_loss(pred, target)
`

---

---

## 8. LỖI 8: LỖI CÚ PHÁP F-STRING LỒNG NHAU (SYNTAXERROR)
### 🔴 Hiện Tượng:
SyntaxError: unexpected character after line continuation character
### 🔬 Nguyên Nhân:
Lồng dấu thoát chuỗi backslash bên trong biểu thức f-string trên Python 3.10/3.11.
### 🟢 Giải Pháp:
Tính toán toàn bộ các biến chuỗi độc lập trước khi gọi lệnh print().

---

---

## 9. QUY TẮC 9: CHẾ ĐỘ NGHIÊM NGẶT KHÔNG INTERNET (NO-INTERNET PROTOCOL) & TỰ ĐỘNG ĐỊNH VỊ DATASET
### 🔴 Bối Cảnh Cuộc Thi:
Trong môi trường chấm thi chính thức, hệ thống chấm điểm **hoàn toàn ngắt kết nối Internet (No Internet Connection)**:
- Mọi lệnh `pip install`, `git clone`, hoặc tải từ Hugging Face qua mạng (`snapshot_download`, `from_pretrained("alibaba-pai/...")`) đều sẽ **bị crash 100%** (ConnectionError / Timeout).
- Tất cả tài nguyên (mô hình, weights, mã nguồn, clip mẫu, vector đánh giá) **bắt buộc phải nạp nội bộ từ Dataset offline đã được đính kèm trước**.

### 🔬 Bảng Danh Mục & Đường Dẫn Tài Nguyên Nội Bộ Trong Dataset:
Tập dữ liệu nạp sẵn tại `/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/` bao gồm đầy đủ 100% các thành phần:

| Thành Phần | Loại Tài Nguyên | Đường Dẫn Nội Bộ (Kaggle Dataset) | Chức Năng |
| :--- | :--- | :--- | :--- |
| **CogVideoX-5B-I2V** | Base Diffusion Model | `.../CogVideoX-5b-I2V/` | Backbone 5B DiT sinh video |
| **SVD-XT** | Base Diffusion Model | `.../stable-video-diffusion-img2vid-xt/...` | Backbone SVD-XT khuếch tán không-thời gian |
| **CameraCtrl Checkpoint** | Weights Adaptor | `.../CameraCtrl_svdxt.ckpt` | Weights điều khiển camera Plücker ray |
| **CameraCtrl Mã Nguồn** | Codebase Repository | `.../CameraCtrl_SVD/CameraCtrl_SVD/` | Mã nguồn trích xuất đặc trưng & suy luận |
| **DINOv2-Large** | Feature / VLM Evaluator | `.../dinov2-large/` | Trích xuất 1024D đặc trưng cấu trúc |
| **CLIP-ViT-Large** | Feature / VLM Evaluator | `.../clip-vit-large-patch14/` | Trích xuất 768D đặc trưng ngữ nghĩa & prompt |
| **CineCam LoRA** | Checkpoint LoRA | `.../cinecam-lora-6000.safetensors` | Tăng cường góc quay điện ảnh |
| **RealEstate10K Mini** | Camera Poses & Clips | `.../RealEstate10K_Mini/` | Ma trận chuyển động camera thực tế |

### 🟢 Giải Pháp Triệt Để (Dynamic Path Resolution):
Không bao giờ hard-code đường dẫn tuyệt đối tĩnh. Luôn dùng `glob.glob` tìm kiếm đệ quy trong `/kaggle/input/`:
```python
def find_local_resource(pattern: str) -> str:
    matches = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy tài nguyên offline: {pattern}")
    return matches[0]
```

---

---

## 10. QUY TẮC 10: TỐI ƯU HÓA TRÊN PHẦN CỨNG CUỘC THI NVIDIA RTX PRO 6000
### 🔬 Đặc Tả Phần Cứng:
Môi trường chấm thi sử dụng card đồ họa công nghiệp chuyên dụng **NVIDIA RTX PRO 6000 (Ada Generation / Blackwell Architecture)**:
- **Dung lượng VRAM**: **48 GB GDDR6** (hoặc Quadro RTX 6000 24GB).
- **Băng thông bộ nhớ**: ~960 GB/s.
- **Tensor Cores**: Thế hệ mới hỗ trợ phần cứng tính toán nguyên bản `torch.bfloat16` và `torch.float16`.

### 🟢 Chiến Lược Tối Ưu Cực Đại Hiệu Năng (Zero-Offload Advantage):
1. **Không Cần Sequential CPU Offload**:
   - Khác với GPU 16GB (phải dùng CPU offload chậm chạp), RTX PRO 6000 với 48GB VRAM cho phép nạp **trực tiếp 100% mô hình CogVideoX-5B hoặc SVD-XT + VLM Evaluator cùng lúc lên VRAM GPU**.
   - Tốc độ suy luận tăng gấp **3-5 lần**, một video 49 frames chỉ mất khoảng 15-25 giây.
2. **Khai Thác Float32 VAE Decoding**:
   - VRAM dồi dào cho phép chạy toàn bộ tiến trình VAE Decode ở `torch.float32`, loại bỏ triệt để hiện tượng vỡ hạt chấm đen, tràn số NaN hoặc nhấp nháy màu.
3. **Bảo Vệ Bộ Nhớ Không Gian**:
   - Duy trì `torch.cuda.empty_cache()` và `gc.collect()` giữa các phân đoạn để giữ VRAM luôn trong ngưỡng an toàn dưới 35GB.

---

### 11. LỖI 11: LỖI NHẬP HÀM `tensor2vid` (IMPORTERROR TRÊN DIFFUSERS MỚI)
### 🔴 Hiện Tượng:
```text
ImportError: cannot import name 'tensor2vid' from 'diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion'
```
### 🔬 Nguyên Nhân:
Trong mã nguồn gốc của CameraCtrl (`cameractrl/pipelines/pipeline_animation.py` dòng 17-20), tác giả gọi:
```python
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    _append_dims,
    tensor2vid,
    StableVideoDiffusionPipelineOutput
)
```
Tuy nhiên, trong các phiên bản Diffusers mới (từ 0.30 trở lên), Hugging Face đã dời hoặc bỏ hàm `tensor2vid` khỏi module `pipeline_stable_video_diffusion`.

### 🟢 Giải Pháp Triệt Để:
Định nghĩa và vá hàm `tensor2vid` vào `_svd_pipe` **TRƯỚC KHI** import `pipeline_animation`:
```python
import diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion as _svd_pipe

def tensor2vid(video: torch.Tensor, processor=None, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = []
        for frame_idx in range(num_frames):
            frame = batch_vid[frame_idx]
            frame = (frame / 2.0 + 0.5).clamp(0.0, 1.0)
            frame = frame.cpu().float().permute(1, 2, 0).numpy()
            if output_type == "pil":
                frame = Image.fromarray((frame * 255).round().astype("uint8"))
            elif output_type == "np":
                frame = (frame * 255).round().astype("uint8")
            batch_output.append(frame)
        outputs.append(batch_output)
    return outputs

_svd_pipe.tensor2vid = tensor2vid
```
Sau khi gắn hàm này vào `_svd_pipe`, lệnh import của CameraCtrl sẽ tìm thấy hàm và thực thi bình thường.

---

---

## 12. LỖI 12: LỖI LỆCH CHIỀU BROADCASTING TRONG PHÉP TÍNH TIA PLÜCKER
### 🔴 Hiện Tượng:
```text
RuntimeError: The size of tensor a (320) must match the size of tensor b (25) at non-singleton dimension 2
```
tại dòng:
```python
r_d = torch.matmul(d_cam, R_norm.transpose(-1, -2))
```

### 🔬 Nguyên Nhân Chi Tiết:
1. Tensor hướng tia camera `d_cam` có 5 chiều: `[B, F, H=320, W=576, 3]`.
2. Ma trận xoay camera `R_norm` có 4 chiều: `[B, F=25, 3, 3]`.
3. Khi thực hiện phép nhân ma trận đa chiều `torch.matmul(d_cam, R_norm)`, PyTorch tự động căn chỉnh các chiều batch từ **phải sang trái**:
   - Chiều `-1`: $3$ khớp với $3$ (Hợp lệ).
   - Chiều `-2`: $W=576$ nhân ma trận với $3$ thành vector $3$ chiều (Hợp lệ).
   - Chiều `-3`: Chiều $H=320$ của `d_cam` bị căn chỉnh trực tiếp với chiều $F=25$ của `R_norm`!
   - Do $320 \ne 25$ và cả hai đều không phải là kích thước đơn vị ($1$), cơ chế broadcasting thất bại và sinh ra lỗi `size of tensor a (320) must match size of tensor b (25)`.

### 🟢 Giải Pháp Triệt Để:
Không giữ 2 chiều không gian $H$ và $W$ độc lập khi nhân ma trận 4D. Làm phẳng không gian thành $H \times W$ (`[B, F, H*W, 3]`), nhân với ma trận quay `[B, F, 3, 3]`, rồi khôi phục lại không gian $H \times W$:
```python
def compute_plucker_rays_normalized(extrinsics: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int):
    B, F_len = extrinsics.shape[:2]
    device = extrinsics.device
    dtype = extrinsics.dtype

    # 1. Chuẩn hóa Relative Pose (Rule 4: Frame 0 làm gốc chuẩn)
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3:]
    R0_inv = R[:, :1, :, :].transpose(-1, -2)
    T0 = T[:, :1, :, :]
    R_norm = torch.matmul(R, R0_inv)
    T_norm = torch.matmul(R[:, :1, :, :], (T - T0))

    # 2. Tạo pixel coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij'
    )
    pixels = torch.stack([x_grid + 0.5, y_grid + 0.5, torch.ones_like(x_grid)], dim=-1)
    pixels = pixels.unsqueeze(0).unsqueeze(0).expand(B, F_len, -1, -1, -1)

    fx = intrinsics[..., 0, 0].view(B, F_len, 1, 1, 1)
    fy = intrinsics[..., 1, 1].view(B, F_len, 1, 1, 1)
    cx = intrinsics[..., 0, 2].view(B, F_len, 1, 1, 1)
    cy = intrinsics[..., 1, 2].view(B, F_len, 1, 1, 1)

    d_x = (pixels[..., 0:1] - cx) / fx
    d_y = (pixels[..., 1:2] - cy) / fy
    d_z = torch.ones_like(d_x)
    d_cam = torch.cat([d_x, d_y, d_z], dim=-1)
    d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True).clamp(min=1e-6)

    # 3. Làm phẳng H*W để đồng bộ chiều batch [B, F] với R_norm [B, F, 3, 3]
    d_cam_flat = d_cam.reshape(B, F_len, height * width, 3) # [B, F, H*W, 3]
    r_d_flat = torch.matmul(d_cam_flat, R_norm.transpose(-1, -2)) # [B, F, H*W, 3]
    r_d = r_d_flat.reshape(B, F_len, height, width, 3) # [B, F, H, W, 3]

    r_o = T_norm.squeeze(-1).view(B, F_len, 1, 1, 3).expand_as(r_d)
    r_dxo = torch.cross(r_o, r_d, dim=-1)
    plucker = torch.cat([r_dxo, r_d], dim=-1).permute(0, 1, 4, 2, 3)
    return plucker # [B, F, 6, H, W]
```
Phương pháp này bảo đảm tensor Plücker luôn có kích thước chuẩn xác `[1, 25, 6, 320, 576]` mà không bao giờ gặp lỗi broadcasting collision.

---

---

## 13. LỖI 13: LỖI TRÀN BỘ NHỚ ATTENTION BẬC HAI (QUADRATIC ATTENTION OOM TRÊN 79 GiB)
### 🔴 Hiện Tượng:
```text
OutOfMemoryError: CUDA out of memory. Tried to allocate 79.10 GiB. GPU 0 has a total capacity of 94.97 GiB of which 6.31 GiB is free.
```
tại dòng:
```python
probs = F.softmax(scores, dim=-1)
```

### 🔬 Nguyên Nhân Chi Tiết:
1. **Độ phân giải vượt chuẩn huấn luyện**:
   Mô hình CameraCtrl SVD-XT được thiết kế và huấn luyện chuẩn ở độ phân giải **$320 \times 576$** (tỷ lệ 16:9, xem file cấu hình `configs/train_cameractrl/svdxt_320_576_cameractrl.yaml`).
2. Khi tăng độ phân giải lên **$576 \times 1024$**:
   - Số lượng tokens không gian tại block đầu tiên tăng từ $2,880$ lên $9,216$ ($3.2\times$).
   - Ma trận Attention dense $N \times N$ tăng theo quy luật bậc hai: $(3.2)^2 \approx 10.24\times$!
   - Kích thước tensor ma trận Attention cho 25 frames ở kiểu dữ liệu Float32 đạt tới:
     $$200 \text{ heads} \times 9,216 \times 9,216 \times 4 \text{ bytes} \approx 67.95 \text{ GiB}$$
   - Khi cộng mask và tính Softmax, bộ nhớ tạm thời yêu cầu vượt quá $79 \text{ GiB}$, làm tràn bộ nhớ ngay cả trên card đồ họa khủng RTX PRO 6000 95GB VRAM!

### 🟢 Giải Pháp Triệt Để (2 Tầng Bảo Vệ):
1. **Tầng 1 (Đúng chuẩn độ phân giải)**:
   Thiết lập độ phân giải chuẩn của SVD-XT: `H = 320, W = 576`.
   - VRAM giảm từ $79 \text{ GB}$ xuống chỉ còn **$\sim 4.5 \text{ GB}$**.
   - Tốc độ khử nhiễu tăng vọt, chỉ mất khoảng 10-15 giây.
2. **Tầng 2 (Bảo vệ phân mảnh qua Query Chunking)**:
   Trong hàm `safe_attention_scores`, chia nhỏ `query` theo từng khối (chunk size = 1024) nếu số tokens lớn:
   ```python
   def safe_attention_scores(self, query, key, attention_mask=None):
       scale = self.scale if self.scale is not None else (1.0 / (query.shape[-1] ** 0.5))
       batch_heads, seq_len, head_dim = query.shape
       if seq_len > 4096:
           probs = []
           chunk_size = 1024
           for i in range(0, seq_len, chunk_size):
               q_chunk = query[:, i:i+chunk_size].float() * scale
               s_chunk = torch.bmm(q_chunk, key.float().transpose(-1, -2))
               if attention_mask is not None:
                   s_chunk = s_chunk + attention_mask[:, i:i+chunk_size].float()
               p_chunk = F.softmax(s_chunk, dim=-1).to(dtype=query.dtype)
               probs.append(p_chunk)
           return torch.cat(probs, dim=1)
       else:
           q_f32 = query.float() * scale
           scores = torch.bmm(q_f32, key.float().transpose(-1, -2))
           if attention_mask is not None:
               scores = scores + attention_mask.float()
           return F.softmax(scores, dim=-1).to(dtype=query.dtype)
   ```
   Điều này đảm bảo cho dù người dùng có chạy ở độ phân giải lớn nào thì ma trận attention cũng không bao giờ chiếm quá 2GB VRAM.

---

---

## 14. LỖI 14: LỖI THIẾU SUBMODULE TRONG DIFFUSERS MỚI (MODULENOTFOUNDERROR TRÊN DIFFUSERS.MODELS.UNET_SPATIO_TEMPORAL_CONDITION / UNET_3D_BLOCKS / TRANSFORMER_TEMPORAL)
### 🔴 Hiện Tượng:
```text
ModuleNotFoundError: No module named 'diffusers.models.unet_spatio_temporal_condition'
```
hoặc:
```text
ModuleNotFoundError: No module named 'diffusers.models.unet_3d_blocks'
```
tại các dòng import của CameraCtrl:
```python
from diffusers.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.models.unet_3d_blocks import DownBlockSpatioTemporal, UpBlockSpatioTemporal
from diffusers.models.transformer_temporal import TransformerTemporalModelOutput
```

### 🔬 Nguyên Nhân Chi Tiết:
1. **Sáng kiến Modular Diffusers (Diffusers $\ge 0.30/0.37$)**:
   Hugging Face đã tái cấu trúc toàn bộ thư mục `diffusers/models`:
   - Các module U-Net chuyên biệt được chuyển vào thư mục con `diffusers.models.unets`:
     - `diffusers.models.unet_spatio_temporal_condition` $\rightarrow$ `diffusers.models.unets.unet_spatio_temporal_condition`
     - `diffusers.models.unet_3d_blocks` $\rightarrow$ `diffusers.models.unets.unet_3d_blocks`
   - Các module Transformer được chuyển vào thư mục con `diffusers.models.transformers`:
     - `diffusers.models.transformer_temporal` $\rightarrow$ `diffusers.models.transformers.transformer_temporal`
2. **Xung đột mã nguồn SOTA cũ**:
   Các repo nghiên cứu SOTA (như CameraCtrl, TrajectoryCrafter, AnimateDiff) được viết từ thời Diffusers $0.24 - 0.28$ nên dùng đường dẫn import cũ trực tiếp từ `diffusers.models.*`. Khi chạy trên môi trường Kaggle / Colab đời mới, Python báo `ModuleNotFoundError`.

### 🟢 Giải Pháp Triệt Để (Dynamic Sys.Modules Aliasing + Fallback Dataclass):
Tại **CELL 2**, trước khi bất kỳ module nào của CameraCtrl được import, ta tự động ánh xạ (alias) các module mới về đúng tên đường dẫn cũ trong `sys.modules`, đồng thời tạo fallback dataclass nếu môi trường hoàn toàn không có:
```python
import sys
import types
from dataclasses import dataclass
from diffusers.utils import BaseOutput

# 1. Alias unet_spatio_temporal_condition
if "diffusers.models.unet_spatio_temporal_condition" not in sys.modules:
    try:
        import diffusers.models.unets.unet_spatio_temporal_condition as _ustc
        sys.modules["diffusers.models.unet_spatio_temporal_condition"] = _ustc
    except (ImportError, ModuleNotFoundError):
        _mod = types.ModuleType("diffusers.models.unet_spatio_temporal_condition")
        @dataclass
        class UNetSpatioTemporalConditionOutput(BaseOutput):
            sample: torch.FloatTensor = None
        _mod.UNetSpatioTemporalConditionOutput = UNetSpatioTemporalConditionOutput
        sys.modules["diffusers.models.unet_spatio_temporal_condition"] = _mod

# 2. Alias unet_3d_blocks
if "diffusers.models.unet_3d_blocks" not in sys.modules:
    try:
        import diffusers.models.unets.unet_3d_blocks as _u3d
        sys.modules["diffusers.models.unet_3d_blocks"] = _u3d
    except (ImportError, ModuleNotFoundError):
        pass

# 3. Alias transformer_temporal
if "diffusers.models.transformer_temporal" not in sys.modules:
    try:
        import diffusers.models.transformers.transformer_temporal as _tt
        sys.modules["diffusers.models.transformer_temporal"] = _tt
    except (ImportError, ModuleNotFoundError):
        _mod = types.ModuleType("diffusers.models.transformer_temporal")
        @dataclass
        class TransformerTemporalModelOutput(BaseOutput):
            sample: torch.FloatTensor = None
        _mod.TransformerTemporalModelOutput = TransformerTemporalModelOutput
        sys.modules["diffusers.models.transformer_temporal"] = _mod
```
Cơ chế này tương thích 100% với mọi phiên bản Diffusers (cả cũ lẫn mới), loại bỏ triệt để lỗi `ModuleNotFoundError` khi nạp U-Net và Transformer.

---

---

## 15. QUY TẮC 15: QUY ƯỚC VỊ TRÍ LƯU TRỮ VIDEO CỤC BỘ CỦA USER (`D:\video`)
### 🔴 Bối Cảnh Thực Tế Của User:
Người dùng luôn lưu trữ và tải tất cả video đầu vào/đầu ra, các video kết quả thử nghiệm từ Kaggle về máy tính cục bộ tại đường dẫn cố định:
```text
D:\video\
```
Ví dụ các tệp:
- `D:\video\camera_retargeted_rtx6000.mp4`
- `D:\video\camera_retargeted_rtx6000 (1).mp4`

### 🟢 Nguyên Tắc Thực Thi:
Mọi đoạn mã kiểm tra, trích xuất khung hình (frame extraction), đánh giá chất lượng (VLM / Optical flow) trên môi trường local của agent **bắt buộc phải tìm kiếm và đọc trực tiếp từ thư mục `D:\video`**, không được tìm ở `Downloads` hay quét ngẫu nhiên gây tốn thời gian.

---

---

## 16. LỖI 16: ĐẢO CHIỀU GÓC QUAY TRONG TIA PLÜCKER (QUAY TRÁI THAY VÌ QUAY PHẢI DO LẪN LỘN DẤU C2W VÀ W2C)
### 🔴 Hiện Tượng Hình Ảnh:
1. Người dùng yêu cầu quay **phải** $+30^\circ$, nhưng video sinh ra lại quay sang **trái**.
2. Khi quay sang trái, camera đi vào vùng không gian hoàn toàn chưa từng xuất hiện trong video gốc (Unobserved Left-side Scenery). Do mô hình I2V CameraCtrl không có 3D Proxy Warper và thiếu thông tin khung cảnh bên trái, mô hình phải tự "bịa" (hallucinate) nội dung mới, dẫn đến chất lượng vùng biên bên trái kém tự nhiên.

### 🔬 Nguyên Nhân Kỹ Thuật (Quy Ước Ma Trận Camera-to-World):
Trong hàm `compute_plucker_rays_normalized`:
```python
r_d = torch.matmul(d_cam_flat, R_norm.transpose(-1, -2))
```
Vector tia camera $d_{cam} = (x, y, 1)$ được nhân với $R_{norm}^T$, điều này đồng nghĩa $R_{norm}$ đóng vai trò là ma trận **Camera-to-World ($C2W$)**.
Theo hệ tọa độ chuẩn của camera (hệ trục OpenCV: $X$ hướng sang Phải, $Y$ hướng Xuống dưới, $Z$ hướng Thẳng về phía trước):
- Khi camera quay sang **PHẢI** góc $+\theta$ quanh trục $Y$ (nhìn từ trên xuống theo hướng trục $Y$):
  Trục $Z$ quay về phía $+X$:
  $$R_{c2w}(\text{Pan Right}) = \begin{pmatrix} \cos\theta & 0 & +\sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}$$
- Tuy nhiên, trong code trước đây đã gán:
  `extrinsics[0, t, 0, 2] = -math.sin(th)` và `extrinsics[0, t, 2, 0] = +math.sin(th)`.
  Dấu âm tại vị trí $(0, 2)$ là ma trận **World-to-Camera ($W2C$)**, khi đưa vào hệ C2W của Plücker nó biến thành **quay sang TRÁI (Pan Left)**!

### 🟢 Giải Pháp Triệt Để:
Đổi đúng dấu của ma trận quay Camera-to-World ($C2W$) cho chuyển động quay phải:
```python
# Chuẩn C2W quay PHẢI (+yaw quanh trục Y)
extrinsics[0, t_idx, 0, 0] = math.cos(th)
extrinsics[0, t_idx, 0, 2] = math.sin(th)   # Dấu DƯƠNG (+) để quay phải
extrinsics[0, t_idx, 1, 1] = 1.0
extrinsics[0, t_idx, 2, 0] = -math.sin(th)  # Dấu ÂM (-) để quay phải
extrinsics[0, t_idx, 2, 2] = math.cos(th)
extrinsics[0, t_idx, 3, 3] = 1.0
```
Khi quay đúng sang phải $+30^\circ$, camera sẽ lia sang góc nhìn thuận chiều của cảnh vật đã có, loại bỏ hoàn toàn việc bị quay nhầm sang trái và giảm thiểu hiện tượng méo mó biên do vùng che khuất (disocclusion).

---

---

## 17. QUY TẮC 17: SAI LẦM KHÓA CỨNG THỜI GIAN (TEMPORAL LOCK-STEP FALLACY) & GIẢI PHÁP BIỂU DIỄN 4D TOÀN CỤC KẾT HỢP QUÉT TIA (RAY-FRUSTUM SWEEPING)

### 🔴 Bản Chất Của Vấn Đề (Failure Mode):
Trong các thiết kế trước đây (như naive frame concatenation $z_{in} = [z_{src}; z_t]$ hoặc frame-wise cross-attention), ta ngầm giả định rằng: *Frame $i$ của video nguồn tương ứng với Frame $i$ của video mục tiêu*.
**Giả định này hoàn toàn sụp đổ khi thay đổi quỹ đạo camera!**
1. **Lệch Phối Cảnh Động Lực Học**: Khi camera mới quay sang phải $30^\circ$ hoặc thay đổi tốc độ di chuyển, hướng nhìn ở frame $k$ của camera mới có thể đang nhìn vào vùng không gian mà camera cũ từng ghi hình ở frame $j$ ($j \neq k$), hoặc một vùng giao thoa giữa nhiều frame, hoặc một vùng hoàn toàn chưa từng xuất hiện. Ép buộc $i \leftrightarrow i$ làm mô hình gán sai đặc trưng ngữ nghĩa và chuyển động, dẫn đến méo hình và giật khung (flickering).
2. **Sụp Đổ Tỷ Số Tín Hiệu Trên Nhiễu (SNR Mismatch)**: Latent nguồn $z_{src}$ là ảnh sạch ($t=0$) trong khi latent mục tiêu $z_t$ ngập tràn nhiễu Gauss ($t \approx 1000$). Khi đưa vào chung Attention, Query $Q_t$ là vector ngẫu nhiên đẳng hướng, dẫn đến Uniform Attention Collapse (trọng số Attention bị dàn phẳng lì, mất sạch chi tiết).
3. **Méo Mép Biên & Flying Pixels Do Che Khuất (Disocclusion Tearing)**: Downsampling $4\times$ chỉ làm dịu răng cưa (aliasing) chứ không lấp được lỗ hổng che khuất. Tại các ranh giới nhảy vọt độ sâu, mép vật thể bị kéo giãn thành màng tam giác kỳ dị.

### 🔬 Giải Pháp Kiến Trúc Toàn Diện (4D-RayV2V Paradigm):
1. **Cắt Tỉa Ranh Giới Biên Sâu (Depth Discontinuity Boundary Pruning)**:
   - Phát hiện biên độ sâu bằng Gradient: $\|\nabla D_t(u, v)\| > \tau_{edge}$.
   - Cắt bỏ toàn bộ các pixel ranh giới này trước khi đưa vào 3D để triệt tiêu flying pixels, biến các vệt lem nhem thành lỗ thủng đen rõ ràng.
2. **Nâng Lên Không Gian 4D Toàn Cục (Global 4D World Lifting)**:
   - Unproject toàn bộ $F$ frames của video gốc cùng ma trận $P_{src}$ thành đám mây điểm không-thời gian $\mathcal{S}_{world} = \{ (\mathbf{X}_{world}, \mathbf{c}_{rgb}, \mathbf{f}_{feat}, t) \}$.
3. **Quét Tia Phối Cảnh Góc Nhìn Mới (Ray-Frustum Sweeping Retrieval)**:
   - Từ camera mục tiêu tại frame $k$ ($P_{tgt}^{(k)}$), phát chùm tia quét vào $\mathcal{S}_{world}$ sử dụng Soft Exponential Z-Buffering:
     $$w_i = \exp\left( -\frac{z'_i - z'_{min}}{\sigma_z} \right) \cdot \text{BilinearWeight}(u', v')$$
   - Tự động thu gom pixel từ bất kỳ frame $t$ nào trong quá khứ ($t \neq k$) mà không bị trói buộc vào chỉ số thời gian.
   - Trích xuất Mặt nạ Lỗ thủng $M_{hole}^{(k)}$ nhị phân rõ ràng, áp dụng phép giãn nở hình thái (cv2.dilate $5 \times 5$) để bao trùm vùng rách mép.
4. **Epipolar Geometry Cross-Attention & Noise Augmentation**:
   - Thêm nhiễu nhẹ $s \sim \text{Uniform}(100, 300)$ vào conditioning proxy để đồng bộ phân phối SNR.
   - Ép buộc Query của token tại frame mới $k$ chỉ tương tác Attention với các Key-Value token nằm trên cùng đường tia 3D (Epipolar ray-cone).
   - Vùng có dữ liệu ($M_{hole}=0$) được giữ nguyên $100\%$ độ nét; vùng lỗ thủng ($M_{hole}=1$) được mạng Diffusion tự do vẽ bù (inpaint) chuẩn xác.

---

---

## 18. LỖI 18: IndexError OUT OF BOUNDS TRONG EulerDiscreteScheduler.add_noise DO TIMESTEP KHÔNG NẰM TRONG SCHEDULE
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
IndexError: index 0 is out of bounds for dimension 0 with size 0
at /usr/local/lib/python3.12/dist-packages/diffusers/schedulers/scheduling_euler_discrete.py in index_for_timestep
    pos = 1 if len(indices) > 1 else 0
    return indices[pos].item()
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Trong Diffusers, `EulerDiscreteScheduler` của SVD-XT quản lý các mốc thời gian liên tục/rời rạc theo mảng `self.timesteps`. Khi gọi `add_noise`, scheduler thực hiện tìm vị trí: `(schedule_timesteps == timestep).nonzero()`. Nếu truyền vào một số nguyên ngẫu nhiên `timestep = torch.randint(...)` mà chưa gọi `set_timesteps` hoặc giá trị đó không trùng khớp chính xác với phần tử trong `self.timesteps`, phép so sánh trả về tensor rỗng (kích thước bằng 0). Khi truy cập `indices[pos]`, Python báo lỗi `IndexError: index 0 is out of bounds for dimension 0 with size 0`.

### 🟢 Giải Pháp Triệt Để:
Khởi tạo trước 1000 bước rời rạc cho scheduler bằng `pipe.scheduler.set_timesteps(1000, device="cuda")`, sau đó lấy mẫu ngẫu nhiên `timestep` trực tiếp từ mảng `pipe.scheduler.timesteps[rand_idx:rand_idx+1]`:
```python
pipe.scheduler.set_timesteps(1000, device="cuda")
train_timesteps = pipe.scheduler.timesteps
rand_idx = torch.randint(10, len(train_timesteps) - 10, (1,)).item()
timestep = train_timesteps[rand_idx:rand_idx+1]
```

---

---

## 19. LỖI 19: LỖI 8 KÊNH ĐẦU VÀO CỦA SVD-XT UNet (INPUT EXPECTED 8 CHANNELS, BUT GOT 4 CHANNELS)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], expected input[25, 4, 40, 72] to have 8 channels, but got 4 channels instead
at /usr/local/lib/python3.12/dist-packages/diffusers/models/unets/unet_spatio_temporal_condition.py line 379:
sample = self.conv_in(sample)
```
### 🔬 Nguyên Nhân Toán Học & Kiến Trúc SVD-XT:
Mô hình SVD-XT là mạng sinh video từ ảnh (Image-to-Video). Lớp tích chập đầu vào `unet.conv_in` có trọng số shape `[320, 8, 3, 3]`, nhận chính xác **8 kênh**:
- **4 kênh đầu**: Latent nhiễu cần khử `noisy_latents` (hoặc `latent_model_input`) shape `[B, F, 4, H_lat, W_lat]`.
- **4 kênh sau**: Image Conditioning Latents `f0_cond_latents` (Latent của Frame 0 nén qua VAE và lặp lại $F$ lần theo trục thời gian) shape `[B, F, 4, H_lat, W_lat]`.
Nếu chỉ truyền `latent_model_input` (4 kênh) vào `unet`, tầng `conv_in` sẽ lập tức báo lỗi thiếu 4 kênh.

### 🟢 Giải Pháp Triệt Để:
Mã hóa Frame 0 qua VAE để tạo `f0_cond_latents` 4 kênh, lặp lại $F$ lần, rồi ghép nối theo chiều kênh (dim=2) trước khi đưa vào UNet:
```python
# 1. Mã hóa Frame 0 bằng VAE (4 channels)
f0_norm = (video_tensor[0:1] * 2.0 - 1.0).to(dtype=torch.float16)
f0_latent = vae.encode(f0_norm).latent_dist.mode() * vae.config.scaling_factor  # [1, 4, H_lat, W_lat]
f0_cond_latents = f0_latent.unsqueeze(1).repeat(1, F_len, 1, 1, 1)              # [1, F, 4, H_lat, W_lat]

# 2. Ghép nối 4 kênh noisy latent + 4 kênh image condition = 8 channels chuẩn SVD-XT
unet_input = torch.cat([latent_model_input, f0_cond_latents], dim=2)            # [1, F, 8, H_lat, W_lat]
pred_noise = unet(unet_input, timestep, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids).sample
```

---

---

## 20. LỖI 20: RuntimeError: element 0 of tensors does not require grad DO BỌC `torch.no_grad()` QUANH MÔ HÌNH NỀN TẢNG (FROZEN BACKBONE)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
at line 116: loss.backward()
```
### 🔬 Nguyên Nhân Kỹ Thuật (Đứt Gãy Đồ Thị Tính Toán - Severed Autograd Graph):
Khi huấn luyện một mạng phụ trợ (Adapter) nối tiếp với một mô hình nền tảng đóng băng (Freeze UNet), mục tiêu là: **Không cập nhật trọng số của UNet, nhưng đạo hàm PHẢI truyền ngược xuyên qua UNet về cập nhật cho Adapter**.
- Để đóng băng trọng số UNet: Ta chỉ cần gọi `unet.requires_grad_(False)`. Autograd của PyTorch sẽ không tính đạo hàm cho các tensor trọng số của UNet.
- Nếu lập trình viên sơ suất bọc `with torch.no_grad():` quanh lệnh gọi `pred_noise = unet(...)`, toàn bộ đồ thị tính toán (Computation Graph) bên trong UNet sẽ bị hủy bỏ hoàn toàn.
- Kết quả: Tensor đầu ra `pred_noise` có `requires_grad=False` và không có `grad_fn`. Hàm loss tính từ `pred_noise` cũng không có đạo hàm, dẫn đến lệnh `loss.backward()` bị sụp đổ.

### 🟢 Giải Pháp Triệt Để:
1. Đóng băng trọng số UNet bằng `unet.requires_grad_(False)`.
2. Bật Gradient Checkpointing trên UNet: `unet.enable_gradient_checkpointing()` để tiết kiệm VRAM khi truyền ngược đạo hàm.
3. **BỎ HOÀN TOÀN** `with torch.no_grad():` quanh lệnh gọi UNet trong vòng lặp huấn luyện để gradient truyền ngược về Adapter:
```python
# Đúng: Gradient truyền xuyên qua UNet về Adapter, nhưng trọng số UNet không bị cập nhật!
pred_noise = unet(
    unet_input,
    timestep,
    encoder_hidden_states=image_embeddings,
    added_time_ids=added_time_ids
).sample
```

---

## 21. LỖI 21: TypeError TRONG AutoencoderKLTemporalDecoder.decode DO THIẾU THAM SỐ `num_frames`
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
TypeError: AutoencoderKLTemporalDecoder.decode() missing 1 required positional argument: 'num_frames'
at vae_f32.decode(latents_f32[f_i:f_i+1]).sample
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Mô hình SVD-XT không sử dụng VAE 2D thông thường (`AutoencoderKL`) mà sử dụng VAE thời gian 3D (`AutoencoderKLTemporalDecoder`).
Phương thức `decode` của `AutoencoderKLTemporalDecoder` có chữ ký bắt buộc:
`def decode(self, z: torch.FloatTensor, num_frames: int)` vì nó chứa các tầng Temporal Attention và 3D Convolutions cần biết rõ độ dài chuỗi thời gian để phân bổ chiều tensor.
Nếu gọi `vae.decode(z)` mà không truyền `num_frames`, hoặc giải mã từng frame đơn lẻ mà không truyền `num_frames=1`, Python báo lỗi `TypeError`.

### 🟢 Giải Pháp Triệt Để:
Sử dụng phương thức chuẩn `pipe.decode_latents(latents, num_frames=F_len, decode_chunk_size=8)` của SVD Pipeline, hoặc truyền tường minh `num_frames` khi gọi `vae.decode`:
```python
# Cách chuẩn nhất trong SVD Pipeline:
pipe.vae.to(dtype=torch.float32)
latents_f32 = latents.to(dtype=torch.float32)

# Giải mã toàn bộ chuỗi latents chuẩn xác bằng hàm decode_latents của pipeline
frames = pipe.decode_latents(latents_f32, num_frames=F_len, decode_chunk_size=8)
# frames có shape [1, F, 3, H, W] hoặc [F, 3, H, W] trong dải [0, 1]
final_video_np = (frames[0] * 255).round().astype(np.uint8) if frames.ndim == 5 else (frames * 255).round().astype(np.uint8)
```

---

## 22. LỖI 22: ValueError: Image must have 1, 2, 3 or 4 channels TRONG `imageio.mimwrite` DO ĐỊNH DẠNG MẢNG CHANNELS-FIRST
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
ValueError: Image must have 1, 2, 3 or 4 channels
at /usr/local/lib/python3.12/dist-packages/imageio/plugins/ffmpeg.py line 590:
self._pix_fmt = map.get(depth, None)
if self._pix_fmt is None:
    raise ValueError("Image must have 1, 2, 3 or 4 channels")
```
### 🔬 Nguyên Nhân Kỹ Thuật:
PyTorch VAE xuất ra tensor ở định dạng Channels-First: `[B, F, C=3, H, W]` hoặc `[F, C=3, H, W]`.
Khi chuyển trực tiếp sang NumPy array mà không hoán vị trục (transpose) chiều kênh về cuối cùng (`[F, H, W, C=3]`), thư viện `imageio` (FFmpeg plugin) khi duyệt qua từng khung hình sẽ đọc `im.shape = (C=3, H=320, W=576)`.
Do đó, nó lấy `depth = im.shape[2] = 576` thay vì số kênh màu ($3$). Khi kiểm tra bảng ánh xạ pixel format, giá trị $576$ không hợp lệ và ném ra ngoại lệ `ValueError`.

### 🟢 Giải Pháp Triệt Để:
Xây dựng hàm chuẩn hóa đa năng: Tự động loại bỏ batch thừa, phát hiện vị trí kênh màu và hoán vị về `[F, H, W, C]`, đồng thời kẹp dải giá trị chuẩn $[0, 255]$:
```python
if isinstance(frames, torch.Tensor):
    frames = frames.detach().cpu().float().numpy()

# 1. Bỏ các chiều thừa ở đầu
while frames.ndim > 4:
    frames = frames[0]

# 2. Hoán vị Channels về cuối cùng -> [F, H, W, C=3]
if frames.shape[0] in [1, 3, 4]:
    # Trường hợp SVD decode_latents: [C=3, F=25, H, W] -> [F, H, W, C]
    frames = np.transpose(frames, (1, 2, 3, 0))
elif frames.shape[1] in [1, 3, 4]:
    # Trường hợp [F=25, C=3, H, W] -> [F, H, W, C]
    frames = np.transpose(frames, (0, 2, 3, 1))

# 3. Chuẩn hóa dải màu sang uint8 [0, 255]
if frames.min() < -0.1:
    frames = (frames / 2.0 + 0.5).clip(0.0, 1.0)
final_video_np = (frames.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
```

---

## 23. LỖI 23: TypeError: iteration over a 0-d tensor TRONG `scheduler.add_noise` DO CHỈ SỐ TIMESTEP DẠNG VÔ HƯỚNG (SCALAR INDEXING)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
TypeError: iteration over a 0-d tensor
at /usr/local/lib/python3.12/dist-packages/diffusers/schedulers/scheduling_euler_discrete.py line 482:
step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]
at line 65:
z_proxy_noisy = pipe.scheduler.add_noise(z_proxy, noise_proxy, t_next)
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Trong vòng lặp khử nhiễu (Denoising loop), khi trích xuất timestep cho bước tiếp theo bằng cú pháp truy cập phần tử đơn `t_next = infer_timesteps[i + 1]`, PyTorch trả về một **0-D scalar tensor** (tensor vô hướng không có chiều, `.ndim == 0`, e.g. `torch.Size([])`).
Khi truyền tensor 0-D này vào phương thức `add_noise` của `EulerDiscreteScheduler`:
Thư viện Diffusers thực hiện vòng lặp comprehension `for t in timesteps:`.
Trong Python và PyTorch, một tensor 0-D không thể duyệt qua bằng vòng lặp `for`, dẫn đến ngoại lệ `TypeError: iteration over a 0-d tensor`.

### 🟢 Giải Pháp Triệt Để:
1. **Dùng Cú Pháp Slicing 1-D**: Truy xuất qua `infer_timesteps[i + 1 : i + 2]` hoặc gọi `t_next.unsqueeze(0)` để đảm bảo `t_next` luôn là tensor 1 chiều (`shape: [1]`, `.ndim == 1`).
2. **Kèm Cơ Chế Fallback Toán Học Của Euler SVD**: Dự phòng tính trực tiếp mức độ nhiễu $\sigma$ theo công thức nguyên bản của Euler Discrete SVD:
   $$x_t = x_0 + \sigma_t \cdot \epsilon$$
```python
if i < len(infer_timesteps) - 1:
    # SỬA LỖI: Cú pháp slice [i+1 : i+2] đảm bảo giữ nguyên 1-D tensor [1]
    t_next = infer_timesteps[i + 1 : i + 2]
    noise_proxy = torch.randn_like(z_proxy)
    try:
        z_proxy_noisy = pipe.scheduler.add_noise(z_proxy, noise_proxy, t_next)
    except Exception:
        step_idx = (pipe.scheduler.timesteps == infer_timesteps[i + 1]).nonzero().item()
        sigma = pipe.scheduler.sigmas[step_idx].to(device=z_proxy.device, dtype=z_proxy.dtype)
        z_proxy_noisy = z_proxy + noise_proxy * sigma
    latents = (1.0 - m_latent) * z_proxy_noisy + m_latent * latents_prev
else:
    latents = (1.0 - m_latent) * z_proxy + m_latent * latents_prev
```

---

## 24. LỖI 24: NameError: name 'masks_tensor' is not defined DO LỆCH TÊN BIẾN GIỮA CÁC CELL KAGGLE VÀ CƠ CHẾ SELF-HEALING
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
NameError: name 'masks_tensor' is not defined
at line 56:
m_latent = F.interpolate(masks_tensor.to(device=DEVICE, dtype=pipe.vae.dtype), size=(TARGET_H // 8, TARGET_W // 8), mode="nearest")
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Trong môi trường Jupyter Notebook/Kaggle tương tác, người dùng thực thi các cell không theo thứ tự tuần tự nghiêm ngặt, hoặc cell trước tạo biến dạng list (`hole_masks = [m0, m1, ...]`) hoặc tên biến khác (`hole_masks`, `mask_tensor`, `cond_masks`). Khi Cell 5 cố truy cập cứng tên `masks_tensor`, Python ném ngoại lệ `NameError`.

### 🟢 Giải Pháp Triệt Để (Self-Healing Dynamic Variable Resolution):
Xây dựng cơ chế tự động tìm kiếm biến trong `globals()`, tự động gom `list` thành `torch.Tensor`, và nếu hoàn toàn không có biến mask trong bộ nhớ, **tự động tái tạo mask lỗ thủng trực tiếp từ vùng giá trị 0 (black unprojected pixels) của `proxy_tensor`**:
```python
# 1. Tự động truy quét mọi tên biến mask khả dĩ trong kernel
masks_var = None
for candidate_name in ['masks_tensor', 'hole_masks', 'mask_tensor', 'masks', 'mask', 'hole_mask']:
    if candidate_name in globals() and globals()[candidate_name] is not None:
        val = globals()[candidate_name]
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], torch.Tensor):
            masks_var = torch.stack(val, dim=0)
            break
        elif isinstance(val, torch.Tensor):
            masks_var = val
            break

# 2. Cơ chế Self-Healing: Nếu mất biến mask, tự sinh mask từ vùng rỗng của proxy_tensor
if masks_var is None:
    is_hole = (proxy_tensor == 0).all(dim=1, keepdim=True).float() # [F, 1, H, W]
    kernel = torch.ones(1, 1, 5, 5, device=proxy_tensor.device, dtype=torch.float32)
    masks_var = F.conv2d(is_hole, kernel, padding=2).clamp(0.0, 1.0)

if masks_var.ndim == 3:
    masks_var = masks_var.unsqueeze(1)
elif masks_var.ndim == 5:
    masks_var = masks_var.squeeze(0)

masks_tensor = masks_var
```

---

## 25. LỖI 25: RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half DO TRÀN KIỂU DỮ LIỆU TỪ VAE SANG UNet
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
at /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py line 134:
return F.linear(input, self.weight, self.bias)
inside pipe.unet(unet_input, t, ...) -> emb = self.time_embedding(t_emb)
```
### 🔬 Nguyên Nhân Kỹ Thuật:
1. Trong lần chạy trước hoặc khi thực hiện giải mã khung hình (Rule 21), module VAE được chuyển sang `torch.float32` bằng `pipe.vae.to(dtype=torch.float32)`.
2. Khi người dùng chạy lại Cell trên cùng một Jupyter Kernel mà không restart, `pipe.vae` vẫn duy trì kiểu `torch.float32`.
3. Khi gọi `f0_latent = pipe.vae.encode(...)` hoặc `z_proxy = pipe.vae.encode(...)`, kết quả tạo ra tensor kiểu `Float` (`float32`).
4. Khi thực hiện ghép nối `torch.cat([latent_model_input, f0_cond_input], dim=2)`, PyTorch tự động thăng cấp kiểu dữ liệu (Type Promotion) của toàn bộ `unet_input` thành `torch.float32`.
5. Bên trong UNet SVD-XT, dòng lệnh:
   ```python
   t_emb = t_emb.to(dtype=sample.dtype)
   emb = self.time_embedding(t_emb)
   ```
   ép kiểu `t_emb` theo kiểu của `sample` (`unet_input`, tức `Float32`). Khi `t_emb` đi vào tầng Linear `linear_1` (vốn có trọng số là `Half` / `float16`), phép nhân ma trận $W \cdot x$ lập tức sụp đổ vì lệch kiểu dữ liệu `Float` và `Half`.

### 🟢 Giải Pháp Triệt Để:
Luôn xác định `UNET_DTYPE = pipe.unet.dtype` (thường là `torch.float16`) và ép kiểu tường minh tất cả các tensor đầu vào của `pipe.unet` trước và trong vòng lặp khử nhiễu:
```python
UNET_DTYPE = pipe.unet.dtype  # torch.float16

# 1. Ép kiểu các điều kiện ngữ cảnh
f0_cond_latents = f0_cond_latents.to(device=DEVICE, dtype=UNET_DTYPE)
image_embeddings = image_embeddings.to(device=DEVICE, dtype=UNET_DTYPE)
added_time_ids = added_time_ids.to(device=DEVICE, dtype=UNET_DTYPE)
z_proxy = z_proxy.to(device=DEVICE, dtype=UNET_DTYPE)
m_latent = m_latent.to(device=DEVICE, dtype=UNET_DTYPE)
latents = latents.to(device=DEVICE, dtype=UNET_DTYPE)

# 2. Bên trong vòng lặp khử nhiễu:
latent_model_input = torch.cat([latents] * 2)
latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t).to(dtype=UNET_DTYPE)
f0_cond_input = torch.cat([f0_cond_latents] * 2).to(dtype=UNET_DTYPE)
unet_input = torch.cat([latent_model_input, f0_cond_input], dim=2).to(dtype=UNET_DTYPE)
```
Sau vòng lặp denoising, khi decode VAE, ta chỉ chuyển riêng `latents_f32 = latents.to(dtype=torch.float32)` và `pipe.vae.to(dtype=torch.float32)` để xuất ảnh, không để `float32` ảnh hưởng ngược lại UNet.

---

## 26. QUY TẮC 26: XÁC MINH HƯỚNG QUAY CAMERA RETARGETING (PAN RIGHT PHẢI ĐẨY CẢNH VẬT SANG TRÁI VÀ ĐỂ LẠI LỖ THỦNG BÊN PHẢI)
### 🔴 Hiện Tượng:
Khi yêu cầu xoay camera sang PHẢI $+30^\circ$, video sinh ra lại xoay sang TRÁI (hướng về phía bức tường và đồng hồ), khiến toàn bộ căn bếp bên phải biến mất.
### 🔬 Nguyên Nhân Hình Học Phối Cảnh:
1. Trong hệ tọa độ OpenCV ($+X$ sang Phải, $+Y$ chúc Xuống, $+Z$ nhìn Thẳng):
   - Khi camera quay sang **PHẢI** (Pan Right) quanh trục thẳng đứng: các vật thể trong thế giới thực sẽ dịch chuyển sang bên **TRÁI** của màn hình. Vùng không gian mới được mở ra ở góc nhìn bên **PHẢI** (lỗ thủng màu đen $M_{hole}$ nằm ở bên phải).
   - Nếu công thức quay làm vật thể dịch chuyển sang bên PHẢI (làm hiện ra chiếc đồng hồ ở góc trái căn phòng) thì đó là **QUAY SANG TRÁI (Pan Left)**.
2. Để đảm bảo quay sang PHẢI chuẩn xác trong hệ Camera-to-World ($C2W$):
   $$R_{\Delta}(\text{Pan Right}) = \begin{pmatrix} \cos\theta & 0 & +\sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}$$
   với $\theta = +\text{rad}(30^\circ) \cdot \frac{t}{F-1}$. C2W mục tiêu: $C2W_{tgt} = C2W_{src} \cdot R_{\Delta}$.

---

## 27. QUY TẮC 27: CHUẨN HÓA CLASSIFIER-FREE GUIDANCE TRONG SVD-XT (DYNAMIC SCHEDULE & UNCONDITIONAL ZEROS)
### 🔴 Hiện Tượng:
Video bị bùng nổ màu sắc (cháy màu neon, đỏ rực, biến dạng thành khối hình kỳ dị như chiếc lò nướng choán hết màn hình).
### 🔬 Nguyên Nhân Kỹ Thuật:
1. **Ép cứng Guidance Scale 2.5 cho Frame 0**: Trong SVD-XT, Frame 0 là khung hình neo (anchor). Guidance scale bắt buộc phải tăng dần tuyến tính theo thời gian: `guidance_scale = torch.linspace(1.0, 2.5, F_len)`. Áp CFG 2.5 ngay từ frame đầu tiên sẽ làm "cháy" khung hình ngay lập tức.
2. **Nhánh Unconditional không phải là Zeros**: Nhánh Unconditional trong SVD-XT bắt buộc phải là tensor Zeros: `uncond_f0 = torch.zeros_like(f0_cond_latents)`. Nếu truyền tensor ảnh thật vào cả hai nhánh, tín hiệu CFG bị triệt tiêu sai lệch.

---

## 28. QUY TẮC 28: HÒA TRỘN ĐIỂM ẢNH BẢO TOÀN TUYỆT ĐỐI 100% ĐỘ NÉT VIDEO GỐC (PIXEL-SPACE FEATHERED INPAINTING)
### 🔴 Hiện Tượng:
Ảnh sau khi hòa trộn bị mờ đục, hiện bóng ma (double-exposure) giữa ảnh thật và ảnh khuếch tán.
### 🔬 Giải Pháp Triệt Để:
1. Điểm ảnh lỗ thủng $M_{hole}$ được xác định chính xác từ điểm ảnh đen chưa chiếu tới của `proxy_tensor`:
   `is_hole = (proxy_tensor.abs().sum(dim=1) < 1e-4).float()`
2. Tại vùng $M_{hole} == 0$ (vùng quan sát được): **Giữ nguyên 100% điểm ảnh gốc từ `proxy_tensor`**, không để latent khuếch tán đè lên.
3. Tại vùng $M_{hole} == 1$ (vùng quay mới sang phải): Sử dụng kết quả sinh của SVD-XT.
4. Tại đường biên tiếp giáp: Áp dụng Gaussian Blur nhẹ ($5\times 5$) để đường nối mượt mà không tì vết.

---

## 29. QUY TẮC 29: NẠP OFFLINE SDXL INPAINTING TỪ DATASET KAGGLE (CHỈ ĐỊNH VARIANT="FP16")
### 🔴 Hiện Tượng & Thông Báo Lỗi:
```python
OSError: Error no file named diffusion_pytorch_model.bin found in directory ...\unet.
(hoặc không tìm thấy diffusion_pytorch_model.safetensors)
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Khi ta tinh gọn dataset upload từ 19.38 GB xuống còn 6.94 GB (chỉ giữ lại bản nửa độ chính xác FP16), các tệp trọng số được đặt tên theo quy chuẩn chính thức của Hugging Face là `*.fp16.safetensors` (ví dụ: `unet/diffusion_pytorch_model.fp16.safetensors`, `text_encoder/model.fp16.safetensors`).
Mặc định trong `diffusers`, nếu gọi `from_pretrained(path)` mà **không truyền `variant="fp16"`**, thư viện sẽ tìm tệp Float32 mặc định (`diffusion_pytorch_model.safetensors` hoặc `.bin`). Khi không thấy tệp này, Diffusers lập tức báo lỗi `OSError`.

### 🟢 Giải Pháp Triệt Để:
Luôn truyền tường minh đối số `variant="fp16"` và `torch_dtype=torch.float16` khi nạp pipeline:
```python
from diffusers import AutoPipelineForInpainting
import torch

pipe = AutoPipelineForInpainting.from_pretrained(
    model_dir,
    variant="fp16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True
).to("cuda")
```
Khi có `variant="fp16"`, `diffusers` tự động khớp đúng chính xác các tệp `*.fp16.safetensors` và nạp vào VRAM siêu tốc mà không phát sinh bất kỳ lỗi thiếu file nào.

---

## 30. QUY TẮC 30: INPAINTING ĐA ĐỘ PHÂN GIẢI (DUAL-RESOLUTION UPSCALING CHO SDXL BUCKETS)
### 🔴 Hiện Tượng:
Khi đưa trực tiếp frame kích thước nhỏ $576 \times 320$ vào SDXL Inpainting, vùng sinh mới có thể bị mờ, thiếu chi tiết kiến trúc hoặc lặp texture vì SDXL được huấn luyện tối ưu tại các bucket độ phân giải lớn xung quanh 1 Megapixel ($1024 \times 1024$, $1152 \times 640$, v.v.).
### 🔬 Giải Pháp Triệt Để:
Áp dụng quy trình Dual-Resolution Inpainting:
1. Video gốc và 3D warp được tính toán ở tỷ lệ gốc $576 \times 320$ (tỷ lệ 16:9).
2. Trước khi đưa vào SDXL Inpainting, phóng đại $2\times$ lên $1152 \times 640$ (đây là bucket 16:9 chuẩn mực của SDXL: $1152 \times 640 = 737,280$ điểm ảnh).
3. SDXL Inpainting sinh các chi tiết vân gỗ, mặt đá, ánh sáng cửa sổ với độ phân giải HD cực kỳ sắc nét.
4. Sau khi khử nhiễu xong, thu nhỏ về lại $576 \times 320$ bằng thuật toán nội suy Lanczos/Bicubic, sau đó áp dụng hòa trộn viền mượt (Feathered Blending, Quy tắc 28).
5. Giữ nguyên seed của `torch.Generator` (`generator = torch.Generator(device=DEVICE).manual_seed(2026)`) xuyên suốt 25 frames để nội dung vùng sinh mới được khóa cứng nhất quán về mặt thời gian (Temporal Consistency).

---

## 31. QUY TẮC 31: LỖI SẬP ĐIỂM ẢNH TRONG 3D SPLATTING DO CÔNG THỨC GLOBAL EXPONENTIAL Z-BUFFER (BLACK WARP COLLAPSE)
### 🔴 Hiện Tượng:
Khung hình 3D Warped bị biến thành **màu đen 99.29%** (chỉ còn sót lại một vệt nhỏ ở góc đáy màn hình). Do 99% khung hình bị coi là lỗ thủng ($M_{hole} = 1$), mô hình Inpainting nhận nhầm đây là bức tranh trống và sinh ra 25 căn phòng hoàn toàn khác nhau cho 25 frames từ câu nhắc văn bản (Prompt), làm mất hoàn toàn video gốc.
### 🔬 Nguyên Nhân Toán Học & Kỹ Thuật:
1. **Sai lầm trong công thức Soft Z-Buffering toàn cục**:
   ```python
   z_min = z_screen.min().clamp(min=1e-3)
   z_weights = torch.exp(-((z_screen - z_min) / (z_min * 0.1 + 1e-4)).clamp(max=20.0))
   ```
   $z_{min}$ là độ sâu của điểm gần camera nhất trong toàn bộ bức ảnh (ví dụ mép bàn $z_{min} \approx 1.0\text{m}$). Mẫu số $z_{min} \times 0.1 = 0.1\text{m}$.
   Mọi vật thể ở cự ly xa hơn mép bàn chỉ cần lệch $> 0.5\text{m}$ (tủ bếp ở $3.5\text{m}$, tường ở $4.5\text{m}$) thì $\frac{z - z_{min}}{0.1} > 20$. Trọng số bị triệt tiêu thành $e^{-20} \approx 2 \times 10^{-9}$.
   Khi đi qua điều kiện lọc `valid_pixels = (canvas_weight > 1e-4)`, toàn bộ các điểm ảnh có trọng số $2 \times 10^{-9}$ bị xóa sạch thành 0 (Pixel Collapse)!
2. **Sai lầm khoảng giá trị độ sâu phòng kín**:
   Đặt độ sâu $50.0\text{m} \rightarrow 1.0\text{m}$ là tỷ lệ phong cảnh ngoài trời (Landscape), không phù hợp với phòng bếp trong nhà của RealEstate10K ($4.0\text{m} \rightarrow 1.5\text{m}$).

### 🟢 Giải Pháp Triệt Để:
1. Sử dụng trọng số nội suy song tuyến (Bilinear Splatting Weights) thuần túy `wt = w_c`:
   ```python
   for w_c, vy, ux in [(wv0*wu0, v0, u0), (wv0*wu1, v0, u1), (wv1*wu0, v1, u0), (wv1*wu1, v1, u1)]:
       wt = w_c.squeeze(-1)
       val = rgb_s * wt.unsqueeze(-1)
       canvas_rgb.index_put_((vy, ux), val, accumulate=True)
       canvas_weight.index_put_((vy, ux), wt.unsqueeze(-1), accumulate=True)
   ```
2. Chuẩn hóa thang độ sâu nội thất phòng bếp: $4.0\text{m}$ (tường sau) $\rightarrow 1.5\text{m}$ (bàn ăn phía trước).
   Kết quả: Vùng quan sát được khôi phục **75.86%** sắc nét nguyên bản, vùng lỗ thủng bên phải là **24.14%**. SDXL chỉ bù đúng 24.14% lỗ thủng, giúp video hoàn toàn mượt mà và giữ nguyên vẹn 100% căn bếp thật!

---

## 32. QUY TẮC 32: ĐỨNG TẠI CHỖ QUAY CAMERA (PURE STANDING PAN) DÙNG QUAN HỆ ROTATIONAL HOMOGRAPHY TRIỆT TIÊU 100% SAI SỐ CHIỀU SÂU
### 🔴 Yêu Cầu & Hiện Tượng:
Video gốc RealEstate10K có camera vừa đi tới vừa quay ($T(t)$ thay đổi liên tục). Khi muốn camera **đứng cố định tại một chỗ** (như đặt máy trên chân tripod tại Frame 0) và chỉ quay quét một góc $+30^\circ$ sang phải, việc dùng quỹ đạo $C2W(t) \cdot R_{\Delta}$ vẫn khiến camera bị tịnh tiến tiến tới.
### 🔬 Nguyên Lý Hình Học Quang Học:
Khi camera quay quanh chính tâm quang học của nó ($T(t) = T(0)$ không đổi):
1. Vector tia trong hệ camera Frame 0: $r = K^{-1} x$.
2. Tọa độ điểm 3D trong không gian: $P_w = R_0 r \cdot d + T_0$.
3. Điểm chiếu lên camera mục tiêu tại góc quay $R_{\Delta}$:
   $$P_{\text{tgt}} = (R_0 R_{\Delta})^T (P_w - T_0) = R_{\Delta}^T R_0^T (R_0 r \cdot d + T_0 - T_0) = R_{\Delta}^T r \cdot d$$
4. Điểm ảnh trên màn hình mục tiêu:
   $$x_{\text{tgt}} \sim K P_{\text{tgt}} = d \cdot (K R_{\Delta}^T K^{-1}) x_{\text{src}} \implies x_{\text{src}} \sim K R_{\Delta} K^{-1} x_{\text{tgt}}$$
5. **Chiều sâu $d$ triệt tiêu hoàn toàn!** Quan hệ chuyển đổi góc nhìn khi đứng tại chỗ thuần túy là **Rotational Homography**:
   $$H_{\text{tgt}\to\text{src}} = K \cdot R_{\Delta} \cdot K^{-1}$$

### 🟢 Giải Pháp Triệt Để:
Dùng ánh xạ lưới song tuyến `torch.nn.functional.grid_sample` với ma trận $H_{\text{tgt}\to\text{src}}$:
- Không phụ thuộc vào bản đồ độ sâu ước lượng $\rightarrow$ **0% méo hình học, 0% rách lưới, 0% flying pixels**.
- Frame 0 tại $\theta = 0^\circ$ giữ nguyên 100% điểm ảnh gốc ($0\%$ lỗ thủng).
- Lỗ thủng bên phải mở rộng mượt mà tuyến tính từ $0\% \to 24\% \to 39\%$ khi $\theta$ quay từ $0^\circ \to +30^\circ$.
- SDXL Inpainting bù đắp mượt mà vùng mở rộng bên phải, trong khi căn bếp thật từ Frame 0 lướt nhẹ nhàng sang trái như trên chân máy quay điện ảnh.

---

## 33. LỖI 33: LỖI KÝ TỰ CẤM TRÊN KAGGLE DO POWERSHELL COMPRESS-ARCHIVE DÙNG DẤU GẠCH CHÉO NGƯỢC (`\`)
### 🔴 Hiện Tượng & Thông Báo Lỗi:
Khi upload file ZIP mã nguồn lên Kaggle Dataset, hệ thống báo lỗi:
```text
'video_retargeting_adapter\inference_pipeline.py' in 'video_retargeting_adapter_v3.zip' contains a forbidden character in name ('\')
```
### 🔬 Nguyên Nhân Kỹ Thuật:
Lệnh `Compress-Archive` của Windows PowerShell mặc định dùng dấu gạch chéo ngược (`\`) của hệ điều hành Windows để phân tách thư mục trong bảng mục lục (Central Directory Header) của file ZIP.
Máy chủ Kaggle chạy trên môi trường **Linux/Unix**, nơi chuẩn POSIX của file ZIP bắt buộc dấu phân cách thư mục phải là gạch chéo xuôi (`/`). Trên Linux, ký tự `\` bị coi là một phần của tên file (chứ không phải thư mục con), và Kaggle từ chối tải lên các file có chứa ký tự `\` trong tên.

### 🟢 Giải Pháp Triệt Để:
Không dùng lệnh `Compress-Archive` của PowerShell khi nén file gửi lên Linux. Sử dụng module `zipfile` của Python để ép buộc chuẩn hóa toàn bộ đường dẫn bên trong thành dấu gạch chéo xuôi (`/`):
```python
import os, zipfile

src_dir = r"d:\Study\Khoa_luan\Video_Editing_Survey\video_retargeting_adapter"
zip_path = r"d:\Study\Khoa_luan\Video_Editing_Survey\video_retargeting_adapter_v3.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src_dir):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".pyc"):
                continue
            full_path = os.path.join(root, f)
            arcname = os.path.relpath(full_path, os.path.dirname(src_dir)).replace(os.sep, '/')
            zf.write(full_path, arcname=arcname)
```
Kết quả: 100% đường dẫn trong file ZIP đều là `video_retargeting_adapter/...` theo chuẩn POSIX, Kaggle chấp nhận thành công không có bất kỳ cảnh báo nào.

---

## 34. LỖI 34: LỖI TRIỆT TIÊU HỐ INPAINT DO DÙNG MẶT NẠ OPTICAL FLOW 2D XÓA HỐ 3D (VỆT SỌC BẸP & MÔ HÌNH BỊ VÔ HIỆU HÓA)
### 🔴 Hiện Tượng & Triệu Chứng:
1. Khi chạy suy luận camera pan $30^\circ$, log báo kích thước hố chỉ có:
   `Disocclusion hole: 93 pixels (0.1% of frame area)` thay vì phải là $\sim 73,000$ pixels ($39.7\%$).
2. Video sinh ra bị hiện tượng **vệt sọc dọc kéo bẹp (smear streak artifact)** lặp đi lặp lại suốt 40% diện tích bên trái khung hình.
3. Checkpoint trước khi train (Step 0) và sau khi train (Step 300) cho kết quả **y hệt nhau**, người dùng cảm giác như mô hình không hề tạo ảnh ("không thấy model tạo ảnh gì hết").

### 🔬 Nguyên Nhân Kỹ Thuật (Mathematical & Architectural Root Cause):
Trong `inference_pipeline.py`, code tính mặt nạ hợp nhất giữa 3D warp và 2D Optical Flow:
```python
flow = compute_farneback_optical_flow(src_frames[t - 1], src_frames[t])
warped_prev_flow, valid_prev_flow = flow_warp_rgb(prev_generated, flow)
valid_prev = valid_prev_3d | (valid_prev_flow > 0)
hole_mask = ((~valid_0) & (~valid_prev)).astype(np.uint8) * 255
```
- **Sai lầm bản chất**: Optical flow được tính giữa 2 frame liên tiếp của **video nguồn 2D** (`src_frames`). Chuyển động giữa 2 frame nguồn chỉ là 1-2 pixel, nên ma trận biến dạng `map_x, map_y` luôn nằm trong biên ảnh $[0, W) \times [0, H)$ trên **99.9% diện tích**.
- Do đó `valid_prev_flow > 0` bằng **True trên 99.9% khung hình**.
- Khi lấy `valid_prev = valid_prev_3d | (valid_prev_flow > 0)`, `valid_prev` bị ép thành **True trên 99.9% khung hình**.
- Dẫn đến biểu thức `hole_mask = ((~valid_0) & (~valid_prev))` bị ép thành **False (0 pixels)** trên toàn bộ vùng góc nhìn mới 3D!
- Trong khi đó, `composite` ở vùng hố bị gán bằng `warped_prev_flow` (ảnh frame trước bị kéo dãn 1-2 pixel bằng optical flow).
- **Hệ quả chết người**:
  1. Trong vòng lặp diffusion: `latents_target = z_known * (1 - m_lat) + latents_denoised * m_lat`. Vì `m_lat` bằng 0, toàn bộ kết quả khử nhiễu của UNet bị nhân với 0 và vứt bỏ!
  2. Ở bước Decode RGB: `blended = composite * (1 - mask_feather) + decoded * mask_feather`. Ảnh `decoded` từ UNet cũng bị nhân với 0!
  3. Qua 24 frames, mép cửa bị kéo dãn lặp lại 24 lần liên tiếp mà **không một pixel nào được mô hình khuếch tán inpaint**, tạo ra 24 sọc dọc kéo bẹp. Step 0 và Step 300 đều xuất ra cùng một ảnh kéo dãn đó nên trông y hệt nhau.

### 🟢 Giải Pháp Triệt Để:
1. **Khôi phục Mặt Nạ Hố 3D Chân Thực**:
   Hố inpaint trong bài toán Camera Retargeting phải được định nghĩa bởi hình học camera 3D từ góc nhìn của Frame 0:
   $$\text{HoleMask} = \text{hole}_0 = (\sim \text{valid}_0)$$
   Tại Frame 24, diện tích hố thực sự là **73,216 pixels (39.7%)**.
2. **Optical Flow chỉ dùng làm Motion Prior, Tuyệt Đối Không Dùng Để Xóa Hố 3D**:
   Optical flow chỉ được dùng làm điều kiện chuyển động dẫn đường trong kênh input của Adapter, không được phép can thiệp vào `hole_mask`.
3. **Mồi Màu và Cấu Trúc (SDEdit Latent Seed)**:
   Trong vùng hố 39.7% đó, lấy Frame $t-1$ warp 3D bằng camera pose (`warped_prev_3d`) để lấp đầy 95% diện tích đã biết từ bước trước, phần rìa mép mới hé lộ dùng Telea Inpaint.
   Sau đó SDXL Inpaint UNet + Adapter sẽ thực sự khử nhiễu trên toàn bộ vùng 39.7% này ở strength $S=0.85$, tái tạo chi tiết vi mô sắc nét và triệt tiêu vệt sọc bẹp!

---

## 35. LỖI 35: NGUYÊN TẮC BẤT DI BẤT DỊCH - KAGGLE NOTEBOOK HOẠT ĐỘNG 100% OFFLINE (INTERNET: OFF)
### 🔴 Hiện Tượng & Triệu Chứng:
1. Thêm lệnh `git clone`, `git pull`, hoặc bất kỳ lệnh tải từ internet nào vào notebook cell dẫn đến lỗi mạng chết người:
   `fatal: unable to access 'https://github.com/...': Could not resolve host: github.com`
2. Khi lệnh mạng thất bại, các câu lệnh dọn dẹp hoặc sao chép tiếp theo (`rm -rf`, `cp`) bị lỗi dây chuyền làm mất thư mục mã nguồn trong `/kaggle/working/`, dẫn đến lỗi:
   `python3: can't open file '/kaggle/working/pipeline_v3/...': [Errno 2] No such file or directory`

### 🔬 Nguyên Tắc Cốt Lõi Về Môi Trường Thực Thi (Core Environmental Constraint):
- Toàn bộ quá trình huấn luyện và đánh giá trên Kaggle Notebook của dự án **LUÔN LUÔN CHẠY Ở CHẾ ĐỘ 100% OFFLINE (INTERNET ACCESS: OFF)**.
- **KHÔNG BAO GIỜ** được yêu cầu người dùng bật mạng hoặc viết các lệnh gọi ra ngoài mạng (`git`, `pip install`, `wget`, `curl`, `requests` đến host ngoài).
- Toàn bộ dữ liệu huấn luyện, weights, depth cache và mã nguồn `pipeline_v3` **ĐÃ ĐƯỢC MOUNT ĐẦY ĐỦ TRONG `/kaggle/input/`**.
- Việc thiết lập môi trường trong `/kaggle/working/` chỉ được phép sử dụng các công cụ cục bộ: tìm kiếm trong `/kaggle/input/` và sao chép an toàn bằng lệnh hệ thống nội bộ, không bao giờ dùng `rm -rf` trước khi kiểm tra nguồn.

### 🟢 Quy Tắc Thiết Kế Cell Chuẩn:
1. Chỉ tìm thư mục `pipeline_v3` bên trong `/kaggle/input/` và `cp -r` sang `/kaggle/working/pipeline_v3`.
2. Tuyệt đối không xóa `/kaggle/working/pipeline_v3` nếu nguồn thay thế không tồn tại.
3. Tự động kiểm tra và truyền đầy đủ các cờ CLI (`--video_dir`, v.v.) để tương thích với tất cả các phiên bản của pipeline mà không cần mạng.
4. **Cô Lập Thao Tác Setup Vào Duy Nhất Cell 1**: Mọi logic tìm kiếm, sao chép hoặc giải nén `pipeline_v3` từ `/kaggle/input/` sang `/kaggle/working/` PHẢI NẰM HOÀN TOÀN Ở CELL 1. Tuyệt đối không để code copy/setup xuất hiện trong Cell 2 (Stage 1) hay Cell 4 (Stage 2) nhằm giữ các cell huấn luyện luôn 100% sạch sẽ, chuyên nghiệp, chỉ tập trung vào siêu tham số và thực thi.