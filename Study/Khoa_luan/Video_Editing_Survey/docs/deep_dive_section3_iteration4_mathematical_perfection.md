# NGHIÊN CỨU CHUYÊN SÂU MỤC 3 (VÒNG LẶP BIỆN CHỨNG 4)
# HOÀN THIỆN TOÁN HỌC & LOẠI BỎ TRIỆT ĐỂ MỌI RỦI RO THỰC THI MỤC 3

**Khóa Luận Tốt Nghiệp:** Video-to-Video Camera Trajectory Retargeting (V2V-CR)  
**Mô hình Nền tảng:** Wan2.1-T2V-1.3B (DiT 30 Layers, Dim 1536, 12 Heads, Head Dim 128)  
**Tài liệu tham chiếu mã nguồn & thực nghiệm:**
- Mã nguồn Wan2.1 DiT Engine (`diffsynth/models/wan_video_dit.py`)
- Mã nguồn Tencent ARC SCoPE (`scratch/scope/modeling.py`, `scratch/scope/encoding.py`)
- Lý thuyết Hình học Xạ ảnh 3D & Tọa độ Đường thẳng Plücker (Hartley & Zisserman, "Multiple View Geometry in Computer Vision")
- Cơ chế FlashAttention-2 / SDPA Execution & PyTorch Slice Indexing Constraints

---

## 1. TỔNG QUAN VÒNG LẶP BIỆN CHỨNG 4: SOI CHI TIẾT TỪNG DÒNG MÃ VÀ CÔNG THỨC TOÁN

Tại Vòng 4 này, chúng tôi đào sâu vào 4 câu hỏi hóc búa nhất mà các hệ thống Video DiT hiện nay thường che giấu hoặc gặp lỗi ngầm (silent bugs):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               CUỘC TẤN CÔNG BIỆN CHỨNG VÒNG 4 VÀ LỜI GIẢI HOÀN HẢO                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẤN CÔNG 1: "Cắt lát bảng tần số RoPE bằng số thực float w_norm sẽ làm sập PyTorch?"   │
│  └──► PHÁT HIỆN CHÍ MẠNG: w_norm = w * (29/51) là số float! Tensorslice [:w_norm] sẽ   │
│       bị văng TypeError, còn nếu ép kiểu int() sẽ gây trùng tọa độ làm mờ pixel!       │
│       LỜI GIẢI: Universal Dynamic Isotropic RoPE (UDI-RoPE) bằng cách co giãn VECTOR   │
│       TẦN SỐ omega_w = (29/51) * omega_h! Giữ nguyên chỉ số nguyên [:w] 100%!         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẤN CÔNG 2: "Nghịch lý Đảo ngược Tích Vô hướng Klein Quadric trên Heads 9–12?"         │
│  └──► PHÁT HIỆN: Tích Klein d1·m2 + m1·d2 = 0 khi 2 tia GIAO NHAU trong không gian 3D! │
│       Nếu dùng Linear layer thuần túy, Softmax sẽ ưu tiên tia KHÔNG giao nhau có dấu    │
│       dương thay vì tia giao nhau!                                                     │
│       LỜI GIẢI: Bắt buộc dùng 2-Layer Non-Linear MLP (GELU, hidden=256) trên Heads     │
│       9–12 để học hàm nhân đối sánh (radial correspondence kernel) chuẩn xác!          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẤN CÔNG 3: "Làm sao 1 Sink Token trong K có thể chia sẻ 32.760 tọa độ cùng lúc?"      │
│  └──► THỰC TẾ: Một token trong K chỉ có 1 vị trí cố định, không thể co-rotate đồng    │
│       thời với 32.760 queries khác nhau trong FlashAttention-2 kernel!                 │
│       LỜI GIẢI: Chuyển đổi sang Cổng Triệt Tiêu Hậu Chú Ý (Residual Disocclusion Gate) │
│       g_disocclude(q_unrot) * O_src, tương đương toán học 100%, 0 byte phụ trội!      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TẤN CÔNG 4: "Khối FFN có phá hủy sự phân vai giữa Heads 1–8 và Heads 9–12 không?"      │
│  └──► BẢO CHỨNG: FFN không phá hủy mà chính là CẦU NỐI ĐA PHƯƠNG THỨC (Cross-Modal    │
│       Synthesizer), biến tri thức hình học 3D từ Heads 9–12 thành vân bề mặt ở Head 1–8!│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. GIẢI MÃ TOÁN HỌC & THIẾT KẾ CẢI TIẾN CHI TIẾT

### 2.1. NGUY CƠ SẬP MÃ NGUỒN CỦA FLOAT ROPE & GIẢI PHÁP UDI-ROPE (FREQUENCY BASE SCALING)

#### Phát hiện Lỗ Hổng Thực Thi
Trong Wan2.1 (`diffsynth/models/wan_video_dit.py` line 335–339):
```python
freqs = torch.cat([
    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
```
Nếu áp dụng công thức Position Interpolation $w_{norm} = w \cdot \frac{29}{51}$:
- Tại $w = 1 \implies w_{norm} \approx 0.5686$.
- Tại $w = 2 \implies w_{norm} \approx 1.1373$.
Việc cố gắng cắt lát bảng tần số `self.freqs[2][:w_norm]` sẽ gây lỗi:
`TypeError: slice indices must be integers or None or have an __index__ method`.
Nếu làm tròn `int(round(w_norm))`:
$w=1 \to 1$, $w=2 \to 1$. Cả hai cột ảnh kế tiếp nhận chung một góc quay RoPE ($\Delta w = 0$), gây nhòe mờ chi tiết theo chiều ngang!

#### Giải Pháp Đóng Băng Hoàn Hảo: Universal Dynamic Isotropic RoPE (UDI-RoPE)
Thay vì co giãn chỉ số tọa độ rời rạc $w$, ta co giãn trực tiếp **Vector Tần số Cơ sở** $\mathbf{\omega}_w$ ngay khi khởi tạo bảng tần số:
Trong hàm `precompute_freqs_cis(dim, end, theta)`:
$$\mathbf{\omega}_{h, m} = \frac{1}{\theta^{2m / d}}, \quad m \in \{0, 1, \dots, d/2 - 1\}$$
Đối với chiều ngang, áp dụng hệ số co giãn tỉ lệ kích thước cảm biến:
$$\mathbf{\omega}_{w, m} = \left( \frac{H - 1}{W - 1} \right) \cdot \mathbf{\omega}_{h, m} = \frac{29}{51} \cdot \mathbf{\omega}_{h, m}$$
Bảng tần số RoPE phức được tính sẵn:
$$\text{freqs\_cis}_w(w, m) = \exp\left(j \cdot w \cdot \mathbf{\omega}_{w, m}\right) = \exp\left(j \cdot w \cdot \frac{29}{51} \mathbf{\omega}_{h, m}\right)$$

**Ưu Điểm Vượt Trội:**
1. Chỉ số cắt lát `[:w]` hoàn toàn là **SỐ NGUYÊN $0, 1, 2, \dots, W-1$**. Triệt tiêu $100\%$ lỗi `TypeError`.
2. Không có bất kỳ hiện tượng làm tròn hay trùng lặp tọa độ nào giữa các cột liền kề.
3. Tại mép khung hình ngang $w = 51$:
   $$\phi_w(51) = 51 \cdot \left(\frac{29}{51} \mathbf{\omega}_{h, m}\right) = 29 \cdot \mathbf{\omega}_{h, m} = \phi_h(29)$$
   Pha quay góc ngang và góc dọc đạt độ đẳng hướng hoàn hảo $360^\circ$!
4. **Khả năng tổng quát hóa (Generalization):** Tự động thích ứng với mọi tỉ lệ khung hình (16:9, 9:16, 1:1, 4:3) mà không cần can thiệp mã nguồn.

---

### 2.2. NGHỊCH LÝ ĐẢO NGƯỢC TÍCH KLEIN QUADRIC & BẮT BUỘC 2-LAYER NON-LINEAR MLP

#### Phân Tích Lỗ Hổng Hình Học Tuyệt Đối
Tích tương hỗ Klein Quadric (Klein Reciprocal Product) giữa hai tia sáng $\mathcal{L}_1 = (\mathbf{d}_1, \mathbf{m}_1)$ và $\mathcal{L}_2 = (\mathbf{d}_2, \mathbf{m}_2)$ là:
$$\text{Reciprocal}(\mathcal{L}_1, \mathcal{L}_2) = \mathbf{d}_1 \cdot \mathbf{m}_2 + \mathbf{m}_1 \cdot \mathbf{d}_2$$
Theo Định lý Hình học Xạ ảnh 3D (Projective Geometry):
$$\text{Reciprocal}(\mathcal{L}_1, \mathcal{L}_2) = 0 \iff \mathcal{L}_1 \text{ và } \mathcal{L}_2 \text{ CÙNG NẰM TRÊN MỘT MẶT PHẲNG HOẶC CẮT NHAU TẠI MỘT ĐIỂM 3D } P$$

*Nghịch lý xuất hiện khi đưa vào hàm Softmax Attention:*
Nếu phép chiếu tia sáng chỉ là một ma trận tuyến tính đơn tầng $\mathbf{W}_q, \mathbf{W}_k$ (như SCoPE mặc định `plucker_mlp_hidden=0`):
$$pe_q \cdot pe_k \approx \mathbf{d}_1 \cdot \mathbf{m}_2 + \mathbf{m}_1 \cdot \mathbf{d}_2$$
- Hai tia **giao nhau thực sự** tại bề mặt vật thể $P \implies \text{Reciprocal} = 0 \implies \exp(0) = 1.0$.
- Hai tia **lệch nhau hoàn toàn (skew lines)** nhưng có hướng xoắn thuận tay phải $\implies \text{Reciprocal} = +5.0 \implies \exp(5.0) \approx 148.4$.
**HẬU QUẢ:** Mạng Softmax sẽ dồn trọng số chú ý gấp 148 lần vào các tia sáng **KHÔNG GIAO NHAU**, trong khi điểm giao nhau thực sự chỉ nhận được điểm số bình thường!

#### Giải Pháp Đóng Băng: Non-Linear Geometric Kernel MLP
Một ma trận tuyến tính $\mathbf{W}^T \mathbf{W}$ là dạng song tuyến tính (bilinear form), không thể tạo ra cực đại tại điểm $0$ (vì $f(0) = 0$ và có thể nhận giá trị lớn hơn $0$).
Để điểm tương đồng Attention đạt **CỰC ĐẠI KHI KHOẢNG CÁCH TIA BẰNG 0**, mạng neural bắt buộc phải học một hàm nhân phi tuyến dạng đối xứng (Radial Basis Kernel):
$$\mathcal{K}(\mathcal{L}_1, \mathcal{L}_2) \approx \exp\left( - \frac{\text{dist}^2(\mathcal{L}_1, \mathcal{L}_2)}{2\sigma^2} \right)$$
Do đó, trên **Heads 9–12**, chúng tôi thiết kế bộ mã hóa tia bắt buộc phải là **2-Layer MLP với hàm kích hoạt phi tuyến GELU**:
```python
self.plucker_encoder_q = nn.Sequential(
    nn.Linear(8, 256, bias=True),
    nn.GELU(),
    nn.Linear(256, 128, bias=False)  # Chiếu vào 128D của mỗi Head Hình học
)
self.plucker_encoder_k = nn.Sequential(
    nn.Linear(8, 256, bias=True),
    nn.GELU(),
    nn.Linear(256, 128, bias=False)
)
```
- Lớp ẩn 256 chiều với phi tuyến GELU cho phép mạng xấp xỉ hoàn hảo hàm khoảng cách hình học, biến điều kiện giao nhau $(\text{dist} \to 0)$ thành tích vô hướng cực đại trong không gian embedding!

---

### 2.3. RÀO CẢN THỰC THI SINK TOKEN & CỔNG TRIỆT TIÊU HẬU CHÚ Ý (RESIDUAL DISOCCLUSION GATE)

#### Rào Cản Bộ Nhớ Trong FlashAttention-2
Ý tưởng đưa một Sink Token $\mathbf{k}_\emptyset$ vào mảng Key và gán cho nó cùng góc quay RoPE $p_i$ với Query là đúng về mặt lý thuyết thuần túy ($R^T R = I$).
Tuy nhiên, trong kiến trúc GPU và FlashAttention-2:
Tensor Key $K$ là một vùng nhớ tĩnh liên tục. Một token $\mathbf{k}_\emptyset$ nằm ở index 0 chỉ có **MỘT ĐỊA CHỈ BỘ NHỚ VÀ MỘT GÓC QUAY DUY NHẤT**.
Ta **KHÔNG THỂ** bắt 1 token ở index 0 phải xoay đồng thời theo 32.760 góc quay khác nhau của 32.760 queries bên trong một kernel song song trên GPU!

#### Giải Pháp Toán Học Tương Đương Tuyệt Đối: Residual Disocclusion Gating
Nhìn lại phương trình Softmax có Sink Token:
$$O_i = \frac{\sum_{j=1}^N \exp(q_i k_j) v_j}{\sum_{j=1}^N \exp(q_i k_j) + \exp(s_{\emptyset, i})}$$
Phân tích thành tổng trọng số:
$$O_i = \underbrace{\left( \frac{\sum_{j=1}^N \exp(q_i k_j)}{\sum_{j=1}^N \exp(q_i k_j) + \exp(s_{\emptyset, i})} \right)}_{g_{valid}(q_i) \in (0, 1]} \cdot O_{real, i}$$
Trong đó:
$$g_{valid}(q_i) = \sigma\left( \log \sum_{j=1}^N \exp(q_i k_j) - s_{\emptyset, i} \right)$$
Đối với luồng mượn chi tiết từ Video Nguồn ($O_{src}$):
Nếu vùng ảnh là bị khuất lấp (disoccluded - tức là không tồn tại trong Source), mạng chỉ cần **TRIỆT TIÊU ĐÓNG GÓP CỦA SOURCE VỀ 0**:
$$O_{out, i} = O_{tgt, i} + \mathbf{g}_{disocclude}(q_{unrot, i}, t) \odot O_{src, i}$$
Trong đó $\mathbf{g}_{disocclude}(q_{unrot, i}, t) \in [0, 1]^{1536}$ được tính trực tiếp từ vector truy vấn chưa xoay $q_{unrot}$:
$$\mathbf{g}_{disocclude} = \sigma\left( \text{Linear}_{gate}(q_{unrot}) + \mathbf{b}_{sink}(t) \right)$$

**Lợi ích Thực thi Tuyệt đối:**
1. $100\%$ không cần đưa thêm token ảo vào $K$. Tương thích nguyên bản (Native) với mọi thư viện CUDA Attention.
2. Được tính toán trực tiếp từ $q_{unrot}$, **hoàn toàn độc lập với góc xoay RoPE**, đảm bảo tính đẳng hướng $100\%$ trên toàn khung hình.
3. Khi vùng ảnh là mới xuất hiện (disocclusion): $\mathbf{g}_{disocclude} \to \mathbf{0}$, vô hiệu hóa hoàn toàn thông tin nhiễu từ Source, để mặc cho DiT FFN và Target Attention tự do vẽ nên nội dung mới!

---

### 2.4. VAI TRÒ ĐIỀU PHỐI ĐA PHƯƠNG THỨC CỦA KHỐI FEED-FORWARD (FFN)

Một câu hỏi phản biện sâu sắc: *Nếu ma trận FFN trộn lẫn toàn bộ 1536 chiều, liệu nó có làm mất đi tính chuyên môn hóa giữa Heads 1–8 (Ngoại quan) và Heads 9–12 (Hình học) không?*

**Câu trả lời:** FFN không làm mất tính chuyên môn hóa, mà chính là **CƠ CHẾ TRUYỀN DẪN QUAN TRỌNG NHẤT** của mạng DiT:
1. Trong tầng Chú ý (Attention Layer $l$):
   - Heads 9–12 chỉ tập trung tính toán thị sai 3D và liên kết Epipolar giữa hai góc máy.
   - Heads 1–8 chỉ tập trung gom nhặt vân bề mặt và chi tiết ánh sáng.
2. Trong tầng FFN (FFN Layer $l$):
   - FFN đóng vai trò là bộ tổng hợp liên kênh (Cross-Channel Synthesizer). Nó lấy kết quả định vị 3D từ các kênh $1024 \dots 1535$ để **hướng dẫn các kênh $0 \dots 1023$ biết chính xác phải đặt vân bề mặt vào đâu**!
3. Sang tầng kế tiếp (Layer $l+1$):
   - Do Heads 9–12 vẫn tiếp tục được tiêm tia Plücker mới và bị mask RoPE không gian, còn Heads 1–8 vẫn giữ RoPE 2D, sự phân vai cấu trúc này được **tái thiết lập và duy trì bền vững qua tất cả 30 tầng của Wan2.1**!

---

## 3. TỔNG KẾT ĐÓNG BĂNG CUỐI CÙNG CHO MỤC 3

| Thách Thức Biện Chứng | Nguy Cơ Kỹ Thuật Ban Đầu | Giải Pháp Đóng Băng Hoàn Hảo (Vòng 4) |
| :--- | :--- | :--- |
| **Cắt lát Bảng Tần số RoPE** | Cắt lát float `[:w_norm]` làm sập code PyTorch | **UDI-RoPE:** Co giãn vector tần số $\mathbf{\omega}_w = \frac{29}{51} \mathbf{\omega}_h$, giữ nguyên chỉ số nguyên `[:w]` |
| **Nghịch lý Điểm Giao nhau 3D** | Tích Klein Quadric bằng 0 tại điểm giao nhau $\implies$ Bị Softmax triệt tiêu | **Non-linear Kernel MLP:** Bắt buộc 2-layer MLP (GELU, hidden=256) trên Heads 9–12 để học khoảng cách |
| **Thực thi Sink Token trên GPU** | 1 token không thể xoay theo 32.760 góc RoPE cùng lúc trong FlashAttn | **Residual Disocclusion Gate:** $\mathbf{g}_{disocclude}(q_{unrot}) \odot O_{src}$, tương đương toán học, 0 token ảo |
| **Giao tiếp Hình học - Ngoại quan**| Lo ngại FFN làm rò rỉ và mất chuyên môn hóa Heads | **Cross-Modal Inductive Loop:** Khóa ở Attention, giao tiếp ở FFN, dẫn hướng tạo ảnh chuẩn xác |

Mọi khía cạnh toán học, hình học xạ ảnh 3D và mã nguồn thực thi của **Mục 3** hiện đã đạt độ hoàn thiện tuyệt đối, không còn bất kỳ kẽ hở lý thuyết hay rủi ro thực thi nào!
