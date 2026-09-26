# AGENT SKILLS & AUTONOMOUS SCIENTIFIC DISCOVERY: SOTA ARCHITECTURES, BENCHMARKS, AND METHODOLOGY

> **Tài liệu Nghiên cứu Toàn diện về Kỹ năng của AI Agent (Agent Skills) trong Nghiên cứu Khoa học Tự động, Đọc hiểu Chuyên sâu và Suy luận Tiến hóa**  
> **Tổng hợp và phân tích từ 7 công trình nghiên cứu đột phá SOTA (2023 - 2026)**  
> **Mã nguồn và tệp văn bản trích xuất đầy đủ:** `docs/papers/extracted_text/`

---

## 1. TỔNG QUAN VÀ DANH MỤC 7 CÔNG TRÌNH NGHIÊN CỨU ĐÃ TẢI VÀ ĐỌC

Hệ thống đã tải về toàn bộ tài liệu gốc (PDF) và trích xuất nguyên văn văn bản (`extracted_text/`) của 7 công trình tiêu biểu nhất định hình toàn bộ lĩnh vực **AI Agent Skills & Autonomous Science**:

| STT | Tên bài báo | Tác giả & Tổ chức | arXiv ID | Số trang | Trọng tâm cốt lõi |
|---|---|---|---|---|---|
| 1 | **Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward** | Renjun Xu, Yang Yan (Zhejiang Univ, Westlake Univ) | `2602.12430` | 9 | Khung lý thuyết toàn diện về Agent Skills, Progressive Disclosure 3 cấp độ, chuẩn `SKILL.md`, và sự tương hỗ với Model Context Protocol (MCP). |
| 2 | **Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills** | Jingwei Ni, Yihao Liu, Mengyu Zhou et al. (Qwen Alibaba, ETH Zürich, Peking Univ) | `2603.25158` | 25 | Cơ chế chưng cất tự động (distillation) từ lịch sử thực thi (trajectories) thành các kỹ năng tái sử dụng được thông qua suy luận quy nạp (inductive reasoning). |
| 3 | **Language Agents Achieve Superhuman Synthesis of Scientific Knowledge (PaperQA2)** | Michael D. Skarlinski, Sam Cox, Andrew D. White et al. (FutureHouse, Univ of Rochester) | `2409.13740` | 25 | Kiến trúc Agent đọc và tổng hợp y văn/khoa học vượt trình độ chuyên gia con người; công cụ Gather Evidence với tóm tắt ngữ cảnh liên quan (RCS) và phát hiện mâu thuẫn (ContraDetect). |
| 4 | **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** | Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha (Sakana AI, Oxford, UBC) | `2408.06292` | 186 | Hệ thống đầu tiên tự động hóa toàn bộ vòng đời nghiên cứu khoa học: Phát kiến ý tưởng (Ideation) $\to$ Viết mã thực nghiệm (Experimentation) $\to$ Viết bài báo LaTeX (Paper Write-up) $\to$ Tự động bình duyệt (Automated Peer Review). |
| 5 | **VOYAGER: An Open-Ended Embodied Agent with Large Language Models** | Guanzhi Wang, Linxi Fan, Yuke Zhu, Anima Anandkumar et al. (NVIDIA, Caltech, Stanford) | `2305.16291` | 42 | Công trình đặt nền móng khai sinh khái niệm **Skill Library** (Kho kỹ năng code thực thi), cơ chế tự sửa lỗi qua phản hồi môi trường (environment feedback) và truy xuất vector (vector retrieval). |
| 6 | **SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources** | Shuaike Shen, Wenduo Cheng, Jian Ma et al. (Carnegie Mellon University - CMU) | `2604.03964` | 19 | Khung tự tiến hóa biến tài nguyên khoa học phân tán (GitHub repos, Jupyter notebooks, paper protocols) thành thư viện kỹ năng agent có cấu trúc với kiểm định đa tầng (MoSciBench). |
| 7 | **HypoForge: A Self-Improving Multi-Agent Framework for Automated Hypothesis Generation and Testing via Scientific Skill Learning** | Ziqing Qian, Jiaying Lei, Yifang Wang, Nan Cao (Tongji Univ, FSU) | `2608.25770` | 9 | Khung đa Agent tự học kỹ năng khoa học: Generator–Discriminator đối kháng cho việc sinh giả thuyết mới và giám sát thực nghiệm cho kiểm thử giả thuyết. |

---

## 2. BẢN CHẤT CỦA "AGENT SKILL" & KIẾN TRÚC TIÊU CHUẨN

### 2.1. Agent Skill là gì? Sự khác biệt giữa Prompt, Tool và Skill
Theo khảo cứu lý thuyết từ **Xu & Yan (arXiv:2602.12430)** và **Voyager (arXiv:2305.16291)**:
*   **Prompt tĩnh / Few-shot**: Chỉ cung cấp chỉ dẫn tạm thời trong một phiên hội thoại, dễ bị trôi ngữ cảnh khi cửa sổ context dài ra.
*   **Tool (Công cụ / Function Calling / MCP Tools)**: Là các hàm kết nối I/O thô (ví dụ: `run_command`, `read_file`, `search_arxiv`, `execute_python`). Tool trả lời câu hỏi **"Làm thế nào để kết nối?" (How to connect)**.
*   **Skill (Kỹ năng của Agent)**: Là các module tri thức thủ tục (procedural knowledge) đóng gói hoàn chỉnh gồm: mục tiêu, điều kiện kích hoạt, quy trình suy nghĩ từng bước, quy tắc xử lý ngoại lệ, kịch bản phối hợp nhiều công cụ và cách tự xác minh kết quả. Skill trả lời câu hỏi **"Cần suy nghĩ và hành động như thế nào?" (How to think and act)**.

### 2.2. Cơ chế Tiết lộ Lũy tiến (Progressive Disclosure Architecture)
Để tránh làm tràn ngập Context Window của LLM với hàng trăm kỹ năng cùng lúc, chuẩn thiết kế SOTA sử dụng kiến trúc **3 cấp độ**:

```
+-------------------------------------------------------------------------+
| Level 1: System Prompt (Discovery Metadata)                             |
| - Tên Skill (name), Mô tả 1-2 dòng (description), Khi nào nên dùng.   |
| - Chi phí Token: Cực thấp (~20 - 50 tokens/skill).                      |
| - Agent quét qua toàn bộ Skill Library để nhận diện kỹ năng phù hợp.    |
+-------------------------------------------------------------------------+
                                    |
                                    v (Kích hoạt khi gặp tác vụ tương ứng)
+-------------------------------------------------------------------------+
| Level 2: Body SKILL.md (Procedural Instructions)                        |
| - Nạp nội dung chi tiết của SKILL.md vào Context khi kích hoạt.          |
| - Định nghĩa luồng làm việc (Workflows), quy tắc kiểm tra, lỗi thường gặp.|
| - Chi phí Token: Trung bình (~1,000 - 3,000 tokens).                    |
+-------------------------------------------------------------------------+
                                    |
                                    v (Gọi khi cần thực thi chi tiết)
+-------------------------------------------------------------------------+
| Level 3: Executable Scripts & Reference Assets                          |
| - Chạy các script Python/Bash trong `scripts/`, đọc tài liệu chuyên sâu |
|   trong `references/`, nạp mẫu templates trong `examples/`.             |
| - Chi phí Token: Thực thi độc lập ngoài ngữ cảnh LLM, chỉ trả về output.|
+-------------------------------------------------------------------------+
```

### 2.3. Cấu trúc chuẩn của một Thư mục Skill (`SKILL.md`)
Chuẩn chung được áp dụng bởi Antigravity, Claude Code, và các hệ thống mở:
```
my-specialized-skill/
├── SKILL.md               # Tệp chỉ dẫn chính với YAML Frontmatter (bắt buộc)
├── scripts/                # Các mã nguồn tự động hóa, kiểm định, công cụ phụ trợ
│   └── verify_pipeline.py
├── references/             # Tài liệu tra cứu chuyên sâu, trích yếu lý thuyết
│   └── architecture_specs.md
└── examples/               # Mẫu đầu vào/đầu ra, test cases chuẩn
    └── sample_config.json
```

**Mẫu tệp `SKILL.md`:**
```yaml
---
name: video-editing-survey-research
description: Tự động hóa nghiên cứu văn hiến, đọc hiểu paper SOTA, chuyển đổi trọng số mô hình Diffusion và viết báo cáo luận văn.
---

# Video Editing Survey & Research Skill

## When to Use This Skill
Kích hoạt kỹ năng này khi:
1. Người dùng yêu cầu tìm kiếm, tải về và đọc hiểu các bài báo về Video Diffusion, Camera Control.
2. Cần chuyển đổi, kiểm định kiến trúc mạng nơ-ron hoặc phân tích sai số (zero-fallback).
3. Cần tạo báo cáo kỹ thuật hoặc các chương luận văn chuẩn LaTeX (AAAI/CVPR).

## Procedural Workflows
### Bước 1: Quét và Lọc Văn hiến (Literature Ingestion)
...
### Bước 2: Tự động Kiểm tra & Chẩn đoán Lỗi
...
```

---

## 3. CƠ CHẾ ĐỌC VÀ TỔNG HỢP VĂN HIẾN VƯỢT TRÌNH ĐỘ CON NGƯỜI (PAPERQA2)

Trong công trình đột phá **PaperQA2 (arXiv:2409.13740)**, FutureHouse đã chỉ ra điểm yếu chí mạng của hệ thống RAG thông thường (Naive RAG):
*   RAG truyền thống nhồi nhét trực tiếp các đoạn văn bản (chunks) vào context, dẫn đến ảo giác (hallucination) trích dẫn sai, nhầm lẫn tác giả, và không thể tổng hợp tri thức chéo giữa nhiều bài báo.

### 3.1. Luồng Tổng hợp 4 giai đoạn của PaperQA2
1.  **Phân tách cấu trúc (Document Parsing with Grobid)**: Không đọc thô bằng PDF text thuần, PaperQA2 sử dụng parser học máy (Grobid) để bóc tách rõ ràng: Tiêu đề, Tác giả, Abstract, Phương pháp, Kết quả, Bảng biểu và Danh mục tài liệu tham khảo (BibTeX).
2.  **Công cụ Thu thập Bằng chứng (Gather Evidence Tool with RCS)**:
    *   Truy xuất Top-K đoạn văn bản theo độ tương đồng ngữ nghĩa.
    *   **Relevant Context Summary (RCS)**: Với *từng* đoạn văn bản, một LLM chuyên biệt sẽ đọc và tạo bản tóm tắt trích xuất: *"Đoạn này có cung cấp bằng chứng cho câu hỏi không? Bằng chứng cụ thể là gì? Mức độ tin cậy (1-10) là bao nhiêu?"*.
    *   Chỉ các mẩu bằng chứng chắt lọc này mới được đưa vào context tổng hợp, loại bỏ 90% nhiễu thông tin.
3.  **Duyệt Đồ thị Trích dẫn Đệ quy (Recursive Citation Expansion)**:
    *   Agent tự động phát hiện các bài báo bản lề được trích dẫn trong bằng chứng vừa tìm được.
    *   Nếu bằng chứng chưa đủ tin cậy, Agent chủ động gọi API tải thêm các bài báo được trích dẫn đó và chạy lại vòng lặp Gather Evidence.
4.  **Bộ Phát hiện Mâu thuẫn (ContraDetect / ContraCrow)**:
    *   Tự động so khớp các tuyên bố khoa học đối nghịch nhau giữa các bài báo (ví dụ: Paper A nói phương pháp X cải thiện 15%, Paper B nói X gây bất ổn định khi mở rộng quy mô).
    *   Gắn nhãn rõ ràng: *Supported*, *Contradicted*, hoặc *Lack of Evidence*, đảm bảo luận văn học thuật có tính khách quan và trung thực.

---

## 4. TỰ ĐỘNG HÓA TOÀN TRÌNH KHÁM PHÁ KHOA HỌC (THE AI SCIENTIST)

Công trình **The AI Scientist (arXiv:2408.06292)** của Sakana AI và Oxford hiện thực hóa giấc mơ tự động hóa toàn bộ một chu trình nghiên cứu của một nhà khoa học máy tính:

```
[Bắt đầu: Template Mã nguồn & Đề tài]
                  |
                  v
       +--------------------+
       | 1. IDEA GENERATION | <---+ Tự phản biện & Lọc độ mới
       +--------------------+     | (Semantic Scholar API)
                  |               |
                  v               |
    +---------------------------+ |
    | 2. EXPERIMENTAL ITERATION | |
    | - Sửa code bằng Aider      | |
    | - Chạy train & kiểm thử   | |
    | - Tự sửa lỗi (4 retries)  | |
    | - Vẽ đồ thị kết quả (.png)| |
    +---------------------------+ |
                  |               |
                  v               |
     +-------------------------+  |
     | 3. PAPER WRITE-UP       |  |
     | - Viết từng mục LaTeX   |  |
     | - Tìm trích dẫn BibTeX  |  |
     | - Compile pdflatex      |  |
     +-------------------------+  |
                  |               |
                  v               |
     +-------------------------+  |
     | 4. AUTOMATED REVIEW     |--+ (Phản hồi để lặp lại ý tưởng mới)
     | - Chấm điểm NeurIPS     |
     | - Soundness, Novelty    |
     +-------------------------+
```

### 4.1. Pha 1: Phát kiến Ý tưởng (Ideation)
*   Sử dụng toán tử đột biến tiến hóa (evolutionary mutation) trên kho lưu trữ các ý tưởng trước đó.
*   Mỗi ý tưởng gồm: Giả thuyết cốt lõi, Kế hoạch thực nghiệm, Điểm tự đánh giá về tính mới lạ và tính khả thi.
*   **Lọc trùng lặp**: Gọi trực tiếp Semantic Scholar API để tra cứu xem ý tưởng này đã có ai công bố chưa. Nếu độ tương đồng quá cao, ý tưởng lập tức bị loại bỏ.

### 4.2. Pha 2: Lặp Thực nghiệm (Experimental Iteration)
*   Tích hợp coding assistant (Aider) để chỉnh sửa mã nguồn mô hình.
*   Chạy thực nghiệm trên GPU: nếu gặp lỗi CUDA Out-Of-Memory hoặc vỡ tensor, stack trace được bắt lại tự động và nạp vào prompt để sửa code (lặp tối đa 4 lần).
*   Ghi nhật ký thực nghiệm (experimental notes) và tự động sinh mã vẽ đồ thị kết quả chuẩn publication (matplotlib/seaborn).

### 4.3. Pha 3: Viết Báo cáo Khoa học (Paper Write-up)
*   Viết bài báo theo từng phân đoạn (Per-Section Generation) dựa trên cấu trúc chuẩn IEEE/NeurIPS:
    1. `Introduction`: Đặt vấn đề, động lực và đóng góp cốt lõi.
    2. `Background & Related Work`: Tổng hợp văn hiến qua Semantic Scholar.
    3. `Methodology`: Trình bày công thức toán học và kiến trúc mạng.
    4. `Experimental Setup & Results`: Nhúng bảng số liệu thực và hình vẽ từ Pha 2.
    5. `Conclusion & Limitations`: Đánh giá hạn chế trung thực.
*   Tự động sửa lỗi biên dịch LaTeX linter cho đến khi xuất ra file PDF hoàn hảo.

### 4.4. Pha 4: Tự động Bình duyệt (Automated Peer Review)
*   Sử dụng LLM Reviewer Agent với barem điểm của hội nghị hạng A (NeurIPS/ICLR): Soundness (1-4), Presentation (1-4), Contribution (1-4), Overall Score (1-10), Confidence (1-5).
*   Tỷ lệ tương quan chấm điểm tiệm cận reviewer người thật (65% vs 66% balanced accuracy trên tập ICLR OpenReview).

---

## 5. CƠ CHẾ HỌC KỸ NĂNG & TỰ TIẾN HÓA CỦA AGENT

### 5.1. Voyager (arXiv:2305.16291): Skill Library bằng Code Thực thi
Voyager chứng minh rằng: **Cách lưu trữ kỹ năng tốt nhất cho Agent không phải là trọng số mô hình hay văn bản mơ hồ, mà là các hàm mã nguồn (executable code) có thể gọi lại được.**
*   Mỗi kỹ năng được lưu thành một hàm Python/JS độc lập, có docstring mô tả chi tiết tham số vào/ra.
*   Khi cần thực hiện tác vụ mới, Agent dùng Vector Search để tìm các kỹ năng con có sẵn trong thư viện và ghép nối chúng lại thành một kỹ năng phức tạp hơn (compositionality).

### 5.2. Trace2Skill (arXiv:2603.25158): Chưng cất Kinh nghiệm Thực thi thành Kỹ năng Chuyển giao
Trace2Skill giải quyết bài toán: *Làm sao để Agent tự thông minh hơn sau mỗi lần chạy mà không cần con người viết tay kỹ năng?*
*   **Stage 1 - Trajectory Logging**: Thu thập toàn bộ log thực thi của Agent (các lệnh đã chạy, kết quả, lỗi vấp phải, cách sửa).
*   **Stage 2 - Parallel Patch Proposal**:
    *   *Error Analyst ($A^-$)*: Mổ xẻ các phiên thất bại để tìm nguyên nhân gốc rễ và chỉ ra các "bẫy" (pitfalls) cần tránh.
    *   *Success Analyst ($A^+$)*: Phân tích các phiên thành công để chuẩn hóa thành quy trình thao tác chuẩn (SOP - Standard Operating Procedure).
*   **Stage 3 - Inductive Consolidation**: Hợp nhất các bài học thành kỹ năng mới, chuyển giao được cho các mô hình kích thước khác nhau (từ 35B đến 122B).

### 5.3. SkillFoundry (arXiv:2604.03964) & HypoForge (arXiv:2608.25770)
*   **SkillFoundry**: Tự động chuyển đổi tài nguyên khoa học phi cấu trúc (GitHub repositories, Notebooks, tài liệu API) thành các Skill Package có kiểm định đa tầng (Execution validation, System testing, Synthetic data testing).
*   **HypoForge**: Phân lập hai cơ chế học kỹ năng:
    *   *Kỹ năng sinh giả thuyết*: Dùng mô hình đối kháng **Generator–Discriminator** để liên tục phản biện tính mới và tính chặt chẽ logic khi không có nhãn giám sát.
    *   *Kỹ năng kiểm thử giả thuyết*: Học trực tiếp từ kết quả thực thi mã nguồn và số liệu đo đạc thực tế (Ground-truth feedback).

---

## 6. HƯỚNG DẪN ỨNG DỤNG CHO DỰ ÁN VIDEO EDITING SURVEY & KHÓA LUẬN

Để áp dụng tri thức từ 7 bài báo này vào đề tài Nghiên cứu Video Editing, Camera Control và Wan2.1 của chúng ta:

### 6.1. Thiết lập Cấu trúc Kỹ năng (.agents/skills/)
Chúng ta đã khởi tạo kỹ năng nghiên cứu chuyên biệt tại:
`d:\Study\Khoa_luan\Video_Editing_Survey\.agents\skills\video-editing-survey-research\SKILL.md`

Kỹ năng này hiện thực hóa đầy đủ:
1.  **Workflow Đọc hiểu Văn hiến**: Tự động hóa tải paper từ arXiv, trích xuất text qua PyMuPDF, tổng hợp theo cơ chế PaperQA2 (RCS, phát hiện mâu thuẫn giữa 3D Warping vs Pure Conditioning).
2.  **Workflow Kiểm định Hệ thống & Zero-Fallback**: Áp dụng bài học từ Trace2Skill để đóng băng các SOP kiểm tra VAE latent, rank DoRA, timestep scaling ($t = \sigma \times 1000$).
3.  **Workflow Viết Luận văn Tự động**: Tích hợp luồng viết bài của The AI Scientist để tạo các chương báo cáo học thuật chuẩn LaTeX.

### 6.2. Ma trận So sánh Kỹ thuật được Đúc kết từ 19 Bài báo
Từ 12 bài báo Video Diffusion và 7 bài báo Agent Skills, toàn bộ cơ sở tri thức đã sẵn sàng phục vụ viết các chương:
*   Chương 1: Giới thiệu & Tổng quan đề tài.
*   Chương 2: Cơ sở lý thuyết về Video Diffusion Models (Wan2.1 Flow Matching, SVD, DiT).
*   Chương 3: Nghiên cứu Văn hiến & Phân loại SOTA về Camera Trajectory Control (CameraCtrl, TrajectoryCrafter, MotionCtrl, CamCo).
*   Chương 4: Kiến trúc Đề xuất (Pipeline v2 Wan2.1 + DoRA + Camera Injection).
*   Chương 5: Thực nghiệm, Chẩn đoán và Đánh giá Định lượng.
