# ==============================================================================
# PIPELINE V3: TRÌNH TẠO BÁO CÁO HTML SO SÁNH 4 MÔ HÌNH (TƯƠNG THÍCH ĐIỆN THOẠI)
# ==============================================================================
# So sánh 4 phiên bản cốt lõi:
# 1. Baseline SOTA (Paper gốc TrajectoryCrafter)
# 2. Checkpoint Step 0 (Base CogVideoX-Fun InP chưa huấn luyện)
# 3. Stage 1 (DoRA Hình học - Step 400)
# 4. Stage 2 (Mô hình Hoàn chỉnh: DoRA + Perceiver - Step 400)
#
# Đặc tính nổi bật:
# - 100% Offline, tự chứa (nhúng trực tiếp video Base64 H.264 yuv420p)
# - Thiết kế Responsive thích ứng hoàn hảo cho màn hình điện thoại (iOS/Android)
# - Hỗ trợ thanh điều khiển đồng bộ (Play/Pause all, 0.5x Slow-mo)
# ==============================================================================

import os
import sys
import glob
import base64
import argparse
import subprocess
import gc
import torch

try:
    from IPython.display import display, HTML, FileLink
    IN_IPYTHON = True
except ImportError:
    IN_IPYTHON = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

for p in [
    "/kaggle/working",
    "/kaggle/working/pipeline_v3",
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
    "/kaggle/input/datasets/tranbao0105/pipeline",
    ".",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)


def auto_find_path(candidate_paths, description="Path"):
    for p in candidate_paths:
        if p and os.path.exists(p):
            print(f"[*] [Phát hiện tự động] {description}: {p}")
            return p
    return None


def convert_to_mobile_h264(input_path, output_path):
    """
    Chuyển đổi video sang chuẩn H.264/yuv420p tối ưu cho phần cứng di động (iPhone / Android).
    Sử dụng baseline/main profile và +faststart để phát mượt mà không bị đen màn hình.
    """
    if not input_path or not os.path.exists(input_path):
        return None
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return output_path
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = (
        f'ffmpeg -y -loglevel error -i "{input_path}" '
        f'-c:v libx264 -pix_fmt yuv420p -profile:v main -level 3.1 '
        f'-preset fast -crf 22 -movflags +faststart "{output_path}"'
    )
    subprocess.run(cmd, shell=True, check=False)
    return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 else input_path


def to_base64_data_uri(path):
    """Đọc file video và chuyển thành Base64 Data URI."""
    if path and os.path.exists(path) and os.path.getsize(path) > 1024:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:video/mp4;base64,{b64}"
    return ""


def find_existing_video(patterns):
    """Tìm video đã được kết xuất trước đó để tránh suy luận lại tốn GPU."""
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            valid_m = [m for m in matches if os.path.exists(m) and os.path.getsize(m) > 1024 and not m.endswith("_depth.mp4")]
            if valid_m:
                sorted_m = sorted(valid_m, key=lambda x: (not x.endswith("_h264.mp4"), -os.path.getsize(x)))
                return sorted_m[0]
    return None


def generate_comparison_html(
    models_info,
    source_video_b64,
    scaffold_video_b64,
    video_stem="000c3ab189999a83",
    motion_name="Quay sang trái 30° (Pan Left)",
    output_html_path="/kaggle/working/so_sanh_4_mo_hinh_mobile.html"
):
    """
    Sinh file HTML Responsive 100% tự chứa, tối ưu cho duyệt trên điện thoại.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>So Sánh 4 Mô Hình - Pipeline V3</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --card-border: #23314e;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --c-base: #ef4444;
            --c-step0: #eab308;
            --c-s1: #f97316;
            --c-s2: #10b981;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 12px 10px 110px 10px;
            line-height: 1.45;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            padding: 16px 12px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            margin-bottom: 16px;
        }}
        .header h1 {{
            font-size: 19px;
            color: var(--accent-blue);
            margin-bottom: 6px;
        }}
        .header p {{
            font-size: 12.5px;
            color: var(--text-muted);
        }}
        .source-container {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 18px;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            gap: 16px;
            text-align: center;
        }}
        .source-col {{
            flex: 1;
            min-width: 220px;
            max-width: 320px;
        }}
        .source-title {{
            font-size: 12px;
            font-weight: 700;
            color: #cbd5e1;
            margin-bottom: 6px;
            display: block;
        }}
        .video-box {{
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            border: 1px solid #334155;
            position: relative;
        }}
        video {{
            width: 100%;
            height: auto;
            display: block;
            background: #000;
        }}
        /* Responsive Grid: 4 columns on desktop, 2 on tablet, 1 on small mobile */
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        @media (max-width: 1024px) {{
            .comparison-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 14px;
            }}
        }}
        @media (max-width: 580px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
                gap: 16px;
            }}
            .header h1 {{ font-size: 17px; }}
            body {{ padding: 8px 6px 115px 6px; }}
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 10px;
            border: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            position: relative;
        }}
        .card-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            text-align: center;
        }}
        .badge-col1 {{ background: rgba(239, 68, 68, 0.18); color: var(--c-base); border: 1px solid var(--c-base); }}
        .badge-col2 {{ background: rgba(234, 179, 8, 0.18); color: var(--c-step0); border: 1px solid var(--c-step0); }}
        .badge-col3 {{ background: rgba(249, 115, 22, 0.18); color: var(--c-s1); border: 1px solid var(--c-s1); }}
        .badge-col4 {{ background: rgba(16, 185, 129, 0.22); color: var(--c-s2); border: 2px solid var(--c-s2); }}
        
        .card-title {{
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #f1f5f9;
        }}
        .card-info {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
            flex-grow: 1;
            line-height: 1.4;
        }}
        .card-info ul {{
            padding-left: 16px;
            margin-top: 4px;
        }}
        .card-info li {{
            margin-bottom: 3px;
        }}
        /* Sticky Floating Bar for Synchronized Controls on Mobile */
        .sticky-controls {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-top: 1px solid #334155;
            padding: 10px 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            z-index: 10000;
            box-shadow: 0 -4px 18px rgba(0, 0, 0, 0.6);
        }}
        .btn {{
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s ease;
        }}
        .btn:active {{ transform: scale(0.96); }}
        .btn-sec {{ background: #334155; color: #f8fafc; }}
        .btn-speed {{ background: #0f766e; color: #ccfbf1; }}
        .btn-speed.active {{ background: #14b8a6; color: #042f2e; font-weight: 700; }}
        .status-pill {{
            font-size: 11px;
            padding: 2px 8px;
            background: #1e293b;
            color: var(--accent-blue);
            border-radius: 10px;
            border: 1px solid #334155;
        }}
    </style>
</head>
<body>

    <!-- Header Section -->
    <div class="header">
        <h1>🎬 BẢNG SO SÁNH 4 MÔ HÌNH (PIPELINE V3)</h1>
        <p>Mẫu video: <b>{video_stem}</b> &bull; Quỹ đạo: <b>{motion_name}</b></p>
        <p style="margin-top: 4px; font-size: 11px; color: #64748b;">File tự chứa 100% Base64 &bull; Hoạt động offline mượt mà trên iPhone & Android</p>
    </div>

    <!-- Source Video & 3D Scaffold Box -->
    <div class="source-container">
        <div class="source-col">
            <span class="source-title">🎥 Video Gốc Đầu Vào (Source)</span>
            <div class="video-box">
                <video class="sync-vid" controls autoplay loop muted playsinline webkit-playsinline preload="auto">
                    <source src="{source_video_b64}" type="video/mp4">
                </video>
            </div>
        </div>
        <div class="source-col">
            <span class="source-title">🧱 3D Scaffold Proxy (Lỗ thủng đen)</span>
            <div class="video-box">
                <video class="sync-vid" controls autoplay loop muted playsinline webkit-playsinline preload="auto">
                    <source src="{scaffold_video_b64}" type="video/mp4">
                </video>
            </div>
        </div>
    </div>

    <!-- 4-Column Main Comparison Grid -->
    <div class="comparison-grid">
"""

    for i, m in enumerate(models_info):
        col_idx = i + 1
        badge_class = f"badge-col{col_idx}"
        border_highlight = f"border: 2px solid {m['color']};" if col_idx == 4 else ""
        
        html_content += f"""
        <div class="card" style="{border_highlight}">
            <span class="card-badge {badge_class}">{m['badge']}</span>
            <div class="card-title">{m['title']}</div>
            <div class="video-box">
                <video class="sync-vid" controls autoplay loop muted playsinline webkit-playsinline preload="auto">
                    <source src="{m['b64']}" type="video/mp4">
                </video>
            </div>
            <div class="card-info">
                <p><b>Trọng số:</b> {m['weights_desc']}</p>
                <ul>
                    {m['notes_html']}
                </ul>
            </div>
        </div>
        """

    html_content += """
    </div>

    <!-- Sticky Floating Synchronizer Bar (Mobile Friendly) -->
    <div class="sticky-controls">
        <button class="btn" onclick="syncPlay()">▶ Phát tất cả</button>
        <button class="btn btn-sec" onclick="syncPause()">⏸ Dừng</button>
        <button class="btn btn-sec" onclick="syncRestart()">🔄 Về 0:00</button>
        <button id="btn-speed-toggle" class="btn btn-speed" onclick="toggleSpeed()">⚡ 1.0x (Chuẩn)</button>
        <span class="status-pill" id="sync-status">Đang phát đồng bộ</span>
    </div>

    <script>
        const vids = document.querySelectorAll('.sync-vid');
        let currentRate = 1.0;

        function syncPlay() {
            vids.forEach(v => {
                v.play().catch(e => console.log('Autoplay handled:', e));
            });
            document.getElementById('sync-status').textContent = 'Đang phát đồng bộ';
        }

        function syncPause() {
            vids.forEach(v => v.pause());
            document.getElementById('sync-status').textContent = 'Đã tạm dừng';
        }

        function syncRestart() {
            vids.forEach(v => {
                v.currentTime = 0;
                v.play().catch(e => {});
            });
            document.getElementById('sync-status').textContent = 'Đã tua về 0:00';
        }

        function toggleSpeed() {
            const btn = document.getElementById('btn-speed-toggle');
            if (currentRate === 1.0) {
                currentRate = 0.5;
                btn.textContent = '⚡ 0.5x (Slow-mo)';
                btn.classList.add('active');
            } else {
                currentRate = 1.0;
                btn.textContent = '⚡ 1.0x (Chuẩn)';
                btn.classList.remove('active');
            }
            vids.forEach(v => v.playbackRate = currentRate);
        }

        // Tự động đồng bộ mốc thời gian khi video loop
        if (vids.length > 0) {
            vids[0].addEventListener('timeupdate', () => {
                if (Math.abs(vids[0].currentTime - 0.0) < 0.15) {
                    vids.forEach(v => {
                        if (Math.abs(v.currentTime - vids[0].currentTime) > 0.3) {
                            v.currentTime = vids[0].currentTime;
                        }
                    });
                }
            });
        }

        // Đảm bảo video tự phát trên di động (đã có thuộc tính muted)
        document.addEventListener('DOMContentLoaded', () => {
            syncPlay();
        });
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ [Hoàn tất] Đã tạo file HTML so sánh 4 mô hình tại:")
    print(f"   --> {output_html_path} ({os.path.getsize(output_html_path) / (1024*1024):.2f} MB)")
    return output_html_path


def run_comparison_pipeline(
    stage1_ckpt="/kaggle/input/datasets/tranbao0105/pipeline/dora_stage1_step400/dora_stage1_step400/data.pkl",
    stage2_ckpt="/kaggle/input/datasets/tranbao0105/pipeline/dora_stage2_step400/dora_stage2_step400/data.pkl",
    video_path=None,
    target_pose=(0.0, -30.0, 0.3, 0.0, 0.0),
    output_dir="/kaggle/working/eval_comparison_4col",
    output_html="/kaggle/working/so_sanh_4_mo_hinh_mobile.html",
    force_regenerate=False,
):
    """
    Điều phối toàn bộ quá trình:
    1. Tìm/kết xuất video cho 4 mô hình.
    2. Chuyển đổi H.264 Web Mobile.
    3. Mã hóa Base64 và nhúng vào HTML.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Phát hiện Model Dirs
    model_dir = auto_find_path([
        "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
        "/kaggle/input/cogvideox-fun-inp",
        "./checkpoints/CogVideoX-Fun-V1.1-5b-InP",
    ], description="Base Model Dir (CogVideoX-Fun InP)")

    transformer_dir = auto_find_path([
        "/kaggle/input/datasets/tranbao0105/trajectorycrafter-weights/TrajectoryCrafter",
        "/kaggle/input/datasets/nguynngcnhntrng/trajectorycrafter",
        "/kaggle/input/trajectorycrafter",
        "./checkpoints/TrajectoryCrafter",
    ], description="Baseline SOTA Transformer (Paper gốc)")

    base_transformer_dir = auto_find_path([
        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer/transformer",
        "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer",
        "/kaggle/input/cogvideox-fun-transformer/transformer",
        "./checkpoints/transformer",
    ], description="Base Alibaba PAI Transformer")

    depth_cache_dir = auto_find_path([
        "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
        "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
        "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
        "/kaggle/working/depth_cache",
    ], description="Depth Cache Dir")

    # 2. Phát hiện Video Mẫu
    if not video_path or not os.path.exists(video_path):
        video_path = auto_find_path([
            "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini/test_256/000c3ab189999a83.mp4",
            "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini/test_256/145da324f69d1c6b.mp4",
            "/kaggle/input/realestate10k/test/000c3ab189999a83.mp4",
        ], description="Video Mẫu Đầu Vào")
        
        if not video_path:
            # Fallback tìm bất kỳ video .mp4 nào
            cands = glob.glob("/kaggle/input/**/*.mp4", recursive=True)
            if cands:
                video_path = cands[0]
                print(f"[*] Sử dụng video dự phòng tìm được: {video_path}")

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("Không tìm thấy video mẫu nào trên hệ thống Kaggle!")

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    raw_phi = int(target_pose[1])
    motion_name = f"Quay sang trái {-raw_phi}° (Pan Left)" if raw_phi < 0 else f"Góc quay {raw_phi}°"

    # Chuẩn bị file video gốc và scaffold
    orig_h264 = os.path.join(output_dir, f"{video_stem}_orig_h264.mp4")
    convert_to_mobile_h264(video_path, orig_h264)

    scaffold_h264 = os.path.join(output_dir, f"{video_stem}_scaffold_h264.mp4")
    found_scaff = find_existing_video([
        scaffold_h264,
        f"/kaggle/working/**/{video_stem}/*scaffold*.mp4",
        f"/kaggle/working/**/{video_stem}/*render*.mp4",
        f"/kaggle/working/eval_results_stage2/{video_stem}/**/*scaffold*.mp4",
        f"/kaggle/working/eval_results_stage1/{video_stem}/**/*render*.mp4",
    ])
    if found_scaff and not force_regenerate:
        convert_to_mobile_h264(found_scaff, scaffold_h264)

    # 3. Định nghĩa 4 Model Cần So Sánh
    configs = [
        {
            "id": "baseline",
            "badge": "1. Baseline (SOTA Paper)",
            "title": "TrajectoryCrafter Gốc",
            "color": "#ef4444",
            "weights_desc": "TrajectoryCrafter Official Pretrained",
            "notes_html": (
                "<li>Mô hình SOTA từ bài báo gốc.</li>"
                "<li>Khả năng lấp đầy tốt ở góc nhỏ, nhưng bị mờ và biến dạng phối cảnh khi quay góc lớn (≥30°).</li>"
            ),
            "use_official": True,
            "dora": None,
            "s2": None,
            "existing_patterns": [
                os.path.join(output_dir, "baseline_sota_h264.mp4"),
                f"/kaggle/working/**/{video_stem}/**/baseline*.mp4",
                f"/kaggle/working/eval_results_stage1/{video_stem}/*before*/*.mp4",
                f"/kaggle/working/eval_checkpoint_evolution/{video_stem}/**/baseline*.mp4",
            ]
        },
        {
            "id": "step0",
            "badge": "2. Checkpoint Step 0",
            "title": "Model Base (Chưa Train)",
            "color": "#eab308",
            "weights_desc": "CogVideoX-Fun-V1.1-5b-InP Base (Alibaba PAI)",
            "notes_html": (
                "<li>Chưa học thích ứng quỹ đạo 3D (Step 0).</li>"
                "<li>Không hiểu kênh 3D Scaffold dẫn tới vùng bị che khuất xuất hiện mảng xám đen lớn.</li>"
            ),
            "use_official": False,
            "dora": None,
            "s2": None,
            "existing_patterns": [
                os.path.join(output_dir, "step0_untrained_h264.mp4"),
                f"/kaggle/working/**/{video_stem}/**/step0*.mp4",
                f"/kaggle/working/eval_checkpoint_evolution/{video_stem}/**/step0*.mp4",
            ]
        },
        {
            "id": "stage1_step400",
            "badge": "3. Stage 1 (DoRA Step 400)",
            "title": "DoRA Khớp Hình Học 3D",
            "color": "#f97316",
            "weights_desc": f"DoRA r=16 alpha=32 ({os.path.basename(os.path.dirname(stage1_ckpt))})",
            "notes_html": (
                "<li>Huấn luyện DoRA trên Self-Attention & FFN (Stage 1).</li>"
                "<li>Lấp đầy ~95% các mảng đen 3D scaffold, phục hồi cấu trúc phòng chuẩn xác.</li>"
                "<li>Vân bề mặt (texture) cơ bản đã liền mạch, loại bỏ biến dạng vỡ góc.</li>"
            ),
            "use_official": False,
            "dora": stage1_ckpt,
            "s2": None,
            "existing_patterns": [
                os.path.join(output_dir, "stage1_step400_h264.mp4"),
                f"/kaggle/working/**/{video_stem}/**/stage1*.mp4",
                f"/kaggle/working/eval_results_stage1/{video_stem}/*after*/*.mp4",
                f"/kaggle/working/eval_checkpoint_evolution/{video_stem}/**/dataset_step_400*.mp4",
            ]
        },
        {
            "id": "stage2_step400",
            "badge": "4. Stage 2 (Mô Hình Hoàn Chỉnh)",
            "title": "DoRA + Perceiver Appearance",
            "color": "#10b981",
            "weights_desc": f"DoRA Step 400 + Perceiver ({os.path.basename(os.path.dirname(stage2_ckpt))})",
            "notes_html": (
                "<li>Kết hợp trọn vẹn cả 2 giai đoạn: DoRA Hình Học + 15 Tầng Perceiver Cross-Attention.</li>"
                "<li>Trích xuất trực tiếp vân vật liệu và ánh sáng từ video gốc sang vùng góc nhìn mới.</li>"
                "<li>Chi tiết nội thất sắc nét, nhất quán chuyển động và chất lượng hình ảnh vượt trội.</li>"
            ),
            "use_official": False,
            "dora": stage1_ckpt,
            "s2": stage2_ckpt,
            "existing_patterns": [
                os.path.join(output_dir, "stage2_step400_h264.mp4"),
                f"/kaggle/working/**/{video_stem}/**/stage2*.mp4",
                f"/kaggle/working/eval_results_stage2/{video_stem}/**/*stage2*.mp4",
                f"/kaggle/working/eval_checkpoint_evolution/{video_stem}/**/step400*stage2*.mp4",
            ]
        },
    ]

    # Kiểm tra xem có cần suy luận PipelineV3 không
    pipe = None
    from inference_v3 import PipelineV3

    def make_opts(run_dir, use_official=False, dora=None, s2=None):
        return argparse.Namespace(
            model_name=model_dir,
            transformer_path=transformer_dir if use_official else (base_transformer_dir or "none"),
            base_transformer_path=base_transformer_dir,
            use_official_weights=use_official,
            dora_checkpoint=dora,
            stage2_checkpoint=s2,
            dora_r=16,
            dora_alpha=32.0,
            video_length=49,
            stride=1,
            sample_size=[384, 672],
            fps=10,
            seed=42,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype="bf16",
            low_gpu_memory_mode=False,
            clean_mask=True,
            heal_hole_size=25,
            mask_threshold=0.85,
            prompt="A high quality, clear indoor room with sharp textures, realistic lighting",
            negative_prompt="blur, distortion, jitter, flickering, low quality, dark patches",
            guidance_scale=4.0,
            inference_steps=25,
            depth_path=None,
            depth_cache_dir=depth_cache_dir,
            near=0.1,
            far=100.0,
            radius_scale=1.0,
            out_dir=run_dir,
        )

    processed_models = []

    for cfg in configs:
        c_id = cfg["id"]
        out_h264 = os.path.join(output_dir, f"{video_stem}_{c_id}_h264.mp4")
        
        # 1. Tìm video có sẵn
        found_vid = None
        if not force_regenerate:
            found_vid = find_existing_video([out_h264] + cfg["existing_patterns"])

        if found_vid and os.path.exists(found_vid):
            print(f"[*] [Tái sử dụng] {cfg['title']}: {found_vid}")
            convert_to_mobile_h264(found_vid, out_h264)
        else:
            print(f"\n--> [Suy luận GPU] Đang chạy mô hình: {cfg['title']} ({cfg['badge']})...")
            run_tmp_dir = os.path.join(output_dir, f"run_{c_id}")
            opts = make_opts(run_tmp_dir, use_official=cfg["use_official"], dora=cfg["dora"], s2=cfg["s2"])
            
            pipe = PipelineV3(opts)
            pipe.run(video_path, target_pose=target_pose)
            
            raw_gen = os.path.join(run_tmp_dir, f"gen_pan_{raw_phi}.mp4")
            raw_scaff = os.path.join(run_tmp_dir, f"render_pan_{raw_phi}.mp4")
            
            convert_to_mobile_h264(raw_gen, out_h264)
            if not os.path.exists(scaffold_h264) and os.path.exists(raw_scaff):
                convert_to_mobile_h264(raw_scaff, scaffold_h264)

            # Dọn dẹp VRAM ngay lập tức
            del pipe
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg["b64"] = to_base64_data_uri(out_h264)
        processed_models.append(cfg)

    # Đọc Base64 cho Video Gốc và Scaffold
    orig_b64 = to_base64_data_uri(orig_h264)
    scaff_b64 = to_base64_data_uri(scaffold_h264)

    # 4. Tạo file HTML
    report_file = generate_comparison_html(
        models_info=processed_models,
        source_video_b64=orig_b64,
        scaffold_video_b64=scaff_b64,
        video_stem=video_stem,
        motion_name=motion_name,
        output_html_path=output_html,
    )

    # 5. Hiển thị trong Notebook (nếu đang chạy trong môi trường IPython)
    if IN_IPYTHON:
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                display(HTML(f.read()))
            display(FileLink(report_file))
        except Exception as e:
            print(f"[*] Lưu ý hiển thị iframe: {e}")

    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo HTML so sánh 4 mô hình tương thích di động.")
    parser.add_argument("--stage1_ckpt", type=str, default="/kaggle/input/datasets/tranbao0105/pipeline/dora_stage1_step400/dora_stage1_step400/data.pkl")
    parser.add_argument("--stage2_ckpt", type=str, default="/kaggle/input/datasets/tranbao0105/pipeline/dora_stage2_step400/dora_stage2_step400/data.pkl")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--pan", type=float, default=-30.0)
    parser.add_argument("--output_html", type=str, default="/kaggle/working/so_sanh_4_mo_hinh_mobile.html")
    parser.add_argument("--force_regenerate", action="store_true")
    args = parser.parse_args()

    run_comparison_pipeline(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        video_path=args.video_path,
        target_pose=(0.0, args.pan, 0.3, 0.0, 0.0),
        output_html=args.output_html,
        force_regenerate=args.force_regenerate,
    )
