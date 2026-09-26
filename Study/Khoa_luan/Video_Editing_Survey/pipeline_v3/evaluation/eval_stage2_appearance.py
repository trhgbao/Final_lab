# ==============================================================================
# PIPELINE V3: ĐÁNH GIÁ CHI TIẾT STAGE 2 (STAGE 2 APPEARANCE EVALUATION)
# ==============================================================================
# Đánh giá năng lực phục hồi chi tiết vân bề mặt (Appearance & Texture)
# của Perceiver Cross-Attention trên nền tảng Base Model đã merge DoRA Stage 1.
# ==============================================================================

import os
import sys
import glob
import base64
import argparse
import subprocess
import torch

try:
    from IPython.display import display, HTML
    IN_IPYTHON = True
except ImportError:
    IN_IPYTHON = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("🎨 [PIPELINE V3] KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ STAGE 2: APPEARANCE RESTORATION")
print("=" * 80)


# 1. THIẾT LẬP SYS.PATH
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

try:
    from pipeline_v3.inference_v3 import PipelineV3
except ImportError:
    try:
        from inference_v3 import PipelineV3
    except ImportError:
        raise ImportError("Không tìm thấy PipelineV3 trong sys.path!")


# 2. AUTO-DISCOVERY CÁC ĐƯỜNG DẪN TÀI NGUYÊN
def auto_find_path(candidate_paths, description="Path"):
    for p in candidate_paths:
        if os.path.exists(p):
            print(f"[*] Tìm thấy {description}: {p}")
            return p
    return None

def auto_find_glob(patterns, description="Pattern"):
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            found = sorted(matches)[-1]
            print(f"[*] Tìm thấy {description}: {found}")
            return found
    return None


MODEL_DIR = auto_find_path([
    "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
    "/kaggle/input/cogvideox-fun-inp",
    "./checkpoints/CogVideoX-Fun-V1.1-5b-InP",
], description="Base Model Dir")

BASE_TRANSFORMER_DIR = auto_find_path([
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer/transformer",
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer",
    "/kaggle/input/cogvideox-fun-transformer/transformer",
    "./checkpoints/transformer",
], description="Base Alibaba PAI Transformer Dir")

if not BASE_TRANSFORMER_DIR:
    BASE_TRANSFORMER_DIR = auto_find_glob([
        "/kaggle/input/**/cogvideox_fun_transformer/transformer",
        "/kaggle/input/**/cogvideox-fun-transformer/**/transformer",
        "/kaggle/input/**/cogvideox_fun_transformer",
        "/kaggle/input/**/transformer",
    ], description="Base Alibaba PAI Transformer Dir Glob")


VIDEO_DIR = auto_find_path([
    "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini/test_256",
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k/test",
    "/kaggle/input/realestate10k/test",
    "./examples",
], description="Video Test Dir")

DEPTH_CACHE_DIR = auto_find_path([
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
    "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
    "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache",
    "/kaggle/working/depth_cache",
], description="Depth Cache Dir")

DORA_STAGE1_CKPT = auto_find_path([
    "/kaggle/working/checkpoints_stage1/dora_stage1_final.pt",
    "/kaggle/working/checkpoints_stage1/dora_checkpoint_final.pt",
    "/kaggle/working/checkpoints_dora_stage1/dora_stage1_final.pt",
    "/kaggle/working/dora_stage1_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/dora_stage1_final.pt",
], description="Stage 1 DoRA Checkpoint")

if not DORA_STAGE1_CKPT:
    DORA_STAGE1_CKPT = auto_find_glob([
        "/kaggle/working/**/dora_checkpoint*.pt",
        "/kaggle/working/**/dora_stage1*.pt",
        "/kaggle/input/**/dora_stage1*.pt",
    ], description="Stage 1 Checkpoint Glob")

STAGE2_CKPT = auto_find_path([
    "/kaggle/working/checkpoints_stage2/stage2_final.pt",
    "/kaggle/working/checkpoints_stage2/stage2_checkpoint_final.pt",
    "/kaggle/working/checkpoints_dora_stage2/stage2_checkpoint_final.pt",
    "/kaggle/working/stage2_checkpoint_final.pt",
    "/kaggle/working/checkpoints_dora_stage2/dora_stage2_final.pt",
    "/kaggle/working/dora_stage2_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/stage2_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/stage2_checkpoint_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/dora_stage2_final.pt",
], description="Stage 2 Appearance Checkpoint")

if not STAGE2_CKPT:
    STAGE2_CKPT = auto_find_glob([
        "/kaggle/working/**/stage2_final*.pt",
        "/kaggle/working/**/stage2_checkpoint*.pt",
        "/kaggle/working/**/dora_stage2*.pt",
        "/kaggle/input/**/stage2_checkpoint*.pt",
        "/kaggle/input/**/dora_stage2*.pt",
    ], description="Stage 2 Checkpoint Glob")

OUT_STAGE2 = "/kaggle/working/eval_results_stage2"
STAGE1_BASE = "/kaggle/working/eval_results_stage1"
os.makedirs(OUT_STAGE2, exist_ok=True)

print(f"\n[Configuration Summary]")
print(f"  • Base Model        : {MODEL_DIR}")
print(f"  • DoRA Stage 1 CKPT : {DORA_STAGE1_CKPT}")
print(f"  • Stage 2 CKPT      : {STAGE2_CKPT}")
print(f"  • Depth Cache       : {DEPTH_CACHE_DIR}")
print(f"  • Thư mục kết quả   : {OUT_STAGE2}\n")

assert STAGE2_CKPT, "Lỗi: Không tìm thấy checkpoint của Stage 2!"

TEST_SAMPLES = [
    {"name": "Train_Video (In-Domain)", "id": "000c3ab189999a83"},
    {"name": "Unseen_Video (Out-of-Domain)", "id": "145da324f69d1c6b"},
]
EVAL_TASKS = [
    {
        "name": "Quay sang trái 30° (Pan Left)",
        "tag": "pan30",
        "pose": (0.0, -30.0, 0.3, 0.0, 0.0),
        "raw_phi": -30,
        "h264_name": "gen_pan_30_stage2_h264.mp4",
        "scaff_name": "scaffold_30_h264.mp4",
        "s1_name": "gen_pan_30_h264.mp4",
    },
    {
        "name": "Đi thẳng (Forward)",
        "tag": "forward",
        "pose": (0.0, 0.0, 0.5, 0.0, 0.0),
        "raw_phi": 0,
        "h264_name": "gen_forward_stage2_h264.mp4",
        "scaff_name": "scaffold_forward_h264.mp4",
        "s1_name": "gen_forward_h264.mp4",
    },
]


# 3. HÀM TIỆN ÍCH VIDEO H.264
def convert_to_h264(input_path, output_path):
    if not os.path.exists(input_path):
        return None
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f"ffmpeg -y -loglevel error -i '{input_path}' -vcodec libx264 -pix_fmt yuv420p '{output_path}'"
    subprocess.run(cmd, shell=True, check=False)
    return output_path if os.path.exists(output_path) else input_path


def find_video_path(video_id, video_dir):
    if video_dir:
        cand = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(cand):
            return cand
    cand_glob = glob.glob(f"/kaggle/**/{video_id}.mp4", recursive=True)
    if cand_glob:
        return cand_glob[0]
    if video_dir and os.path.exists(video_dir):
        all_vids = glob.glob(os.path.join(video_dir, "*.mp4"))
        if all_vids:
            return all_vids[0]
    return None


# 4. KHỞI TẠO PIPELINE CHUẨN STAGE 2
opts_stage2 = argparse.Namespace(
    model_name=MODEL_DIR,
    transformer_path=BASE_TRANSFORMER_DIR or "none",
    base_transformer_path=BASE_TRANSFORMER_DIR,
    use_official_weights=False,
    dora_checkpoint=DORA_STAGE1_CKPT,
    stage2_checkpoint=STAGE2_CKPT,
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
    prompt="A high quality, clear indoor room with sharp textures, realistic lighting",
    negative_prompt="blur, distortion, jitter, flickering, low quality, dark patches",
    guidance_scale=4.0,
    inference_steps=25,
    depth_path=None,
    depth_cache_dir=DEPTH_CACHE_DIR,
    near=0.1,
    far=100.0,
    radius_scale=1.0,
    out_dir=OUT_STAGE2,
)

print("--> Đang nạp mô hình kết hợp Stage 1 + Stage 2...")
pipe = PipelineV3(opts_stage2)
print("✅ Nạp hoàn chỉnh Pipeline Two-Stage thành công 100%!")


# 5. THỰC THI SUY LUẬN VÀ THU THẬP KẾT QUẢ
eval_matrix = {}

for sample in TEST_SAMPLES:
    s_name = sample["name"]
    s_id = sample["id"]
    eval_matrix[s_id] = {}
    v_path = find_video_path(s_id, VIDEO_DIR)

    if not v_path or not os.path.exists(v_path):
        print(f"[Warning] Bỏ qua mẫu {s_id} do không tìm thấy file video.")
        continue

    print("\n" + "=" * 75)
    print(f"🎬 XỬ LÝ MẪU: {s_name} | ID: {s_id}")
    print("=" * 75)

    orig_h264 = os.path.join(OUT_STAGE2, f"{s_id}_orig.mp4")
    convert_to_h264(v_path, orig_h264)

    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        eval_matrix[s_id][t_tag] = {"orig": orig_h264}

        # A. Thu thập kết quả Stage 1 & Scaffold từ phiên trước (nếu có)
        prev_s1_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_after", task["s1_name"])
        prev_scaff_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_before", task["scaff_name"])
        if not os.path.exists(prev_scaff_h264):
            prev_scaff_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_after", task["scaff_name"])

        eval_matrix[s_id][t_tag]["stage1"] = prev_s1_h264 if os.path.exists(prev_s1_h264) else None
        eval_matrix[s_id][t_tag]["scaffold"] = prev_scaff_h264 if os.path.exists(prev_scaff_h264) else None

        # B. Suy luận Stage 2 Hoàn chỉnh
        out_s2_dir = os.path.join(OUT_STAGE2, s_id, f"{t_tag}_stage2")
        os.makedirs(out_s2_dir, exist_ok=True)
        h264_s2 = os.path.join(out_s2_dir, task["h264_name"])

        if os.path.exists(h264_s2):
            print(f"   [Stage 2] Đã có video Stage 2 ({t_name}).")
        else:
            print(f"   [Stage 2] Đang suy luận mô hình Stage 2 ({t_name})...")
            pipe.opts.out_dir = out_s2_dir
            pipe.run(v_path, target_pose=task["pose"])

            raw_s2 = os.path.join(out_s2_dir, f"gen_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_s2, h264_s2)

        if not eval_matrix[s_id][t_tag]["scaffold"]:
            scaff_cand = os.path.join(out_s2_dir, task["scaff_name"])
            raw_scaff = os.path.join(out_s2_dir, f"render_pan_{task['raw_phi']}.mp4")
            if os.path.exists(scaff_cand):
                eval_matrix[s_id][t_tag]["scaffold"] = scaff_cand
            elif os.path.exists(raw_scaff):
                convert_to_h264(raw_scaff, scaff_cand)
                eval_matrix[s_id][t_tag]["scaffold"] = scaff_cand

        eval_matrix[s_id][t_tag]["stage2"] = h264_s2


# 6. HIỂN THỊ BẢNG HTML SO SÁNH
def to_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

has_stage1 = any(
    eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("stage1") is not None
    for s in TEST_SAMPLES
)

html_output = """
<div style="font-family: Arial, sans-serif; background: #09090b; color: white; padding: 24px; border-radius: 12px;">
    <h2 style="color: #4ade80; text-align: center; margin-bottom: 6px;">BẢNG ĐÁNH GIÁ STAGE 2: APPEARANCE DETAIL RESTORATION</h2>
    <p style="text-align: center; color: #a1a1aa; font-size: 14px; margin-bottom: 24px;">
        Đánh giá bước nhảy vọt về độ sắc nét vân bề mặt từ DoRA Stage 1 ➔ Perceiver Ref-DiT Stage 2
    </p>
"""

for sample in TEST_SAMPLES:
    s_name = sample["name"]
    s_id = sample["id"]
    if s_id not in eval_matrix or not eval_matrix[s_id]:
        continue

    b64_orig = to_b64(eval_matrix[s_id][EVAL_TASKS[0]["tag"]]["orig"])

    html_output += f"""
    <div style="background: #18181b; padding: 18px; border-radius: 10px; margin-bottom: 25px; border: 1px solid #27272a;">
        <h3 style="color: #facc15; margin-top: 0;">🎬 Mẫu: {s_name} (ID: {s_id})</h3>
        <div style="text-align: center; margin-bottom: 16px;">
            <p style="color: #cbd5e1; font-weight: bold; margin-bottom: 6px;">Video Gốc Đầu Vào (Monocular Source)</p>
            <video style="width: 380px; border-radius: 6px; border: 2px solid #3f3f46;" controls autoplay loop muted>
                <source src="data:video/mp4;base64,{b64_orig}" type="video/mp4">
            </video>
        </div>
        <table style="width: 100%; border-collapse: collapse; text-align: center; margin-top: 10px;">
            <thead>
                <tr style="background: #27272a; color: #f4f4f5; font-size: 13px;">
                    <th style="padding: 10px; border: 1px solid #3f3f46;">Quỹ Đạo Camera</th>
                    <th style="padding: 10px; border: 1px solid #3f3f46;">3D Scaffold</th>
    """
    if has_stage1:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #fbbf24;">Stage 1 (DoRA Hình học)</th>"""
    
    html_output += """
                    <th style="padding: 10px; border: 1px solid #3f3f46; color: #4ade80;">Stage 2 (Mô hình Hoàn chỉnh)</th>
                </tr>
            </thead>
            <tbody>
    """
    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        b64_scaff = to_b64(eval_matrix[s_id][t_tag]["scaffold"])
        b64_s1 = to_b64(eval_matrix[s_id][t_tag]["stage1"])
        b64_s2 = to_b64(eval_matrix[s_id][t_tag]["stage2"])

        html_output += f"""
                <tr>
                    <td style="padding: 10px; font-weight: bold; font-size: 14px; color: #38bdf8; border: 1px solid #3f3f46; vertical-align: middle;">
                        {t_name}
                    </td>
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 250px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_scaff}" type="video/mp4">
                        </video>
                    </td>
        """
        if has_stage1:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 250px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_s1}" type="video/mp4">
                        </video>
                    </td>
            """
        html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46; background: rgba(74, 222, 128, 0.05);">
                        <video style="width: 100%; max-width: 250px; border-radius: 6px; border: 1px solid #4ade80;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_s2}" type="video/mp4">
                        </video>
                    </td>
                </tr>
        """
    html_output += """
            </tbody>
        </table>
    </div>
    """

html_output += "</div>"

report_path = os.path.join(OUT_STAGE2, "eval_stage2_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_output)
print(f"\n[Báo cáo hoàn tất] Đã lưu bảng đánh giá HTML tại: {report_path}")

if IN_IPYTHON:
    display(HTML(html_output))
print("✅ [HOÀN THÀNH] Đánh giá Stage 2 Appearance Restoration thành công 100%!")
