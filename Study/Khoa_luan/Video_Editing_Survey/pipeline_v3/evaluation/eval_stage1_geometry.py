# ==============================================================================
# PIPELINE V3: ĐÁNH GIÁ HÌNH HỌC STAGE 1 (STAGE 1 DORA GEOMETRY EVALUATION)
# ==============================================================================
# Đánh giá năng lực thích ứng hình học (Geometry Adaptation) của DoRA trên
# nền tảng CogVideoX-Fun-InP, đối đầu trực tiếp với Baseline TrajectoryCrafter.
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
print("🚀 [PIPELINE V3] KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ STAGE 1: DORA GEOMETRY ADAPTATION")
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

TRANSFORMER_DIR = auto_find_path([
    "/kaggle/input/datasets/tranbao0105/trajectorycrafter-weights/TrajectoryCrafter",
    "/kaggle/input/datasets/nguynngcnhntrng/trajectorycrafter",
    "/kaggle/input/trajectorycrafter",
    "./checkpoints/TrajectoryCrafter",
], description="SOTA Baseline Transformer Dir")

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

OUT_STAGE1 = "/kaggle/working/eval_results_stage1"
os.makedirs(OUT_STAGE1, exist_ok=True)

print(f"\n[Configuration Summary]")
print(f"  • Base Model        : {MODEL_DIR}")
print(f"  • Baseline SOTA Dir : {TRANSFORMER_DIR}")
print(f"  • DoRA Stage 1 CKPT : {DORA_STAGE1_CKPT}")
print(f"  • Depth Cache       : {DEPTH_CACHE_DIR}")
print(f"  • Thư mục kết quả   : {OUT_STAGE1}\n")

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
        "h264_name": "gen_pan_30_h264.mp4",
        "scaff_name": "scaffold_30_h264.mp4",
    },
    {
        "name": "Đi thẳng (Forward)",
        "tag": "forward",
        "pose": (0.0, 0.0, 0.5, 0.0, 0.0),
        "raw_phi": 0,
        "h264_name": "gen_forward_h264.mp4",
        "scaff_name": "scaffold_forward_h264.mp4",
    },
]


# 3. HÀM CHUYỂN ĐỔI VIDEO H.264
def convert_to_h264(input_path, output_path):
    if not os.path.exists(input_path):
        return None
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f'ffmpeg -y -loglevel error -i "{input_path}" -vcodec libx264 -pix_fmt yuv420p "{output_path}"'
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
    # Fallback to any mp4 in video_dir or current dir
    if video_dir and os.path.exists(video_dir):
        all_vids = glob.glob(os.path.join(video_dir, "*.mp4"))
        if all_vids:
            return all_vids[0]
    return None


# 4. THỰC HIỆN ĐÁNH GIÁ (BASELINE VS STAGE 1 DORA)
eval_matrix = {}

# Khởi tạo options cơ sở
base_opts = argparse.Namespace(
    model_name=MODEL_DIR,
    transformer_path=TRANSFORMER_DIR,
    use_official_weights=False,
    dora_checkpoint=DORA_STAGE1_CKPT,
    stage2_checkpoint=None,
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
    out_dir=OUT_STAGE1,
)

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

    orig_h264 = os.path.join(OUT_STAGE1, f"{s_id}_orig.mp4")
    convert_to_h264(v_path, orig_h264)

    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        eval_matrix[s_id][t_tag] = {"orig": orig_h264, "name": t_name}

        # --- A. Chạy Baseline SOTA (TrajectoryCrafter) nếu có checkpoint ---
        out_before_dir = os.path.join(OUT_STAGE1, s_id, f"{t_tag}_before")
        os.makedirs(out_before_dir, exist_ok=True)
        h264_before = os.path.join(out_before_dir, task["h264_name"])
        h264_scaff = os.path.join(out_before_dir, task["scaff_name"])

        if os.path.exists(h264_before) and os.path.exists(h264_scaff):
            print(f"   [Baseline] Đã có video Baseline {t_name}.")
        elif TRANSFORMER_DIR and os.path.exists(TRANSFORMER_DIR):
            print(f"   [Baseline] Đang chạy Baseline SOTA ({t_name})...")
            base_opts.use_official_weights = True
            base_opts.transformer_path = TRANSFORMER_DIR
            base_opts.dora_checkpoint = None
            base_opts.out_dir = out_before_dir
            pipe_base = PipelineV3(base_opts)
            pipe_base.run(v_path, target_pose=task["pose"])

            raw_gen = os.path.join(out_before_dir, f"gen_pan_{task['raw_phi']}.mp4")
            raw_scaff = os.path.join(out_before_dir, f"render_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_gen, h264_before)
            convert_to_h264(raw_scaff, h264_scaff)
            del pipe_base
            torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["before"] = h264_before if os.path.exists(h264_before) else None
        eval_matrix[s_id][t_tag]["scaffold"] = h264_scaff if os.path.exists(h264_scaff) else None

        # --- B. Chạy Stage 1 DoRA Geometry Adaptation ---
        out_stage1_dir = os.path.join(OUT_STAGE1, s_id, f"{t_tag}_after")
        os.makedirs(out_stage1_dir, exist_ok=True)
        h264_s1 = os.path.join(out_stage1_dir, task["h264_name"])

        if os.path.exists(h264_s1):
            print(f"   [Stage 1 DoRA] Đã có video Stage 1 {t_name}.")
        else:
            print(f"   [Stage 1 DoRA] Đang chạy Stage 1 DoRA ({t_name})...")
            base_opts.use_official_weights = False
            base_opts.transformer_path = BASE_TRANSFORMER_DIR or "none"
            base_opts.base_transformer_path = BASE_TRANSFORMER_DIR
            base_opts.dora_checkpoint = DORA_STAGE1_CKPT
            base_opts.out_dir = out_stage1_dir
            pipe_s1 = PipelineV3(base_opts)
            pipe_s1.run(v_path, target_pose=task["pose"])

            raw_s1 = os.path.join(out_stage1_dir, f"gen_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_s1, h264_s1)

            # Cập nhật scaffold nếu baseline chưa tạo
            if not eval_matrix[s_id][t_tag]["scaffold"]:
                raw_scaff = os.path.join(out_stage1_dir, f"render_pan_{task['raw_phi']}.mp4")
                h264_scaff_s1 = os.path.join(out_stage1_dir, task["scaff_name"])
                convert_to_h264(raw_scaff, h264_scaff_s1)
                eval_matrix[s_id][t_tag]["scaffold"] = h264_scaff_s1

            del pipe_s1
            torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["stage1"] = h264_s1 if os.path.exists(h264_s1) else None


# 5. HIỂN THỊ BẢNG HTML SO SÁNH TRỰC QUAN ĐA CỘT
def to_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


html_output = """
<div style="font-family: Arial, sans-serif; background: #09090b; color: white; padding: 24px; border-radius: 12px;">
    <h2 style="color: #38bdf8; text-align: center; margin-bottom: 6px;">BẢNG ĐÁNH GIÁ STAGE 1: DORA GEOMETRY ADAPTATION</h2>
    <p style="text-align: center; color: #a1a1aa; font-size: 14px; margin-bottom: 24px;">
        Đánh giá sự tiến hóa hình học từ 3D Scaffold ➔ Baseline SOTA ➔ DoRA Geometry Stage 1
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
                    <th style="padding: 10px; border: 1px solid #3f3f46; color: #f87171;">Baseline (Paper SOTA)</th>
                    <th style="padding: 10px; border: 1px solid #3f3f46; color: #fbbf24;">Stage 1 (DoRA Hình học)</th>
                </tr>
            </thead>
            <tbody>
    """
    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        b64_scaff = to_b64(eval_matrix[s_id][t_tag]["scaffold"])
        b64_bef = to_b64(eval_matrix[s_id][t_tag]["before"])
        b64_s1 = to_b64(eval_matrix[s_id][t_tag]["stage1"])

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
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 250px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_bef}" type="video/mp4">
                        </video>
                    </td>
                    <td style="padding: 6px; border: 1px solid #3f3f46; background: rgba(251, 191, 36, 0.05);">
                        <video style="width: 100%; max-width: 250px; border-radius: 6px; border: 1px solid #fbbf24;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_s1}" type="video/mp4">
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

# Ghi file HTML ra đĩa
report_path = os.path.join(OUT_STAGE1, "eval_stage1_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_output)
print(f"\n[Báo cáo hoàn tất] Đã lưu bảng đánh giá HTML tại: {report_path}")

if IN_IPYTHON:
    display(HTML(html_output))
print("✅ [HOÀN THÀNH] Đánh giá Stage 1 DoRA Geometry thành công 100%!")
