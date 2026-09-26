# ==============================================================================
# PIPELINE V3: KIỂM THỬ HOÀN CHỈNH TOÀN DIỆN (FINAL TWO-STAGE EVALUATION)
# ==============================================================================
# Đánh giá đa chiều (Ablation Study) toàn bộ tiến trình tiến hóa:
# 3D Scaffold ➔ Baseline SOTA (TrajectoryCrafter) ➔ DoRA Stage 1 ➔ Two-Stage Stage 2
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

print("=" * 80)
print("🏆 [PIPELINE V3] KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ TOÀN DIỆN: MÔ HÌNH TWO-STAGE HOÀN CHỈNH")
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

DORA_STAGE2_CKPT = auto_find_path([
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

if not DORA_STAGE2_CKPT:
    DORA_STAGE2_CKPT = auto_find_glob([
        "/kaggle/working/**/stage2_final*.pt",
        "/kaggle/working/**/stage2_checkpoint*.pt",
        "/kaggle/working/**/dora_stage2*.pt",
        "/kaggle/input/**/stage2_checkpoint*.pt",
        "/kaggle/input/**/dora_stage2*.pt",
    ], description="Stage 2 Checkpoint Glob")

OUT_FINAL = "/kaggle/working/eval_results_final"
STAGE1_BASE = "/kaggle/working/eval_results_stage1"
STAGE2_BASE = "/kaggle/working/eval_results_stage2"
os.makedirs(OUT_FINAL, exist_ok=True)

print(f"\n[Configuration Summary]")
print(f"  • Base Model        : {MODEL_DIR}")
print(f"  • Baseline SOTA Dir : {TRANSFORMER_DIR}")
print(f"  • DoRA Stage 1 CKPT : {DORA_STAGE1_CKPT}")
print(f"  • Stage 2 CKPT      : {DORA_STAGE2_CKPT}")
print(f"  • Depth Cache       : {DEPTH_CACHE_DIR}")
print(f"  • Thư mục kết quả   : {OUT_FINAL}\n")

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


# 3. HÀM CHUYỂN ĐỔI VIDEO H.264
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


# 4. KHỞI TẠO VÀ THỰC THI PIPELINE HOÀN CHỈNH
base_opts = argparse.Namespace(
    model_name=MODEL_DIR,
    transformer_path=TRANSFORMER_DIR,
    use_official_weights=False,
    dora_checkpoint=DORA_STAGE1_CKPT,
    stage2_checkpoint=DORA_STAGE2_CKPT,
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
    out_dir=OUT_FINAL,
)

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

    orig_h264 = os.path.join(OUT_FINAL, f"{s_id}_orig.mp4")
    convert_to_h264(v_path, orig_h264)

    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        eval_matrix[s_id][t_tag] = {"orig": orig_h264}

        # --- A. Thu thập kết quả Baseline SOTA & Scaffold ---
        prev_before_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_before", task["s1_name"])
        prev_scaff_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_before", task["scaff_name"])
        if not os.path.exists(prev_scaff_h264):
            prev_scaff_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_after", task["scaff_name"])
        if not os.path.exists(prev_scaff_h264):
            prev_scaff_h264 = os.path.join(STAGE2_BASE, s_id, f"{t_tag}_stage2", task["scaff_name"])

        # Nếu chưa có sẵn, chạy baseline
        if not os.path.exists(prev_before_h264) and TRANSFORMER_DIR and os.path.exists(TRANSFORMER_DIR):
            out_bef = os.path.join(OUT_FINAL, s_id, f"{t_tag}_baseline")
            os.makedirs(out_bef, exist_ok=True)
            prev_before_h264 = os.path.join(out_bef, task["s1_name"])
            if not os.path.exists(prev_before_h264):
                print(f"   [Baseline] Đang suy luận Baseline SOTA ({t_name})...")
                base_opts.use_official_weights = True
                base_opts.transformer_path = TRANSFORMER_DIR
                base_opts.dora_checkpoint = None
                base_opts.stage2_checkpoint = None
                base_opts.out_dir = out_bef
                pipe_base = PipelineV3(base_opts)
                pipe_base.run(v_path, target_pose=task["pose"])
                convert_to_h264(os.path.join(out_bef, f"gen_pan_{task['raw_phi']}.mp4"), prev_before_h264)
                if not os.path.exists(prev_scaff_h264):
                    prev_scaff_h264 = os.path.join(out_bef, task["scaff_name"])
                    convert_to_h264(os.path.join(out_bef, f"render_pan_{task['raw_phi']}.mp4"), prev_scaff_h264)
                del pipe_base
                torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["before"] = prev_before_h264 if os.path.exists(prev_before_h264) else None
        eval_matrix[s_id][t_tag]["scaffold"] = prev_scaff_h264 if os.path.exists(prev_scaff_h264) else None

        # --- B. Thu thập kết quả Stage 1 DoRA ---
        prev_stage1_h264 = os.path.join(STAGE1_BASE, s_id, f"{t_tag}_after", task["s1_name"])
        if not os.path.exists(prev_stage1_h264) and DORA_STAGE1_CKPT:
            out_s1 = os.path.join(OUT_FINAL, s_id, f"{t_tag}_stage1")
            os.makedirs(out_s1, exist_ok=True)
            prev_stage1_h264 = os.path.join(out_s1, task["s1_name"])
            if not os.path.exists(prev_stage1_h264):
                print(f"   [Stage 1] Đang suy luận Stage 1 DoRA ({t_name})...")
                base_opts.use_official_weights = False
                base_opts.transformer_path = BASE_TRANSFORMER_DIR or "none"
                base_opts.base_transformer_path = BASE_TRANSFORMER_DIR
                base_opts.dora_checkpoint = DORA_STAGE1_CKPT
                base_opts.stage2_checkpoint = None
                base_opts.out_dir = out_s1
                pipe_s1 = PipelineV3(base_opts)
                pipe_s1.run(v_path, target_pose=task["pose"])
                convert_to_h264(os.path.join(out_s1, f"gen_pan_{task['raw_phi']}.mp4"), prev_stage1_h264)
                if not eval_matrix[s_id][t_tag]["scaffold"]:
                    scaff_s1 = os.path.join(out_s1, task["scaff_name"])
                    convert_to_h264(os.path.join(out_s1, f"render_pan_{task['raw_phi']}.mp4"), scaff_s1)
                    eval_matrix[s_id][t_tag]["scaffold"] = scaff_s1
                del pipe_s1
                torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["stage1"] = prev_stage1_h264 if os.path.exists(prev_stage1_h264) else None

        # --- C. Thu thập / Suy luận Stage 2 Hoàn chỉnh ---
        prev_s2_h264 = os.path.join(STAGE2_BASE, s_id, f"{t_tag}_stage2", task["h264_name"])
        out_s2_dir = os.path.join(OUT_FINAL, s_id, f"{t_tag}_stage2")
        os.makedirs(out_s2_dir, exist_ok=True)
        h264_s2 = os.path.join(out_s2_dir, task["h264_name"])

        if os.path.exists(prev_s2_h264):
            eval_matrix[s_id][t_tag]["stage2"] = prev_s2_h264
            print(f"   [Stage 2] Tái sử dụng kết quả Stage 2 sẵn có từ phiên trước.")
        elif os.path.exists(h264_s2):
            eval_matrix[s_id][t_tag]["stage2"] = h264_s2
            print(f"   [Stage 2] Đã có video Stage 2 ({t_name}).")
        elif DORA_STAGE2_CKPT:
            print(f"   [Stage 2] Đang suy luận mô hình Hoàn chỉnh Stage 2 ({t_name})...")
            base_opts.use_official_weights = False
            base_opts.transformer_path = BASE_TRANSFORMER_DIR or "none"
            base_opts.base_transformer_path = BASE_TRANSFORMER_DIR
            base_opts.dora_checkpoint = DORA_STAGE1_CKPT
            base_opts.stage2_checkpoint = DORA_STAGE2_CKPT
            base_opts.out_dir = out_s2_dir
            pipe_s2 = PipelineV3(base_opts)
            pipe_s2.run(v_path, target_pose=task["pose"])

            raw_s2 = os.path.join(out_s2_dir, f"gen_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_s2, h264_s2)

            if not eval_matrix[s_id][t_tag]["scaffold"]:
                raw_scaff = os.path.join(out_s2_dir, f"render_pan_{task['raw_phi']}.mp4")
                h264_scaff = os.path.join(out_s2_dir, task["scaff_name"])
                convert_to_h264(raw_scaff, h264_scaff)
                eval_matrix[s_id][t_tag]["scaffold"] = h264_scaff

            del pipe_s2
            torch.cuda.empty_cache()
            eval_matrix[s_id][t_tag]["stage2"] = h264_s2
        else:
            eval_matrix[s_id][t_tag]["stage2"] = None


# 5. HIỂN THỊ BẢNG HTML SO SÁNH TRỰC QUAN ĐA CỘT (ABLATION STUDY)
def to_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

has_baseline = any(eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("before") is not None for s in TEST_SAMPLES)
has_stage1 = any(eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("stage1") is not None for s in TEST_SAMPLES)
has_stage2 = any(eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("stage2") is not None for s in TEST_SAMPLES)

html_output = """
<div style="font-family: Arial, sans-serif; background: #09090b; color: white; padding: 24px; border-radius: 12px;">
    <h2 style="color: #38bdf8; text-align: center; margin-bottom: 6px;">BẢNG SO SÁNH HIỆU NĂNG: TOÀN BỘ CÁC GIAI ĐOẠN (ABLATION STUDY)</h2>
    <p style="text-align: center; color: #a1a1aa; font-size: 14px; margin-bottom: 24px;">
        Đánh giá sự tiến hóa từ 3D Scaffold ➔ Baseline SOTA ➔ DoRA Geometry (Stage 1) ➔ Ref-DiT Appearance (Stage 2)
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
    if has_baseline:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #f87171;">Baseline (Paper SOTA)</th>"""
    if has_stage1:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #fbbf24;">Stage 1 (DoRA Hình học)</th>"""
    if has_stage2:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #4ade80;">Stage 2 (Mô hình Hoàn chỉnh)</th>"""
    
    html_output += """
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
        b64_s2 = to_b64(eval_matrix[s_id][t_tag]["stage2"])

        html_output += f"""
                <tr>
                    <td style="padding: 10px; font-weight: bold; font-size: 14px; color: #38bdf8; border: 1px solid #3f3f46; vertical-align: middle;">
                        {t_name}
                    </td>
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 240px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_scaff}" type="video/mp4">
                        </video>
                    </td>
        """
        if has_baseline:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 240px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_bef}" type="video/mp4">
                        </video>
                    </td>
            """
        if has_stage1:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 240px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_s1}" type="video/mp4">
                        </video>
                    </td>
            """
        if has_stage2:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46; background: rgba(74, 222, 128, 0.05);">
                        <video style="width: 100%; max-width: 240px; border-radius: 6px; border: 1px solid #4ade80;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_s2}" type="video/mp4">
                        </video>
                    </td>
            """
        html_output += """
                </tr>
        """
    html_output += """
            </tbody>
        </table>
    </div>
    """

html_output += "</div>"

report_path = os.path.join(OUT_FINAL, "eval_final_ablation_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_output)
print(f"\n[Báo cáo hoàn tất] Đã lưu bảng đánh giá HTML tại: {report_path}")

if IN_IPYTHON:
    display(HTML(html_output))
print("🏆 [HOÀN TẤT ĐỈNH CAO] Bảng đánh giá Ablation Study hoàn chỉnh đã được tạo thành công 100%!")
