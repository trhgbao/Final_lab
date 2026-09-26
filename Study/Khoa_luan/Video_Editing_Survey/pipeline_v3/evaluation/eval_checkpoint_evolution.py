# ==============================================================================
# PIPELINE V3: ĐÁNH GIÁ TIẾN HÓA MÔ HÌNH (CHECKPOINT EVOLUTION COMPARATOR)
# ==============================================================================
# So sánh đa chiều quá trình cải tiến mô hình:
# Cột 1: Paper gốc (TrajectoryCrafter SOTA weights)
# Cột 2: Mô hình của chúng ta - Chưa train (Step 0 Base CogVideoX-Fun InP)
# Cột 3..N: Tất cả các checkpoint hoàn chỉnh trong Dataset/Working (Step 400, Step 1000, ...)
# ==============================================================================

import os
import sys
import time
import warnings
import traceback
import re
import glob
import base64
import argparse
import subprocess
import gc
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    from diffusers.utils import logging as d_logging
    from transformers import logging as t_logging
    d_logging.set_verbosity_error()
    t_logging.set_verbosity_error()
except Exception:
    pass

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
print("🚀 [PIPELINE V3] HỆ THỐNG ĐÁNH GIÁ TIẾN HÓA CÁC PHIÊN BẢN CHECKPOINT (TWO-STAGE)")
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
        if p and os.path.exists(p):
            print(f"[*] Tìm thấy {description}: {p}")
            return p
    return None

MODEL_DIR = auto_find_path([
    "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
    "/kaggle/input/cogvideox-fun-inp",
    "./checkpoints/CogVideoX-Fun-V1.1-5b-InP",
], description="Base Model Dir (CogVideoX-Fun InP)")

TRANSFORMER_DIR = auto_find_path([
    "/kaggle/input/datasets/tranbao0105/trajectorycrafter-weights/TrajectoryCrafter",
    "/kaggle/input/datasets/nguynngcnhntrng/trajectorycrafter",
    "/kaggle/input/trajectorycrafter",
    "./checkpoints/TrajectoryCrafter",
], description="SOTA Baseline Transformer Dir (Paper gốc)")

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

OUT_EVAL = "/kaggle/working/eval_checkpoint_evolution"
os.makedirs(OUT_EVAL, exist_ok=True)


# 3. QUÉT VÀ GHÉP CẶP TẤT CẢ CÁC CHECKPOINT TRONG DATASET & WORKING
def parse_step_num(path):
    if not path:
        return 0
    norm_path = path.replace("\\", "/").lower()
    m = re.search(r"step(\d+)", norm_path)
    if m:
        return int(m.group(1))
    base = os.path.basename(norm_path)
    nums = re.findall(r"\d+", base)
    if len(nums) > 1:
        return int(nums[-1])
    elif len(nums) == 1 and not ("stage" in base and nums[0] in ["1", "2"]):
        return int(nums[0])
    return 999999 if "final" in norm_path else 0

candidate_dirs = [
    "/kaggle/input/datasets/tranbao0105/pipeline",
    "/kaggle/input/pipeline",
    "/kaggle/working/checkpoints_stage1",
    "/kaggle/working/checkpoints_stage2",
    "/kaggle/working/checkpoints_dora_stage1",
    "/kaggle/working/checkpoints_dora_stage2",
    "/kaggle/working",
    "./checkpoints_stage1",
    "./checkpoints_stage2",
]

s1_files = []
s2_files = []

for cd in candidate_dirs:
    if os.path.exists(cd):
        for root, dirs, files in os.walk(cd):
            for f in files:
                full = os.path.join(root, f)
                norm_full = full.replace("\\", "/").lower()
                if (f.endswith(".pt") or f.endswith(".pth") or f.endswith(".bin")) and os.path.getsize(full) > 1024:
                    if "stage1" in norm_full or "dora_checkpoint" in norm_full or "stage_1" in norm_full:
                        s1_files.append(full)
                    elif "stage2" in norm_full or "stage_2" in norm_full:
                        s2_files.append(full)
                elif f == "data.pkl":
                    if "stage1" in norm_full or "dora" in norm_full:
                        s1_files.append(full)
                    elif "stage2" in norm_full:
                        s2_files.append(full)

# Lọc trùng lặp file
s1_files = sorted(list(set(s1_files)))
s2_files = sorted(list(set(s2_files)))

def is_working_path(p):
    if not p:
        return False
    norm = os.path.abspath(p).replace("\\", "/")
    return "/working" in norm or "./checkpoints" in norm or "/tmp" in norm

# Phân tách checkpoint từ Dataset và checkpoint vừa train trong working
dataset_s1 = [f for f in s1_files if not is_working_path(f)]
dataset_s2 = [f for f in s2_files if not is_working_path(f)]
working_s1 = [f for f in s1_files if is_working_path(f)]
working_s2 = [f for f in s2_files if is_working_path(f)]

checkpoint_models = []

# A. Thêm tất cả các checkpoint có trong Dataset
dataset_steps = sorted(list(set([parse_step_num(f) for f in dataset_s1 + dataset_s2 if parse_step_num(f) > 0])))
for step in dataset_steps:
    matching_s1 = [f for f in dataset_s1 if parse_step_num(f) == step]
    if not matching_s1:
        cands = [f for f in dataset_s1 if parse_step_num(f) <= step]
        matching_s1 = [sorted(cands, key=parse_step_num)[-1]] if cands else dataset_s1
    matching_s2 = [f for f in dataset_s2 if parse_step_num(f) == step]
    if not matching_s2:
        cands = [f for f in dataset_s2 if parse_step_num(f) <= step]
        matching_s2 = [sorted(cands, key=parse_step_num)[-1]] if cands else dataset_s2

    s1_path = matching_s1[0] if matching_s1 else None
    s2_path = matching_s2[0] if matching_s2 else None

    if s1_path or s2_path:
        checkpoint_models.append({
            "key": f"dataset_step_{step}",
            "label": f"Dataset (Step {step})",
            "step": step,
            "stage1": s1_path,
            "stage2": s2_path,
            "is_working": False,
        })

# B. Thêm checkpoint VỪA HUẤN LUYỆN XONG trong phiên làm việc hiện tại (working)
working_steps = sorted(list(set([parse_step_num(f) for f in working_s1 + working_s2 if parse_step_num(f) > 0])))
for step in working_steps:
    matching_s1 = [f for f in working_s1 if parse_step_num(f) == step]
    if not matching_s1:
        cands = [f for f in working_s1 if parse_step_num(f) <= step]
        matching_s1 = [sorted(cands, key=parse_step_num)[-1]] if cands else (working_s1 or dataset_s1)
    matching_s2 = [f for f in working_s2 if parse_step_num(f) == step]
    if not matching_s2:
        cands = [f for f in working_s2 if parse_step_num(f) <= step]
        matching_s2 = [sorted(cands, key=parse_step_num)[-1]] if cands else (working_s2 or dataset_s2)

    s1_path = matching_s1[0] if matching_s1 else None
    s2_path = matching_s2[0] if matching_s2 else None

    if s1_path or s2_path:
        checkpoint_models.append({
            "key": f"working_step_{step}",
            "label": f"Vừa Train Xong (Step {step})",
            "step": step,
            "stage1": s1_path,
            "stage2": s2_path,
            "is_working": True,
        })

print(f"\n[Checkpoint Discovery] Tìm thấy tổng cộng {len(checkpoint_models)} phiên bản checkpoint:")
for cm in checkpoint_models:
    s1_name = os.path.basename(cm["stage1"]) if cm["stage1"] else "None"
    s2_name = os.path.basename(cm["stage2"]) if cm["stage2"] else "None"
    origin_tag = " [VỪA TRAIN XONG]" if cm["is_working"] else " [DATASET]"
    print(f"  • {cm['label']}{origin_tag}: Stage 1 = {s1_name} | Stage 2 = {s2_name}")


# 4. ĐỊNH NGHĨA TẬP MẪU KIỂM THỬ VÀ NHIỆM VỤ
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
    },
    {
        "name": "Đi thẳng (Forward)",
        "tag": "forward",
        "pose": (0.0, 0.0, 0.5, 0.0, 0.0),
        "raw_phi": 0,
    },
]


# 5. TIỆN ÍCH VIDEO & FFMPEG
def convert_to_h264(input_path, output_path):
    if not input_path or not os.path.exists(input_path):
        return None
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = f'ffmpeg -y -loglevel error -i "{input_path}" -vcodec libx264 -pix_fmt yuv420p "{output_path}"'
    subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 else input_path

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

def to_b64(path):
    if path and os.path.exists(path) and os.path.getsize(path) > 1024:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# 6. KHỞI TẠO VÀ THỰC THI KIỂM THỬ ĐA PHIÊN BẢN
# Base inference options template
def make_opts(out_dir, use_official=False, dora_ckpt=None, s2_ckpt=None, quiet=True):
    return argparse.Namespace(
        model_name=MODEL_DIR,
        transformer_path=TRANSFORMER_DIR if use_official else (BASE_TRANSFORMER_DIR or "none"),
        base_transformer_path=BASE_TRANSFORMER_DIR,
        use_official_weights=use_official,
        dora_checkpoint=dora_ckpt,
        stage2_checkpoint=s2_ckpt,
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
        out_dir=out_dir,
        quiet=quiet,
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

    print("\n" + "=" * 80)
    print(f"🎬 XỬ LÝ MẪU: {s_name} | ID: {s_id}")
    print("=" * 80)

    sample_out_dir = os.path.join(OUT_EVAL, s_id)
    os.makedirs(sample_out_dir, exist_ok=True)
    orig_h264 = os.path.join(sample_out_dir, f"{s_id}_orig_h264.mp4")
    convert_to_h264(v_path, orig_h264)

    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        eval_matrix[s_id][t_tag] = {"orig": orig_h264}

        task_out_dir = os.path.join(sample_out_dir, t_tag)
        os.makedirs(task_out_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1. CỘT 1: PAPER GỐC (TRAJECTORYCRAFTER OFFICIAL SOTA)
        # ----------------------------------------------------------------------
        base_h264 = os.path.join(task_out_dir, "baseline_sota_h264.mp4")
        scaff_h264 = os.path.join(task_out_dir, "scaffold_h264.mp4")

        # Kiểm tra tái sử dụng kết quả đã suy luận trước đó
        reused_baseline = False
        cand_baseline_paths = [
            base_h264,
            os.path.join("/kaggle/working/eval_results_final", s_id, f"{t_tag}_baseline", f"gen_pan_{task['raw_phi']}.mp4"),
            os.path.join("/kaggle/working/eval_results_stage1", s_id, f"{t_tag}_before", f"gen_pan_{task['raw_phi']}.mp4"),
        ]
        for cbp in cand_baseline_paths:
            if os.path.exists(cbp) and os.path.getsize(cbp) > 1024:
                convert_to_h264(cbp, base_h264)
                reused_baseline = True
                print(f"   [Cột 1: Paper gốc] Tái sử dụng video có sẵn ({t_name}): {base_h264}")
                break

        if not reused_baseline and TRANSFORMER_DIR and os.path.exists(TRANSFORMER_DIR):
            print(f"   [Cột 1: Paper gốc] Đang suy luận TrajectoryCrafter Official SOTA ({t_name})...")
            run_dir = os.path.join(task_out_dir, "run_baseline")
            opts = make_opts(run_dir, use_official=True)
            pipe = PipelineV3(opts)
            pipe.run(v_path, target_pose=task["pose"])
            raw_gen = os.path.join(run_dir, f"gen_pan_{task['raw_phi']}.mp4")
            raw_scaff = os.path.join(run_dir, f"render_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_gen, base_h264)
            convert_to_h264(raw_scaff, scaff_h264)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["scaffold"] = scaff_h264 if os.path.exists(scaff_h264) else None
        eval_matrix[s_id][t_tag]["baseline"] = base_h264 if os.path.exists(base_h264) else None

        # ----------------------------------------------------------------------
        # 2. CỘT 2: MODEL CỦA CHÚNG TA - STEP 0 (CHƯA HUẤN LUYỆN GÌ HẾT)
        # ----------------------------------------------------------------------
        step0_h264 = os.path.join(task_out_dir, "step0_untrained_h264.mp4")
        if os.path.exists(step0_h264) and os.path.getsize(step0_h264) > 1024:
            print(f"   [Cột 2: Step 0] Tái sử dụng video có sẵn ({t_name}): {step0_h264}")
        else:
            print(f"   [Cột 2: Step 0] Đang suy luận Mô hình chưa train Step 0 ({t_name})...")
            run_dir = os.path.join(task_out_dir, "run_step0")
            opts = make_opts(run_dir, use_official=False, dora_ckpt=None, s2_ckpt=None)
            pipe = PipelineV3(opts)
            pipe.run(v_path, target_pose=task["pose"])
            raw_gen = os.path.join(run_dir, f"gen_pan_{task['raw_phi']}.mp4")
            raw_scaff = os.path.join(run_dir, f"render_pan_{task['raw_phi']}.mp4")
            convert_to_h264(raw_gen, step0_h264)
            if not eval_matrix[s_id][t_tag]["scaffold"]:
                convert_to_h264(raw_scaff, scaff_h264)
                eval_matrix[s_id][t_tag]["scaffold"] = scaff_h264
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

        eval_matrix[s_id][t_tag]["step0"] = step0_h264 if os.path.exists(step0_h264) else None

        # ----------------------------------------------------------------------
        # 3. CÁC CỘT TIẾP THEO: TẤT CẢ CÁC CHECKPOINT HOÀN CHỈNH TỪNG STEP
        # ----------------------------------------------------------------------
        for cm in checkpoint_models:
            c_key = cm["key"]
            c_label = cm["label"]
            c_h264 = os.path.join(task_out_dir, f"{c_key}_h264.mp4")

            if os.path.exists(c_h264) and os.path.getsize(c_h264) > 1024:
                print(f"   [{c_label}] Tái sử dụng video có sẵn ({t_name}): {c_h264}")
                eval_matrix[s_id][t_tag][c_key] = c_h264
            else:
                print(f"   [{c_label}] Đang suy luận phiên bản hoàn chỉnh ({t_name})...")
                run_dir = os.path.join(task_out_dir, f"run_{c_key}")
                opts = make_opts(run_dir, use_official=False, dora_ckpt=cm["stage1"], s2_ckpt=cm["stage2"])
                pipe = PipelineV3(opts)
                pipe.run(v_path, target_pose=task["pose"])
                raw_gen = os.path.join(run_dir, f"gen_pan_{task['raw_phi']}.mp4")
                raw_scaff = os.path.join(run_dir, f"render_pan_{task['raw_phi']}.mp4")
                convert_to_h264(raw_gen, c_h264)
                if not eval_matrix[s_id][t_tag]["scaffold"]:
                    convert_to_h264(raw_scaff, scaff_h264)
                    eval_matrix[s_id][t_tag]["scaffold"] = scaff_h264
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                eval_matrix[s_id][t_tag][c_key] = c_h264 if os.path.exists(c_h264) else None


# 7. TẠO BẢNG HTML SO SÁNH TRỰC QUAN ĐA CỘT (EVOLUTION STUDY)
has_baseline = any(eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("baseline") is not None for s in TEST_SAMPLES)
has_step0 = any(eval_matrix.get(s["id"], {}).get(EVAL_TASKS[0]["tag"], {}).get("step0") is not None for s in TEST_SAMPLES)

html_output = """
<div style="font-family: Arial, sans-serif; background: #09090b; color: white; padding: 24px; border-radius: 12px;">
    <h2 style="color: #38bdf8; text-align: center; margin-bottom: 6px;">BẢNG SO SÁNH TIẾN HÓA TRỌNG SỐ (CHECKPOINT EVOLUTION BENCHMARK)</h2>
    <p style="text-align: center; color: #a1a1aa; font-size: 14px; margin-bottom: 24px;">
        So sánh trực tiếp: <b>Paper SOTA (TrajectoryCrafter)</b> ➔ <b>Model chúng ta (Step 0)</b> ➔ <b>Các mốc Checkpoint Hoàn chỉnh</b>
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
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; text-align: center; margin-top: 10px; min-width: 850px;">
            <thead>
                <tr style="background: #27272a; color: #f4f4f5; font-size: 13px;">
                    <th style="padding: 10px; border: 1px solid #3f3f46; min-width: 130px;">Quỹ Đạo Camera</th>
                    <th style="padding: 10px; border: 1px solid #3f3f46; min-width: 180px;">3D Scaffold</th>
    """
    if has_baseline:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #f87171; min-width: 190px;">Paper gốc<br><span style="font-size:11px;font-weight:normal;">(TrajectoryCrafter SOTA)</span></th>"""
    if has_step0:
        html_output += """<th style="padding: 10px; border: 1px solid #3f3f46; color: #fbbf24; min-width: 190px;">Model chúng ta (Step 0)<br><span style="font-size:11px;font-weight:normal;">(Chưa huấn luyện)</span></th>"""
    for cm in checkpoint_models:
        if cm.get("is_working"):
            html_output += f"""<th style="padding: 10px; border: 2px solid #22c55e; background: rgba(34, 197, 94, 0.18); color: #86efac; min-width: 200px;">🎯 {cm['label']}<br><span style="font-size:11px;font-weight:bold;color:#4ade80;">[VỪA TRAIN XONG]</span></th>"""
        else:
            html_output += f"""<th style="padding: 10px; border: 1px solid #38bdf8; background: rgba(56, 189, 248, 0.08); color: #38bdf8; min-width: 190px;">📦 {cm['label']}<br><span style="font-size:11px;font-weight:normal;color:#93c5fd;">[DATASET]</span></th>"""

    html_output += """
                </tr>
            </thead>
            <tbody>
    """

    for task in EVAL_TASKS:
        t_tag = task["tag"]
        t_name = task["name"]
        b64_scaff = to_b64(eval_matrix[s_id][t_tag].get("scaffold"))
        b64_base = to_b64(eval_matrix[s_id][t_tag].get("baseline"))
        b64_step0 = to_b64(eval_matrix[s_id][t_tag].get("step0"))

        html_output += f"""
                <tr>
                    <td style="padding: 10px; font-weight: bold; font-size: 13px; color: #38bdf8; border: 1px solid #3f3f46; vertical-align: middle;">
                        {t_name}
                    </td>
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 220px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_scaff}" type="video/mp4">
                        </video>
                    </td>
        """
        if has_baseline:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 220px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_base}" type="video/mp4">
                        </video>
                    </td>
            """
        if has_step0:
            html_output += f"""
                    <td style="padding: 6px; border: 1px solid #3f3f46;">
                        <video style="width: 100%; max-width: 220px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_step0}" type="video/mp4">
                        </video>
                    </td>
            """
        for cm in checkpoint_models:
            b64_cm = to_b64(eval_matrix[s_id][t_tag].get(cm["key"]))
            bg_cell = "background: rgba(34, 197, 94, 0.08); border: 2px solid #22c55e;" if cm.get("is_working") else "background: rgba(56, 189, 248, 0.04); border: 1px solid #3f3f46;"
            html_output += f"""
                    <td style="padding: 6px; {bg_cell}">
                        <video style="width: 100%; max-width: 220px; border-radius: 6px;" controls autoplay loop muted>
                            <source src="data:video/mp4;base64,{b64_cm}" type="video/mp4">
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
    </div>
    """

html_output += "</div>"

report_path = os.path.join(OUT_EVAL, "eval_checkpoint_evolution_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_output)

root_report = "/kaggle/working/eval_checkpoint_evolution_report.html"
try:
    with open(root_report, "w", encoding="utf-8") as f:
        f.write(html_output)
except Exception:
    pass

elapsed = time.time() - start_time
mm, ss = divmod(int(elapsed), 60)
print("\n" + "=" * 75)
print(f"⏱️ Tổng thời gian chạy: {mm:02d}m {ss:02d}s")
print(f"📁 Đường dẫn lưu file kết quả:")
print(f"   - Báo cáo HTML: {report_path}")
print(f"   - Bản sao root: {root_report}")
print("=" * 75 + "\n")

if IN_IPYTHON:
    display(HTML(html_output))
print("🏆 [HOÀN TẤT] Bảng đánh giá tiến hóa Checkpoint đã sẵn sàng!")
