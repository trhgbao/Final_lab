# ==============================================================================
# PIPELINE V3: KIỂM TRA MÔI TRƯỜNG & DỌN DẸP VRAM (SYSTEM & RESOURCE CHECK)
# ==============================================================================

import os
import gc
import sys
import glob
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 80)
print("🧹 [CELL 1] DỌN SẠCH VRAM VÀ KIỂM TRA TÀI NGUYÊN MÔI TRƯỜNG GPU")
print("=" * 80)

# 1. Dọn sạch bộ nhớ
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# 2. Thông tin GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024 ** 3)
    total_gb = total_mem / (1024 ** 3)
    print(f"[*] GPU Model           : {gpu_name}")
    print(f"[*] VRAM khả dụng       : {free_gb:.2f} GB / {total_gb:.2f} GB")
    print(f"[*] PyTorch Version     : {torch.__version__}")
    print(f"[*] CUDA Available      : {torch.cuda.is_available()} (Version: {torch.version.cuda})")
else:
    print("[!] CẢNH BÁO: Không phát hiện GPU CUDA!")

# 3. Kiểm tra các thư mục tài nguyên trọng yếu trên Kaggle
def check_path(name, paths):
    for p in paths:
        if os.path.exists(p):
            print(f"[*] {name:<22}: ✅ {p}")
            return p
    print(f"[*] {name:<22}: ❌ Không tìm thấy!")
    return None

print("\n" + "-" * 80)
print("[*] KIỂM TRA ĐƯỜNG DẪN DỮ LIỆU VÀ TRỌNG SỐ MÔ HÌNH:")
print("-" * 80)

check_path("Base Model (InP)", [
    "/kaggle/input/datasets/tranbao0105/cogvideox-fun-components/CogVideoX-Fun-V1.1-5b-InP",
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-inp",
    "/kaggle/input/cogvideox-fun-inp",
])

check_path("Base Transformer", [
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer/transformer",
    "/kaggle/input/datasets/nguynngcnhntrng/cogvideox-fun-transformer/cogvideox_fun_transformer",
    "/kaggle/input/cogvideox-fun-transformer/transformer",
])

check_path("Video Test Dir", [
    "/kaggle/input/datasets/tranbao0105/cameractrl-sota-weights/RealEstate10K_Mini/real-estate-10k-mini/test_256",
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k/test",
])

check_path("Depth Cache Dir", [
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache/depth_cache",
    "/kaggle/input/datasets/nguynngcnhntrng/realestate10k-depthcrafter-cache",
    "/kaggle/input/datasets/tranbao0105/realestate10k-depthcrafter-cache/depth_cache",
    "/kaggle/working/depth_cache",
])

check_path("Stage 1 Checkpoint", [
    "/kaggle/working/checkpoints_stage1/dora_stage1_final.pt",
    "/kaggle/working/checkpoints_stage1/dora_checkpoint_final.pt",
    "/kaggle/working/dora_stage1_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/dora_stage1_final.pt",
])

check_path("Stage 2 Checkpoint", [
    "/kaggle/working/checkpoints_stage2/stage2_final.pt",
    "/kaggle/working/checkpoints_stage2/stage2_checkpoint_final.pt",
    "/kaggle/working/dora_stage2_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/stage2_final.pt",
    "/kaggle/input/datasets/tranbao0105/pipeline/dora_stage2_final.pt",
])

print("-" * 80)
print("✅ Hệ thống đã sẵn sàng cho quy trình huấn luyện & đánh giá Pipeline v3!\n")
