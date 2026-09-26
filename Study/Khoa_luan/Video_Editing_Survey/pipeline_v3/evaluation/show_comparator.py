# ==============================================================================
# PIPELINE V3: TRÌNH CHIẾU & SO SÁNH TRỰC QUAN ĐA CỘT (SHOW COMPARATOR)
# ==============================================================================
# Tự động quét toàn bộ video kết quả trên Kaggle, chuyển đổi sang H.264
# chuẩn Web và hiển thị bảng so sánh 4 cột (Ablation Study) ngay trong Notebook:
# 1. Baseline SOTA | 2. 3D Scaffold | 3. Stage 1 DoRA | 4. Stage 2 Our Full Model
# ==============================================================================

import os
import sys
import glob
import base64
import subprocess

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

WEB_DIR = "/kaggle/working/web_videos" if (os.path.exists("/kaggle") or os.name != "nt") else "./web_videos"
os.makedirs(WEB_DIR, exist_ok=True)


def to_web_h264(input_path):
    """Chuyển đổi video sang chuẩn H.264/yuv420p để phát mượt mà trên trình duyệt."""
    if not input_path or not os.path.exists(input_path):
        return None
    if input_path.endswith("_web.mp4") and os.path.exists(input_path):
        return input_path
    
    parent_tag = os.path.basename(os.path.dirname(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"{parent_tag}_{base_name}_web.mp4" if parent_tag else f"{base_name}_web.mp4"
    h264_path = os.path.join(WEB_DIR, out_name)
    
    if not os.path.exists(h264_path) or os.path.getsize(h264_path) == 0:
        cmd = f'ffmpeg -y -loglevel error -i "{input_path}" -vcodec libx264 -pix_fmt yuv420p "{h264_path}"'
        subprocess.run(cmd, shell=True, check=False)
        
    return h264_path if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0 else input_path


def to_b64(path):
    """Đọc file video và mã hóa Base64 nhúng vào HTML."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def find_file(patterns, s_id=None):
    """Tìm kiếm file an toàn dựa trên danh sách glob patterns, bắt buộc phải thuộc về s_id nếu s_id được cung cấp."""
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            if s_id:
                matches = [m for m in matches if s_id in m]
            if not matches:
                continue
            sorted_m = sorted(matches, key=lambda x: (not x.endswith("_h264.mp4"), -os.path.getsize(x)))
            for m in sorted_m:
                if os.path.exists(m) and os.path.getsize(m) > 1024:
                    return m
    return None


DEFAULT_TASKS = [
    {
        "tag": "pan30",
        "name": "Quay sang trái 30° (Pan Left)",
        "patterns": ["pan_30", "pan30", "pan_-30", "pan-30", "pan*30"],
    },
    {
        "tag": "forward",
        "name": "Đi thẳng (Forward)",
        "patterns": ["forward", "pan_0", "pan0"],
    },
]


def show(samples=None, tasks=None, angles=None):
    """Quét dữ liệu và hiển thị bảng so sánh 4 cột trực quan trong Notebook."""
    if samples is None:
        samples = [
            {"name": "Train_Video (In-Domain)", "id": "000c3ab189999a83"},
            {"name": "Unseen_Video (Out-of-Domain)", "id": "145da324f69d1c6b"},
        ]
    
    if tasks is None:
        if angles is not None:
            tasks = []
            for a in angles:
                if a == 30:
                    tasks.append(DEFAULT_TASKS[0])
                elif a == 0:
                    tasks.append(DEFAULT_TASKS[1])
                else:
                    tasks.append({
                        "tag": f"pan{a}",
                        "name": f"Quay sang trái {a}° (Pan Left)",
                        "patterns": [f"pan_{a}", f"pan{a}", f"pan_-{a}", f"pan-{a}", f"pan*{a}"],
                    })
        else:
            tasks = DEFAULT_TASKS

    print("=" * 80)
    print("🎬 [CELL 6] BẢNG SO SÁNH TRỰC QUAN 4 CỘT HOÀN CHỈNH (PIPELINE V3 ABLATION)")
    print("=" * 80)

    all_results = {}

    for sample in samples:
        s_id = sample["id"]
        s_name = sample["name"]
        
        orig_path = find_file([
            f"/kaggle/working/eval_results_stage2/{s_id}_orig.mp4",
            f"/kaggle/working/eval_results_stage1/{s_id}_orig.mp4",
            f"/kaggle/working/**/{s_id}*orig*.mp4",
            f"/kaggle/input/**/{s_id}.mp4",
            f"/kaggle/input/**/test_256/{s_id}*.mp4",
        ], s_id=s_id)
        
        sample_data = {"name": s_name, "id": s_id, "orig": to_web_h264(orig_path), "tasks": {}}
        
        for task in tasks:
            t_tag = task["tag"]
            t_name = task["name"]
            
            # 1. Baseline SOTA
            base_path = find_file([
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_before/*gen*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_before/*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}_before/*.mp4",
                f"/kaggle/working/**/{s_id}/**/baseline*{t_tag}*.mp4",
            ], s_id=s_id)

            # 2. 3D Scaffold
            scaff_path = find_file([
                f"/kaggle/working/eval_results_stage2/{s_id}/{t_tag}_stage2/*scaffold*.mp4",
                f"/kaggle/working/eval_results_stage2/{s_id}/{t_tag}_stage2/*render*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_before/*scaffold*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_after/*scaffold*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_before/*render*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_after/*render*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}*render*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}*scaffold*.mp4",
            ], s_id=s_id)
            
            # 3. Stage 1 (DoRA)
            s1_path = find_file([
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_after/*gen*.mp4",
                f"/kaggle/working/eval_results_stage1/{s_id}/{t_tag}_after/*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}_after/*.mp4",
                f"/kaggle/working/**/{s_id}/**/stage1*{t_tag}*.mp4",
            ], s_id=s_id)
            
            # 4. Stage 2 (Our Full Model)
            s2_path = find_file([
                f"/kaggle/working/eval_results_stage2/{s_id}/{t_tag}_stage2/*stage2*.mp4",
                f"/kaggle/working/eval_results_stage2/{s_id}/{t_tag}_stage2/*gen*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}_stage2/*.mp4",
                f"/kaggle/working/**/{s_id}/**/stage2*{t_tag}*.mp4",
            ], s_id=s_id)
            
            # 5. Triptych Video
            viz_path = find_file([
                f"/kaggle/working/eval_results_stage2/{s_id}/{t_tag}_stage2/*viz*.mp4",
                f"/kaggle/working/**/{s_id}/**/{t_tag}*viz*.mp4",
            ], s_id=s_id)
            
            sample_data["tasks"][t_tag] = {
                "name": t_name,
                "baseline": to_web_h264(base_path),
                "scaffold": to_web_h264(scaff_path),
                "stage1": to_web_h264(s1_path),
                "stage2": to_web_h264(s2_path),
                "viz": to_web_h264(viz_path),
            }
            
        all_results[s_id] = sample_data

    # In chẩn đoán trạng thái
    print("\n[Trạng thái phát hiện file video trên đĩa]")
    for s_id, s_data in all_results.items():
        print(f"\n📂 Mẫu: {s_data['name']} (ID: {s_id})")
        print(f"   • Video Gốc : {'✅ Sẵn sàng' if s_data['orig'] else '❌ Không tìm thấy'}")
        for t_tag, t_data in s_data["tasks"].items():
            print(f"   • {t_data['name']}: "
                  f"Baseline: {'✅' if t_data['baseline'] else '❌'} | "
                  f"Scaffold: {'✅' if t_data['scaffold'] else '❌'} | "
                  f"Stage 1: {'✅' if t_data['stage1'] else '⚠️'} | "
                  f"Stage 2 (Full): {'✅' if t_data['stage2'] else '❌'} | "
                  f"Triptych: {'✅' if t_data['viz'] else '❌'}")

    # Tạo giao diện HTML Đa Cột chuyên nghiệp
    html_output = """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; background: #09090b; color: #f4f4f5; padding: 24px; border-radius: 12px; border: 1px solid #27272a;">
        <h2 style="color: #38bdf8; text-align: center; margin: 0 0 8px 0;">🎬 PIPELINE V3: BẢNG SO SÁNH TRỰC QUAN ABLATION STUDY</h2>
        <p style="text-align: center; color: #a1a1aa; font-size: 14px; margin-bottom: 25px;">
            Đánh giá sự tiến hóa 4 Cột: <strong>Baseline SOTA</strong> ➔ <strong>3D Scaffold</strong> ➔ <strong>Stage 1 DoRA</strong> ➔ <strong>Stage 2 Mô Hình Hoàn Chỉnh</strong>
        </p>
    """

    for s_id, s_data in all_results.items():
        b64_orig = to_b64(s_data["orig"])
        if not b64_orig and not any(s_data["tasks"][t]["stage2"] for t in s_data["tasks"]):
            continue

        html_output += f"""
        <div style="background: #18181b; padding: 18px; border-radius: 10px; margin-bottom: 25px; border: 1px solid #27272a;">
            <h3 style="color: #facc15; margin-top: 0;">🎬 Mẫu: {s_data['name']} (ID: {s_id})</h3>
            
            <div style="text-align: center; margin-bottom: 18px;">
                <span style="display: inline-block; background: #27272a; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; color: #38bdf8; margin-bottom: 8px;">
                    Video Gốc Đầu Vào (Monocular Source Video)
                </span><br>
                <video style="width: 320px; border-radius: 8px; border: 2px solid #3f3f46; background: #000;" controls autoplay loop muted playsinline>
                    <source src="data:video/mp4;base64,{b64_orig}" type="video/mp4">
                </video>
            </div>

            <table style="width: 100%; border-collapse: collapse; text-align: center; margin-top: 10px;">
                <thead>
                    <tr style="background: #27272a; color: #f4f4f5; font-size: 13px;">
                        <th style="padding: 10px; border: 1px solid #3f3f46; width: 12%;">Quỹ Đạo Camera</th>
                        <th style="padding: 10px; border: 1px solid #3f3f46; width: 22%; color: #f87171;">1. Baseline (SOTA Paper)</th>
                        <th style="padding: 10px; border: 1px solid #3f3f46; width: 22%; color: #94a3b8;">2. 3D Scaffold (Proxy)</th>
                        <th style="padding: 10px; border: 1px solid #3f3f46; width: 22%; color: #fbbf24;">3. Stage 1 (DoRA Hình học)</th>
                        <th style="padding: 10px; border: 1px solid #3f3f46; width: 22%; color: #4ade80;">4. Stage 2 (Our Full Model)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for t_tag, t_data in s_data["tasks"].items():
            b64_base = to_b64(t_data["baseline"])
            b64_scaff = to_b64(t_data["scaffold"])
            b64_s1 = to_b64(t_data["stage1"])
            b64_s2 = to_b64(t_data["stage2"])
            t_name = t_data["name"]
            
            html_output += f"""
                    <tr>
                        <td style="padding: 10px; font-weight: bold; font-size: 14px; color: #38bdf8; border: 1px solid #3f3f46; vertical-align: middle;">
                            {t_name}
                        </td>
                        <td style="padding: 8px; border: 1px solid #3f3f46; background: rgba(0, 0, 0, 0.2);">
                            <video style="width: 100%; max-width: 240px; border-radius: 6px; border: 1px solid #ef4444;" controls autoplay loop muted playsinline>
                                <source src="data:video/mp4;base64,{b64_base or ''}" type="video/mp4">
                            </video>
                            <p style="font-size: 11px; color: #f87171; margin: 4px 0 0 0;">Mô hình gốc TrajectoryCrafter</p>
                        </td>
                        <td style="padding: 8px; border: 1px solid #3f3f46; background: rgba(0, 0, 0, 0.2);">
                            <video style="width: 100%; max-width: 240px; border-radius: 6px; border: 1px solid #64748b;" controls autoplay loop muted playsinline>
                                <source src="data:video/mp4;base64,{b64_scaff or ''}" type="video/mp4">
                            </video>
                            <p style="font-size: 11px; color: #94a3b8; margin: 4px 0 0 0;">Forward-warping + Lỗ thủng đen</p>
                        </td>
                        <td style="padding: 8px; border: 1px solid #3f3f46; background: rgba(0, 0, 0, 0.2);">
                            <video style="width: 100%; max-width: 240px; border-radius: 6px; border: 1px solid #f59e0b;" controls autoplay loop muted playsinline>
                                <source src="data:video/mp4;base64,{b64_s1 or ''}" type="video/mp4">
                            </video>
                            <p style="font-size: 11px; color: #fbbf24; margin: 4px 0 0 0;">DoRA Self-Attn & FFN (Khớp hình học)</p>
                        </td>
                        <td style="padding: 8px; border: 1px solid #3f3f46; background: rgba(34, 197, 94, 0.08);">
                            <video style="width: 100%; max-width: 240px; border-radius: 6px; border: 2px solid #22c55e;" controls autoplay loop muted playsinline>
                                <source src="data:video/mp4;base64,{b64_s2 or ''}" type="video/mp4">
                            </video>
                            <p style="font-size: 11px; color: #4ade80; font-weight: bold; margin: 4px 0 0 0;">DoRA + 15 Tầng Perceiver (Cực nét vân vật liệu)</p>
                        </td>
                    </tr>
            """
            
        html_output += """
                </tbody>
            </table>
        </div>
        """

    html_output += "</div>"

    report_path = "/kaggle/working/eval_results_stage2/ablation_comparator.html"
    if not os.path.exists("/kaggle/working"):
        report_path = "./eval_results_stage2/ablation_comparator.html"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    if IN_IPYTHON:
        display(HTML(html_output))
    print(f"\n✅ Đã tạo bảng so sánh trực quan thành công!")
    print(f"📁 Báo cáo HTML cũng đã được lưu tại: {report_path}")
    return report_path


if __name__ == "__main__":
    show()
