import os
import zipfile

def create_clean_zip(source_dir, output_zip):
    source_dir = os.path.abspath(source_dir)
    parent_dir = os.path.dirname(source_dir)
    base_folder_name = os.path.basename(source_dir)

    # Exclusions
    excluded_extensions = {".pyc", ".pyo", ".mp4", ".mov", ".avi"}
    excluded_dirs = {"__pycache__", ".git", ".github", ".ipynb_checkpoints"}

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in excluded_extensions:
                    continue
                if file.startswith(".DS_Store"):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, parent_dir)
                
                # CRITICAL: Always use POSIX forward slashes '/' for Linux / Kaggle compatibility!
                arcname = rel_path.replace("\\", "/")
                
                zf.write(full_path, arcname=arcname)
                
    size_kb = os.path.getsize(output_zip) / 1024
    print(f"Created {output_zip}: {size_kb:.2f} KB")

if __name__ == "__main__":
    src = "pipeline_v3"
    create_clean_zip(src, "pipeline_v3_code.zip")
    create_clean_zip(src, "pipeline_v3.zip")
