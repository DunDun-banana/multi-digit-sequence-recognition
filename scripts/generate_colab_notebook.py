"""Generate a self-contained Google Colab (L4/T4/A100 GPU) notebook for Antigravity IDE or Colab."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

EMBEDDED_FILES = [
    "configs/default.yaml",
    "configs/colab_gpu.yaml",
    "src/config.py",
    "src/utils.py",
    "src/metrics.py",
    "src/download_data.py",
    "src/inspect_data.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/demo.py",
    "src/data/__init__.py",
    "src/data/labels.py",
    "src/data/preprocessing.py",
    "src/data/dataset.py",
    "src/data/splitting.py",
    "src/models/__init__.py",
    "src/models/encoder.py",
    "src/models/attention.py",
    "src/models/decoder.py",
    "src/models/seq2seq.py",
]


def _build_sources_bundle_b64() -> tuple[str, int]:
    """Pack all embedded repo files into a compact base64-encoded zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in EMBEDDED_FILES:
            file_path = ROOT_DIR / rel_path
            if not file_path.exists():
                raise FileNotFoundError(f"Missing expected source file: {file_path}")
            zf.write(file_path, arcname=rel_path)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return encoded, len(EMBEDDED_FILES)


def _code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines[:-1]]
        + ([lines[-1]] if lines else []),
    }


def _md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines[:-1]]
        + ([lines[-1]] if lines else []),
    }


def build_notebook() -> dict:
    bundle_b64, file_count = _build_sources_bundle_b64()

    cells = [
        _md_cell([
            "# Multi-Digit Sequence Recognition — Google Colab GPU Runner (L4 / T4 / A100)",
            "",
            "**Hỗ trợ chạy trực tiếp trong Antigravity IDE (qua Google Colab Extension) hoặc trên trình duyệt Colab.**",
            "",
            "- Chọn Kernel **Google Colab** có GPU (**L4 GPU** khuyến nghị, hoặc T4/A100).",
            "- Notebook này **tự chứa toàn bộ mã nguồn `src/` và `configs/`** (tương tự cơ chế turnkey của `cv3d`), tự tải dataset `fellahabdelnour13/multidigit-mnist` (52.7 MB) về ổ đĩa NVMe `/content/` trên VM Colab nên **không cần push GitHub hay upload code thủ công**.",
        ]),
        _code_cell([
            "#@title ⚙️ 0. Cấu hình Pipeline (Colab L4 GPU)",
            "KAGGLE_DATASET_SLUG = \"fellahabdelnour13/multidigit-mnist\"  #@param {type:\"string\"}",
            "EPOCHS = 35  #@param {type:\"integer\"}",
            "BATCH_SIZE = 64  #@param {type:\"integer\"}",
            "NUM_WORKERS = 2  #@param {type:\"integer\"}",
            "AUTO_RESUME = False  #@param {type:\"boolean\"}",
            "SAVE_TO_GOOGLE_DRIVE = True  #@param {type:\"boolean\"}",
            "DRIVE_OUTPUT_DIR = \"multi-digit-sequence-recognition\"  #@param {type:\"string\"}",
            "",
            "print(f\"Dataset slug : {KAGGLE_DATASET_SLUG}\")",
            "print(f\"Epochs       : {EPOCHS} | Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | Auto-resume: {AUTO_RESUME}\")",
            "print(f\"Save to Drive: {SAVE_TO_GOOGLE_DRIVE} (MyDrive/{DRIVE_OUTPUT_DIR})\")",
        ]),
        _code_cell([
            "#@title 🔌 1. Kiểm tra GPU (L4/T4/A100) & Kết nối Google Drive (tuỳ chọn)",
            "import torch",
            "from pathlib import Path",
            "",
            "!nvidia-smi",
            "assert torch.cuda.is_available(), \"Chưa có GPU! Hãy chọn Colab Kernel có GPU (L4/T4/A100).\"",
            "print(f\"✅ GPU sẵn sàng: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)\")",
            "",
            "DRIVE_DIR = None",
            "if SAVE_TO_GOOGLE_DRIVE:",
            "    try:",
            "        from google.colab import drive",
            "        drive.mount(\"/content/drive\")",
            "        DRIVE_DIR = Path(f\"/content/drive/MyDrive/{DRIVE_OUTPUT_DIR}\")",
            "        (DRIVE_DIR / \"checkpoints\").mkdir(parents=True, exist_ok=True)",
            "        (DRIVE_DIR / \"outputs\").mkdir(parents=True, exist_ok=True)",
            "        print(f\"✅ Đã kết nối Google Drive: {DRIVE_DIR}\")",
            "    except Exception as exc:",
            "        print(f\"⚠️ Không mount được Google Drive ({exc}). Kết quả vẫn được lưu tại /content/.\")",
            "        DRIVE_DIR = None",
        ]),
        _code_cell([
            "#@title 📦 2. Khôi phục mã nguồn repo vào /content/ & Cài thư viện",
            "import base64, io, zipfile",
            "from pathlib import Path",
            "",
            "WORK_DIR = Path(\"/content/multi-digit-sequence-recognition\")",
            "WORK_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            f"# Compressed bundle of {file_count} repo files (src/ and configs/)",
            f"REPO_BUNDLE_B64 = \"{bundle_b64}\"",
            "",
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(REPO_BUNDLE_B64))) as zf:",
            "    file_count = len(zf.namelist())",
            "    zf.extractall(WORK_DIR)",
            "",
            "print(f\"✅ Đã giải nén {file_count} file mã nguồn & config vào {WORK_DIR}\")",
            "%cd /content/multi-digit-sequence-recognition",
            "!pip install -q pyyaml tqdm kagglehub kaggle",
        ]),
        _code_cell([
            "#@title ⬇️ 3. Tải dataset & Kiểm tra / Chia tập train-val-test",
            "import os",
            "from pathlib import Path",
            "",
            "# Đọc Kaggle credentials nếu bạn lưu trong Colab Secrets (không bắt buộc với dataset public)",
            "try:",
            "    from google.colab import userdata",
            "    for key in (\"KAGGLE_USERNAME\", \"KAGGLE_KEY\"):",
            "        try:",
            "            val = userdata.get(key)",
            "            if val:",
            "                os.environ[key] = val",
            "        except Exception:",
            "            pass",
            "except Exception:",
            "    pass",
            "",
            "raw_png_count = len(list(Path(\"data/raw\").rglob(\"*.png\"))) if Path(\"data/raw\").exists() else 0",
            "if raw_png_count == 0:",
            "    !python src/download_data.py --config configs/colab_gpu.yaml --slug \"{KAGGLE_DATASET_SLUG}\"",
            "else:",
            "    print(f\"ℹ️ Đã có sẵn {raw_png_count} ảnh trong data/raw/, bỏ qua bước tải lại.\")",
            "",
            "!python src/inspect_data.py --config configs/colab_gpu.yaml",
        ]),
        _code_cell([
            "#@title 🚀 4. Huấn luyện trên Colab GPU (ReduceLROnPlateau + Scheduled Teacher Forcing)",
            "from pathlib import Path",
            "import shutil",
            "",
            "resume_arg = \"\"",
            "local_final = Path(\"checkpoints/final_model.pt\")",
            "if AUTO_RESUME:",
            "    if DRIVE_DIR is not None:",
            "        drive_final = DRIVE_DIR / \"checkpoints\" / \"final_model.pt\"",
            "        drive_best = DRIVE_DIR / \"checkpoints\" / \"best_model.pt\"",
            "        Path(\"checkpoints\").mkdir(parents=True, exist_ok=True)",
            "        if drive_final.exists() and not local_final.exists():",
            "            shutil.copy2(drive_final, local_final)",
            "        if drive_best.exists() and not Path(\"checkpoints/best_model.pt\").exists():",
            "            shutil.copy2(drive_best, Path(\"checkpoints/best_model.pt\"))",
            "    if local_final.exists():",
            "        resume_arg = \"--resume checkpoints/final_model.pt\"",
            "        print(\"♻️ Phát hiện checkpoints/final_model.pt -> tiếp tục train (resume)!\")",
            "else:",
            "    print(\"🆕 Train mới từ Epoch 1 với ReduceLROnPlateau & scheduled teacher forcing (0.8 -> 0.3).\")",
            "",
            "!python src/train.py --config configs/colab_gpu.yaml --device cuda --epochs {EPOCHS} --batch-size {BATCH_SIZE} --num-workers {NUM_WORKERS} {resume_arg}",
        ]),
        _code_cell([
            "#@title 📊 5. Đánh giá trên tập Test & Hiển thị kết quả trực quan",
            "import json",
            "from pathlib import Path",
            "from IPython.display import Image as IPyImage, display",
            "",
            "!python src/evaluate.py --config configs/colab_gpu.yaml --checkpoint checkpoints/best_model.pt --split test --device cuda --batch-size {BATCH_SIZE} --num-workers {NUM_WORKERS}",
            "!python src/demo.py --config configs/colab_gpu.yaml --checkpoint checkpoints/best_model.pt --split test --num-samples 5 --device cuda",
            "",
            "metrics_path = Path(\"outputs/test_metrics.json\")",
            "if metrics_path.exists():",
            "    print(\"\\n=== TEST METRICS ===\")",
            "    print(json.dumps(json.loads(metrics_path.read_text(encoding=\"utf-8\")), indent=2))",
            "",
            "for img_path in [\"outputs/demo_predictions.png\", \"outputs/demo_attention.png\"]:",
            "    if Path(img_path).exists():",
            "        print(f\"\\n--- {img_path} ---\")",
            "        display(IPyImage(filename=img_path))",
        ]),
        _code_cell([
            "#@title 💾 6. Lưu checkpoints & outputs sang Google Drive + Tự tải colab_results.zip",
            "import shutil",
            "import zipfile",
            "from pathlib import Path",
            "",
            "zip_path = Path(\"/content/colab_results.zip\")",
            "with zipfile.ZipFile(zip_path, \"w\", compression=zipfile.ZIP_DEFLATED) as zf:",
            "    for folder in (\"checkpoints\", \"outputs\"):",
            "        src_dir = Path(folder)",
            "        if src_dir.exists():",
            "            for file_path in sorted(src_dir.rglob(\"*\")):",
            "                if file_path.is_file() and file_path.name != \".gitkeep\":",
            "                    zf.write(file_path, arcname=file_path.as_posix())",
            "",
            "print(f\"📦 Đã đóng gói checkpoints/ + outputs/ tại: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)\")",
            "",
            "if DRIVE_DIR is not None:",
            "    for folder in (\"checkpoints\", \"outputs\"):",
            "        src_dir = Path(folder)",
            "        if src_dir.exists():",
            "            shutil.copytree(src_dir, DRIVE_DIR / folder, dirs_exist_ok=True)",
            "    shutil.copy2(zip_path, DRIVE_DIR / zip_path.name)",
            "    print(f\"☁️ Đã lưu lên Google Drive: {DRIVE_DIR / zip_path.name}\")",
            "",
            "# Tự kích hoạt tải xuống khi chạy trên trình duyệt Google Colab",
            "try:",
            "    from google.colab import files",
            "    files.download(str(zip_path))",
            "    print(\"⬇️ Đang tải colab_results.zip về máy (nếu dùng Colab trên trình duyệt).\")",
            "except Exception as exc:",
            "    print(f\"ℹ️ Không tự bật tải file qua IDE ({exc}). Hãy tải {zip_path.name} từ Google Drive về Downloads rồi chạy: python scripts/pull_colab_results.py\")",
        ]),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "L4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = ROOT_DIR / "notebooks" / "colab_gpu.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated turnkey Colab notebook: {out_path}")


if __name__ == "__main__":
    main()
