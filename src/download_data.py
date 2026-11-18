"""Step 1: Download or extract the multi-digit dataset into data/raw/."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import load_config, resolve_path


def _strip_wrapping_quotes(value: str) -> str:
    """Strip whitespace and matching wrapping single/double quotes."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _load_env_file(env_path: Path) -> None:
    """Load key=value pairs from .env into os.environ without overwriting existing vars."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = _strip_wrapping_quotes(val)
        if val:
            os.environ.setdefault(key, val)


def _normalize_dataset_slug(raw_slug: str) -> str:
    """Convert a Kaggle dataset URL or quoted slug into `<owner>/<dataset-slug>`."""
    slug = _strip_wrapping_quotes(raw_slug)
    if not slug:
        return ""
    if slug.startswith(("http://", "https://")):
        parsed = urlparse(slug)
        parts = [p for p in parsed.path.split("/") if p]
        if "datasets" in parts:
            idx = parts.index("datasets")
            if idx + 2 < len(parts):
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return slug.strip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or extract the multi-digit dataset into data/raw/."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Optional Kaggle dataset slug (<owner>/<dataset-slug>). Overrides KAGGLE_DATASET_SLUG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    config = load_config(args.config)
    raw_dir = resolve_path(config["data"]["raw_dir"], base_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    _load_env_file(base_dir / ".env")
    dataset_slug = _normalize_dataset_slug(args.slug or os.environ.get("KAGGLE_DATASET_SLUG", ""))

    if dataset_slug:
        print(f"Downloading Kaggle dataset '{dataset_slug}' via kagglehub...")
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError("kagglehub is required for automatic download. Install with `pip install kagglehub`.") from exc

        downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
        print(f"Downloaded cache path: {downloaded_path}")
        for item in downloaded_path.iterdir():
            dest = raw_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"Dataset copied into: {raw_dir}")
    else:
        print(
            "KAGGLE_DATASET_SLUG is not set.\n"
            "Reference Kaggle notebook: https://www.kaggle.com/code/fellahabdelnour13/multi-digit-sequence-recognition/input\n"
            "Mount path in reference notebook: /kaggle/input/multidigit-mnist (datasetVersion sourceId: 9136180)\n"
            "To download automatically:\n"
            "  1. Copy .env.example to .env and set KAGGLE_DATASET_SLUG=<owner>/<dataset-slug> from the notebook Input tab.\n"
            "  2. Configure ~/.kaggle/kaggle.json (or KAGGLE_USERNAME and KAGGLE_KEY).\n"
            "  3. Re-run: python src/download_data.py\n"
            "Or manually place the downloaded ZIP archive or extracted folders into data/raw/."
        )

    # Extract any .zip archives placed manually in data/raw/
    for zip_path in sorted(raw_dir.glob("*.zip")):
        print(f"Extracting archive: {zip_path.name} -> {raw_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    allowed_exts = {ext.lower() for ext in config["data"].get("supported_extensions", [".png", ".jpg", ".jpeg", ".bmp"])}
    image_files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    subdirs = sorted({p.parent.relative_to(raw_dir).as_posix() for p in image_files})

    print(f"Raw directory     : {raw_dir}")
    print(f"Total image files : {len(image_files)}")
    print(f"Image subfolders  : {subdirs[:20]}")
    if image_files:
        print(f"Sample filenames  : {[p.name for p in image_files[:10]]}")


if __name__ == "__main__":
    main()
