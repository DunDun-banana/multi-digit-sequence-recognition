"""Extract Colab training checkpoints and outputs into the local repository.
Chạy xong cell 6 thì gọi 
python scripts/pull_colab_results.py --drive-url "link folder drive chứa kết quả sau khi chạy cell 6 notebook"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ALLOWED_FOLDERS = ("checkpoints", "outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull or unpack Colab results (checkpoints/ and outputs/) into local repo."
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to colab_results.zip (defaults to ./colab_results.zip or ~/Downloads/colab_results.zip).",
    )
    parser.add_argument(
        "--drive-url",
        type=str,
        default=None,
        help="Google Drive share URL (file or folder) or file ID for Colab results (uses gdown).",
    )
    return parser.parse_args()


def _ensure_gdown():
    try:
        import gdown  # type: ignore

        return gdown
    except ImportError:
        print("Installing gdown to download from Google Drive...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown  # type: ignore

        return gdown


def _extract_drive_file_id(drive_url_or_id: str) -> str:
    match = re.search(r"(?:/d/|[?&]id=)([A-Za-z0-9_-]+)", drive_url_or_id)
    if match:
        return match.group(1)
    return drive_url_or_id.strip()


def _find_local_zip(explicit_path: str | None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Zip file not found: {path}")
        return path

    candidates = [
        ROOT_DIR / "colab_results.zip",
        Path.home() / "Downloads" / "colab_results.zip",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "Could not find colab_results.zip in repository root or Downloads folder.\n"
            "Download colab_results.zip from Google Drive or pass --zip-path / --drive-url."
        )
    return max(existing, key=lambda p: p.stat().st_mtime)


def extract_results_zip(zip_path: Path, target_root: Path) -> list[Path]:
    """Safely extract only checkpoints/ and outputs/ entries into target_root."""
    extracted: list[Path] = []
    allowed_prefixes = tuple(f"{name}/" for name in ALLOWED_FOLDERS)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel = Path(member.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename!r}")
            if not rel.as_posix().startswith(allowed_prefixes):
                continue

            dest = (target_root / rel).resolve()
            if target_root.resolve() not in dest.parents:
                raise ValueError(f"Path escapes target root: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, dest.open("wb") as dst:
                dst.write(src.read())
            extracted.append(dest)
    return extracted


def _copy_folders_into_repo(source_root: Path, target_root: Path) -> list[Path]:
    copied: list[Path] = []
    for folder in ALLOWED_FOLDERS:
        candidates = [source_root / folder] + [
            p for p in source_root.rglob(folder) if p.is_dir()
        ]
        src_dir = next((p for p in candidates if p.is_dir()), None)
        if src_dir is None:
            continue
        for file_path in sorted(src_dir.rglob("*")):
            if not file_path.is_file() or file_path.name == ".gitkeep":
                continue
            rel = file_path.relative_to(src_dir)
            dest = target_root / folder / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            copied.append(dest)
    return copied


def _pull_from_drive(drive_url: str, target_root: Path) -> list[Path]:
    gdown = _ensure_gdown()
    if "/folders/" in drive_url:
        with tempfile.TemporaryDirectory(prefix="colab_pull_") as tmp_name:
            tmp_dir = Path(tmp_name)
            gdown.download_folder(url=drive_url, output=str(tmp_dir), quiet=False)
            zip_candidates = list(tmp_dir.rglob("colab_results.zip"))
            if zip_candidates:
                saved_zip = target_root / "colab_results.zip"
                shutil.copy2(zip_candidates[0], saved_zip)
                return extract_results_zip(saved_zip, target_root)
            copied = _copy_folders_into_repo(tmp_dir, target_root)
            if not copied:
                raise RuntimeError(
                    "Downloaded Drive folder did not contain colab_results.zip, checkpoints/, or outputs/."
                )
            return copied

    dest_zip = target_root / "colab_results.zip"
    file_id = _extract_drive_file_id(drive_url)
    gdown.download(id=file_id, output=str(dest_zip), quiet=False)
    if not dest_zip.exists():
        raise RuntimeError(f"Failed to download {dest_zip} from Google Drive.")
    return extract_results_zip(dest_zip, target_root)


def main() -> None:
    args = parse_args()
    if args.drive_url:
        restored = _pull_from_drive(args.drive_url, ROOT_DIR)
    else:
        zip_path = _find_local_zip(args.zip_path)
        print(f"Extracting results from: {zip_path}")
        restored = extract_results_zip(zip_path, ROOT_DIR)

    print(f"Restored {len(restored)} files into {ROOT_DIR}:")
    for p in restored:
        print(f"  - {p.relative_to(ROOT_DIR).as_posix()} ({p.stat().st_size / 1024:.1f} KB)")

    metrics_file = ROOT_DIR / "outputs" / "test_metrics.json"
    if metrics_file.exists():
        print("\n=== Restored test_metrics.json ===")
        print(json.dumps(json.loads(metrics_file.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
