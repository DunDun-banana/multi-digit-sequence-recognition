"""Step 2: Inspect raw dataset, validate images/labels, plot distributions, and create splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import load_config, resolve_path
from data.labels import build_and_validate_metadata
from data.splitting import create_splits
from utils import plot_sample_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect dataset, validate labels/images, save EDA plots, and generate train/val/test splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Run PIL Image.verify() on every image (slower on large datasets; skipped by default).",
    )
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Legacy no-op flag (image verification is already skipped by default).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample images to save in outputs/sample_images.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path.cwd()

    data_cfg = config["data"]
    train_cfg = config["training"]
    raw_dir = resolve_path(data_cfg["raw_dir"], base_dir)
    processed_dir = resolve_path(data_cfg["processed_dir"], base_dir)
    splits_dir = resolve_path(data_cfg["splits_dir"], base_dir)
    output_dir = resolve_path(train_cfg.get("output_dir", "outputs"), base_dir)
    metadata_csv = processed_dir / data_cfg.get("metadata_filename", "metadata.csv")

    print(f"Inspecting raw dataset in: {raw_dir}")
    metadata_df, report = build_and_validate_metadata(
        raw_dir=raw_dir,
        output_csv=metadata_csv,
        label_config=data_cfg["label_extraction"],
        supported_extensions=data_cfg.get("supported_extensions", [".png", ".jpg", ".jpeg", ".bmp"]),
        min_sequence_length=int(data_cfg.get("min_sequence_length", 1)),
        max_sequence_length=int(data_cfg.get("max_sequence_length", 5)),
        base_dir=base_dir,
        verify_images=args.verify_images and not args.skip_image_verify,
        show_progress=not args.no_progress,
    )

    print("Validation report:")
    print(json.dumps(report, indent=2))

    if metadata_df.empty:
        raise RuntimeError(
            "No valid samples found under data/raw/. Run `python src/download_data.py` first "
            "or check data.label_extraction in configs/default.yaml."
        )

    # Verify labels contain only ASCII digits 0-9
    non_digit_count = int((~metadata_df["label"].str.fullmatch(r"[0-9]+")).sum())
    print(f"Labels with non-digit characters: {non_digit_count}")

    # Save sample image grid & sequence length distribution plot
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = metadata_df.sample(
        min(args.num_samples, len(metadata_df)),
        random_state=int(config.get("project", {}).get("seed", 42)),
    )
    sample_imgs = []
    sample_labels = []
    for _, row in sample_df.iterrows():
        img_path = resolve_path(row["image_path"], base_dir)
        with Image.open(img_path) as im:
            sample_imgs.append(np.asarray(im.convert("L")))
        sample_labels.append(str(row["label"]))

    samples_plot = plot_sample_predictions(
        images=sample_imgs,
        true_labels=sample_labels,
        pred_labels=sample_labels,
        output_path=output_dir / "sample_images.png",
        max_samples=args.num_samples,
    )

    split_cfg = data_cfg.get("split", {})
    split_summary = create_splits(
        metadata=metadata_df,
        splits_dir=splits_dir,
        seed=int(config.get("project", {}).get("seed", 42)),
        val_ratio_from_train=float(split_cfg.get("val_ratio_from_train", 0.15)),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        stratify_by_length=bool(split_cfg.get("stratify_by_length", True)),
    )

    print("Split summary:")
    print(json.dumps(split_summary, indent=2))
    print(f"Saved inspection plot: {samples_plot}")


if __name__ == "__main__":
    main()
