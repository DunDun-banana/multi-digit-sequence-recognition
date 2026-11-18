"""Step 6: Demo visualization of predictions and Bahdanau attention weights on split samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import load_config, resolve_path
from predict import Predictor
from utils import plot_attention_sample, plot_sample_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual prediction grid and attention heatmap on dataset samples."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (.pt) path.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split CSV to sample from.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override compute device ('auto', 'cpu', or 'cuda').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    config = load_config(args.config)

    data_cfg = config["data"]
    train_cfg = config["training"]
    output_dir = resolve_path(train_cfg.get("output_dir", "outputs"), base_dir)
    splits_dir = resolve_path(data_cfg["splits_dir"], base_dir)
    split_csv = splits_dir / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}. Run `python src/inspect_data.py` first."
        )

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"), base_dir)
        / str(train_cfg.get("best_checkpoint_name", "best_model.pt"))
    )

    device_pref = args.device if args.device is not None else str(config.get("project", {}).get("device", "auto"))
    predictor = Predictor.from_checkpoint(
        checkpoint_path=ckpt_path,
        config_override=config,
        device_pref=device_pref,
    )

    df = pd.read_csv(split_csv, dtype={"image_path": str, "label": str}, keep_default_na=False)
    sample_df = df.sample(
        min(args.num_samples, len(df)),
        random_state=int(config.get("project", {}).get("seed", 42)),
    ).reset_index(drop=True)

    images = []
    true_labels = []
    pred_labels = []
    first_result = None

    for idx, row in sample_df.iterrows():
        img_path = resolve_path(row["image_path"], base_dir)
        res = predictor.predict_image(img_path, ground_truth=str(row["label"]))
        if idx == 0:
            first_result = res
        images.append(np.asarray(res["original_image"].convert("L")))
        true_labels.append(str(row["label"]))
        pred_labels.append(res["predicted_sequence"])
        print(f"Sample {idx + 1}: True={row['label']} | Pred={res['predicted_sequence']}")

    grid_path = plot_sample_predictions(
        images=images,
        true_labels=true_labels,
        pred_labels=pred_labels,
        output_path=output_dir / "demo_predictions.png",
        max_samples=args.num_samples,
    )
    print(f"Saved prediction grid: {grid_path}")

    if first_result is not None:
        attn_path = plot_attention_sample(
            image=np.asarray(first_result["original_image"].convert("L")),
            attention_weights=first_result["attention_weights"],
            true_label=first_result["ground_truth"],
            pred_label=first_result["predicted_sequence"],
            output_path=output_dir / "demo_attention.png",
        )
        print(f"Saved attention map  : {attn_path}")


if __name__ == "__main__":
    main()
