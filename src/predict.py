"""Step 5: Single-image inference pipeline for trained CNNSeq2Seq checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import get_device, load_config, resolve_path
from data.labels import DigitVocabulary
from data.preprocessing import preprocess_image
from models.seq2seq import CNNSeq2Seq, build_model_from_config
from utils import plot_attention_sample


class Predictor:
    """Loads a saved checkpoint and predicts digit sequences from image paths."""

    def __init__(
        self,
        model: CNNSeq2Seq,
        vocab: DigitVocabulary,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.config = config
        self.device = device

        img_cfg = config.get("image", {})
        inf_cfg = config.get("inference", {})
        data_cfg = config.get("data", {})

        self.target_height = int(img_cfg.get("height", 32))
        raw_w = img_cfg.get("width", 160)
        self.target_width = int(raw_w) if raw_w is not None else None
        self.pad_mode = str(img_cfg.get("pad_mode", "right"))
        self.pad_value = int(img_cfg.get("pad_value", 0))
        self.normalize_mean = float(img_cfg.get("normalize_mean", 0.0))
        self.normalize_std = float(img_cfg.get("normalize_std", 1.0))
        self.max_decode_length = int(
            inf_cfg.get("max_decode_length", int(data_cfg.get("max_sequence_length", 5)) + 1)
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_override: dict[str, Any] | None = None,
        device_pref: str = "auto",
    ) -> "Predictor":
        """Instantiate a Predictor from a saved .pt checkpoint file."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        device = get_device(device_pref)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        config = config_override if config_override is not None else checkpoint.get("config", {})
        vocab_info = checkpoint.get("vocab", {})
        if vocab_info:
            vocab = DigitVocabulary(
                pad_id=int(vocab_info["pad_id"]),
                sos_id=int(vocab_info["sos_id"]),
                eos_id=int(vocab_info["eos_id"]),
                token_to_id=dict(vocab_info["token_to_id"]),
            )
        else:
            vocab = DigitVocabulary()

        model = build_model_from_config(config, vocab)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, vocab=vocab, config=config, device=device)

    @torch.inference_mode()
    def predict_image(
        self,
        image_path: str | Path,
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """Predict a digit sequence from a single image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        with Image.open(path) as img:
            original_image = img.copy()

        tensor = preprocess_image(
            original_image,
            target_height=self.target_height,
            target_width=self.target_width,
            pad_mode=self.pad_mode,
            pad_value=self.pad_value,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        ).unsqueeze(0).to(self.device)

        pred_ids, attn_weights = self.model.predict(tensor, max_len=self.max_decode_length)
        token_list = pred_ids[0].detach().cpu().tolist()
        predicted_sequence = self.vocab.decode(token_list, stop_at_eos=True)

        return {
            "image_path": path.as_posix(),
            "original_image": original_image,
            "predicted_sequence": predicted_sequence,
            "predicted_length": len(predicted_sequence),
            "ground_truth": ground_truth,
            "token_ids": token_list,
            "attention_weights": attn_weights[0].detach().cpu().numpy().astype(np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict digit sequence from a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (.pt) path.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override compute device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional ground-truth label.")
    parser.add_argument("--save-plot", type=str, default=None, help="Optional output PNG path for attention plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    config = load_config(args.config) if Path(args.config).exists() else {}

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        train_cfg = config.get("training", {})
        ckpt_path = resolve_path(
            train_cfg.get("checkpoint_dir", "checkpoints"),
            base_dir,
        ) / str(train_cfg.get("best_checkpoint_name", "best_model.pt"))

    device_pref = args.device if args.device is not None else str(config.get("project", {}).get("device", "auto"))
    predictor = Predictor.from_checkpoint(
        checkpoint_path=ckpt_path,
        config_override=config if config else None,
        device_pref=device_pref,
    )
    result = predictor.predict_image(args.image, ground_truth=args.ground_truth)

    print(f"Image Path        : {result['image_path']}")
    print(f"Image Size (W x H): {result['original_image'].size}")
    print(f"Predicted Sequence: {result['predicted_sequence']}")
    print(f"Predicted Length  : {result['predicted_length']}")
    if result["ground_truth"] is not None:
        print(f"Ground Truth      : {result['ground_truth']}")

    if args.save_plot:
        out_plot = plot_attention_sample(
            image=np.asarray(result["original_image"].convert("L")),
            attention_weights=result["attention_weights"],
            true_label=result["ground_truth"],
            pred_label=result["predicted_sequence"],
            output_path=args.save_plot,
        )
        print(f"Saved plot        : {out_plot}")


if __name__ == "__main__":
    main()
