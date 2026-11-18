"""Step 4: Evaluate a trained checkpoint on train, validation, or test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import get_device, load_config, resolve_path
from data.dataset import DigitSequenceDataset, create_dataloader
from data.labels import DigitVocabulary
from metrics import compute_all_metrics
from models.seq2seq import build_model_from_config


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    vocab: DigitVocabulary,
    device: torch.device,
    max_decode_length: int = 6,
    criterion: nn.Module | None = None,
    output_dir: str | Path | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    """Evaluate a trained CNNSeq2Seq model on a DataLoader and optionally save reports."""
    model.eval()
    model.to(device)

    all_true: list[str] = []
    all_pred: list[str] = []
    all_paths: list[str] = []
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        decoder_inputs = batch["decoder_inputs"].to(device)
        decoder_targets = batch["decoder_targets"].to(device)
        labels = batch["labels"]
        paths = batch["image_paths"]

        if criterion is not None:
            logits = model(images, decoder_inputs, teacher_forcing_ratio=0.0)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_targets.reshape(-1),
            )
            total_loss += float(loss.item())
            num_batches += 1

        pred_ids, _ = model.predict(images, max_len=max_decode_length)
        pred_ids_list = pred_ids.detach().cpu().tolist()

        for true_str, token_seq, path in zip(labels, pred_ids_list, paths):
            pred_str = vocab.decode(token_seq, stop_at_eos=True)
            all_true.append(true_str)
            all_pred.append(pred_str)
            all_paths.append(path)

    metrics = compute_all_metrics(all_true, all_pred)
    metrics["split"] = split_name
    if criterion is not None and num_batches > 0:
        metrics["loss"] = total_loss / num_batches

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        metrics_file = out_path / f"{split_name}_metrics.json"
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        preds_df = pd.DataFrame(
            {
                "image_path": all_paths,
                "ground_truth": all_true,
                "prediction": all_pred,
                "exact_match": [t == p for t, p in zip(all_true, all_pred)],
            }
        )
        preds_df.to_csv(out_path / f"{split_name}_predictions.csv", index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-digit recognition checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (.pt) path.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split CSV to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override compute device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader num_workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path.cwd()

    data_cfg = config["data"]
    img_cfg = config["image"]
    train_cfg = config["training"]
    inf_cfg = config["inference"]

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"), base_dir)
        / str(train_cfg.get("best_checkpoint_name", "best_model.pt"))
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    split_csv = resolve_path(data_cfg["splits_dir"], base_dir) / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    device_pref = args.device if args.device is not None else str(config.get("project", {}).get("device", "auto"))
    device = get_device(device_pref)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", config)

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

    model = build_model_from_config(ckpt_config, vocab)
    model.load_state_dict(checkpoint["model_state_dict"])

    raw_width = img_cfg.get("width", 160)
    target_width = int(raw_width) if raw_width is not None else None
    dataset = DigitSequenceDataset(
        split_csv,
        vocab=vocab,
        base_dir=base_dir,
        target_height=int(img_cfg.get("height", 32)),
        target_width=target_width,
        pad_mode=str(img_cfg.get("pad_mode", "right")),
        pad_value=int(img_cfg.get("pad_value", 0)),
        normalize_mean=float(img_cfg.get("normalize_mean", 0.0)),
        normalize_std=float(img_cfg.get("normalize_std", 1.0)),
    )
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_id=vocab.pad_id,
        pin_memory=(device.type == "cuda"),
    )

    output_dir = resolve_path(train_cfg.get("output_dir", "outputs"), base_dir)
    max_decode_length = int(
        inf_cfg.get("max_decode_length", int(data_cfg.get("max_sequence_length", 5)) + 1)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        vocab=vocab,
        device=device,
        max_decode_length=max_decode_length,
        criterion=criterion,
        output_dir=output_dir,
        split_name=args.split,
    )

    print(f"Evaluation results on '{args.split}' split:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
