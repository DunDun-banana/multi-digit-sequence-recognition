"""Step 3: Training pipeline for Attention-based CNN Encoder - LSTM Decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import get_device, load_config, resolve_path, seed_everything
from data.dataset import DigitSequenceDataset, create_dataloader
from data.labels import DigitVocabulary
from evaluate import evaluate_model
from models.seq2seq import build_model_from_config


class Trainer:
    """Manages model training, validation, checkpointing, early stopping, and resume."""

    def __init__(
        self,
        config: dict[str, Any],
        vocab: DigitVocabulary | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.vocab = vocab if vocab is not None else DigitVocabulary()

        proj_cfg = config.get("project", {})
        train_cfg = config.get("training", {})
        inf_cfg = config.get("inference", {})

        seed_everything(int(proj_cfg.get("seed", 42)))
        self.device = get_device(str(proj_cfg.get("device", "auto")))

        self.model = build_model_from_config(config, self.vocab).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        self.epochs = int(train_cfg.get("epochs", 15))
        self.teacher_forcing_ratio = float(train_cfg.get("teacher_forcing_ratio", 0.5))
        raw_tf_start = train_cfg.get("teacher_forcing_start")
        raw_tf_end = train_cfg.get("teacher_forcing_end")
        self.teacher_forcing_start = float(raw_tf_start) if raw_tf_start is not None else None
        self.teacher_forcing_end = float(raw_tf_end) if raw_tf_end is not None else None
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))

        sched_cfg = train_cfg.get("lr_scheduler", {})
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None
        if bool(sched_cfg.get("enabled", True)):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 2)),
                min_lr=float(sched_cfg.get("min_lr", 1e-6)),
            )
        self.max_decode_length = int(
            inf_cfg.get("max_decode_length", int(config.get("data", {}).get("max_sequence_length", 5)) + 1)
        )

        self.checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "checkpoints"), self.base_dir)
        self.output_dir = resolve_path(train_cfg.get("output_dir", "outputs"), self.base_dir)
        self.best_checkpoint_name = str(train_cfg.get("best_checkpoint_name", "best_model.pt"))
        self.final_checkpoint_name = str(train_cfg.get("final_checkpoint_name", "final_model.pt"))

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []

    def save_checkpoint(self, path: str | Path, epoch: int) -> Path:
        """Persist model weights, optimizer state, config, and training history."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "vocab": {
                "pad_id": self.vocab.pad_id,
                "sos_id": self.vocab.sos_id,
                "eos_id": self.vocab.eos_id,
                "token_to_id": self.vocab.token_to_id,
            },
            "history": self.history,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> dict[str, Any]:
        """Load a checkpoint to resume training."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.history = list(checkpoint.get("history", []))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        return checkpoint

    def get_teacher_forcing_ratio(self, epoch: int, total_epochs: int) -> float:
        """Compute teacher forcing ratio for a given epoch (linearly scheduled if configured)."""
        if self.teacher_forcing_start is not None and self.teacher_forcing_end is not None:
            if total_epochs <= 1:
                return self.teacher_forcing_start
            progress = min(1.0, max(0.0, (epoch - 1) / float(total_epochs - 1)))
            return self.teacher_forcing_start + (self.teacher_forcing_end - self.teacher_forcing_start) * progress
        return self.teacher_forcing_ratio

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        teacher_forcing_ratio: float | None = None,
        show_progress: bool = True,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tf_ratio = self.teacher_forcing_ratio if teacher_forcing_ratio is None else float(teacher_forcing_ratio)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [train]") if show_progress else train_loader
        for batch in iterator:
            images = batch["images"].to(self.device)
            decoder_inputs = batch["decoder_inputs"].to(self.device)
            decoder_targets = batch["decoder_targets"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(
                images,
                decoder_inputs,
                teacher_forcing_ratio=tf_ratio,
            )
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_targets.reshape(-1),
            )
            loss.backward()

            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            num_batches += 1
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        if num_batches == 0:
            raise ValueError("Train DataLoader is empty.")
        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
        resume_from: str | Path | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run full training loop with validation, checkpointing, and loss curve plotting."""
        if resume_from is not None:
            self.load_checkpoint(resume_from, load_optimizer=True)

        total_epochs = int(epochs) if epochs is not None else self.epochs
        self.epochs = total_epochs

        epochs_without_improvement = 0
        best_ckpt_path = self.checkpoint_dir / self.best_checkpoint_name
        final_ckpt_path = self.checkpoint_dir / self.final_checkpoint_name

        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, total_epochs + 1):
            last_epoch = epoch
            tf_ratio = self.get_teacher_forcing_ratio(epoch=epoch, total_epochs=total_epochs)
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self._train_one_epoch(
                train_loader,
                epoch=epoch,
                teacher_forcing_ratio=tf_ratio,
                show_progress=show_progress,
            )

            val_report = evaluate_model(
                model=self.model,
                dataloader=val_loader,
                vocab=self.vocab,
                device=self.device,
                max_decode_length=self.max_decode_length,
                criterion=self.criterion,
                output_dir=None,
                split_name="val",
            )
            val_loss = float(val_report.get("loss", float("inf")))
            val_exact_acc = float(val_report["recognition"]["exact_sequence_accuracy"])
            val_cer = float(val_report["recognition"]["cer"])

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            next_lr = float(self.optimizer.param_groups[0]["lr"])

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_exact_sequence_accuracy": val_exact_acc,
                "val_cer": val_cer,
                "lr": current_lr,
                "teacher_forcing_ratio": tf_ratio,
            }
            self.history.append(record)

            print(
                f"Epoch {epoch}/{total_epochs} | lr={current_lr:.6g} | tf={tf_ratio:.2f} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_exact_acc={val_exact_acc:.4f} | val_cer={val_cer:.4f}"
            )
            if next_lr < current_lr:
                print(f"ReduceLROnPlateau lowered learning rate: {current_lr:.6g} -> {next_lr:.6g}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(best_ckpt_path, epoch=epoch)
            else:
                epochs_without_improvement += 1

            self.save_checkpoint(final_ckpt_path, epoch=epoch)

            if (
                self.early_stopping_patience > 0
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_loss improvement for {self.early_stopping_patience} epochs)."
                )
                break

        if not best_ckpt_path.exists() and last_epoch >= 1:
            self.save_checkpoint(best_ckpt_path, epoch=last_epoch)

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "epoch_metrics.csv", index=False)
        with (self.output_dir / "epoch_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return history_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-digit sequence recognition model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
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
        help="Override DataLoader num_workers (e.g., 2 or 4 on Linux/Colab GPU).",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.setdefault("project", {})["device"] = args.device
    base_dir = Path.cwd()

    data_cfg = config["data"]
    img_cfg = config["image"]
    train_cfg = config["training"]

    splits_dir = resolve_path(data_cfg["splits_dir"], base_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Split files not found in {splits_dir}. Run `python src/inspect_data.py` first."
        )

    vocab = DigitVocabulary()
    raw_width = img_cfg.get("width", 160)
    target_width = int(raw_width) if raw_width is not None else None

    ds_kwargs = {
        "vocab": vocab,
        "base_dir": base_dir,
        "target_height": int(img_cfg.get("height", 32)),
        "target_width": target_width,
        "pad_mode": str(img_cfg.get("pad_mode", "right")),
        "pad_value": int(img_cfg.get("pad_value", 0)),
        "normalize_mean": float(img_cfg.get("normalize_mean", 0.0)),
        "normalize_std": float(img_cfg.get("normalize_std", 1.0)),
    }

    train_dataset = DigitSequenceDataset(train_csv, **ds_kwargs)
    val_dataset = DigitSequenceDataset(val_csv, **ds_kwargs)

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))

    trainer = Trainer(config=config, vocab=vocab, base_dir=base_dir)
    use_pin_memory = trainer.device.type == "cuda"

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pad_id=vocab.pad_id,
        pin_memory=use_pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_id=vocab.pad_id,
        pin_memory=use_pin_memory,
    )

    print(
        f"Training on device: {trainer.device} | batch_size={batch_size} | "
        f"num_workers={num_workers} | train={len(train_dataset)} | val={len(val_dataset)}"
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        resume_from=args.resume,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
