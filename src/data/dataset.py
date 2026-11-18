"""PyTorch Dataset and DataLoader collation for multi-digit sequence recognition."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.labels import PAD_ID, DigitVocabulary
from data.preprocessing import preprocess_image


class DigitSequenceDataset(Dataset):
    """Dataset returning preprocessed grayscale images and SOS/EOS-wrapped token sequences."""

    def __init__(
        self,
        metadata: pd.DataFrame | str | Path,
        vocab: DigitVocabulary | None = None,
        base_dir: str | Path | None = None,
        target_height: int = 32,
        target_width: int | None = 160,
        pad_mode: str = "right",
        pad_value: int = 0,
        normalize_mean: float = 0.0,
        normalize_std: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(metadata, (str, Path)):
            df = pd.read_csv(metadata, dtype={"image_path": str, "label": str}, keep_default_na=False)
        else:
            df = metadata.copy()

        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("Metadata must contain 'image_path' and 'label' columns.")

        self.df = df.reset_index(drop=True)
        self.vocab = vocab if vocab is not None else DigitVocabulary()
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.target_height = target_height
        self.target_width = target_width
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        raw_path = str(row["image_path"])
        label = str(row["label"]).strip()
        img_path = self._resolve_image_path(raw_path)

        image = preprocess_image(
            img_path,
            target_height=self.target_height,
            target_width=self.target_width,
            pad_mode=self.pad_mode,
            pad_value=self.pad_value,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )

        token_ids = self.vocab.encode(label, add_sos=True, add_eos=True)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return {
            "image": image,
            "tokens": tokens,
            "target_length": len(token_ids) - 1,
            "label": label,
            "image_path": raw_path,
        }


def collate_digit_batch(
    batch: list[dict[str, Any]],
    pad_id: int = PAD_ID,
    image_pad_value: float = 0.0,
) -> dict[str, Any]:
    """Collate variable-width images and variable-length digit sequences into a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)
    batch_size = len(batch)

    images = torch.full(
        (batch_size, 1, max_h, max_w),
        fill_value=image_pad_value,
        dtype=torch.float32,
    )
    for idx, item in enumerate(batch):
        _, h, w = item["image"].shape
        images[idx, :, :h, :w] = item["image"]

    max_token_len = max(item["tokens"].shape[0] for item in batch)
    padded_tokens = torch.full((batch_size, max_token_len), fill_value=pad_id, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        seq = item["tokens"]
        seq_len = seq.shape[0]
        padded_tokens[idx, :seq_len] = seq
        target_lengths[idx] = int(item["target_length"])

    decoder_inputs = padded_tokens[:, :-1]
    decoder_targets = padded_tokens[:, 1:]

    return {
        "images": images,
        "decoder_inputs": decoder_inputs,
        "decoder_targets": decoder_targets,
        "target_lengths": target_lengths,
        "labels": [item["label"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def create_dataloader(
    dataset: DigitSequenceDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_id: int = PAD_ID,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader with the sequence-padding collate function."""
    collate_fn = partial(collate_digit_batch, pad_id=pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
