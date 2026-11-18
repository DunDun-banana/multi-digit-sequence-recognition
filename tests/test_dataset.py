"""Unit tests for preprocessing, Dataset, DataLoader collation, and splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from data.dataset import DigitSequenceDataset, create_dataloader
from data.labels import EOS_ID, PAD_ID, SOS_ID, DigitVocabulary
from data.preprocessing import preprocess_image
from data.splitting import create_splits, verify_no_leakage


def _create_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_preprocess_image_converts_resizes_and_pads(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    _create_image(img_path, size=(96, 32))

    tensor = preprocess_image(img_path, target_height=32, target_width=160, pad_mode="right")
    assert tensor.shape == (1, 32, 160)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert float(tensor[:, :, 96:].max()) == 0.0


def test_dataset_and_collate_variable_lengths(tmp_path: Path) -> None:
    rows = []
    for idx, (label, width) in enumerate([("7", 32), ("42", 64), ("90815", 160)]):
        rel = f"raw/img_{idx}_{label}.png"
        _create_image(tmp_path / rel, size=(width, 32))
        rows.append({"image_path": rel, "label": label, "sequence_length": len(label)})

    df = pd.DataFrame(rows)
    vocab = DigitVocabulary()
    ds = DigitSequenceDataset(df, vocab=vocab, base_dir=tmp_path, target_height=32, target_width=160)
    loader = create_dataloader(ds, batch_size=3, shuffle=False, pad_id=vocab.pad_id)

    batch = next(iter(loader))
    assert batch["images"].shape == (3, 1, 32, 160)
    assert batch["decoder_inputs"].shape == (3, 6)
    assert batch["decoder_targets"].shape == (3, 6)
    assert batch["target_lengths"].tolist() == [2, 3, 6]

    assert batch["decoder_inputs"][0].tolist() == [SOS_ID, 7, EOS_ID, PAD_ID, PAD_ID, PAD_ID]
    assert batch["decoder_targets"][0].tolist() == [7, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]


def test_create_splits_prevents_leakage(tmp_path: Path) -> None:
    rows = [
        {"image_path": f"raw/s_{i}_{i % 10}.png", "label": str(i % 10) * ((i % 3) + 1), "sequence_length": (i % 3) + 1, "original_split": "unassigned"}
        for i in range(40)
    ]
    summary = create_splits(pd.DataFrame(rows), splits_dir=tmp_path / "splits_flat", seed=42)
    assert summary["counts"] == {"train": 28, "val": 6, "test": 6}

    train_df = pd.read_csv(tmp_path / "splits_flat" / "train.csv", dtype={"label": str})
    val_df = pd.read_csv(tmp_path / "splits_flat" / "val.csv", dtype={"label": str})
    test_df = pd.read_csv(tmp_path / "splits_flat" / "test.csv", dtype={"label": str})
    verify_no_leakage(train_df, val_df, test_df)

    rows_official = [
        {"image_path": f"raw/train/{i}.png", "label": "12", "sequence_length": 2, "original_split": "train"}
        for i in range(20)
    ] + [
        {"image_path": f"raw/test/{i}.png", "label": "34", "sequence_length": 2, "original_split": "test"}
        for i in range(5)
    ]
    summary2 = create_splits(
        pd.DataFrame(rows_official),
        splits_dir=tmp_path / "splits_official",
        seed=42,
        val_ratio_from_train=0.2,
    )
    assert summary2["split_mode"] == "official_train_test_with_val_holdout"
    assert summary2["counts"] == {"train": 16, "val": 4, "test": 5}
