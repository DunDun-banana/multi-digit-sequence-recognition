"""Unit tests for vocabulary, label extraction, and dataset validation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from data.labels import (
    EOS_ID,
    PAD_ID,
    SOS_ID,
    DigitVocabulary,
    build_and_validate_metadata,
    extract_label_from_filename,
    validate_label,
)


def _save_dummy_png(path: Path, size: tuple[int, int] = (128, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=0).save(path)


def test_vocabulary_encode_decode_preserves_leading_zeros() -> None:
    vocab = DigitVocabulary()
    assert vocab.vocab_size == 13

    encoded = vocab.encode("0487", add_sos=True, add_eos=True)
    assert encoded == [SOS_ID, 0, 4, 8, 7, EOS_ID]

    padded = vocab.pad_sequence(encoded, max_len=8)
    assert padded == [SOS_ID, 0, 4, 8, 7, EOS_ID, PAD_ID, PAD_ID]

    decoded = vocab.decode(padded + [9, 9], stop_at_eos=True)
    assert decoded == "0487"


def test_extract_label_from_filename_cases() -> None:
    label, err = extract_label_from_filename("00012_78055.png", mode="filename_suffix", delimiter="_")
    assert label == "78055"
    assert err is None

    label, err = extract_label_from_filename("img_123_09.png", mode="filename_suffix", delimiter="_")
    assert label == "09"
    assert err is None

    label, err = extract_label_from_filename("12345.png", mode="filename_suffix", delimiter="_")
    assert label is None
    assert err == "missing_label_delimiter"

    label, err = extract_label_from_filename("12345.png", mode="filename_stem")
    assert label == "12345"
    assert err is None


def test_validate_label_checks() -> None:
    assert validate_label("01234", min_length=1, max_length=5) == (True, None)
    assert validate_label("", min_length=1, max_length=5) == (False, "empty_sequence")
    assert validate_label("12a4", min_length=1, max_length=5) == (False, "invalid_label_characters")
    assert validate_label("123456", min_length=1, max_length=5) == (False, "unexpected_sequence_length")
    assert validate_label(None, min_length=1, max_length=5) == (False, "missing_label")


def test_build_and_validate_metadata_reports_invalid_samples(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _save_dummy_png(raw_dir / "train" / "0001_78055.png", size=(160, 32))
    _save_dummy_png(raw_dir / "train" / "0002_048.png", size=(96, 32))
    _save_dummy_png(raw_dir / "test" / "0003_91.png", size=(64, 32))

    _save_dummy_png(raw_dir / "train" / "nodelimiter.png")
    _save_dummy_png(raw_dir / "train" / "0004_12abc.png")
    _save_dummy_png(raw_dir / "train" / "0005_123456.png")
    (raw_dir / "train" / "0006_123.txt").write_text("not an image", encoding="utf-8")
    corrupted = raw_dir / "train" / "0007_123.png"
    corrupted.write_bytes(b"corrupted-bytes")

    out_csv = tmp_path / "processed" / "metadata.csv"
    df, report = build_and_validate_metadata(
        raw_dir=raw_dir,
        output_csv=out_csv,
        label_config={"mode": "filename_suffix", "delimiter": "_"},
        supported_extensions=[".png", ".jpg"],
        min_sequence_length=1,
        max_sequence_length=5,
        base_dir=tmp_path,
    )

    assert len(df) == 3
    assert set(df.columns) >= {"image_path", "label", "sequence_length", "original_split"}
    assert report["valid_samples"] == 3
    assert report["rejected_samples"] == 5
    assert report["rejection_counts"]["missing_label_delimiter"] == 1
    assert report["rejection_counts"]["invalid_label_characters"] == 1
    assert report["rejection_counts"]["unexpected_sequence_length"] == 1
    assert report["rejection_counts"]["unsupported_format"] == 1
    assert report["rejection_counts"]["corrupted_image"] == 1
    assert (tmp_path / "processed" / "validation_report.json").exists()


def test_annotation_file_label_mode(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _save_dummy_png(raw_dir / "image_001.png", size=(64, 32))
    _save_dummy_png(raw_dir / "image_002.png", size=(96, 32))
    ann_file = raw_dir / "annotations.json"
    ann_file.write_text(
        json.dumps({"image_001.png": "42", "image_002.png": "007"}),
        encoding="utf-8",
    )

    df, report = build_and_validate_metadata(
        raw_dir=raw_dir,
        output_csv=tmp_path / "processed" / "metadata.csv",
        label_config={"mode": "annotation_file", "annotation_file": str(ann_file)},
        base_dir=tmp_path,
    )
    assert len(df) == 2
    assert set(df["label"].tolist()) == {"42", "007"}
    assert report["rejected_samples"] == 0
