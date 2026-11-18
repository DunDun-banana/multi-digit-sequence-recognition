"""Data loading, label extraction, preprocessing, and splitting modules."""

from data.dataset import DigitSequenceDataset, collate_digit_batch, create_dataloader
from data.labels import (
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SOS_ID,
    SOS_TOKEN,
    DigitVocabulary,
    build_and_validate_metadata,
    extract_label_from_filename,
    load_annotation_mapping,
    validate_label,
)
from data.preprocessing import preprocess_image
from data.splitting import create_splits, verify_no_leakage

__all__ = [
    "DigitVocabulary",
    "PAD_ID",
    "SOS_ID",
    "EOS_ID",
    "PAD_TOKEN",
    "SOS_TOKEN",
    "EOS_TOKEN",
    "extract_label_from_filename",
    "load_annotation_mapping",
    "validate_label",
    "build_and_validate_metadata",
    "preprocess_image",
    "DigitSequenceDataset",
    "collate_digit_batch",
    "create_dataloader",
    "create_splits",
    "verify_no_leakage",
]
