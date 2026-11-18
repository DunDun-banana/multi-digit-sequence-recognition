"""Vocabulary, label extraction, and dataset validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DIGITS = tuple("0123456789")
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 10
SOS_ID = 11
EOS_ID = 12


@dataclass(frozen=True)
class DigitVocabulary:
    """Vocabulary mapping digits '0'-'9' and special tokens (PAD, SOS, EOS) to IDs."""

    pad_id: int = PAD_ID
    sos_id: int = SOS_ID
    eos_id: int = EOS_ID
    token_to_id: dict[str, int] = field(
        default_factory=lambda: {
            **{str(d): d for d in range(10)},
            PAD_TOKEN: PAD_ID,
            SOS_TOKEN: SOS_ID,
            EOS_TOKEN: EOS_ID,
        }
    )

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, label: str, add_sos: bool = True, add_eos: bool = True) -> list[int]:
        """Convert a digit string (e.g. '4878') into a list of token IDs."""
        if not label or not all(ch in DIGITS for ch in label):
            raise ValueError(f"Invalid digit sequence label: {label!r}")
        ids = [self.token_to_id[ch] for ch in label]
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, token_ids: Sequence[int], stop_at_eos: bool = True) -> str:
        """Convert token IDs back into a digit string, ignoring special tokens."""
        out: list[str] = []
        for raw_id in token_ids:
            idx = int(raw_id)
            if idx == self.eos_id:
                if stop_at_eos:
                    break
                continue
            if idx in (self.pad_id, self.sos_id):
                continue
            if 0 <= idx <= 9:
                out.append(str(idx))
        return "".join(out)

    def pad_sequence(self, token_ids: Sequence[int], max_len: int) -> list[int]:
        """Pad or validate a token ID sequence to exactly `max_len` using PAD_ID."""
        seq = list(token_ids)
        if len(seq) > max_len:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_len={max_len}")
        return seq + [self.pad_id] * (max_len - len(seq))


def extract_label_from_filename(
    file_path: str | Path,
    mode: str = "filename_suffix",
    delimiter: str = "_",
    allow_no_delimiter: bool = False,
) -> tuple[str | None, str | None]:
    """Extract a digit sequence label from an image filename."""
    stem = Path(file_path).stem
    if mode == "filename_stem":
        return (stem, None) if stem else (None, "empty_sequence")

    if mode == "filename_suffix":
        if delimiter not in stem:
            if allow_no_delimiter:
                return (stem, None) if stem else (None, "empty_sequence")
            return None, "missing_label_delimiter"
        suffix = stem.split(delimiter)[-1]
        if suffix == "":
            return None, "empty_sequence"
        return suffix, None

    raise ValueError(f"Unsupported filename label extraction mode: {mode!r}")


def load_annotation_mapping(
    annotation_file: str | Path,
    image_col: str = "image_path",
    label_col: str = "label",
) -> dict[str, str]:
    """Load an image-to-label mapping from a CSV or JSON annotation file."""
    ann_path = Path(annotation_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    suffix = ann_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(ann_path, dtype={image_col: str, label_col: str}, keep_default_na=False)
        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            key = Path(str(row[image_col])).as_posix()
            mapping[key] = str(row[label_col]).strip()
            mapping[Path(key).name] = str(row[label_col]).strip()
        return mapping

    if suffix == ".json":
        with ann_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        mapping = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                key = Path(str(k)).as_posix()
                val = str(v).strip()
                mapping[key] = val
                mapping[Path(key).name] = val
        elif isinstance(raw, list):
            for item in raw:
                key = Path(str(item[image_col])).as_posix()
                val = str(item[label_col]).strip()
                mapping[key] = val
                mapping[Path(key).name] = val
        else:
            raise ValueError("JSON annotation file must be a dict or a list of records.")
        return mapping

    raise ValueError(f"Unsupported annotation file format: {ann_path.suffix!r}")


def validate_label(
    label: str | None,
    min_length: int = 1,
    max_length: int = 5,
) -> tuple[bool, str | None]:
    """Validate that a label is a non-empty ASCII digit string within length bounds."""
    if label is None:
        return False, "missing_label"
    label_str = str(label).strip()
    if len(label_str) == 0:
        return False, "empty_sequence"
    if not all(ch in DIGITS for ch in label_str):
        return False, "invalid_label_characters"
    if len(label_str) < min_length or len(label_str) > max_length:
        return False, "unexpected_sequence_length"
    return True, None


def _infer_original_split(rel_parts: tuple[str, ...]) -> str:
    """Infer 'train' or 'test' split from directory hierarchy if present."""
    lowered = [p.lower() for p in rel_parts[:-1]]
    if "train" in lowered:
        return "train"
    if "test" in lowered:
        return "test"
    if "val" in lowered or "valid" in lowered or "validation" in lowered:
        return "val"
    return "unassigned"


def build_and_validate_metadata(
    raw_dir: str | Path,
    output_csv: str | Path,
    label_config: dict[str, Any],
    supported_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp"),
    min_sequence_length: int = 1,
    max_sequence_length: int = 5,
    base_dir: str | Path | None = None,
    verify_images: bool = True,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scan raw_dir, validate images and labels, and save metadata CSV + report."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    root_for_rel = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    allowed_exts = {ext.lower() for ext in supported_extensions}

    mode = label_config.get("mode", "filename_suffix")
    delimiter = label_config.get("delimiter", "_")
    allow_no_delimiter = bool(label_config.get("allow_no_delimiter", False))

    annotation_map: dict[str, str] | None = None
    if mode == "annotation_file":
        ann_file = label_config.get("annotation_file")
        if not ann_file:
            raise ValueError("label_extraction.annotation_file must be set when mode='annotation_file'.")
        ann_path = Path(ann_file)
        if not ann_path.is_absolute():
            ann_path = root_for_rel / ann_path
        annotation_map = load_annotation_mapping(
            ann_path,
            image_col=label_config.get("image_col", "image_path"),
            label_col=label_config.get("label_col", "label"),
        )

    rejection_counts: dict[str, int] = {
        "unsupported_format": 0,
        "duplicate_path": 0,
        "corrupted_image": 0,
        "missing_label": 0,
        "missing_label_delimiter": 0,
        "empty_sequence": 0,
        "invalid_label_characters": 0,
        "unexpected_sequence_length": 0,
    }
    rejection_examples: list[dict[str, str]] = []

    def _record_rejection(reason: str, file_path: Path, detail: str = "") -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        if len(rejection_examples) < 25:
            rejection_examples.append(
                {"path": file_path.as_posix(), "reason": reason, "detail": detail}
            )

    seen_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    total_files_scanned = 0

    all_files = sorted(p for p in raw_path.rglob("*") if p.is_file() and p.name != ".gitkeep")
    iterator = tqdm(
        all_files,
        desc="Validating dataset",
        unit="file",
        disable=not show_progress,
    )
    for file_path in iterator:
        total_files_scanned += 1
        ext = file_path.suffix.lower()

        if ext not in allowed_exts:
            if mode == "annotation_file" and ext in (".csv", ".json"):
                continue
            _record_rejection("unsupported_format", file_path, ext)
            continue

        resolved = file_path.resolve()
        try:
            stored_path = resolved.relative_to(root_for_rel).as_posix()
        except ValueError:
            stored_path = resolved.as_posix()

        if stored_path in seen_paths:
            _record_rejection("duplicate_path", file_path)
            continue
        seen_paths.add(stored_path)

        if verify_images:
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                _record_rejection("corrupted_image", file_path, str(exc))
                continue

        if mode == "annotation_file":
            assert annotation_map is not None
            rel_to_raw = file_path.relative_to(raw_path).as_posix()
            label = annotation_map.get(rel_to_raw)
            if label is None:
                label = annotation_map.get(file_path.name)
            err = "missing_label" if label is None else None
        else:
            label, err = extract_label_from_filename(
                file_path,
                mode=mode,
                delimiter=delimiter,
                allow_no_delimiter=allow_no_delimiter,
            )

        if err is not None:
            _record_rejection(err, file_path)
            continue

        valid, reason = validate_label(
            label,
            min_length=min_sequence_length,
            max_length=max_sequence_length,
        )
        if not valid:
            assert reason is not None
            _record_rejection(reason, file_path, str(label))
            continue

        assert label is not None
        clean_label = label.strip()
        rel_parts = file_path.relative_to(raw_path).parts
        records.append(
            {
                "image_path": stored_path,
                "label": clean_label,
                "sequence_length": len(clean_label),
                "original_split": _infer_original_split(rel_parts),
            }
        )

    df = pd.DataFrame(
        records,
        columns=["image_path", "label", "sequence_length", "original_split"],
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    total_rejected = sum(rejection_counts.values())
    length_dist = (
        {str(k): int(v) for k, v in df["sequence_length"].value_counts().sort_index().items()}
        if not df.empty
        else {}
    )
    report: dict[str, Any] = {
        "total_files_scanned": total_files_scanned,
        "valid_samples": len(df),
        "rejected_samples": total_rejected,
        "rejection_counts": rejection_counts,
        "rejection_examples": rejection_examples,
        "sequence_length_distribution": length_dist,
        "metadata_csv": out_path.as_posix(),
    }

    report_path = out_path.with_name("validation_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return df, report
