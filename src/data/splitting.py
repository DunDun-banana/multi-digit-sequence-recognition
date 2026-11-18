"""Reproducible train/validation/test splitting with data-leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame reproducibly, stratifying by sequence_length when valid."""
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    stratify_col = None
    if stratify and "sequence_length" in df.columns:
        counts = df["sequence_length"].value_counts()
        if (counts >= 2).all() and len(df) * test_size >= len(counts):
            stratify_col = df["sequence_length"]

    left, right = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_col,
    )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Raise AssertionError if any image_path appears in more than one split."""
    train_paths = set(train_df["image_path"].tolist())
    val_paths = set(val_df["image_path"].tolist())
    test_paths = set(test_df["image_path"].tolist())

    overlap_tv = train_paths & val_paths
    overlap_tt = train_paths & test_paths
    overlap_vt = val_paths & test_paths
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Data leakage detected! Overlaps: train/val={len(overlap_tv)}, "
            f"train/test={len(overlap_tt)}, val/test={len(overlap_vt)}"
        )


def create_splits(
    metadata: pd.DataFrame | str | Path,
    splits_dir: str | Path,
    seed: int = 42,
    val_ratio_from_train: float = 0.15,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by_length: bool = True,
) -> dict[str, Any]:
    """Create and save reproducible train/val/test CSV splits."""
    if isinstance(metadata, (str, Path)):
        df = pd.read_csv(metadata, dtype={"image_path": str, "label": str}, keep_default_na=False)
    else:
        df = metadata.copy()

    if len(df) < 3:
        raise ValueError(f"Need at least 3 samples to create train/val/test splits, got {len(df)}.")

    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    has_official_train_test = (
        "original_split" in df.columns
        and set(df["original_split"].unique()) >= {"train", "test"}
    )

    if has_official_train_test:
        split_mode = "official_train_test_with_val_holdout"
        official_train = df[df["original_split"] == "train"].reset_index(drop=True)
        test_df = df[df["original_split"] == "test"].reset_index(drop=True)

        if "val" in set(df["original_split"].unique()):
            train_df = official_train
            val_df = df[df["original_split"] == "val"].reset_index(drop=True)
        else:
            train_df, val_df = _safe_stratified_split(
                official_train,
                test_size=val_ratio_from_train,
                seed=seed,
                stratify=stratify_by_length,
            )
    else:
        split_mode = "reproducible_random_70_15_15"
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}"
            )
        train_df, temp_df = _safe_stratified_split(
            df,
            test_size=(val_ratio + test_ratio),
            seed=seed,
            stratify=stratify_by_length,
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = _safe_stratified_split(
            temp_df,
            test_size=relative_test,
            seed=seed,
            stratify=stratify_by_length,
        )

    verify_no_leakage(train_df, val_df, test_df)

    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    def _dist(split_df: pd.DataFrame) -> dict[str, int]:
        if split_df.empty or "sequence_length" not in split_df.columns:
            return {}
        return {str(k): int(v) for k, v in split_df["sequence_length"].value_counts().sort_index().items()}

    summary: dict[str, Any] = {
        "split_mode": split_mode,
        "seed": seed,
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "sequence_length_distributions": {
            "train": _dist(train_df),
            "val": _dist(val_df),
            "test": _dist(test_df),
        },
        "files": {
            "train": train_path.as_posix(),
            "val": val_path.as_posix(),
            "test": test_path.as_posix(),
        },
    }

    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
