"""Evaluation metrics for sequence recognition and assignment-required numerical metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def levenshtein_distance(source: str, target: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    prev_row = list(range(len(target) + 1))
    for i, sc in enumerate(source, start=1):
        curr_row = [i]
        for j, tc in enumerate(target, start=1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (sc != tc)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _extract_digits_only(seq: str) -> list[int]:
    """Convert a digit string to integer values, rejecting non-digit special tokens."""
    digits: list[int] = []
    for ch in seq:
        if "0" <= ch <= "9":
            digits.append(int(ch))
        else:
            raise ValueError(
                f"Found non-digit character {ch!r} in sequence {seq!r}. "
                "PAD, SOS, and EOS tokens must be stripped before computing numerical metrics."
            )
    return digits


def compute_recognition_metrics(
    y_true_seqs: Sequence[str],
    y_pred_seqs: Sequence[str],
) -> dict[str, Any]:
    """Compute sequence-recognition metrics across the entire evaluation set."""
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(
            f"Length mismatch: len(y_true_seqs)={len(y_true_seqs)} != len(y_pred_seqs)={len(y_pred_seqs)}"
        )

    total_samples = len(y_true_seqs)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "exact_matches": 0,
            "exact_sequence_accuracy": 0.0,
            "character_accuracy": 0.0,
            "cer": 0.0,
            "total_true_chars": 0,
            "total_edit_distance": 0,
        }

    exact_matches = 0
    total_edit_distance = 0
    total_true_chars = 0

    for true_seq, pred_seq in zip(y_true_seqs, y_pred_seqs):
        if true_seq == pred_seq:
            exact_matches += 1
        total_edit_distance += levenshtein_distance(pred_seq, true_seq)
        total_true_chars += len(true_seq)

    cer = (total_edit_distance / total_true_chars) if total_true_chars > 0 else 0.0
    char_acc = max(0.0, 1.0 - cer)
    exact_acc = exact_matches / total_samples

    return {
        "total_samples": total_samples,
        "exact_matches": exact_matches,
        "exact_sequence_accuracy": float(exact_acc),
        "character_accuracy": float(char_acc),
        "cer": float(cer),
        "total_true_chars": total_true_chars,
        "total_edit_distance": total_edit_distance,
    }


def compute_numerical_regression_metrics(
    y_true_seqs: Sequence[str],
    y_pred_seqs: Sequence[str],
) -> dict[str, Any]:
    """Compute R^2, NSE, MAE, and RMSE on aligned digit positions of equal-length predictions.

    Only samples where len(pred) == len(true) > 0 are included so that PAD/SOS/EOS
    tokens are never treated as numeric digits.
    """
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(
            f"Length mismatch: len(y_true_seqs)={len(y_true_seqs)} != len(y_pred_seqs)={len(y_pred_seqs)}"
        )

    total_samples = len(y_true_seqs)
    equal_length_samples = 0
    y_true_digits: list[int] = []
    y_pred_digits: list[int] = []

    for true_seq, pred_seq in zip(y_true_seqs, y_pred_seqs):
        t_digits = _extract_digits_only(true_seq)
        p_digits = _extract_digits_only(pred_seq)
        if len(t_digits) > 0 and len(t_digits) == len(p_digits):
            equal_length_samples += 1
            y_true_digits.extend(t_digits)
            y_pred_digits.extend(p_digits)

    coverage = (equal_length_samples / total_samples) if total_samples > 0 else 0.0
    num_positions = len(y_true_digits)

    if num_positions == 0:
        return {
            "status": "insufficient_equal_length_samples",
            "note": (
                "No samples had matching predicted and ground-truth sequence lengths; "
                "R^2, NSE, MAE, and RMSE cannot be computed without treating padding as digits."
            ),
            "total_samples": total_samples,
            "equal_length_samples": 0,
            "coverage": float(coverage),
            "num_digit_positions": 0,
            "r2": None,
            "nse": None,
            "mae": None,
            "rmse": None,
        }

    y_true_arr = np.asarray(y_true_digits, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_digits, dtype=np.float64)

    errors = y_true_arr - y_pred_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(errors**2))))

    ss_res = float(np.sum(errors**2))
    y_mean = float(np.mean(y_true_arr))
    ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))

    if ss_tot == 0.0:
        return {
            "status": "zero_target_variance",
            "note": (
                "Ground-truth digit values in the equal-length subset have zero variance "
                "(sum((y - y_bar)^2) == 0), so R^2 and NSE are undefined."
            ),
            "total_samples": total_samples,
            "equal_length_samples": equal_length_samples,
            "coverage": float(coverage),
            "num_digit_positions": num_positions,
            "r2": None,
            "nse": None,
            "mae": mae,
            "rmse": rmse,
        }

    r2 = float(1.0 - (ss_res / ss_tot))
    nse = float(1.0 - (ss_res / ss_tot))

    return {
        "status": "ok",
        "note": (
            "Computed on aligned digit positions from the subset of samples whose predicted "
            "and ground-truth lengths match (excluding PAD, SOS, and EOS tokens)."
        ),
        "total_samples": total_samples,
        "equal_length_samples": equal_length_samples,
        "coverage": float(coverage),
        "num_digit_positions": num_positions,
        "r2": r2,
        "nse": nse,
        "mae": mae,
        "rmse": rmse,
    }


def compute_all_metrics(
    y_true_seqs: Sequence[str],
    y_pred_seqs: Sequence[str],
) -> dict[str, Any]:
    """Compute both sequence-recognition metrics and assignment numerical metrics."""
    return {
        "recognition": compute_recognition_metrics(y_true_seqs, y_pred_seqs),
        "numerical_equal_length": compute_numerical_regression_metrics(y_true_seqs, y_pred_seqs),
    }
