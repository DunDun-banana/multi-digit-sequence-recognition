"""Unit tests for sequence recognition metrics and numerical metrics (R^2, NSE, MAE, RMSE)."""

from __future__ import annotations

import math

import pytest

from metrics import (
    compute_all_metrics,
    compute_numerical_regression_metrics,
    compute_recognition_metrics,
    levenshtein_distance,
)


def test_recognition_metrics_exact_and_cer() -> None:
    assert levenshtein_distance("123", "123") == 0
    assert levenshtein_distance("123", "124") == 1
    assert levenshtein_distance("12", "123") == 1

    y_true = ["123", "456", "78"]
    y_pred = ["123", "450", "7"]
    res = compute_recognition_metrics(y_true, y_pred)
    assert res["total_samples"] == 3
    assert res["exact_matches"] == 1
    assert res["exact_sequence_accuracy"] == pytest.approx(1.0 / 3.0)
    assert res["cer"] == pytest.approx(0.25)
    assert res["character_accuracy"] == pytest.approx(0.75)


def test_numerical_metrics_on_equal_length_subset() -> None:
    y_true = ["13", "57", "999"]
    y_pred = ["14", "55", "99"]

    res = compute_numerical_regression_metrics(y_true, y_pred)
    assert res["status"] == "ok"
    assert res["total_samples"] == 3
    assert res["equal_length_samples"] == 2
    assert res["coverage"] == pytest.approx(2.0 / 3.0)
    assert res["num_digit_positions"] == 4
    assert res["mae"] == pytest.approx(0.75)
    assert res["rmse"] == pytest.approx(math.sqrt(1.25))
    assert res["r2"] == pytest.approx(0.75)
    assert res["nse"] == pytest.approx(0.75)


def test_numerical_metrics_handles_no_equal_lengths() -> None:
    res = compute_numerical_regression_metrics(["123", "45"], ["12", "456"])
    assert res["status"] == "insufficient_equal_length_samples"
    assert res["equal_length_samples"] == 0
    assert res["num_digit_positions"] == 0
    assert res["r2"] is None
    assert res["nse"] is None
    assert res["mae"] is None
    assert res["rmse"] is None


def test_numerical_metrics_handles_zero_variance() -> None:
    res = compute_numerical_regression_metrics(["55", "55"], ["54", "55"])
    assert res["status"] == "zero_target_variance"
    assert res["r2"] is None
    assert res["nse"] is None
    assert res["mae"] == pytest.approx(0.25)
    assert res["rmse"] == pytest.approx(0.5)


def test_numerical_metrics_rejects_special_tokens() -> None:
    with pytest.raises(ValueError, match="non-digit character"):
        compute_all_metrics(["12"], ["1<PAD>"])
