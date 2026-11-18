"""Visualization helpers for dataset inspection, loss curves, predictions, and attention."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot and save training vs validation loss curves."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    if val_losses:
        ax.plot(epochs, val_losses, marker="s", label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sample_predictions(
    images: Sequence[np.ndarray],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    output_path: str | Path,
    max_samples: int = 8,
) -> Path:
    """Save a grid comparing input images, ground-truth labels, and predictions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = min(len(images), len(true_labels), len(pred_labels), max_samples)
    if count == 0:
        raise ValueError("Cannot plot sample predictions with zero samples.")

    fig, axes = plt.subplots(nrows=count, ncols=1, figsize=(8, 2.2 * count))
    if count == 1:
        axes = [axes]

    for ax, img, true_seq, pred_seq in zip(axes, images[:count], true_labels[:count], pred_labels[:count]):
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        ax.imshow(arr, cmap="gray")
        match_mark = "OK" if true_seq == pred_seq else "MISMATCH"
        ax.set_title(f"True: {true_seq} | Pred: {pred_seq} ({match_mark})")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_attention_sample(
    image: np.ndarray,
    attention_weights: np.ndarray,
    true_label: str | None,
    pred_label: str,
    output_path: str | Path,
) -> Path:
    """Save a two-panel plot of an input image and its decoder attention heatmap."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    ax1.imshow(arr, cmap="gray")
    title = f"Pred: {pred_label}" if true_label is None else f"True: {true_label} | Pred: {pred_label}"
    ax1.set_title(title)
    ax1.axis("off")

    ax2.imshow(attention_weights, aspect="auto", cmap="viridis")
    ax2.set_title("Decoder Attention Weights over CNN Horizontal Positions")
    ax2.set_xlabel("Encoder Spatial Position")
    ax2.set_ylabel("Decoding Step")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_length_distribution(
    length_counts: dict[int, int],
    output_path: str | Path,
) -> Path:
    """Save a bar chart of digit sequence lengths."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lengths = sorted(length_counts.keys())
    counts = [length_counts[k] for k in lengths]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in lengths], counts)
    ax.set_title("Sequence Length Distribution")
    ax.set_xlabel("Number of Digits")
    ax.set_ylabel("Image Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
