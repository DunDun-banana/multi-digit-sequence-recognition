"""Image preprocessing utilities for grayscale multi-digit sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_grayscale_image(image_source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load an image from disk, PIL Image, or NumPy array as grayscale ('L')."""
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            return img.convert("L")
    if isinstance(image_source, Image.Image):
        return image_source.convert("L")
    if isinstance(image_source, np.ndarray):
        arr = image_source
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 1:
                arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("L")
    raise TypeError(f"Unsupported image source type: {type(image_source).__name__}")


def resize_preserving_aspect(
    img: Image.Image,
    target_height: int = 32,
    max_width: int | None = 160,
) -> Image.Image:
    """Resize a PIL image to `target_height` preserving aspect ratio."""
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {(orig_w, orig_h)}")

    scale = target_height / float(orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = target_height

    if max_width is not None and new_w > max_width:
        down_scale = max_width / float(orig_w)
        new_w = max_width
        new_h = max(1, int(round(orig_h * down_scale)))

    if (new_w, new_h) != (orig_w, orig_h):
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return img


def pad_image_array(
    arr: np.ndarray,
    target_height: int,
    target_width: int,
    pad_mode: str = "right",
    pad_value: int = 0,
) -> np.ndarray:
    """Pad a 2D grayscale array (H, W) to (target_height, target_width)."""
    h, w = arr.shape
    if h > target_height or w > target_width:
        raise ValueError(
            f"Array shape {(h, w)} exceeds target shape {(target_height, target_width)}."
        )
    if (h, w) == (target_height, target_width):
        return arr

    canvas = np.full((target_height, target_width), fill_value=pad_value, dtype=arr.dtype)
    pad_y = (target_height - h) // 2

    if pad_mode == "right":
        pad_x = 0
    elif pad_mode == "left":
        pad_x = target_width - w
    elif pad_mode == "center":
        pad_x = (target_width - w) // 2
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode!r}. Use 'right', 'left', or 'center'.")

    canvas[pad_y : pad_y + h, pad_x : pad_x + w] = arr
    return canvas


def preprocess_image(
    image_source: str | Path | Image.Image | np.ndarray,
    target_height: int = 32,
    target_width: int | None = 160,
    pad_mode: str = "right",
    pad_value: int = 0,
    normalize_mean: float = 0.0,
    normalize_std: float = 1.0,
) -> torch.Tensor:
    """Preprocess an image into a normalized grayscale tensor of shape [1, H, W]."""
    img = load_grayscale_image(image_source)
    img = resize_preserving_aspect(img, target_height=target_height, max_width=target_width)
    arr = np.asarray(img, dtype=np.uint8)

    if target_width is not None:
        arr = pad_image_array(
            arr,
            target_height=target_height,
            target_width=target_width,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

    tensor = torch.from_numpy(arr.copy()).float().unsqueeze(0) / 255.0
    if normalize_mean != 0.0 or normalize_std != 1.0:
        if normalize_std == 0.0:
            raise ValueError("normalize_std must be non-zero.")
        tensor = (tensor - normalize_mean) / normalize_std
    return tensor
