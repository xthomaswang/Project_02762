"""Lightweight color-space utilities for optimization and diagnostics.

This module intentionally has no image-processing dependencies so that
ML-only code can import color metrics without pulling in OpenCV.
"""

from __future__ import annotations

import numpy as np


_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255) to CIELAB using a D65 white point."""
    v = np.clip(np.asarray(rgb, dtype=float), 0, 255) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T
    t = xyz / _D65_WHITE
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def color_distance(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    metric: str = "rgb_euclidean",
) -> float:
    """Compute color distance between two sRGB values."""
    a = np.asarray(rgb_a, dtype=float)
    b = np.asarray(rgb_b, dtype=float)
    if metric == "delta_e_lab":
        return float(np.sqrt(np.sum((srgb_to_lab(a) - srgb_to_lab(b)) ** 2)))
    if metric == "rgb_euclidean":
        return float(np.sqrt(np.sum((a - b) ** 2)))
    raise ValueError(
        f"Unknown metric: {metric!r}. Use 'rgb_euclidean' or 'delta_e_lab'."
    )


def batch_color_distances(
    rgb_batch: np.ndarray,
    target_rgb: np.ndarray,
    metric: str = "rgb_euclidean",
) -> np.ndarray:
    """Compute distances from a batch of sRGB values to one target."""
    batch = np.clip(np.asarray(rgb_batch, dtype=float), 0, 255)
    target = np.asarray(target_rgb, dtype=float)
    if metric == "delta_e_lab":
        lab_batch = srgb_to_lab(batch)
        lab_target = srgb_to_lab(target.reshape(1, 3))
        return np.sqrt(np.sum((lab_batch - lab_target) ** 2, axis=1))
    if metric == "rgb_euclidean":
        return np.sqrt(np.sum((batch - target) ** 2, axis=1))
    raise ValueError(
        f"Unknown metric: {metric!r}. Use 'rgb_euclidean' or 'delta_e_lab'."
    )
