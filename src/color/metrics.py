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


# ======================================================================
# Linear RGB helpers
# ======================================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255) to linear RGB (0-1)."""
    v = np.clip(np.asarray(rgb, dtype=float), 0, 255) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB (0-1) to sRGB (0-255)."""
    v = np.clip(linear, 0, 1)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1.0 / 2.4) - 0.055)
    return srgb * 255.0


# ======================================================================
# Color gamut (reachable colors from dye mixing)
# ======================================================================

def predict_mix_rgb(
    pure_rgbs: dict,
    fractions: np.ndarray,
    water_rgb: np.ndarray = None,
) -> np.ndarray:
    """Predict RGB of a dye mixture via linear mixing in linear RGB space.

    Args:
        pure_rgbs: {"red": (3,), "green": (3,), "blue": (3,)} sRGB 0-255.
        fractions: (3,) array [fr, fg, fb] where sum <= 1. Remainder is water.
        water_rgb: (3,) sRGB of water well. Default white (255,255,255).

    Returns:
        (3,) predicted sRGB clipped to [0, 255].
    """
    if water_rgb is None:
        water_rgb = np.array([255.0, 255.0, 255.0])

    dyes = [np.asarray(pure_rgbs[k], dtype=float) for k in ("red", "green", "blue")]
    water_frac = max(0.0, 1.0 - float(np.sum(fractions)))

    dye_lin = [srgb_to_linear(d) for d in dyes]
    water_lin = srgb_to_linear(np.asarray(water_rgb, dtype=float))

    mixed = water_frac * water_lin
    for i in range(3):
        mixed = mixed + float(fractions[i]) * dye_lin[i]

    return np.clip(linear_to_srgb(mixed), 0, 255)


def compute_reachable_gamut(
    pure_rgbs: dict,
    water_rgb: np.ndarray = None,
    n_samples: int = 5000,
    seed: int = 42,
) -> dict:
    """Sample the reachable color gamut from pure dye controls.

    Args:
        pure_rgbs: {"red": (3,), "green": (3,), "blue": (3,)} sRGB 0-255.
        water_rgb: sRGB of water well.
        n_samples: number of random mixtures to sample.
        seed: RNG seed.

    Returns:
        dict with keys: samples_rgb (n,3), samples_lab (n,3),
        samples_fractions (n,3), pure_rgbs, water_rgb, suggested_targets.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet(1,1,1,1) -> uniform over simplex including water fraction
    proportions = rng.dirichlet(np.ones(4), size=n_samples)
    fractions = proportions[:, :3]  # dye fractions; col 3 is water

    samples_rgb = np.array([
        predict_mix_rgb(pure_rgbs, f, water_rgb) for f in fractions
    ])
    samples_lab = srgb_to_lab(samples_rgb)

    suggested = _pick_spread_targets(samples_rgb, samples_lab, fractions, n=12)

    return {
        "samples_rgb": samples_rgb.tolist(),
        "samples_lab": samples_lab.tolist(),
        "samples_fractions": fractions.tolist(),
        "pure_rgbs": {k: np.asarray(v).tolist() for k, v in pure_rgbs.items()},
        "water_rgb": np.asarray(water_rgb if water_rgb is not None else [255, 255, 255]).tolist(),
        "suggested_targets": suggested,
    }


def _pick_spread_targets(samples_rgb, samples_lab, fractions, n=12):
    """Pick n well-spread target suggestions via greedy farthest-point in Lab."""
    if len(samples_rgb) == 0:
        return []
    indices = [0]
    for _ in range(min(n - 1, len(samples_rgb) - 1)):
        min_dists = np.full(len(samples_lab), np.inf)
        for idx in indices:
            d = np.sum((samples_lab - samples_lab[idx]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
        indices.append(int(np.argmax(min_dists)))

    targets = []
    for idx in indices:
        rgb = samples_rgb[idx]
        lab = samples_lab[idx]
        f = fractions[idx]
        targets.append({
            "rgb": [round(float(rgb[0])), round(float(rgb[1])), round(float(rgb[2]))],
            "lab": [round(float(lab[0]), 1), round(float(lab[1]), 1), round(float(lab[2]), 1)],
            "fractions": [round(float(f[0]), 3), round(float(f[1]), 3), round(float(f[2]), 3)],
        })
    return targets
