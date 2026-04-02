"""Plate grid geometry calibration.

Two grid representations:

- :class:`PlateGrid` — axis-aligned model (origin + spacing). Fast,
  backward-compatible with ``color_extraction.DEFAULT_GRID``.
- :class:`PlateGrid4` — 4-corner model (A1, A12, H1, H12). Handles
  rotation and perspective skew via bilinear interpolation.

Both share the same public API surface: ``grid_to_dict``,
``get_well_center``, ``get_labeled_well_centers``, ``render_grid_overlay``,
and JSON persistence.

Usage::

    from src.vision.geometry import (
        compute_grid_from_corners,
        manual_calibrate_grid_from_image,
        save_grid_calibration,
    )

    # From known coordinates
    grid = compute_grid_from_corners(
        a1=(380, 240), a12=(1550, 240), h1=(375, 925), h12=(1545, 920),
    )

    # Interactive — click 4 corners on an image
    grid = manual_calibrate_grid_from_image("data/plate.jpg")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math

ROW_LABELS = list("ABCDEFGH")
N_ROWS = 8
N_COLS = 12


# ======================================================================
# Grid dataclasses
# ======================================================================

@dataclass(frozen=True)
class PlateGrid:
    """Axis-aligned pixel-space grid for a 96-well plate.

    Backward-compatible with ``color_extraction.DEFAULT_GRID``.
    """
    origin_x: float
    origin_y: float
    spacing_x: float
    spacing_y: float
    roi_radius: float


@dataclass(frozen=True)
class PlateGrid4:
    """4-corner pixel-space grid for a 96-well plate.

    Uses bilinear interpolation from the four corner wells to compute
    any well centre. Handles slight rotation and perspective skew.
    """
    a1: Tuple[float, float]
    a12: Tuple[float, float]
    h1: Tuple[float, float]
    h12: Tuple[float, float]
    roi_radius: float


# Type alias for either grid representation.
AnyGrid = Union[PlateGrid, PlateGrid4]


# ======================================================================
# Compute from corners
# ======================================================================

def compute_grid_from_corners(
    a1: Tuple[float, float],
    a12: Tuple[float, float],
    h1: Tuple[float, float],
    h12: Optional[Tuple[float, float]] = None,
    roi_scale: float = 0.35,
) -> AnyGrid:
    """Compute grid from corner-well centres.

    With 3 corners (``h12=None``) returns a :class:`PlateGrid` (axis-aligned).
    With 4 corners returns a :class:`PlateGrid4` (rotation-robust).

    Args:
        a1: (x, y) pixel centre of well A1 (top-left).
        a12: (x, y) pixel centre of well A12 (top-right).
        h1: (x, y) pixel centre of well H1 (bottom-left).
        h12: (x, y) pixel centre of well H12 (bottom-right). Optional.
        roi_scale: Fraction of the smaller spacing to use as ROI radius.

    Returns:
        :class:`PlateGrid` or :class:`PlateGrid4`.
    """
    if h12 is None:
        # Legacy 3-corner axis-aligned model
        spacing_x = (a12[0] - a1[0]) / 11.0
        spacing_y = (h1[1] - a1[1]) / 7.0
        roi_radius = min(abs(spacing_x), abs(spacing_y)) * roi_scale
        return PlateGrid(
            origin_x=a1[0],
            origin_y=a1[1],
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            roi_radius=roi_radius,
        )

    # 4-corner bilinear model
    top_dist = math.hypot(a12[0] - a1[0], a12[1] - a1[1]) / 11.0
    left_dist = math.hypot(h1[0] - a1[0], h1[1] - a1[1]) / 7.0
    roi_radius = min(top_dist, left_dist) * roi_scale
    return PlateGrid4(
        a1=tuple(a1), a12=tuple(a12),
        h1=tuple(h1), h12=tuple(h12),
        roi_radius=roi_radius,
    )


# ======================================================================
# Dict conversion (backward-compatible with color_extraction)
# ======================================================================

def grid_to_dict(grid: AnyGrid) -> Dict[str, float]:
    """Convert any grid to the dict format used by ``color_extraction``.

    For :class:`PlateGrid4`, this produces the axis-aligned approximation
    needed by ``color_extraction.get_well_center()``, plus the 4 corner
    fields so the full model can be round-tripped.
    """
    if isinstance(grid, PlateGrid):
        return {
            "origin_x": grid.origin_x,
            "origin_y": grid.origin_y,
            "spacing_x": grid.spacing_x,
            "spacing_y": grid.spacing_y,
            "roi_radius": grid.roi_radius,
        }

    # PlateGrid4 — include both the corner model and the axis-aligned
    # approximation for backward compatibility.
    return {
        "origin_x": grid.a1[0],
        "origin_y": grid.a1[1],
        "spacing_x": (grid.a12[0] - grid.a1[0]) / 11.0,
        "spacing_y": (grid.h1[1] - grid.a1[1]) / 7.0,
        "roi_radius": grid.roi_radius,
        # 4-corner fields for round-trip
        "corners": {
            "a1": list(grid.a1),
            "a12": list(grid.a12),
            "h1": list(grid.h1),
            "h12": list(grid.h12),
        },
    }


def grid_from_dict(d: Dict) -> AnyGrid:
    """Build a grid from a dict.

    If the dict contains a ``"corners"`` key with 4 entries, returns a
    :class:`PlateGrid4`. Otherwise returns a :class:`PlateGrid`.
    """
    corners = d.get("corners")
    if corners and all(k in corners for k in ("a1", "a12", "h1", "h12")):
        return PlateGrid4(
            a1=tuple(corners["a1"]),
            a12=tuple(corners["a12"]),
            h1=tuple(corners["h1"]),
            h12=tuple(corners["h12"]),
            roi_radius=d["roi_radius"],
        )
    return PlateGrid(
        origin_x=d["origin_x"],
        origin_y=d["origin_y"],
        spacing_x=d["spacing_x"],
        spacing_y=d["spacing_y"],
        roi_radius=d["roi_radius"],
    )


# ======================================================================
# JSON persistence
# ======================================================================

def save_grid_calibration(path: str, grid: AnyGrid) -> str:
    """Save grid calibration to a JSON file. Returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(grid_to_dict(grid), f, indent=2)
    return path


def load_grid_calibration(path: str) -> AnyGrid:
    """Load grid calibration from a JSON file.

    Automatically detects whether the file contains a 4-corner or
    axis-aligned grid.
    """
    with open(path) as f:
        return grid_from_dict(json.load(f))


# ======================================================================
# Well centres
# ======================================================================

def _bilinear(
    a1: Tuple[float, float],
    a12: Tuple[float, float],
    h1: Tuple[float, float],
    h12: Tuple[float, float],
    u: float,
    v: float,
) -> Tuple[float, float]:
    """Bilinear interpolation across the plate surface.

    ``u`` runs 0→1 along columns (A1→A12), ``v`` runs 0→1 along rows
    (A1→H1).
    """
    x = (
        (1 - u) * (1 - v) * a1[0]
        + u * (1 - v) * a12[0]
        + (1 - u) * v * h1[0]
        + u * v * h12[0]
    )
    y = (
        (1 - u) * (1 - v) * a1[1]
        + u * (1 - v) * a12[1]
        + (1 - u) * v * h1[1]
        + u * v * h12[1]
    )
    return x, y


def get_well_center(
    row: int,
    col: int,
    grid: AnyGrid,
) -> Tuple[int, int]:
    """Pixel centre of well at (row, col) indices (0-based)."""
    if isinstance(grid, PlateGrid4):
        u = col / 11.0
        v = row / 7.0
        x, y = _bilinear(grid.a1, grid.a12, grid.h1, grid.h12, u, v)
        return int(round(x)), int(round(y))

    cx = grid.origin_x + col * grid.spacing_x
    cy = grid.origin_y + row * grid.spacing_y
    return int(round(cx)), int(round(cy))


def get_labeled_well_centers(grid: AnyGrid) -> Dict[str, Tuple[int, int]]:
    """Return ``{label: (cx, cy)}`` for all 96 wells (A1 … H12)."""
    centers: Dict[str, Tuple[int, int]] = {}
    for r, row_letter in enumerate(ROW_LABELS):
        for c in range(N_COLS):
            label = f"{row_letter}{c + 1}"
            centers[label] = get_well_center(r, c, grid)
    return centers


# ======================================================================
# Overlay rendering
# ======================================================================

def render_grid_overlay(
    image_path: str,
    grid: AnyGrid,
    output_path: Optional[str] = None,
    label_wells: bool = True,
) -> str:
    """Draw the well grid on an image and save it.

    Args:
        image_path: Source plate image.
        grid: Grid calibration to overlay.
        output_path: Where to write the annotated image.
            Defaults to ``<image_path>_grid_overlay.<ext>``.
        label_wells: Whether to draw well labels (e.g. A1).

    Returns:
        Path to the saved overlay image.
    """
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    radius = int(round(grid.roi_radius))

    for r, row_letter in enumerate(ROW_LABELS):
        for c in range(N_COLS):
            cx, cy = get_well_center(r, c, grid)
            cv2.circle(image, (cx, cy), radius, (0, 255, 0), 1)
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
            if label_wells:
                label = f"{row_letter}{c + 1}"
                cv2.putText(
                    image, label, (cx - 12, cy - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
                )

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_grid_overlay{ext}"

    cv2.imwrite(output_path, image)
    return output_path


# ======================================================================
# Interactive calibration from image (matplotlib click)
# ======================================================================

def manual_calibrate_grid_from_image(
    image_path: str,
    save_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
    roi_scale: float = 0.35,
) -> PlateGrid4:
    """Interactively calibrate by clicking 4 corner wells on a plate image.

    Opens a matplotlib figure. The operator clicks wells in order:
    **A1 → A12 → H1 → H12**. After 4 clicks the grid is computed,
    an overlay is rendered, and the calibration is saved to JSON.

    Args:
        image_path: Path to a plate image.
        save_path: Where to save the calibration JSON.
            Defaults to ``<image_dir>/grid_calibration.json``.
        overlay_path: Where to save the overlay image.
            Defaults to ``<image_path>_grid_overlay.<ext>``.
        roi_scale: Fraction of well spacing for the ROI radius.

    Returns:
        A :class:`PlateGrid4`.
    """
    import cv2
    import matplotlib.pyplot as plt

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    corner_names = ["A1 (top-left)", "A12 (top-right)",
                    "H1 (bottom-left)", "H12 (bottom-right)"]
    clicks: List[Tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(image_rgb)
    ax.set_title(f"Click well {corner_names[0]}")
    ax.axis("off")

    def _on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        clicks.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "r+", markersize=14, markeredgewidth=2)
        ax.annotate(
            corner_names[len(clicks) - 1].split(" ")[0],
            (event.xdata, event.ydata),
            textcoords="offset points", xytext=(8, -8),
            color="red", fontsize=10, fontweight="bold",
        )
        fig.canvas.draw()

        if len(clicks) < 4:
            ax.set_title(f"Click well {corner_names[len(clicks)]}")
        else:
            ax.set_title("All 4 corners captured — close this window")
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show(block=True)

    if len(clicks) < 4:
        raise RuntimeError(
            f"Only {len(clicks)} of 4 corners clicked. "
            "Please click A1, A12, H1, H12 in order."
        )

    a1, a12, h1, h12 = clicks[0], clicks[1], clicks[2], clicks[3]
    grid = compute_grid_from_corners(a1=a1, a12=a12, h1=h1, h12=h12,
                                     roi_scale=roi_scale)

    # Save calibration JSON
    if save_path is None:
        base_dir = os.path.dirname(image_path) or "."
        save_path = os.path.join(base_dir, "grid_calibration.json")
    save_grid_calibration(save_path, grid)
    print(f"Grid calibration saved: {save_path}")

    # Save overlay
    overlay = render_grid_overlay(image_path, grid, output_path=overlay_path)
    print(f"Grid overlay saved: {overlay}")

    return grid
