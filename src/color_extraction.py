"""
Color extraction from well plate images.

Extracts mean RGB values from individual wells of a 96-well plate
captured by a fixed overhead USB camera. Uses a configurable grid
layout that should be calibrated once for the camera position.

Task A4 from TASK_DISTRIBUTION.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ======================================================================
# Well grid definition
# ======================================================================

# Default grid parameters for a Corning 96-well plate in slot 5
# captured by a 1920x1080 fixed overhead camera.
# These MUST be calibrated for your specific camera position.
# Run: python -m src.color_extraction --calibrate <image_path>
DEFAULT_GRID = {
    # Top-left corner of well A1 (pixel coordinates)
    "origin_x": 400,
    "origin_y": 150,
    # Spacing between well centers (pixels)
    "spacing_x": 90,   # column spacing (1→12, left to right)
    "spacing_y": 85,    # row spacing (A→H, top to bottom)
    # Well ROI radius (pixels) — use ~60-70% of actual well radius
    # to avoid edge artifacts from the well rim
    "roi_radius": 25,
}

ROW_LABELS = list("ABCDEFGH")
N_ROWS = 8
N_COLS = 12


def get_well_center(
    row: int,
    col: int,
    grid: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Get pixel coordinates of a well center.

    Args:
        row: Row index (0=A, 7=H).
        col: Column index (0=col1, 11=col12).
        grid: Grid parameters dict. Uses DEFAULT_GRID if None.

    Returns:
        (cx, cy) pixel coordinates.
    """
    g = grid or DEFAULT_GRID
    cx = g["origin_x"] + col * g["spacing_x"]
    cy = g["origin_y"] + row * g["spacing_y"]
    return int(cx), int(cy)


def extract_well_rgb(
    image: np.ndarray,
    row: int,
    col: int,
    grid: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract mean RGB from a single well.

    Uses a circular ROI centered on the well to avoid edge/rim artifacts.

    Args:
        image: BGR image (as loaded by OpenCV).
        row: Row index (0=A).
        col: Column index (0=col1).
        grid: Grid parameters.

    Returns:
        (3,) array of mean [R, G, B] values (0-255).
    """
    g = grid or DEFAULT_GRID
    cx, cy = get_well_center(row, col, g)
    radius = g["roi_radius"]

    # Create circular mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Extract mean color within the circular ROI (OpenCV uses BGR)
    mean_bgr = cv2.mean(image, mask=mask)[:3]
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])  # BGR → RGB


def extract_column_rgb(
    image: np.ndarray,
    col: int,
    rows: Optional[List[int]] = None,
    grid: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract RGB from all wells in a column.

    Args:
        image: BGR image.
        col: Column index (0-based).
        rows: List of row indices to extract. Default: all 8 rows (A-H).
        grid: Grid parameters.

    Returns:
        (n_rows, 3) array of RGB values.
    """
    if rows is None:
        rows = list(range(N_ROWS))
    return np.array([extract_well_rgb(image, r, col, grid) for r in rows])


def extract_plate_rgb(
    image_path: str,
    grid: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract RGB from all 96 wells of a plate image.

    Args:
        image_path: Path to plate image.
        grid: Grid parameters.

    Returns:
        (8, 12, 3) array: plate_rgb[row, col] = [R, G, B].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    plate = np.zeros((N_ROWS, N_COLS, 3))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            plate[r, c] = extract_well_rgb(image, r, c, grid)
    return plate


def white_balance(
    rgb: np.ndarray,
    water_rgb: np.ndarray,
    reference: float = 255.0,
) -> np.ndarray:
    """
    Apply white balance correction using a water (clear) reference well.

    Scales each channel so that the water well maps to ``reference``
    (default 255 = pure white). This removes lighting drift between
    imaging sessions.

    Args:
        rgb: (..., 3) RGB values to correct.
        water_rgb: (3,) RGB from the water/clear control well.
        reference: Target white value (default 255).

    Returns:
        Corrected RGB array, same shape as *rgb*, clipped to [0, 255].
    """
    safe = np.maximum(water_rgb, 1.0)
    scale = reference / safe
    return np.clip(rgb * scale, 0, 255)


def extract_experiment_rgb(
    image_path: str,
    col: int,
    experiment_rows: List[str] = None,
    control_rows: List[str] = None,
    grid: Optional[Dict] = None,
    apply_white_balance: bool = True,
    water_row: str = "D",
) -> Dict[str, np.ndarray]:
    """
    Extract RGB for one experiment column: controls + experiment replicates.

    When *apply_white_balance* is True (default), the water control well
    (row D) is used as a white reference to normalize lighting.

    Args:
        image_path: Path to plate image.
        col: Column index (0-based, so column 1 = index 0).
        experiment_rows: Row labels for experiment wells. Default: ["E","F","G","H"].
        control_rows: Row labels for control wells. Default: ["A","B","C","D"].
        grid: Grid parameters.
        apply_white_balance: Use water well for white balance correction.
        water_row: Row label of the water/clear control well.

    Returns:
        Dict with:
            "experiment_rgb": (n_exp, 3) RGB per experiment well
            "experiment_mean": (3,) mean RGB across replicates
            "experiment_std": (3,) std RGB across replicates
            "control_rgb": dict mapping row label → (3,) RGB
    """
    if experiment_rows is None:
        experiment_rows = ["E", "F", "G", "H"]
    if control_rows is None:
        control_rows = ["A", "B", "C", "D"]

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Extract control wells (raw)
    ctrl_rgb = {}
    for label in control_rows:
        ri = ROW_LABELS.index(label)
        ctrl_rgb[label] = extract_well_rgb(image, ri, col, grid)

    # Extract experiment wells (raw)
    exp_indices = [ROW_LABELS.index(r) for r in experiment_rows]
    exp_rgb = np.array([extract_well_rgb(image, ri, col, grid) for ri in exp_indices])

    # White balance using water control
    if apply_white_balance and water_row in ctrl_rgb:
        water_rgb = ctrl_rgb[water_row]
        exp_rgb = white_balance(exp_rgb, water_rgb)
        for label in ctrl_rgb:
            ctrl_rgb[label] = white_balance(ctrl_rgb[label], water_rgb)

    return {
        "experiment_rgb": exp_rgb,
        "experiment_mean": exp_rgb.mean(axis=0),
        "experiment_std": exp_rgb.std(axis=0),
        "control_rgb": ctrl_rgb,
    }


# ======================================================================
# Grid calibration helper
# ======================================================================

def visualize_grid(
    image_path: str,
    output_path: Optional[str] = None,
    grid: Optional[Dict] = None,
) -> str:
    """
    Draw the well grid overlay on an image for calibration verification.

    Args:
        image_path: Path to plate image.
        output_path: Where to save annotated image. Default: adds '_grid' suffix.
        grid: Grid parameters to visualize.

    Returns:
        Path to the annotated image.
    """
    g = grid or DEFAULT_GRID
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cx, cy = get_well_center(r, c, g)
            # Draw ROI circle
            cv2.circle(image, (cx, cy), g["roi_radius"], (0, 255, 0), 1)
            # Draw center dot
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
            # Label
            label = f"{ROW_LABELS[r]}{c + 1}"
            cv2.putText(
                image, label, (cx - 12, cy - g["roi_radius"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
            )

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_grid{ext}"

    cv2.imwrite(output_path, image)
    print(f"[VISION] Grid overlay saved: {output_path}")
    return output_path


# ======================================================================
# CLI for calibration
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Well plate grid calibration tool")
    parser.add_argument("image", help="Path to a plate image for calibration")
    parser.add_argument("--origin-x", type=int, default=DEFAULT_GRID["origin_x"])
    parser.add_argument("--origin-y", type=int, default=DEFAULT_GRID["origin_y"])
    parser.add_argument("--spacing-x", type=int, default=DEFAULT_GRID["spacing_x"])
    parser.add_argument("--spacing-y", type=int, default=DEFAULT_GRID["spacing_y"])
    parser.add_argument("--roi-radius", type=int, default=DEFAULT_GRID["roi_radius"])
    args = parser.parse_args()

    cal_grid = {
        "origin_x": args.origin_x,
        "origin_y": args.origin_y,
        "spacing_x": args.spacing_x,
        "spacing_y": args.spacing_y,
        "roi_radius": args.roi_radius,
    }
    out = visualize_grid(args.image, grid=cal_grid)
    print(f"Check the overlay at: {out}")
    print("Adjust --origin-x/y --spacing-x/y --roi-radius until circles align with wells.")
