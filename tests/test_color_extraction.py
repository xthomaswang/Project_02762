"""
Pre-tests for color extraction — validates that the grid-based well
detection correctly identifies wells, assigns positions, and extracts
colors from plate images.

Works with:
  1. Synthetic plate images (generated automatically, always runs).
  2. Real plate images (scanned from ``data/test_images/``).

Run::

    pytest tests/test_color_extraction.py -v
    pytest tests/test_color_extraction.py -v -k synthetic   # synthetic only
    pytest tests/test_color_extraction.py -v -k real        # real images only

Or standalone with a detailed visual report::

    python tests/test_color_extraction.py [--image_dir data/synthetic]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.color_extraction import (
    DEFAULT_GRID,
    ROW_LABELS,
    N_ROWS,
    N_COLS,
    get_well_center,
    extract_well_rgb,
    extract_column_rgb,
    extract_plate_rgb,
    extract_experiment_rgb,
    white_balance,
)

SYNTHETIC_DIR = str(Path(__file__).resolve().parent.parent / "data" / "synthetic")

# ======================================================================
# Named colors (RGB)
# ======================================================================

KNOWN_COLORS_RGB = {
    "red":     np.array([220, 30, 30]),
    "green":   np.array([30, 200, 30]),
    "blue":    np.array([40, 40, 220]),
    "yellow":  np.array([230, 230, 30]),
    "purple":  np.array([160, 30, 200]),
    "cyan":    np.array([30, 210, 210]),
    "orange":  np.array([240, 150, 30]),
    "white":   np.array([245, 245, 245]),
}

# Short labels for grid display (max 7 chars)
COLOR_SHORT = {
    "red": "RED", "green": "GRN", "blue": "BLU",
    "yellow": "YEL", "purple": "PUR", "cyan": "CYN",
    "orange": "ORG", "white": "WHT", "empty": "---",
}


def _classify_color(rgb: np.ndarray, threshold: float = 60) -> str:
    """Classify an RGB value into a named color."""
    if np.mean(rgb) < 30:
        return "empty"
    best_name, best_dist = "?", float("inf")
    for name, ref in KNOWN_COLORS_RGB.items():
        d = np.linalg.norm(rgb - ref)
        if d < best_dist:
            best_dist, best_name = d, name
    if best_dist > threshold:
        return "mix"
    return best_name


def _short_label(rgb: np.ndarray) -> str:
    """Short color label for grid display."""
    name = _classify_color(rgb)
    return COLOR_SHORT.get(name, f"{int(rgb[0]):>3d}")


# ======================================================================
# Synthetic plate image generators
# ======================================================================

def generate_plate_image(
    color_map: Dict[Tuple[int, int], np.ndarray],
    grid: Optional[Dict] = None,
    width: int = 1920,
    height: int = 1080,
    bg_color: Tuple[int, int, int] = (40, 40, 40),
    well_radius: int = 35,
) -> np.ndarray:
    """Generate a synthetic plate image with colored wells."""
    g = grid or DEFAULT_GRID
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cx, cy = get_well_center(r, c, g)
            cv2.circle(image, (cx, cy), well_radius, (80, 80, 80), 2)

    for (r, c), rgb in color_map.items():
        cx, cy = get_well_center(r, c, g)
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.circle(image, (cx, cy), well_radius - 2, bgr, -1)

    return image


def generate_full_plate(grid: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """All 96 wells filled: A=red, B=green, C=blue, D=white, E-H=gradient."""
    g = grid or DEFAULT_GRID
    expected = np.zeros((N_ROWS, N_COLS, 3))
    color_map = {}
    row_colors = {
        0: KNOWN_COLORS_RGB["red"], 1: KNOWN_COLORS_RGB["green"],
        2: KNOWN_COLORS_RGB["blue"], 3: KNOWN_COLORS_RGB["white"],
    }
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if r < 4:
                rgb = row_colors[r].copy()
            else:
                frac = c / (N_COLS - 1)
                rgb = np.array([
                    int(255 * frac),
                    int(50 + 150 * (1 - frac)),
                    int(100 + 100 * frac),
                ])
            color_map[(r, c)] = rgb
            expected[r, c] = rgb
    return generate_plate_image(color_map, grid=g), expected


def generate_experiment_plate(
    n_columns: int = 3,
    grid: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Experiment plate with controls + colored experiment wells."""
    g = grid or DEFAULT_GRID
    color_map = {}
    exp_colors = [
        KNOWN_COLORS_RGB["purple"],
        KNOWN_COLORS_RGB["orange"],
        KNOWN_COLORS_RGB["cyan"],
        KNOWN_COLORS_RGB["yellow"],
        np.array([180, 100, 60]),
        np.array([60, 180, 120]),
    ]
    for col in range(min(n_columns, N_COLS)):
        color_map[(0, col)] = KNOWN_COLORS_RGB["red"]
        color_map[(1, col)] = KNOWN_COLORS_RGB["green"]
        color_map[(2, col)] = KNOWN_COLORS_RGB["blue"]
        color_map[(3, col)] = KNOWN_COLORS_RGB["white"]
        exp_c = exp_colors[col % len(exp_colors)]
        for r in range(4, 8):
            color_map[(r, col)] = exp_c
    return generate_plate_image(color_map, grid=g), color_map


def generate_checkerboard_plate(grid: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Alternating red/blue checkerboard for position verification."""
    g = grid or DEFAULT_GRID
    expected = np.zeros((N_ROWS, N_COLS, 3))
    color_map = {}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rgb = KNOWN_COLORS_RGB["red"] if (r + c) % 2 == 0 else KNOWN_COLORS_RGB["blue"]
            color_map[(r, c)] = rgb
            expected[r, c] = rgb
    return generate_plate_image(color_map, grid=g), expected


def generate_random_plate(
    seed: int = 42,
    grid: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random known colors in every well."""
    g = grid or DEFAULT_GRID
    rng = np.random.default_rng(seed)
    names = list(KNOWN_COLORS_RGB.keys())
    expected = np.zeros((N_ROWS, N_COLS, 3))
    color_map = {}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rgb = KNOWN_COLORS_RGB[rng.choice(names)].copy()
            color_map[(r, c)] = rgb
            expected[r, c] = rgb
    return generate_plate_image(color_map, grid=g), expected


def save_all_synthetic(grid: Optional[Dict] = None):
    """Generate and save all synthetic plate images to data/synthetic/."""
    g = grid or DEFAULT_GRID
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    plates = {
        "01_full_plate.png": generate_full_plate(g),
        "02_experiment_1col.png": generate_experiment_plate(1, g),
        "03_experiment_3col.png": generate_experiment_plate(3, g),
        "04_experiment_6col.png": generate_experiment_plate(6, g),
        "05_checkerboard.png": generate_checkerboard_plate(g),
        "06_random_colors.png": generate_random_plate(42, g),
    }
    paths = []
    for name, (image, _) in plates.items():
        path = os.path.join(SYNTHETIC_DIR, name)
        cv2.imwrite(path, image)
        paths.append(path)
    return paths


# ======================================================================
# Grid-format report
# ======================================================================

def print_grid_report(
    plate_rgb: np.ndarray,
    title: str = "",
    expected: Optional[np.ndarray] = None,
    mode: str = "label",
):
    """
    Print a compact grid table (rows A-H x columns 1-12).

    Args:
        plate_rgb: (8, 12, 3) extracted RGB values.
        title: Header text.
        expected: (8, 12, 3) expected RGB for accuracy check.
        mode: "label" for color names, "rgb" for raw values.
    """
    col_w = 8 if mode == "label" else 16

    if title:
        print(f"\n  {title}")
        print(f"  {'=' * (5 + N_COLS * col_w)}")

    # Column headers
    header = "     "
    for c in range(N_COLS):
        header += f"{c+1:>{col_w}}"
    print(header)
    print("     " + "-" * (N_COLS * col_w))

    n_correct, n_total = 0, 0
    for r in range(N_ROWS):
        row_str = f"  {ROW_LABELS[r]}  "
        for c in range(N_COLS):
            obs = plate_rgb[r, c]
            n_total += 1
            if mode == "rgb":
                cell = f"({int(obs[0]):>3},{int(obs[1]):>3},{int(obs[2]):>3})"
                row_str += f"{cell:>{col_w}}"
            else:
                label = _short_label(obs)
                if expected is not None:
                    dist = np.linalg.norm(obs - expected[r, c])
                    ok = dist < 15
                    if ok:
                        n_correct += 1
                    mark = label if ok else f"*{label}"
                    row_str += f"{mark:>{col_w}}"
                else:
                    row_str += f"{label:>{col_w}}"
        print(row_str)

    # Summary
    print()
    colors = {}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            name = _classify_color(plate_rgb[r, c])
            colors[name] = colors.get(name, 0) + 1

    dist_str = "  Colors: " + ", ".join(
        f"{name}={cnt}" for name, cnt in sorted(colors.items(), key=lambda x: -x[1])
    )
    print(dist_str)

    if expected is not None:
        acc = n_correct / n_total * 100
        print(f"  Accuracy: {n_correct}/{n_total} ({acc:.1f}%) {'PASS' if acc > 90 else 'FAIL'}")

    return n_correct if expected is not None else n_total


# ======================================================================
# Pytest fixtures
# ======================================================================

@pytest.fixture
def grid():
    return DEFAULT_GRID.copy()


@pytest.fixture
def full_plate(grid):
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    image, expected = generate_full_plate(grid)
    path = os.path.join(SYNTHETIC_DIR, "01_full_plate.png")
    cv2.imwrite(path, image)
    return path, image, expected


@pytest.fixture
def experiment_plate(grid):
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    image, color_map = generate_experiment_plate(1, grid)
    path = os.path.join(SYNTHETIC_DIR, "02_experiment_1col.png")
    cv2.imwrite(path, image)
    return path, image, color_map


# ======================================================================
# Tests: Synthetic plate
# ======================================================================

class TestSyntheticWellDetection:
    """Validate grid-based extraction on synthetic images."""

    def test_well_center_positions(self, grid):
        """Well centers follow grid formula."""
        for r in range(N_ROWS):
            for c in range(N_COLS):
                cx, cy = get_well_center(r, c, grid)
                assert cx == grid["origin_x"] + c * grid["spacing_x"]
                assert cy == grid["origin_y"] + r * grid["spacing_y"]

    def test_all_96_wells_detected(self, full_plate, grid):
        """All 96 wells should be extractable and return valid RGB."""
        path, image, expected = full_plate
        plate_rgb = extract_plate_rgb(path, grid=grid)
        assert plate_rgb.shape == (8, 12, 3)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                assert np.all(np.isfinite(plate_rgb[r, c])), \
                    f"Non-finite at {ROW_LABELS[r]}{c+1}"

    def test_color_accuracy(self, full_plate, grid):
        """Extracted colors should match expected within tolerance."""
        path, image, expected = full_plate
        plate_rgb = extract_plate_rgb(path, grid=grid)
        n_correct = print_grid_report(plate_rgb, "Color Accuracy Test", expected)
        accuracy = n_correct / 96 * 100
        assert accuracy > 90, f"Only {accuracy:.1f}% wells matched (need >90%)"

    def test_well_index_order(self, full_plate, grid):
        """Wells indexed in correct order: row A=red, B=green, C=blue."""
        path, image, expected = full_plate
        plate_rgb = extract_plate_rgb(path, grid=grid)
        for c in range(N_COLS):
            assert plate_rgb[0, c, 0] > 150, f"A{c+1} should be red"
            assert plate_rgb[1, c, 1] > 150, f"B{c+1} should be green"
            assert plate_rgb[2, c, 2] > 150, f"C{c+1} should be blue"

    def test_column_extraction(self, full_plate, grid):
        """extract_column_rgb returns correct shape and order."""
        path, image, expected = full_plate
        col_rgb = extract_column_rgb(image, col=0, grid=grid)
        assert col_rgb.shape == (8, 3)
        assert col_rgb[0, 0] > 150, "Row A col 0 should be red"
        assert col_rgb[1, 1] > 150, "Row B col 0 should be green"
        assert col_rgb[2, 2] > 150, "Row C col 0 should be blue"

    def test_checkerboard_positions(self, grid):
        """Checkerboard pattern verifies row/col are not swapped."""
        os.makedirs(SYNTHETIC_DIR, exist_ok=True)
        image, expected = generate_checkerboard_plate(grid)
        path = os.path.join(SYNTHETIC_DIR, "05_checkerboard.png")
        cv2.imwrite(path, image)
        plate_rgb = extract_plate_rgb(path, grid=grid)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                obs = _classify_color(plate_rgb[r, c])
                exp = "red" if (r + c) % 2 == 0 else "blue"
                assert obs == exp, f"{ROW_LABELS[r]}{c+1}: expected {exp}, got {obs}"


class TestSyntheticExperimentExtraction:
    """Validate experiment-specific extraction."""

    def test_experiment_rgb_structure(self, experiment_plate, grid):
        """extract_experiment_rgb returns correct dict structure."""
        path, image, color_map = experiment_plate
        result = extract_experiment_rgb(path, col=0, grid=grid, apply_white_balance=False)
        assert result["experiment_rgb"].shape == (4, 3)
        assert result["experiment_mean"].shape == (3,)
        assert result["experiment_std"].shape == (3,)
        assert set(result["control_rgb"].keys()) == {"A", "B", "C", "D"}

    def test_control_colors_correct(self, experiment_plate, grid):
        """Control rows should match expected colors."""
        path, image, color_map = experiment_plate
        result = extract_experiment_rgb(path, col=0, grid=grid, apply_white_balance=False)
        ctrl = result["control_rgb"]
        assert ctrl["A"][0] > 150, f"Row A red channel too low: {ctrl['A']}"
        assert ctrl["B"][1] > 150, f"Row B green channel too low: {ctrl['B']}"
        assert ctrl["C"][2] > 150, f"Row C blue channel too low: {ctrl['C']}"
        assert np.mean(ctrl["D"]) > 200, f"Row D should be white: {ctrl['D']}"

    def test_experiment_replicates_consistent(self, experiment_plate, grid):
        """All 4 experiment wells (E-H) should be similar."""
        path, image, color_map = experiment_plate
        result = extract_experiment_rgb(path, col=0, grid=grid, apply_white_balance=False)
        assert np.all(result["experiment_std"] < 20), \
            f"Replicate std too high: {result['experiment_std']}"

    def test_white_balance_normalization(self, experiment_plate, grid):
        """White balance should scale colors using water well."""
        path, image, color_map = experiment_plate
        raw = extract_experiment_rgb(path, col=0, grid=grid, apply_white_balance=False)
        bal = extract_experiment_rgb(path, col=0, grid=grid, apply_white_balance=True)
        water_bal = bal["control_rgb"]["D"]
        assert np.allclose(water_bal, 255, atol=5), \
            f"Water after WB should be ~255, got {water_bal}"
        assert not np.allclose(raw["experiment_mean"], bal["experiment_mean"], atol=1)


class TestWhiteBalance:
    """Test the white_balance function directly."""

    def test_perfect_lighting(self):
        corrected = white_balance(np.array([128, 64, 200]), np.array([255, 255, 255]))
        np.testing.assert_array_almost_equal(corrected, [128, 64, 200])

    def test_dim_lighting(self):
        corrected = white_balance(np.array([100, 50, 150]), np.array([200, 200, 200]))
        np.testing.assert_array_almost_equal(corrected, [127.5, 63.75, 191.25])

    def test_uneven_lighting(self):
        corrected = white_balance(np.array([100, 100, 100]), np.array([200, 250, 180]))
        assert corrected[2] > corrected[0] > corrected[1]

    def test_clipping(self):
        corrected = white_balance(np.array([200, 200, 200]), np.array([100, 100, 100]))
        assert np.all(corrected <= 255)

    def test_batch(self):
        corrected = white_balance(
            np.array([[100, 50, 150], [200, 100, 50]]), np.array([200, 200, 200]),
        )
        assert corrected.shape == (2, 3)


# ======================================================================
# Tests: Real images (skipped if none found)
# ======================================================================

def find_plate_images(search_dirs: Optional[List[str]] = None) -> List[str]:
    """Scan directories for plate images (png, jpg, jpeg)."""
    if search_dirs is None:
        root = Path(__file__).resolve().parent.parent
        search_dirs = [str(root / "data" / "test_images"), str(root / "data")]
    images = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(d, f))
    return images


@pytest.fixture
def real_images():
    images = find_plate_images()
    if not images:
        pytest.skip("No real plate images found in data/test_images/")
    return images


class TestRealImages:
    def test_real_image_extraction(self, real_images, grid):
        for img_path in real_images:
            plate_rgb = extract_plate_rgb(img_path, grid=grid)
            assert plate_rgb.shape == (8, 12, 3)
            print_grid_report(plate_rgb, os.path.basename(img_path))


# ======================================================================
# Standalone CLI
# ======================================================================

def run_pretest_report(image_path: Optional[str] = None, grid: Optional[Dict] = None):
    """Run a pretest on a plate image, printing a compact grid table."""
    g = grid or DEFAULT_GRID

    if image_path is None:
        print("No image provided — generating synthetic plate...\n")
        image, expected = generate_full_plate(g)
        os.makedirs(SYNTHETIC_DIR, exist_ok=True)
        image_path = os.path.join(SYNTHETIC_DIR, "pretest_plate.png")
        cv2.imwrite(image_path, image)
    else:
        expected = None

    plate_rgb = extract_plate_rgb(image_path, grid=g)

    # Color label grid
    print_grid_report(plate_rgb, f"{os.path.basename(image_path)}  [color labels]", expected)

    # RGB value grid
    print_grid_report(plate_rgb, f"{os.path.basename(image_path)}  [RGB values]", mode="rgb")

    return plate_rgb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Color extraction pretest")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to a plate image.")
    parser.add_argument("--image-dir", "--image_dir", type=str, default=None,
                        help="Directory to scan for plate images (png/jpg).")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Generate all synthetic images to data/synthetic/.")
    args = parser.parse_args()

    if args.generate:
        paths = save_all_synthetic()
        print(f"Generated {len(paths)} synthetic images in {SYNTHETIC_DIR}/")
        for p in paths:
            print(f"  {os.path.basename(p)}")

    if args.image_dir:
        images = find_plate_images([args.image_dir])
        if not images:
            print(f"No images found in {args.image_dir}")
        for img in images:
            run_pretest_report(img)
    elif args.image:
        run_pretest_report(args.image)
    elif not args.generate:
        run_pretest_report()
