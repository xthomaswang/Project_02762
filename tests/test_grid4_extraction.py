"""Tests proving PlateGrid4 bilinear geometry is used in real extraction.

A PlateGrid4 with slight rotation should produce different well centers
than the axis-aligned approximation, and those differences should
flow through to extract_well_rgb and extract_experiment_rgb.
"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from src.vision.geometry import (
    PlateGrid4,
    compute_grid_from_corners,
    get_well_center as pg_get_well_center,
    grid_to_dict,
)
from src.vision.extraction import (
    get_well_center,
    extract_well_rgb,
    extract_experiment_rgb,
    extract_plate_rgb,
    _resolve_grid,
)


# A plate with ~10px of rotation: top-right is 10px lower, bottom-left
# is 10px further left.  This makes axis-aligned spacing incorrect for
# interior wells.
ROTATED_A1 = (200, 100)
ROTATED_A12 = (1300, 110)  # 10px lower than A1
ROTATED_H1 = (190, 800)    # 10px left of A1
ROTATED_H12 = (1290, 810)  # consistent with parallelogram


def _make_grid4():
    return compute_grid_from_corners(
        a1=ROTATED_A1, a12=ROTATED_A12,
        h1=ROTATED_H1, h12=ROTATED_H12,
    )


def _make_axis_aligned_dict():
    """A plain axis-aligned dict (no corners key) — the lossy approximation."""
    d = grid_to_dict(_make_grid4())
    # Strip corners so _resolve_grid treats this as a plain dict,
    # exercising the axis-aligned path.
    return {k: v for k, v in d.items() if k != "corners"}


class TestGrid4VsAxisAlignedCenters(unittest.TestCase):
    """PlateGrid4 produces different centers from the axis-aligned approx."""

    def test_corner_wells_match(self):
        """Corners should be identical between both models."""
        g4 = _make_grid4()
        d = _make_axis_aligned_dict()

        # A1 (row=0, col=0)
        self.assertEqual(get_well_center(0, 0, g4), get_well_center(0, 0, d))

    def test_interior_well_differs(self):
        """An interior well (D6 = row 3, col 5) should differ due to rotation."""
        g4 = _make_grid4()
        d = _make_axis_aligned_dict()

        cx4, cy4 = get_well_center(3, 5, g4)
        cxd, cyd = get_well_center(3, 5, d)

        # With 10px rotation across the plate, interior wells should differ
        # by at least 1 pixel in at least one axis.
        diff = abs(cx4 - cxd) + abs(cy4 - cyd)
        self.assertGreater(diff, 0,
            f"D6 should differ: grid4=({cx4},{cy4}) vs dict=({cxd},{cyd})")

    def test_h12_corner_exact(self):
        """H12 should match the input corner exactly for PlateGrid4."""
        g4 = _make_grid4()
        cx, cy = get_well_center(7, 11, g4)
        self.assertEqual((cx, cy), ROTATED_H12)

    def test_h12_differs_from_axis_aligned(self):
        """H12 via axis-aligned won't match the true corner."""
        g4 = _make_grid4()
        d = _make_axis_aligned_dict()

        cx4, cy4 = get_well_center(7, 11, g4)
        cxd, cyd = get_well_center(7, 11, d)

        # Axis-aligned H12 = origin + 11*spacing_x, origin + 7*spacing_y
        # which doesn't account for the parallelogram shape.
        self.assertNotEqual((cx4, cy4), (cxd, cyd))


class TestResolveGrid(unittest.TestCase):
    """_resolve_grid upgrades dicts with corners to PlateGrid4."""

    def test_dict_with_corners_becomes_grid4(self):
        d = grid_to_dict(_make_grid4())  # full dict including corners
        resolved = _resolve_grid(d)
        self.assertIsInstance(resolved, PlateGrid4)

    def test_plain_dict_stays_dict(self):
        d = {"origin_x": 100, "origin_y": 50, "spacing_x": 90,
             "spacing_y": 85, "roi_radius": 25}
        resolved = _resolve_grid(d)
        self.assertIsInstance(resolved, dict)

    def test_grid4_passes_through(self):
        g4 = _make_grid4()
        self.assertIs(_resolve_grid(g4), g4)

    def test_none_returns_default(self):
        from src.vision.extraction import DEFAULT_GRID
        self.assertIs(_resolve_grid(None), DEFAULT_GRID)


class TestGrid4Extraction(unittest.TestCase):
    """extract_well_rgb and friends use PlateGrid4 geometry."""

    def _make_synthetic_image(self, grid):
        """Create an image with a bright dot at each PlateGrid4 well center."""
        img = np.zeros((900, 1400, 3), dtype=np.uint8)
        for r in range(8):
            for c in range(12):
                cx, cy = pg_get_well_center(r, c, grid)
                # Draw a small colored circle: color varies by position
                blue = int(50 + 200 * (c / 11.0))
                green = int(50 + 200 * (r / 7.0))
                red = 150
                cv2.circle(img, (cx, cy), 20, (blue, green, red), -1)
        return img

    def test_extract_well_rgb_uses_bilinear_center(self):
        """Extraction at bilinear centers should hit colored pixels."""
        g4 = _make_grid4()
        img = self._make_synthetic_image(g4)

        # Extract from a well that exists at the bilinear center
        rgb = extract_well_rgb(img, row=3, col=5, grid=g4)
        # Should get the colored dot, not black background
        self.assertGreater(rgb.sum(), 100,
            f"Expected colored pixel, got {rgb}")

    def test_extract_well_rgb_axis_aligned_may_miss(self):
        """Axis-aligned centers may partially miss the rotated dots."""
        g4 = _make_grid4()
        d = _make_axis_aligned_dict()
        img = self._make_synthetic_image(g4)

        rgb_g4 = extract_well_rgb(img, row=3, col=5, grid=g4)
        rgb_dict = extract_well_rgb(img, row=3, col=5, grid=d)

        # Both should extract something, but they may differ
        # because the dict centers don't account for the corners field
        # (wait — _resolve_grid upgrades dicts with corners!)
        # So let's use a truly plain dict without corners:
        plain = {k: v for k, v in d.items() if k != "corners"}
        rgb_plain = extract_well_rgb(img, row=3, col=5, grid=plain)

        # PlateGrid4 extraction should differ from plain axis-aligned
        # for interior wells on a rotated plate.
        # At minimum, both should be non-black.
        self.assertGreater(rgb_g4.sum(), 100)
        # The two may or may not differ much depending on dot size vs offset,
        # but the well centers are definitely different:
        cx4, cy4 = get_well_center(3, 5, g4)
        cxp, cyp = get_well_center(3, 5, plain)
        self.assertNotEqual((cx4, cy4), (cxp, cyp))

    def test_extract_experiment_rgb_with_grid4(self):
        """extract_experiment_rgb works end-to-end with PlateGrid4."""
        g4 = _make_grid4()
        img = self._make_synthetic_image(g4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "plate.png")
            cv2.imwrite(path, img)

            result = extract_experiment_rgb(
                path, col=0, grid=g4, apply_white_balance=False,
            )
            self.assertEqual(result["experiment_rgb"].shape, (4, 3))
            self.assertEqual(result["experiment_mean"].shape, (3,))
            # Experiment wells E-H (rows 4-7) should have extracted color
            for i, rgb in enumerate(result["experiment_rgb"]):
                self.assertGreater(rgb.sum(), 50,
                    f"Experiment well {i} should be colored, got {rgb}")

    def test_extract_plate_rgb_with_grid4(self):
        """extract_plate_rgb works with PlateGrid4."""
        g4 = _make_grid4()
        img = self._make_synthetic_image(g4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "plate.png")
            cv2.imwrite(path, img)

            plate = extract_plate_rgb(path, grid=g4)
            self.assertEqual(plate.shape, (8, 12, 3))
            # All wells should have color
            for r in range(8):
                for c in range(12):
                    self.assertGreater(plate[r, c].sum(), 50,
                        f"Well ({r},{c}) should be colored")


if __name__ == "__main__":
    unittest.main()
