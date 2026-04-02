"""Tests for src.plate_geometry — no hardware needed."""

import json
import os
import tempfile
import unittest

import numpy as np

from src.vision.geometry import (
    PlateGrid,
    PlateGrid4,
    compute_grid_from_corners,
    grid_to_dict,
    grid_from_dict,
    save_grid_calibration,
    load_grid_calibration,
    get_well_center,
    get_labeled_well_centers,
    render_grid_overlay,
    _bilinear,
)


# ======================================================================
# PlateGrid (axis-aligned, 3-corner legacy)
# ======================================================================

class TestComputeGridFromCorners3(unittest.TestCase):
    """3-corner (legacy) axis-aligned model."""

    def setUp(self):
        self.grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240), h1=(375, 925),
        )

    def test_returns_plate_grid(self):
        self.assertIsInstance(self.grid, PlateGrid)

    def test_spacing_x(self):
        expected = (1550 - 380) / 11.0
        self.assertAlmostEqual(self.grid.spacing_x, expected, places=4)

    def test_spacing_y(self):
        expected = (925 - 240) / 7.0
        self.assertAlmostEqual(self.grid.spacing_y, expected, places=4)

    def test_origin(self):
        self.assertAlmostEqual(self.grid.origin_x, 380)
        self.assertAlmostEqual(self.grid.origin_y, 240)

    def test_roi_radius_positive(self):
        self.assertGreater(self.grid.roi_radius, 0)

    def test_roi_scale_respected(self):
        grid_05 = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240), h1=(375, 925), roi_scale=0.5,
        )
        self.assertGreater(grid_05.roi_radius, self.grid.roi_radius)

    def test_frozen(self):
        with self.assertRaises(AttributeError):
            self.grid.origin_x = 999


# ======================================================================
# PlateGrid4 (4-corner, rotation-robust)
# ======================================================================

class TestComputeGridFromCorners4(unittest.TestCase):
    """4-corner bilinear model."""

    def setUp(self):
        self.grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240),
            h1=(375, 925), h12=(1545, 920),
        )

    def test_returns_plate_grid4(self):
        self.assertIsInstance(self.grid, PlateGrid4)

    def test_corners_stored(self):
        self.assertEqual(self.grid.a1, (380, 240))
        self.assertEqual(self.grid.a12, (1550, 240))
        self.assertEqual(self.grid.h1, (375, 925))
        self.assertEqual(self.grid.h12, (1545, 920))

    def test_roi_radius_positive(self):
        self.assertGreater(self.grid.roi_radius, 0)

    def test_frozen(self):
        with self.assertRaises(AttributeError):
            self.grid.a1 = (0, 0)


class TestBilinearInterpolation(unittest.TestCase):
    """Verify bilinear math at corners and interior."""

    def setUp(self):
        # Slightly rotated plate
        self.a1 = (100.0, 50.0)
        self.a12 = (1200.0, 55.0)
        self.h1 = (95.0, 750.0)
        self.h12 = (1195.0, 755.0)

    def test_corners_exact(self):
        self.assertEqual(_bilinear(*self._corners(), 0, 0), self.a1)
        self.assertEqual(_bilinear(*self._corners(), 1, 0), self.a12)
        self.assertEqual(_bilinear(*self._corners(), 0, 1), self.h1)
        self.assertEqual(_bilinear(*self._corners(), 1, 1), self.h12)

    def test_midpoint(self):
        mx, my = _bilinear(*self._corners(), 0.5, 0.5)
        # Midpoint should be roughly the average of all 4 corners
        avg_x = (self.a1[0] + self.a12[0] + self.h1[0] + self.h12[0]) / 4
        avg_y = (self.a1[1] + self.a12[1] + self.h1[1] + self.h12[1]) / 4
        self.assertAlmostEqual(mx, avg_x, places=1)
        self.assertAlmostEqual(my, avg_y, places=1)

    def _corners(self):
        return self.a1, self.a12, self.h1, self.h12


class TestGrid4WellCenters(unittest.TestCase):
    """Well centre computation with 4-corner model."""

    def setUp(self):
        self.grid = compute_grid_from_corners(
            a1=(100, 50), a12=(1200, 55),
            h1=(95, 750), h12=(1195, 755),
        )

    def test_a1_is_corner(self):
        cx, cy = get_well_center(0, 0, self.grid)
        self.assertEqual(cx, 100)
        self.assertEqual(cy, 50)

    def test_a12_is_corner(self):
        cx, cy = get_well_center(0, 11, self.grid)
        self.assertEqual(cx, 1200)
        self.assertEqual(cy, 55)

    def test_h1_is_corner(self):
        cx, cy = get_well_center(7, 0, self.grid)
        self.assertEqual(cx, 95)
        self.assertEqual(cy, 750)

    def test_h12_is_corner(self):
        cx, cy = get_well_center(7, 11, self.grid)
        self.assertEqual(cx, 1195)
        self.assertEqual(cy, 755)

    def test_labeled_count(self):
        centers = get_labeled_well_centers(self.grid)
        self.assertEqual(len(centers), 96)

    def test_handles_rotation(self):
        """Top-right corner is 5px lower than top-left — y should increase."""
        centers = get_labeled_well_centers(self.grid)
        self.assertGreater(centers["A12"][1], centers["A1"][1])


# ======================================================================
# Dict conversion and backward compatibility
# ======================================================================

class TestDictConversion(unittest.TestCase):
    def test_plate_grid_round_trip(self):
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(1200, 50), h1=(100, 750),
        )
        d = grid_to_dict(grid)
        restored = grid_from_dict(d)
        self.assertEqual(grid, restored)

    def test_plate_grid4_round_trip(self):
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(1200, 55),
            h1=(95, 750), h12=(1195, 755),
        )
        d = grid_to_dict(grid)
        restored = grid_from_dict(d)
        self.assertIsInstance(restored, PlateGrid4)
        self.assertEqual(grid, restored)

    def test_plate_grid4_dict_has_compat_keys(self):
        """PlateGrid4 dict includes origin_x/spacing_x for color_extraction."""
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(1200, 55),
            h1=(95, 750), h12=(1195, 755),
        )
        d = grid_to_dict(grid)
        self.assertIn("origin_x", d)
        self.assertIn("spacing_x", d)
        self.assertIn("spacing_y", d)
        self.assertIn("roi_radius", d)
        self.assertAlmostEqual(d["origin_x"], 100)
        self.assertAlmostEqual(d["spacing_x"], (1200 - 100) / 11.0, places=2)

    def test_old_format_loads_as_plate_grid(self):
        """Dict without corners key loads as PlateGrid."""
        d = {
            "origin_x": 400, "origin_y": 150,
            "spacing_x": 90, "spacing_y": 85,
            "roi_radius": 25,
        }
        grid = grid_from_dict(d)
        self.assertIsInstance(grid, PlateGrid)
        self.assertEqual(grid.origin_x, 400)


# ======================================================================
# JSON persistence
# ======================================================================

class TestJsonPersistence(unittest.TestCase):
    def test_save_and_load_plate_grid(self):
        grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240), h1=(375, 925),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grid.json")
            save_grid_calibration(path, grid)
            loaded = load_grid_calibration(path)
            self.assertEqual(grid, loaded)

    def test_save_and_load_plate_grid4(self):
        grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240),
            h1=(375, 925), h12=(1545, 920),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grid4.json")
            save_grid_calibration(path, grid)
            loaded = load_grid_calibration(path)
            self.assertIsInstance(loaded, PlateGrid4)
            self.assertEqual(grid, loaded)

    def test_json_is_valid(self):
        grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240),
            h1=(375, 925), h12=(1545, 920),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grid.json")
            save_grid_calibration(path, grid)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("corners", data)
            self.assertIn("roi_radius", data)


# ======================================================================
# Well centers (legacy PlateGrid)
# ======================================================================

class TestWellCentersLegacy(unittest.TestCase):
    def setUp(self):
        self.grid = compute_grid_from_corners(
            a1=(380, 240), a12=(1550, 240), h1=(375, 925),
        )

    def test_a1_is_origin(self):
        cx, cy = get_well_center(0, 0, self.grid)
        self.assertEqual(cx, 380)
        self.assertEqual(cy, 240)

    def test_labeled_well_count(self):
        centers = get_labeled_well_centers(self.grid)
        self.assertEqual(len(centers), 96)

    def test_specific_labels(self):
        centers = get_labeled_well_centers(self.grid)
        for label in ("A1", "A12", "H1", "H12"):
            self.assertIn(label, centers)

    def test_a12_x_greater_than_a1(self):
        centers = get_labeled_well_centers(self.grid)
        self.assertGreater(centers["A12"][0], centers["A1"][0])

    def test_h1_y_greater_than_a1(self):
        centers = get_labeled_well_centers(self.grid)
        self.assertGreater(centers["H1"][1], centers["A1"][1])


# ======================================================================
# Overlay rendering
# ======================================================================

class TestRenderGridOverlay(unittest.TestCase):
    def test_overlay_plate_grid(self):
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(650, 50), h1=(100, 400),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            import cv2
            img = np.zeros((500, 800, 3), dtype=np.uint8)
            img[:] = (200, 200, 200)
            src_path = os.path.join(tmpdir, "plate.png")
            cv2.imwrite(src_path, img)

            out_path = os.path.join(tmpdir, "overlay.png")
            result = render_grid_overlay(src_path, grid, output_path=out_path)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))

    def test_overlay_plate_grid4(self):
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(650, 55),
            h1=(95, 400), h12=(645, 405),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            import cv2
            img = np.zeros((500, 800, 3), dtype=np.uint8)
            src_path = os.path.join(tmpdir, "plate.png")
            cv2.imwrite(src_path, img)

            out_path = os.path.join(tmpdir, "overlay4.png")
            result = render_grid_overlay(src_path, grid, output_path=out_path)
            self.assertTrue(os.path.exists(out_path))

    def test_default_output_path(self):
        grid = compute_grid_from_corners(
            a1=(100, 50), a12=(650, 50), h1=(100, 400),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            import cv2
            img = np.zeros((500, 800, 3), dtype=np.uint8)
            src_path = os.path.join(tmpdir, "plate.jpg")
            cv2.imwrite(src_path, img)

            result = render_grid_overlay(src_path, grid)
            self.assertTrue(result.endswith("_grid_overlay.jpg"))
            self.assertTrue(os.path.exists(result))


if __name__ == "__main__":
    unittest.main()
