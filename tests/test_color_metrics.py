"""Tests for CIELAB color distance metrics in src.color_metrics."""

import unittest

import numpy as np

from src.color.metrics import (
    srgb_to_lab,
    color_distance,
    batch_color_distances,
    srgb_to_linear,
    linear_to_srgb,
    predict_mix_rgb,
    compute_reachable_gamut,
)


class TestSrgbToLab(unittest.TestCase):
    """Verify sRGB → CIELAB conversion against known reference values."""

    def test_white(self):
        """White (255,255,255) → Lab ≈ (100, 0, 0)."""
        lab = srgb_to_lab(np.array([255.0, 255.0, 255.0]))
        self.assertAlmostEqual(lab[0], 100.0, delta=0.5)
        self.assertAlmostEqual(lab[1], 0.0, delta=0.5)
        self.assertAlmostEqual(lab[2], 0.0, delta=0.5)

    def test_black(self):
        """Black (0,0,0) → Lab ≈ (0, 0, 0)."""
        lab = srgb_to_lab(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(lab[0], 0.0, delta=0.5)
        self.assertAlmostEqual(lab[1], 0.0, delta=0.5)
        self.assertAlmostEqual(lab[2], 0.0, delta=0.5)

    def test_pure_red(self):
        """Pure red should have positive a*, near-zero b*."""
        lab = srgb_to_lab(np.array([255.0, 0.0, 0.0]))
        self.assertGreater(lab[1], 50)   # a* strongly positive
        self.assertGreater(lab[0], 40)   # L* moderate

    def test_pure_green(self):
        """Pure green should have negative a*."""
        lab = srgb_to_lab(np.array([0.0, 255.0, 0.0]))
        self.assertLess(lab[1], -50)     # a* strongly negative

    def test_batch_shape(self):
        """Batch conversion (n, 3) → (n, 3)."""
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=float)
        lab = srgb_to_lab(rgb)
        self.assertEqual(lab.shape, (3, 3))

    def test_clipping(self):
        """Out-of-range inputs are clipped to [0, 255]."""
        lab = srgb_to_lab(np.array([-10.0, 300.0, 128.0]))
        self.assertTrue(np.all(np.isfinite(lab)))


class TestColorDistance(unittest.TestCase):
    """Test unified color_distance with both metrics."""

    def test_identical_rgb_euclidean(self):
        a = np.array([128.0, 64.0, 200.0])
        self.assertAlmostEqual(color_distance(a, a, "rgb_euclidean"), 0.0)

    def test_identical_delta_e(self):
        a = np.array([128.0, 64.0, 200.0])
        self.assertAlmostEqual(color_distance(a, a, "delta_e_lab"), 0.0, places=3)

    def test_different_colors_delta_e_nonzero(self):
        """Obviously different colors should have large Delta E."""
        red = np.array([255.0, 0.0, 0.0])
        blue = np.array([0.0, 0.0, 255.0])
        de = color_distance(red, blue, "delta_e_lab")
        self.assertGreater(de, 50)

    def test_both_metrics_positive(self):
        a = np.array([100.0, 150.0, 200.0])
        b = np.array([110.0, 140.0, 190.0])
        self.assertGreater(color_distance(a, b, "rgb_euclidean"), 0)
        self.assertGreater(color_distance(a, b, "delta_e_lab"), 0)

    def test_default_is_rgb_euclidean(self):
        a = np.array([100.0, 0.0, 0.0])
        b = np.array([200.0, 0.0, 0.0])
        self.assertAlmostEqual(color_distance(a, b), 100.0)

    def test_unknown_metric_raises(self):
        a = np.array([100.0, 100.0, 100.0])
        with self.assertRaises(ValueError):
            color_distance(a, a, "unknown_metric")


class TestBatchColorDistances(unittest.TestCase):
    """Test vectorised batch distance computation."""

    def test_rgb_euclidean_batch(self):
        batch = np.array([[0, 0, 0], [255, 255, 255]], dtype=float)
        target = np.array([0, 0, 0], dtype=float)
        d = batch_color_distances(batch, target, "rgb_euclidean")
        self.assertEqual(d.shape, (2,))
        self.assertAlmostEqual(d[0], 0.0)
        self.assertAlmostEqual(d[1], np.sqrt(3 * 255**2), places=1)

    def test_delta_e_batch(self):
        batch = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=float)
        target = np.array([128, 128, 128], dtype=float)
        d = batch_color_distances(batch, target, "delta_e_lab")
        self.assertEqual(d.shape, (3,))
        self.assertTrue(np.all(d > 0))

    def test_identical_batch_zero(self):
        target = np.array([100, 200, 50], dtype=float)
        batch = np.tile(target, (5, 1))
        d = batch_color_distances(batch, target, "delta_e_lab")
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_unknown_metric_raises(self):
        batch = np.array([[100, 100, 100]], dtype=float)
        target = np.array([100, 100, 100], dtype=float)
        with self.assertRaises(ValueError):
            batch_color_distances(batch, target, "bad")


class TestLinearRGBConversion(unittest.TestCase):
    """Test srgb_to_linear and linear_to_srgb round-trip."""

    def test_white_roundtrip(self):
        linear = srgb_to_linear(np.array([255, 255, 255]))
        np.testing.assert_allclose(linear, [1, 1, 1], atol=0.01)
        back = linear_to_srgb(linear)
        np.testing.assert_allclose(back, [255, 255, 255], atol=1)

    def test_black_roundtrip(self):
        linear = srgb_to_linear(np.array([0, 0, 0]))
        np.testing.assert_allclose(linear, [0, 0, 0], atol=0.01)
        back = linear_to_srgb(linear)
        np.testing.assert_allclose(back, [0, 0, 0], atol=1)

    def test_mid_gray_roundtrip(self):
        rgb = np.array([128, 128, 128])
        back = linear_to_srgb(srgb_to_linear(rgb))
        np.testing.assert_allclose(back, rgb, atol=1)

    def test_batch_roundtrip(self):
        rgb = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=float)
        back = linear_to_srgb(srgb_to_linear(rgb))
        np.testing.assert_allclose(back, rgb, atol=1)


class TestPredictMixRGB(unittest.TestCase):
    """Test predict_mix_rgb mixing model."""

    def test_pure_water(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        # fractions = [0, 0, 0] → 100% water
        result = predict_mix_rgb(pure, np.array([0, 0, 0]), np.array([240, 240, 240]))
        np.testing.assert_allclose(result, [240, 240, 240], atol=2)

    def test_pure_red(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = predict_mix_rgb(pure, np.array([1, 0, 0]))
        np.testing.assert_allclose(result, [200, 30, 30], atol=5)

    def test_equal_mix(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = predict_mix_rgb(pure, np.array([0.33, 0.33, 0.34]))
        # Should be roughly grayish
        self.assertTrue(np.std(result) < 50)


class TestComputeReachableGamut(unittest.TestCase):
    """Test compute_reachable_gamut."""

    def test_returns_required_keys(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = compute_reachable_gamut(pure, n_samples=100)
        self.assertIn("samples_rgb", result)
        self.assertIn("samples_lab", result)
        self.assertIn("suggested_targets", result)
        self.assertIn("pure_rgbs", result)

    def test_samples_count(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = compute_reachable_gamut(pure, n_samples=500)
        self.assertEqual(len(result["samples_rgb"]), 500)

    def test_suggested_targets_exist(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = compute_reachable_gamut(pure, n_samples=1000)
        self.assertGreater(len(result["suggested_targets"]), 0)
        for t in result["suggested_targets"]:
            self.assertIn("rgb", t)
            self.assertEqual(len(t["rgb"]), 3)

    def test_samples_in_valid_range(self):
        pure = {"red": np.array([200, 30, 30]), "green": np.array([30, 200, 30]), "blue": np.array([30, 30, 200])}
        result = compute_reachable_gamut(pure, n_samples=200)
        rgb = np.array(result["samples_rgb"])
        self.assertTrue(np.all(rgb >= 0))
        self.assertTrue(np.all(rgb <= 255))


if __name__ == "__main__":
    unittest.main()
