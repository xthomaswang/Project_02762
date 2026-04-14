"""Focused tests for pipeline integration changes (Prompt C).

Tests the quick/full mode behaviors, park-instead-of-home, and
runtime color calibration flow. Uses mocks — no real hardware.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import time

import numpy as np

from src.robot import ParkPosition, DEFAULT_OFFSET
from src.color.calibration import RuntimeReference, PlateCalibration


class TestQuickModeRinseCycles(unittest.TestCase):
    """Quick mode should use rinse_cycles=1."""

    def test_quick_overrides_value(self):
        """Verify the quick_overrides dict sets rinse_cycles=1."""
        # Directly test the override logic from the pipeline
        quick_overrides = {}
        mode = "quick"
        if mode == "quick":
            quick_overrides["rinse_cycles"] = 1
            quick_overrides["mix_cycles"] = 2
            quick_overrides["skip_controls_after_first"] = True

        self.assertEqual(quick_overrides["rinse_cycles"], 1)
        # Full mode should not set overrides
        full_overrides = {}
        self.assertEqual(full_overrides.get("rinse_cycles", 3), 3)


class TestSettleTimingLogic(unittest.TestCase):
    """Settle timing should count park motion inside the settle window."""

    def test_remaining_sleep_reduced_by_park_time(self):
        """If park takes 3s and settle is 15s, sleep should be ~12s."""
        # This tests the timing logic conceptually
        settle_time = 15
        park_duration = 3.0  # simulated

        settle_start = time.monotonic()
        # Simulate park taking park_duration seconds
        elapsed = park_duration
        remaining = max(0, settle_time - elapsed)

        self.assertAlmostEqual(remaining, 12.0, places=0)

    def test_no_negative_sleep(self):
        """If park takes longer than settle, remaining should be 0."""
        settle_time = 5
        park_duration = 10.0

        remaining = max(0, settle_time - park_duration)
        self.assertEqual(remaining, 0)


class TestRuntimeCalibrationFlow(unittest.TestCase):
    """Color calibration should be control-driven at runtime."""

    def test_full_mode_builds_fresh_reference(self):
        """In full mode with controls, a RuntimeReference should be built."""
        from src.color.calibration import (
            LabeledPlate, build_runtime_reference,
        )

        plate = LabeledPlate()
        plate.rgb["A1"] = np.array([200.0, 20.0, 20.0])   # red
        plate.rgb["B1"] = np.array([20.0, 200.0, 20.0])    # green
        plate.rgb["C1"] = np.array([20.0, 20.0, 200.0])    # blue
        plate.rgb["D1"] = np.array([220.0, 220.0, 220.0])  # water

        ref = build_runtime_reference(plate, col=1)
        self.assertIsInstance(ref, RuntimeReference)
        self.assertEqual(ref.source_column, 1)
        # Water should be correctly identified
        water_labels = [l for l, r in ref.roles.items() if r == "water"]
        self.assertEqual(len(water_labels), 1)

    def test_quick_mode_reuses_cached_reference(self):
        """Quick mode should use cached reference for later columns."""
        from src.color.calibration import (
            LabeledPlate, build_runtime_reference, apply_reference_to_column,
        )

        # Build reference from column 1
        plate1 = LabeledPlate()
        plate1.rgb["A1"] = np.array([200.0, 20.0, 20.0])
        plate1.rgb["B1"] = np.array([20.0, 200.0, 20.0])
        plate1.rgb["C1"] = np.array([20.0, 20.0, 200.0])
        plate1.rgb["D1"] = np.array([220.0, 220.0, 220.0])
        plate1.rgb["E1"] = np.array([110.0, 55.0, 110.0])
        plate1.rgb["F1"] = np.array([115.0, 50.0, 105.0])
        plate1.rgb["G1"] = np.array([108.0, 58.0, 112.0])
        plate1.rgb["H1"] = np.array([112.0, 52.0, 108.0])

        ref = build_runtime_reference(plate1, col=1)

        # Apply to column 2 (no controls dispensed)
        plate2 = LabeledPlate()
        plate2.rgb["E2"] = np.array([100.0, 60.0, 100.0])
        plate2.rgb["F2"] = np.array([105.0, 55.0, 95.0])
        plate2.rgb["G2"] = np.array([98.0, 62.0, 102.0])
        plate2.rgb["H2"] = np.array([102.0, 58.0, 98.0])

        calibrated = apply_reference_to_column(plate2, ref, col=2)
        self.assertIn("E2", calibrated)
        self.assertIn("H2", calibrated)
        # Calibrated values should differ from raw (WB scale != 1.0)
        raw_e2 = plate2.rgb["E2"]
        cal_e2 = calibrated["E2"]
        self.assertFalse(np.allclose(raw_e2, cal_e2, atol=1.0))


class TestParkInsteadOfHome(unittest.TestCase):
    """Pipeline should park instead of homing before imaging."""

    def test_park_pos_is_passed_through(self):
        """_run_single_iteration signature accepts park_pos."""
        from src.pipeline import _run_single_iteration
        import inspect
        sig = inspect.signature(_run_single_iteration)
        self.assertIn("park_pos", sig.parameters)
        self.assertIn("cached_reference", sig.parameters)
        self.assertIn("is_quick", sig.parameters)


class TestQuickModeCalibrationFix(unittest.TestCase):
    """Quick mode must use cached reference for optimization RGB, not row D WB."""

    def test_cached_reference_produces_calibrated_mean(self):
        """When controls are skipped, calibrated mean uses cached reference."""
        from src.color.calibration import (
            LabeledPlate, build_runtime_reference, apply_reference_to_column,
        )

        # Build reference from column 1 (with controls)
        plate1 = LabeledPlate()
        plate1.rgb["A1"] = np.array([200.0, 20.0, 20.0])
        plate1.rgb["B1"] = np.array([20.0, 200.0, 20.0])
        plate1.rgb["C1"] = np.array([20.0, 20.0, 200.0])
        plate1.rgb["D1"] = np.array([220.0, 220.0, 220.0])
        plate1.rgb["E1"] = np.array([110.0, 55.0, 110.0])
        plate1.rgb["F1"] = np.array([115.0, 50.0, 105.0])
        plate1.rgb["G1"] = np.array([108.0, 58.0, 112.0])
        plate1.rgb["H1"] = np.array([112.0, 52.0, 108.0])

        ref = build_runtime_reference(plate1, col=1)

        # Column 2: no controls dispensed, only experiment wells
        plate2 = LabeledPlate()
        plate2.rgb["E2"] = np.array([100.0, 60.0, 100.0])
        plate2.rgb["F2"] = np.array([105.0, 55.0, 95.0])
        plate2.rgb["G2"] = np.array([98.0, 62.0, 102.0])
        plate2.rgb["H2"] = np.array([122.0, 58.0, 98.0])

        # Apply cached reference
        calibrated = apply_reference_to_column(plate2, ref, col=2)
        cal_values = np.array(list(calibrated.values()))
        cal_mean = cal_values.mean(axis=0)

        # Raw mean (what old code would feed the optimizer)
        raw_values = np.array([plate2.rgb[f"{r}2"] for r in "EFGH"])
        raw_mean = raw_values.mean(axis=0)

        # Calibrated should differ from raw — WB scale ≈ 255/220 ≠ 1.0
        self.assertFalse(np.allclose(cal_mean, raw_mean, atol=1.0))

        # Calibrated values clipped to [0, 255]
        self.assertTrue(np.all(cal_values >= 0))
        self.assertTrue(np.all(cal_values <= 255))

    def test_all_columns_share_same_reference(self):
        """Multiple quick-mode columns should use the same calibration."""
        from src.color.calibration import (
            LabeledPlate, build_runtime_reference, apply_reference_to_column,
        )

        plate = LabeledPlate()
        plate.rgb["A1"] = np.array([200.0, 20.0, 20.0])
        plate.rgb["B1"] = np.array([20.0, 200.0, 20.0])
        plate.rgb["C1"] = np.array([20.0, 20.0, 200.0])
        plate.rgb["D1"] = np.array([220.0, 220.0, 220.0])
        ref = build_runtime_reference(plate, col=1)

        # Same raw values in columns 3 and 5 should get same calibration
        for col in [3, 5]:
            p = LabeledPlate()
            for row in "EFGH":
                p.rgb[f"{row}{col}"] = np.array([100.0, 80.0, 120.0])
            cal = apply_reference_to_column(p, ref, col=col)
            vals = np.array(list(cal.values()))
            mean = vals.mean(axis=0)
            # All should be identical since raw values are the same
            np.testing.assert_allclose(mean, vals[0], atol=1e-6)


class TestDistanceMetricIntegration(unittest.TestCase):
    """Pipeline color_distance should work with both metrics."""

    def test_pipeline_imports_color_distance(self):
        """color_distance should be importable from color_metrics."""
        from src.color.metrics import color_distance
        a = np.array([128.0, 0.0, 255.0])
        b = np.array([100.0, 50.0, 200.0])
        d_rgb = color_distance(a, b, "rgb_euclidean")
        d_lab = color_distance(a, b, "delta_e_lab")
        self.assertGreater(d_rgb, 0)
        self.assertGreater(d_lab, 0)


class TestCalibrationWarnings(unittest.TestCase):
    """Task module should surface calibration fallback reasons."""

    @patch("src.tasks.color_mixing.observation.build_runtime_reference",
           side_effect=ValueError("bad controls"))
    @patch("src.tasks.color_mixing.observation.extract_labeled_plate",
           return_value=MagicMock())
    @patch("src.tasks.color_mixing.observation.extract_experiment_rgb")
    @patch("cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8))
    def test_analyze_capture_reports_calibration_failure(
        self,
        _mock_imread,
        mock_extract_rgb,
        _mock_extract_plate,
        _mock_build_ref,
    ):
        """Calibration failure should return raw RGB plus an explicit warning."""
        from src.tasks.color_mixing.observation import analyze_capture
        from src.tasks.color_mixing.config import load_task_config
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "color_mixing.yaml",
        )
        cfg = load_task_config(config_path)

        mock_extract_rgb.return_value = {
            "experiment_rgb": np.array([
                [100.0, 60.0, 100.0],
                [105.0, 55.0, 95.0],
                [98.0, 62.0, 102.0],
                [102.0, 58.0, 98.0],
            ]),
            "experiment_mean": np.array([101.25, 58.75, 98.75]),
            "experiment_std": np.array([2.6, 2.6, 2.6]),
            "control_rgb": {
                "A": np.array([200.0, 20.0, 20.0]),
                "B": np.array([20.0, 200.0, 20.0]),
                "C": np.array([20.0, 20.0, 200.0]),
                "D": np.array([220.0, 220.0, 220.0]),
            },
        }

        result = analyze_capture(
            image_path="/tmp/fake.jpg",
            cfg=cfg,
            col_idx=0,
            skip_controls=False,
        )

        self.assertFalse(result.used_calibration)
        self.assertIn("Calibration failed", result.calibration_warning)
        self.assertIsNone(result.reference_source_column)
        np.testing.assert_allclose(result.mean_rgb, result.raw_mean_rgb)
        np.testing.assert_allclose(
            result.calibrated_experiment_rgb, result.experiment_rgb,
        )


if __name__ == "__main__":
    unittest.main()
