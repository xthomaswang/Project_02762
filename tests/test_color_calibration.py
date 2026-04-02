"""Tests for src.color_calibration — synthetic data, no hardware."""

import csv
import os
import tempfile
import unittest

import numpy as np

from src.color.calibration import (
    LabeledPlate,
    PlateCalibration,
    ColumnSummary,
    RuntimeReference,
    extract_labeled_plate,
    build_plate_calibration,
    apply_calibration,
    summarize_column,
    write_measurements_csv,
    infer_control_roles,
    build_runtime_reference,
    apply_reference_to_column,
    DEFAULT_CONTROL_ROWS,
    DEFAULT_EXPERIMENT_ROWS,
)


def _make_synthetic_plate() -> LabeledPlate:
    """Build a LabeledPlate with known RGB values for column 1."""
    plate = LabeledPlate()
    # Controls for column 1
    plate.rgb["A1"] = np.array([200.0, 20.0, 20.0])   # red
    plate.rgb["B1"] = np.array([20.0, 200.0, 20.0])    # green
    plate.rgb["C1"] = np.array([20.0, 20.0, 200.0])    # blue
    plate.rgb["D1"] = np.array([220.0, 220.0, 220.0])  # water (white-ish)
    # Experiment replicates
    plate.rgb["E1"] = np.array([110.0, 55.0, 110.0])
    plate.rgb["F1"] = np.array([115.0, 50.0, 105.0])
    plate.rgb["G1"] = np.array([108.0, 58.0, 112.0])
    plate.rgb["H1"] = np.array([112.0, 52.0, 108.0])
    return plate


class TestLabeledPlate(unittest.TestCase):
    def test_wells_for_row(self):
        plate = _make_synthetic_plate()
        row_a = plate.wells_for_row("A")
        self.assertIn("A1", row_a)
        self.assertNotIn("B1", row_a)

    def test_wells_for_col(self):
        plate = _make_synthetic_plate()
        col1 = plate.wells_for_col(1)
        self.assertIn("A1", col1)
        self.assertIn("H1", col1)
        self.assertEqual(len(col1), 8)


class TestExtractLabeledPlate(unittest.TestCase):
    def test_basic_extraction(self):
        """Extract from a synthetic image with known pixel values."""
        # Create a small synthetic BGR image (solid color)
        image = np.full((500, 800, 3), (100, 150, 200), dtype=np.uint8)  # BGR
        grid = {
            "origin_x": 50, "origin_y": 50,
            "spacing_x": 60, "spacing_y": 50,
            "roi_radius": 15,
        }
        plate = extract_labeled_plate(image, grid=grid, rows=["A"], cols=[1])
        self.assertIn("A1", plate.rgb)
        rgb = plate.rgb["A1"]
        # BGR (100,150,200) -> RGB (200,150,100)
        self.assertAlmostEqual(rgb[0], 200, delta=5)
        self.assertAlmostEqual(rgb[1], 150, delta=5)
        self.assertAlmostEqual(rgb[2], 100, delta=5)


class TestBuildPlateCalibration(unittest.TestCase):
    def test_white_balance_scale(self):
        plate = _make_synthetic_plate()
        cal = build_plate_calibration(plate, col=1)
        # Water is (220,220,220), so scale ≈ 255/220 ≈ 1.159
        expected = 255.0 / 220.0
        np.testing.assert_allclose(cal.white_scale, [expected] * 3, atol=0.01)

    def test_channel_scaling(self):
        plate = _make_synthetic_plate()
        cal = build_plate_calibration(plate, col=1, use_channel_scaling=True)
        self.assertIsNotNone(cal.channel_scales)
        # Red control is (200,20,20), after WB red channel = 200*255/220 ≈ 231.8
        # So channel_scales[0] ≈ 255/231.8 ≈ 1.1
        self.assertGreater(cal.channel_scales[0], 1.0)

    def test_no_water_raises(self):
        plate = LabeledPlate(rgb={"A1": np.array([200, 0, 0])})
        with self.assertRaises(KeyError):
            build_plate_calibration(plate, col=1)


class TestApplyCalibration(unittest.TestCase):
    def test_applies_white_balance(self):
        plate = _make_synthetic_plate()
        cal = PlateCalibration(white_scale=np.array([1.0, 2.0, 1.0]))
        result = apply_calibration(plate, cal, col=1)

        self.assertIn("E1", result)
        # Green channel should double
        raw_g = plate.rgb["E1"][1]
        np.testing.assert_allclose(result["E1"][1], min(raw_g * 2.0, 255))

    def test_clips_to_255(self):
        plate = _make_synthetic_plate()
        cal = PlateCalibration(white_scale=np.array([5.0, 5.0, 5.0]))
        result = apply_calibration(plate, cal, col=1)
        for label, rgb in result.items():
            self.assertTrue(np.all(rgb <= 255))
            self.assertTrue(np.all(rgb >= 0))


class TestSummarizeColumn(unittest.TestCase):
    def test_basic_summary(self):
        plate = _make_synthetic_plate()
        summary = summarize_column(plate, col=1)
        self.assertEqual(summary.col, 1)
        self.assertEqual(len(summary.experiment_rgb), 4)
        self.assertEqual(len(summary.control_rgb), 4)
        self.assertEqual(summary.mean_rgb.shape, (3,))
        self.assertEqual(summary.std_rgb.shape, (3,))

    def test_with_calibration(self):
        plate = _make_synthetic_plate()
        cal = build_plate_calibration(plate, col=1)
        summary = summarize_column(plate, col=1, calibration=cal)
        # Calibrated values should differ from raw
        raw_summary = summarize_column(plate, col=1)
        self.assertFalse(np.allclose(summary.mean_rgb, raw_summary.mean_rgb))

    def test_control_labels(self):
        plate = _make_synthetic_plate()
        summary = summarize_column(plate, col=1)
        self.assertIn("A1", summary.control_rgb)
        self.assertIn("D1", summary.control_rgb)

    def test_experiment_labels(self):
        plate = _make_synthetic_plate()
        summary = summarize_column(plate, col=1)
        self.assertIn("E1", summary.experiment_rgb)
        self.assertIn("H1", summary.experiment_rgb)


class TestWriteMeasurementsCsv(unittest.TestCase):
    def test_writes_valid_csv(self):
        plate = _make_synthetic_plate()
        summary = summarize_column(plate, col=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "measurements.csv")
            result = write_measurements_csv(path, [summary])

            self.assertEqual(result, path)
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 4 controls + 4 experiments = 8 rows
            self.assertEqual(len(rows), 8)
            labels = [r["well"] for r in rows]
            self.assertIn("A1", labels)
            self.assertIn("H1", labels)

            # Check column values
            for r in rows:
                self.assertIn(r["row_type"], ("control", "experiment"))
                self.assertEqual(r["column"], "1")

    def test_multiple_summaries(self):
        plate = _make_synthetic_plate()
        # Add column 2 data
        for label in list(plate.rgb.keys()):
            row = label[0]
            plate.rgb[f"{row}2"] = plate.rgb[label] + np.array([5, 5, 5])

        s1 = summarize_column(plate, col=1)
        s2 = summarize_column(plate, col=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi.csv")
            write_measurements_csv(path, [s1, s2])

            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)  # 8 per column


# ==================================================================
# Runtime control-driven calibration tests
# ==================================================================

def _make_swapped_plate() -> LabeledPlate:
    """Plate where green and blue controls are swapped vs default layout.

    Default assumes A=red, B=green, C=blue, D=water.
    Here B is blue and C is green — physical swap.
    """
    plate = LabeledPlate()
    plate.rgb["A1"] = np.array([200.0, 20.0, 20.0])   # red (same)
    plate.rgb["B1"] = np.array([20.0, 20.0, 200.0])    # blue (swapped!)
    plate.rgb["C1"] = np.array([20.0, 200.0, 20.0])    # green (swapped!)
    plate.rgb["D1"] = np.array([220.0, 220.0, 220.0])  # water (same)
    plate.rgb["E1"] = np.array([110.0, 55.0, 110.0])
    plate.rgb["F1"] = np.array([115.0, 50.0, 105.0])
    plate.rgb["G1"] = np.array([108.0, 58.0, 112.0])
    plate.rgb["H1"] = np.array([112.0, 52.0, 108.0])
    return plate


class TestInferControlRoles(unittest.TestCase):
    """Tests for infer_control_roles — runtime role assignment."""

    def test_standard_layout(self):
        """Normal A=red, B=green, C=blue, D=water is inferred correctly."""
        wells = {
            "A1": np.array([200.0, 20.0, 20.0]),
            "B1": np.array([20.0, 200.0, 20.0]),
            "C1": np.array([20.0, 20.0, 200.0]),
            "D1": np.array([220.0, 220.0, 220.0]),
        }
        roles = infer_control_roles(wells)
        self.assertEqual(roles["A1"], "red")
        self.assertEqual(roles["B1"], "green")
        self.assertEqual(roles["C1"], "blue")
        self.assertEqual(roles["D1"], "water")

    def test_swapped_green_blue(self):
        """Green and blue physically swapped — inference still correct."""
        wells = {
            "A1": np.array([200.0, 20.0, 20.0]),    # red
            "B1": np.array([20.0, 20.0, 200.0]),     # blue (in green's row)
            "C1": np.array([20.0, 200.0, 20.0]),     # green (in blue's row)
            "D1": np.array([220.0, 220.0, 220.0]),   # water
        }
        roles = infer_control_roles(wells)
        self.assertEqual(roles["A1"], "red")
        self.assertEqual(roles["B1"], "blue")
        self.assertEqual(roles["C1"], "green")
        self.assertEqual(roles["D1"], "water")

    def test_water_identified_as_brightest_neutral(self):
        """Water well is the brightest and most neutral."""
        wells = {
            "X": np.array([180.0, 30.0, 30.0]),      # red, bright but not neutral
            "Y": np.array([30.0, 180.0, 30.0]),       # green
            "Z": np.array([30.0, 30.0, 180.0]),       # blue
            "W": np.array([200.0, 200.0, 200.0]),     # water
        }
        roles = infer_control_roles(wells)
        self.assertEqual(roles["W"], "water")

    def test_all_roles_assigned(self):
        """All four roles are present in the output."""
        wells = {
            "A1": np.array([200.0, 20.0, 20.0]),
            "B1": np.array([20.0, 200.0, 20.0]),
            "C1": np.array([20.0, 20.0, 200.0]),
            "D1": np.array([220.0, 220.0, 220.0]),
        }
        roles = infer_control_roles(wells)
        role_set = set(roles.values())
        self.assertEqual(role_set, {"water", "red", "green", "blue"})

    def test_too_few_wells_raises(self):
        """Fewer than 4 wells raises ValueError."""
        wells = {
            "A1": np.array([200.0, 20.0, 20.0]),
            "B1": np.array([20.0, 200.0, 20.0]),
        }
        with self.assertRaises(ValueError):
            infer_control_roles(wells)

    def test_completely_shuffled_order(self):
        """All four wells in a non-standard order are still classified."""
        wells = {
            "D1": np.array([20.0, 200.0, 20.0]),    # green (in D!)
            "C1": np.array([230.0, 230.0, 230.0]),   # water (in C!)
            "B1": np.array([200.0, 20.0, 20.0]),     # red (in B!)
            "A1": np.array([20.0, 20.0, 200.0]),     # blue (in A!)
        }
        roles = infer_control_roles(wells)
        self.assertEqual(roles["D1"], "green")
        self.assertEqual(roles["C1"], "water")
        self.assertEqual(roles["B1"], "red")
        self.assertEqual(roles["A1"], "blue")


class TestBuildRuntimeReference(unittest.TestCase):
    """Tests for build_runtime_reference."""

    def test_basic_reference(self):
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        self.assertIsInstance(ref, RuntimeReference)
        self.assertEqual(ref.source_column, 1)
        self.assertEqual(set(ref.roles.values()), {"water", "red", "green", "blue"})

    def test_water_rgb_stored(self):
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        np.testing.assert_array_equal(ref.water_rgb, np.array([220.0, 220.0, 220.0]))

    def test_white_scale_matches_legacy(self):
        """Runtime reference produces same white_scale as legacy for standard layout."""
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        legacy_cal = build_plate_calibration(plate, col=1)
        np.testing.assert_allclose(
            ref.calibration.white_scale,
            legacy_cal.white_scale,
            atol=0.01,
        )

    def test_with_channel_scaling(self):
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1, use_channel_scaling=True)
        self.assertIsNotNone(ref.calibration.channel_scales)

    def test_swapped_layout_builds_correctly(self):
        """Reference from a swapped plate still builds valid calibration."""
        plate = _make_swapped_plate()
        ref = build_runtime_reference(plate, col=1)
        # Water should still be D1 (brightest + neutral)
        self.assertEqual(ref.roles["D1"], "water")
        # Roles should still cover all four
        self.assertEqual(set(ref.roles.values()), {"water", "red", "green", "blue"})

    def test_missing_well_raises(self):
        plate = LabeledPlate(rgb={"A1": np.array([200.0, 0.0, 0.0])})
        with self.assertRaises(KeyError):
            build_runtime_reference(plate, col=1)

    def test_custom_control_labels(self):
        """Can specify explicit control labels instead of using defaults."""
        plate = LabeledPlate()
        plate.rgb["X1"] = np.array([200.0, 20.0, 20.0])
        plate.rgb["Y1"] = np.array([20.0, 200.0, 20.0])
        plate.rgb["Z1"] = np.array([20.0, 20.0, 200.0])
        plate.rgb["W1"] = np.array([220.0, 220.0, 220.0])
        plate.rgb["E1"] = np.array([100.0, 100.0, 100.0])

        ref = build_runtime_reference(
            plate, col=1,
            control_labels=["X1", "Y1", "Z1", "W1"],
        )
        self.assertEqual(ref.roles["W1"], "water")
        self.assertEqual(set(ref.roles.values()), {"water", "red", "green", "blue"})


class TestApplyReferenceToColumn(unittest.TestCase):
    """Tests for apply_reference_to_column."""

    def test_basic_apply(self):
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        result = apply_reference_to_column(plate, ref, col=1)
        self.assertIn("E1", result)
        self.assertIn("H1", result)
        self.assertEqual(len(result), 4)

    def test_results_match_legacy(self):
        """Runtime path produces same calibrated values as legacy for standard layout."""
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        runtime_result = apply_reference_to_column(plate, ref, col=1)

        legacy_cal = build_plate_calibration(plate, col=1)
        legacy_result = apply_calibration(plate, legacy_cal, col=1)

        for label in legacy_result:
            np.testing.assert_allclose(
                runtime_result[label], legacy_result[label], atol=0.01,
            )

    def test_reference_reuse_across_columns(self):
        """Reference built from column 1 can be applied to column 2."""
        plate = _make_synthetic_plate()
        # Add column 2 experiment data with slightly different values
        plate.rgb["E2"] = np.array([120.0, 60.0, 100.0])
        plate.rgb["F2"] = np.array([125.0, 55.0, 95.0])
        plate.rgb["G2"] = np.array([118.0, 62.0, 102.0])
        plate.rgb["H2"] = np.array([122.0, 58.0, 98.0])

        ref = build_runtime_reference(plate, col=1)
        # Apply col-1 reference to col-2 experiments
        result = apply_reference_to_column(plate, ref, col=2)

        self.assertIn("E2", result)
        self.assertIn("H2", result)
        self.assertEqual(len(result), 4)
        # Values should be calibrated (different from raw)
        self.assertFalse(np.allclose(result["E2"], plate.rgb["E2"]))

    def test_clipping(self):
        """Calibrated values are clipped to [0, 255]."""
        plate = _make_synthetic_plate()
        ref = build_runtime_reference(plate, col=1)
        result = apply_reference_to_column(plate, ref, col=1)
        for rgb in result.values():
            self.assertTrue(np.all(rgb >= 0))
            self.assertTrue(np.all(rgb <= 255))


class TestLegacyApiStillWorks(unittest.TestCase):
    """Ensure the fixed-layout API is not broken by the new code."""

    def test_build_plate_calibration_default(self):
        """Legacy build_plate_calibration with default rows still works."""
        plate = _make_synthetic_plate()
        cal = build_plate_calibration(plate, col=1)
        expected = 255.0 / 220.0
        np.testing.assert_allclose(cal.white_scale, [expected] * 3, atol=0.01)

    def test_apply_calibration_default(self):
        """Legacy apply_calibration with default experiment rows still works."""
        plate = _make_synthetic_plate()
        cal = build_plate_calibration(plate, col=1)
        result = apply_calibration(plate, cal, col=1)
        self.assertEqual(len(result), 4)
        self.assertIn("E1", result)

    def test_summarize_column_default(self):
        """Legacy summarize_column with default rows still works."""
        plate = _make_synthetic_plate()
        summary = summarize_column(plate, col=1)
        self.assertEqual(summary.col, 1)
        self.assertEqual(len(summary.control_rgb), 4)
        self.assertEqual(len(summary.experiment_rgb), 4)

    def test_explicit_control_rows(self):
        """Passing explicit control_rows to legacy API still works."""
        plate = _make_synthetic_plate()
        custom = {"A": "red", "B": "green", "C": "blue", "D": "water"}
        cal = build_plate_calibration(plate, col=1, control_rows=custom)
        result = apply_calibration(plate, cal, col=1)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
