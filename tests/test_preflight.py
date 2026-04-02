"""Tests for src.preflight — path resolution and config wiring."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yaml

from src.preflight import (
    load_config,
    _resolve_config_path,
    run_device_precheck_from_config,
    load_or_create_grid_calibration,
)
from src.vision.geometry import PlateGrid, PlateGrid4, save_grid_calibration, compute_grid_from_corners


class TestLoadConfig(unittest.TestCase):
    def test_stores_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "experiment.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump({"target_color": [128, 0, 255]}, f)

            config = load_config(cfg_path)
            self.assertEqual(config["_config_dir"], tmpdir)
            self.assertEqual(config["target_color"], [128, 0, 255])

    def test_config_dir_is_absolute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "sub", "experiment.yaml")
            os.makedirs(os.path.dirname(cfg_path))
            with open(cfg_path, "w") as f:
                yaml.dump({"seed": 42}, f)

            config = load_config(cfg_path)
            self.assertTrue(os.path.isabs(config["_config_dir"]))


class TestResolveConfigPath(unittest.TestCase):
    def test_relative_path_resolved_to_config_dir(self):
        config = {"_config_dir": "/project/configs"}
        result = _resolve_config_path(config, "calibrations/grid.json")
        self.assertEqual(result, "/project/configs/calibrations/grid.json")

    def test_absolute_path_unchanged(self):
        config = {"_config_dir": "/project/configs"}
        result = _resolve_config_path(config, "/absolute/path/grid.json")
        self.assertEqual(result, "/absolute/path/grid.json")

    def test_missing_config_dir_falls_back_to_cwd(self):
        config = {}
        result = _resolve_config_path(config, "grid.json")
        self.assertEqual(result, os.path.join(os.getcwd(), "grid.json"))


class TestGridPathResolution(unittest.TestCase):
    """Grid calibration path is resolved relative to config file, not cwd."""

    def test_load_grid_relative_to_config(self):
        """A grid.json next to the config file is found regardless of cwd."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create calibration file relative to config dir
            cal_dir = os.path.join(tmpdir, "calibrations")
            os.makedirs(cal_dir)
            grid = compute_grid_from_corners(
                a1=(380, 240), a12=(1550, 240), h1=(375, 925),
            )
            grid_path = os.path.join(cal_dir, "grid.json")
            save_grid_calibration(grid_path, grid)

            # Config references a relative path
            config = {
                "_config_dir": tmpdir,
                "calibration": {"grid_path": "calibrations/grid.json"},
            }

            loaded = load_or_create_grid_calibration(config)
            self.assertEqual(loaded, grid)

    def test_create_grid_saves_relative_to_config(self):
        """When creating a new grid, it saves relative to config dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "_config_dir": tmpdir,
                "calibration": {
                    "grid_path": "calibrations/grid.json",
                    "roi_scale": 0.35,
                },
            }
            corners = {
                "a1": (380, 240),
                "a12": (1550, 240),
                "h1": (375, 925),
            }

            grid = load_or_create_grid_calibration(config, corners=corners)

            expected_path = os.path.join(tmpdir, "calibrations", "grid.json")
            self.assertTrue(os.path.exists(expected_path))
            self.assertIsInstance(grid, PlateGrid)

    def test_4_corners_creates_plate_grid4(self):
        """Passing h12 produces PlateGrid4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "_config_dir": tmpdir,
                "calibration": {
                    "grid_path": "calibrations/grid.json",
                    "roi_scale": 0.35,
                },
            }
            corners = {
                "a1": (380, 240),
                "a12": (1550, 240),
                "h1": (375, 925),
                "h12": (1545, 920),
            }

            grid = load_or_create_grid_calibration(config, corners=corners)
            self.assertIsInstance(grid, PlateGrid4)

            # Reload from saved file — should also be PlateGrid4
            grid2 = load_or_create_grid_calibration(config)
            self.assertIsInstance(grid2, PlateGrid4)
            self.assertEqual(grid, grid2)

    def test_no_calibration_no_corners_no_image_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "_config_dir": tmpdir,
                "calibration": {"grid_path": "calibrations/grid.json"},
            }
            with self.assertRaises(ValueError):
                load_or_create_grid_calibration(config)


class TestForceRecalibrate(unittest.TestCase):
    """force_recalibrate=True skips preset and creates fresh calibration."""

    def _make_config_with_saved_grid(self, tmpdir):
        """Create a config + saved grid file, return (config, saved_grid)."""
        cal_dir = os.path.join(tmpdir, "calibrations")
        os.makedirs(cal_dir)
        old_grid = compute_grid_from_corners(
            a1=(100, 100), a12=(1100, 100), h1=(100, 800),
        )
        grid_path = os.path.join(cal_dir, "grid.json")
        save_grid_calibration(grid_path, old_grid)

        config = {
            "_config_dir": tmpdir,
            "calibration": {
                "grid_path": "calibrations/grid.json",
                "roi_scale": 0.35,
            },
        }
        return config, old_grid

    def test_default_loads_preset(self):
        """force_recalibrate=False (default) loads saved preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config, old_grid = self._make_config_with_saved_grid(tmpdir)
            loaded = load_or_create_grid_calibration(config)
            self.assertEqual(loaded, old_grid)

    def test_force_recalibrate_uses_corners(self):
        """force_recalibrate=True ignores preset and uses new corners."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config, old_grid = self._make_config_with_saved_grid(tmpdir)

            new_corners = {
                "a1": (200, 200),
                "a12": (1200, 200),
                "h1": (200, 900),
                "h12": (1200, 900),
            }
            new_grid = load_or_create_grid_calibration(
                config, corners=new_corners, force_recalibrate=True,
            )

            # Should NOT be the old grid
            self.assertNotEqual(new_grid, old_grid)
            self.assertIsInstance(new_grid, PlateGrid4)

    def test_force_recalibrate_overwrites_saved_file(self):
        """force_recalibrate=True saves the new grid to the preset path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config, old_grid = self._make_config_with_saved_grid(tmpdir)

            new_corners = {
                "a1": (200, 200),
                "a12": (1200, 200),
                "h1": (200, 900),
            }
            new_grid = load_or_create_grid_calibration(
                config, corners=new_corners, force_recalibrate=True,
            )

            # Reload without force — should get the NEW grid
            reloaded = load_or_create_grid_calibration(config)
            self.assertEqual(reloaded, new_grid)
            self.assertNotEqual(reloaded, old_grid)

    def test_force_recalibrate_without_source_raises(self):
        """force_recalibrate=True with no corners or image raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config, _ = self._make_config_with_saved_grid(tmpdir)
            with self.assertRaises(ValueError):
                load_or_create_grid_calibration(
                    config, force_recalibrate=True,
                )


class TestPrecheckConfigWiring(unittest.TestCase):
    """precheck.robot_timeout and precheck.max_camera_id are consumed."""

    @patch("src.preflight.run_device_precheck")
    def test_default_precheck_values(self, mock_precheck):
        mock_precheck.return_value = MagicMock()
        config = {
            "robot": {"ip": "10.0.0.1", "port": 31950},
            "camera": {"device_id": 2},
        }

        run_device_precheck_from_config(config)

        mock_precheck.assert_called_once_with(
            robot_ip="10.0.0.1",
            port=31950,
            timeout=5.0,
            max_camera_id=10,
            expected_camera_id=2,
            preview=False,
        )

    @patch("src.preflight.run_device_precheck")
    def test_precheck_config_overrides(self, mock_precheck):
        mock_precheck.return_value = MagicMock()
        config = {
            "robot": {"ip": "10.0.0.1", "port": 31950},
            "camera": {"device_id": 0},
            "precheck": {
                "robot_timeout": 15.0,
                "max_camera_id": 3,
            },
        }

        run_device_precheck_from_config(config, preview=True)

        mock_precheck.assert_called_once_with(
            robot_ip="10.0.0.1",
            port=31950,
            timeout=15.0,
            max_camera_id=3,
            expected_camera_id=0,
            preview=True,
        )


if __name__ == "__main__":
    unittest.main()
