"""Notebook-facing helper functions for experiment setup and recovery.

Keeps the notebook thin — most logic lives here or in the modules
this file imports.

Usage (in notebook)::

    from src.preflight import (
        load_config,
        run_device_precheck_from_config,
        load_or_create_grid_calibration,
        capture_seed_column,
    )

    config = load_config("../configs/experiment.yaml")
    report = run_device_precheck_from_config(config)
    grid = load_or_create_grid_calibration(config)
    seed_X, seed_Y = capture_seed_column(config, grid, volumes=[100, 50, 50])
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from openot2 import OT2Client
from openot2.precheck import PrecheckReport, run_device_precheck

from src.vision.geometry import (
    AnyGrid,
    PlateGrid,
    PlateGrid4,
    compute_grid_from_corners,
    load_grid_calibration,
    manual_calibrate_grid_from_image,
    render_grid_overlay,
    save_grid_calibration,
)


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the experiment YAML config.

    Stores ``_config_dir`` (absolute path to the directory containing
    the config file) so that relative paths in the config can be
    resolved unambiguously.
    """
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["_config_dir"] = os.path.dirname(config_path)
    return config


def _resolve_config_path(config: Dict[str, Any], rel_path: str) -> str:
    """Resolve *rel_path* relative to the config file directory.

    If *rel_path* is already absolute it is returned as-is.
    """
    if os.path.isabs(rel_path):
        return rel_path
    config_dir = config.get("_config_dir", os.getcwd())
    return os.path.join(config_dir, rel_path)


# ------------------------------------------------------------------
# Device precheck
# ------------------------------------------------------------------

def run_device_precheck_from_config(
    config: Dict[str, Any],
    preview: bool = False,
) -> PrecheckReport:
    """Run device precheck using robot/camera settings from config.

    Args:
        config: Loaded experiment config dict.
        preview: Show matplotlib camera previews.

    Returns:
        :class:`~openot2.precheck.PrecheckReport`.
    """
    robot_cfg = config.get("robot", {})
    cam_cfg = config.get("camera", {})
    precheck_cfg = config.get("precheck", {})

    return run_device_precheck(
        robot_ip=robot_cfg.get("ip", "169.254.8.56"),
        port=robot_cfg.get("port", 31950),
        timeout=precheck_cfg.get("robot_timeout", 5.0),
        max_camera_id=precheck_cfg.get("max_camera_id", 10),
        expected_camera_id=cam_cfg.get("device_id", 0),
        preview=preview,
    )


# ------------------------------------------------------------------
# Grid calibration
# ------------------------------------------------------------------

def load_or_create_grid_calibration(
    config: Dict[str, Any],
    corners: Optional[Dict[str, Tuple[float, float]]] = None,
    image_path: Optional[str] = None,
    force_recalibrate: bool = False,
) -> AnyGrid:
    """Load an existing grid calibration, or create one.

    Resolution order:

    1. If *force_recalibrate* is ``False`` (default) and a calibration
       JSON exists at ``config["calibration"]["grid_path"]``, load and
       return it.
    2. If *image_path* is provided, open the image and let the operator
       click 4 corner wells interactively (A1, A12, H1, H12).
       Saves the result to the calibration path.
    3. If *corners* dict is provided, compute the grid directly.
       Saves the result to the calibration path.
    4. Otherwise raise ``ValueError``.

    When *force_recalibrate* is ``True``, step 1 is skipped — an
    existing preset is ignored and a new calibration is created from
    *image_path* or *corners*, then saved (overwriting any previous
    file).

    Args:
        config: Experiment config (must include ``_config_dir``).
        corners: Dict with keys ``"a1"``, ``"a12"``, ``"h1"`` and
            optionally ``"h12"`` mapping to ``(x, y)`` pixel positions.
        image_path: Path to a plate image for interactive calibration.
        force_recalibrate: Skip loading the preset and create a fresh
            calibration from *image_path* or *corners*.

    Returns:
        A :class:`~src.plate_geometry.PlateGrid` or
        :class:`~src.plate_geometry.PlateGrid4`.
    """
    cal_cfg = config.get("calibration", {})
    raw_path = cal_cfg.get("grid_path", "calibrations/cv_calibration/grid.json")
    grid_path = _resolve_config_path(config, raw_path)
    roi_scale = cal_cfg.get("roi_scale", 0.35)

    # 1. Try loading existing calibration (unless forced to re-calibrate)
    if not force_recalibrate and os.path.exists(grid_path):
        grid = load_grid_calibration(grid_path)
        print(f"Loaded grid calibration from {grid_path}")
        return grid

    if force_recalibrate:
        print("Force re-calibration requested — ignoring existing preset")

    # 2. Interactive calibration from image
    if image_path is not None:
        grid = manual_calibrate_grid_from_image(
            image_path,
            save_path=grid_path,
            roi_scale=roi_scale,
        )
        return grid

    # 3. Corners provided directly
    if corners is not None:
        grid = compute_grid_from_corners(
            a1=corners["a1"],
            a12=corners["a12"],
            h1=corners["h1"],
            h12=corners.get("h12"),
            roi_scale=roi_scale,
        )
        save_grid_calibration(grid_path, grid)
        print(f"Computed and saved grid calibration to {grid_path}")
        return grid

    raise ValueError(
        f"No saved calibration at {grid_path} and no corners or image "
        "provided. Pass image_path for interactive calibration, or "
        "corners={'a1': (x,y), 'a12': (x,y), 'h1': (x,y), 'h12': (x,y)}."
    )


# ------------------------------------------------------------------
# Fallback: reset / resume helpers
# ------------------------------------------------------------------

def reconnect_robot(config: Dict[str, Any]) -> OT2Client:
    """Create a new OT2Client and reconnect to the last run."""
    robot_cfg = config.get("robot", {})
    client = OT2Client(
        robot_ip=robot_cfg.get("ip", "169.254.8.56"),
        port=robot_cfg.get("port", 31950),
    )
    run_id = client.reconnect_last_run()
    print(f"Reconnected to run {run_id}")
    return client


def emergency_home(config: Dict[str, Any]) -> None:
    """Connect and home the robot — useful after errors."""
    robot_cfg = config.get("robot", {})
    client = OT2Client(
        robot_ip=robot_cfg.get("ip", "169.254.8.56"),
        port=robot_cfg.get("port", 31950),
    )
    run_id = client.reconnect_last_run()
    client.home()
    print(f"Robot homed (run {run_id})")
