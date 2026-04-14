"""Image analysis and GP model update for color mixing.

:func:`analyze_capture` handles the full post-imaging pipeline:
extract RGB → build/apply calibration → return structured result.

:func:`fit_observation` is a thin helper that updates a GP surrogate
with a new (volumes, RGB) data point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.color.calibration import (
    ColumnSummary,
    LabeledPlate,
    RuntimeReference,
    apply_reference_to_column,
    build_runtime_reference,
    extract_labeled_plate,
    summarize_column,
)
from src.vision.extraction import extract_experiment_rgb

from src.tasks.color_mixing.config import ColorMixingConfig


@dataclass
class CaptureResult:
    """Structured output from :func:`analyze_capture`."""

    mean_rgb: np.ndarray
    std_rgb: np.ndarray
    raw_mean_rgb: np.ndarray
    raw_std_rgb: np.ndarray
    experiment_rgb: np.ndarray
    calibrated_experiment_rgb: np.ndarray
    control_rgb: Dict[str, np.ndarray]
    image_path: str
    used_calibration: bool = False
    reference_source_column: Optional[int] = None
    calibration_warning: Optional[str] = None
    summary: Optional[ColumnSummary] = None
    runtime_reference: Optional[RuntimeReference] = None


def analyze_capture(
    image_path: str,
    cfg: ColorMixingConfig,
    col_idx: int,
    grid: Any = None,
    *,
    skip_controls: bool = False,
    cached_reference: Optional[RuntimeReference] = None,
    is_quick: bool = False,
) -> CaptureResult:
    """Analyze a captured plate image for one column.

    Extracts raw RGB, builds or reuses a runtime calibration reference,
    applies it to experiment wells, and returns a :class:`CaptureResult`.

    Args:
        image_path: Path to the captured plate image (BGR).
        cfg: Color mixing config (provides well role assignments).
        col_idx: 0-based column index.
        grid: Grid calibration for well extraction.
        skip_controls: Whether control wells were dispensed this column.
        cached_reference: Previously built RuntimeReference for reuse.
        is_quick: Whether running in quick mode.

    Returns:
        A :class:`CaptureResult` with both raw and calibrated RGB.
    """
    well_col = col_idx + 1
    experiment_rows = cfg.experiment_rows
    control_row_letters = list(cfg.control_rows.keys())

    # --- Raw extraction (no white balance) ---
    raw_result = extract_experiment_rgb(
        image_path=image_path,
        col=col_idx,
        experiment_rows=experiment_rows,
        control_rows=control_row_letters,
        grid=grid,
        apply_white_balance=False,
    )

    raw_mean = raw_result["experiment_mean"]
    raw_std = raw_result["experiment_std"]

    # --- Calibration ---
    mean_rgb = raw_mean
    std_rgb = raw_std
    calibrated_exp_rgb = raw_result["experiment_rgb"]
    used_calibration = False
    reference_source_column = None
    calibration_warning: Optional[str] = None
    col_summary = None
    runtime_ref = None
    plate = None

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            calibration_warning = (
                f"Could not reload {image_path} for calibration; using raw RGB."
            )
        else:
            plate = extract_labeled_plate(img, grid=grid, cols=[well_col])
    except Exception as exc:
        calibration_warning = (
            f"Calibration setup failed for column {well_col}: {exc}; using raw RGB."
        )

    if plate is not None:
        try:
            if not skip_controls:
                runtime_ref = build_runtime_reference(plate, col=well_col)
                calibrated_exp = apply_reference_to_column(
                    plate, runtime_ref, col=well_col,
                )
                reference_source_column = runtime_ref.source_column
            elif is_quick and cached_reference is not None:
                calibrated_exp = apply_reference_to_column(
                    plate, cached_reference, col=well_col,
                )
                reference_source_column = cached_reference.source_column
            else:
                calibrated_exp = None
                calibration_warning = (
                    f"No calibration reference available for column {well_col}; "
                    "using raw RGB."
                )

            if calibrated_exp:
                calibrated_exp_rgb = np.array(list(calibrated_exp.values()))
                mean_rgb = calibrated_exp_rgb.mean(axis=0)
                std_rgb = calibrated_exp_rgb.std(axis=0)
                used_calibration = True
                calibration = (
                    runtime_ref.calibration
                    if runtime_ref is not None
                    else cached_reference.calibration
                )
                col_summary = summarize_column(
                    plate, col=well_col, calibration=calibration,
                )
            elif col_summary is None:
                col_summary = summarize_column(plate, col=well_col, calibration=None)
        except Exception as exc:
            calibration_warning = (
                f"Calibration failed for column {well_col}: {exc}; using raw RGB."
            )
            if col_summary is None:
                try:
                    col_summary = summarize_column(plate, col=well_col, calibration=None)
                except Exception:
                    col_summary = None

    return CaptureResult(
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        raw_mean_rgb=raw_mean,
        raw_std_rgb=raw_std,
        experiment_rgb=raw_result["experiment_rgb"],
        calibrated_experiment_rgb=calibrated_exp_rgb,
        control_rgb=raw_result["control_rgb"],
        image_path=image_path,
        used_calibration=used_calibration,
        reference_source_column=reference_source_column,
        calibration_warning=calibration_warning,
        summary=col_summary,
        runtime_reference=runtime_ref,
    )


def fit_observation(
    surrogate: Any,
    volumes: np.ndarray,
    observed_rgb: np.ndarray,
    *,
    full_refit: bool = False,
    all_X: Optional[np.ndarray] = None,
    all_Y: Optional[np.ndarray] = None,
) -> None:
    """Update a GP surrogate with a new observation.

    Args:
        surrogate: A fitted surrogate (``IndependentMultiOutputGP`` or
            ``CorrelatedMultiOutputGP``).
        volumes: (3,) dye volumes.
        observed_rgb: (3,) observed RGB.
        full_refit: If True, refit from scratch using *all_X*/*all_Y*.
        all_X: (n, 3) all volume observations (required if full_refit).
        all_Y: (n, 3) all RGB observations (required if full_refit).
    """
    if full_refit and all_X is not None and all_Y is not None:
        surrogate.fit(np.atleast_2d(all_X), np.atleast_2d(all_Y))
    else:
        surrogate.update(
            volumes.reshape(1, 3),
            observed_rgb.reshape(1, 3),
        )
