"""Project-specific plate color calibration and well annotation.

Sits on top of ``src.color_extraction`` and adds assay-specific logic:
control / experiment row assignment, white-balance calibration from the
water control, optional per-channel scaling from pure-dye controls,
and labelled measurement output.

Default row assignments (color-mix assay):
    A = pure Red control
    B = pure Green control
    C = pure Blue control
    D = water / white reference
    E-H = experiment replicates

Runtime control-driven API (no fixed row assumptions)::

    from src.color.calibration import (
        infer_control_roles,
        build_runtime_reference,
        apply_reference_to_column,
        RuntimeReference,
    )

Legacy fixed-layout API::

    from src.color.calibration import (
        extract_labeled_plate,
        build_plate_calibration,
        apply_calibration,
        summarize_column,
        write_measurements_csv,
    )
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.vision.extraction import (
    ROW_LABELS,
    N_ROWS,
    N_COLS,
    extract_well_rgb,
)


# ------------------------------------------------------------------
# Default layout
# ------------------------------------------------------------------

DEFAULT_CONTROL_ROWS: Dict[str, str] = {
    "A": "red",
    "B": "green",
    "C": "blue",
    "D": "water",
}
DEFAULT_EXPERIMENT_ROWS: List[str] = ["E", "F", "G", "H"]


# ------------------------------------------------------------------
# Labeled extraction
# ------------------------------------------------------------------

@dataclass
class LabeledPlate:
    """Per-well RGB with explicit labels for one plate image."""
    rgb: Dict[str, np.ndarray] = field(default_factory=dict)
    """Mapping of well label (e.g. ``"A1"``) to (3,) RGB array."""

    def wells_for_row(self, row: str) -> Dict[str, np.ndarray]:
        """Return ``{label: rgb}`` for all wells in *row*."""
        return {k: v for k, v in self.rgb.items() if k.startswith(row)}

    def wells_for_col(self, col: int) -> Dict[str, np.ndarray]:
        """Return ``{label: rgb}`` for all wells in column *col* (1-based)."""
        suffix = str(col)
        return {k: v for k, v in self.rgb.items() if k[1:] == suffix}


def extract_labeled_plate(
    image: np.ndarray,
    grid=None,
    rows: Optional[List[str]] = None,
    cols: Optional[List[int]] = None,
) -> LabeledPlate:
    """Extract labeled per-well RGB from a BGR image.

    Args:
        image: BGR image (OpenCV format).
        grid: Grid (dict, PlateGrid, or PlateGrid4). Uses DEFAULT_GRID if *None*.
        rows: Row letters to extract. Default: all A-H.
        cols: 1-based column numbers. Default: 1-12.

    Returns:
        A :class:`LabeledPlate`.
    """
    if rows is None:
        rows = ROW_LABELS
    if cols is None:
        cols = list(range(1, N_COLS + 1))

    plate = LabeledPlate()
    for row_letter in rows:
        ri = ROW_LABELS.index(row_letter)
        for col_num in cols:
            ci = col_num - 1
            label = f"{row_letter}{col_num}"
            plate.rgb[label] = extract_well_rgb(image, ri, ci, grid)
    return plate


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

@dataclass
class PlateCalibration:
    """Simple per-plate colour calibration.

    Stores a white-balance scale and optional per-channel scaling
    derived from pure-dye controls.
    """
    white_scale: np.ndarray  # (3,) — multiply RGB by this for white balance
    channel_scales: Optional[np.ndarray] = None  # (3,) optional further scaling

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply calibration to an RGB array (..., 3)."""
        result = rgb * self.white_scale
        if self.channel_scales is not None:
            result = result * self.channel_scales
        return np.clip(result, 0, 255)


def build_plate_calibration(
    plate: LabeledPlate,
    col: int,
    control_rows: Optional[Dict[str, str]] = None,
    white_reference: float = 255.0,
    use_channel_scaling: bool = False,
) -> PlateCalibration:
    """Build calibration from a single column's control wells.

    Args:
        plate: Labeled plate with extracted RGB.
        col: 1-based column number.
        control_rows: Mapping of row letter to role
            (default: A=red, B=green, C=blue, D=water).
        white_reference: Target white level.
        use_channel_scaling: If *True*, compute per-channel scaling
            from pure-dye controls so each pure dye maps to
            ``white_reference`` in its dominant channel.

    Returns:
        A :class:`PlateCalibration`.
    """
    if control_rows is None:
        control_rows = DEFAULT_CONTROL_ROWS

    # White balance from water well
    water_row = [r for r, role in control_rows.items() if role == "water"]
    if not water_row:
        raise ValueError("No water control row defined in control_rows")
    water_label = f"{water_row[0]}{col}"
    water_rgb = plate.rgb.get(water_label)
    if water_rgb is None:
        raise KeyError(f"Water control well {water_label} not found in plate")

    safe_water = np.maximum(water_rgb, 1.0)
    white_scale = white_reference / safe_water

    channel_scales = None
    if use_channel_scaling:
        # Each pure-dye control should max out its dominant channel
        dye_channel_map = {"red": 0, "green": 1, "blue": 2}
        scales = np.ones(3)
        for row_letter, role in control_rows.items():
            if role in dye_channel_map:
                ch = dye_channel_map[role]
                label = f"{row_letter}{col}"
                raw = plate.rgb.get(label)
                if raw is not None:
                    wb_value = raw[ch] * white_scale[ch]
                    if wb_value > 1.0:
                        scales[ch] = white_reference / wb_value
        channel_scales = scales

    return PlateCalibration(
        white_scale=white_scale,
        channel_scales=channel_scales,
    )


# ------------------------------------------------------------------
# Apply calibration to experiment wells
# ------------------------------------------------------------------

def apply_calibration(
    plate: LabeledPlate,
    calibration: PlateCalibration,
    col: int,
    experiment_rows: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Apply calibration to experiment wells in one column.

    Returns ``{label: calibrated_rgb}`` for each experiment well.
    """
    if experiment_rows is None:
        experiment_rows = DEFAULT_EXPERIMENT_ROWS

    result: Dict[str, np.ndarray] = {}
    for row_letter in experiment_rows:
        label = f"{row_letter}{col}"
        raw = plate.rgb.get(label)
        if raw is not None:
            result[label] = calibration.apply(raw)
    return result


# ------------------------------------------------------------------
# Column summary
# ------------------------------------------------------------------

@dataclass
class ColumnSummary:
    """Summary of one experiment column."""
    col: int
    experiment_rgb: Dict[str, np.ndarray]
    mean_rgb: np.ndarray
    std_rgb: np.ndarray
    control_rgb: Dict[str, np.ndarray]


def summarize_column(
    plate: LabeledPlate,
    col: int,
    calibration: Optional[PlateCalibration] = None,
    control_rows: Optional[Dict[str, str]] = None,
    experiment_rows: Optional[List[str]] = None,
) -> ColumnSummary:
    """Summarize one column: control RGB + calibrated experiment stats.

    Args:
        plate: Labeled plate.
        col: 1-based column number.
        calibration: Optional calibration to apply. If *None*, raw values.
        control_rows: Control row mapping.
        experiment_rows: Experiment row letters.

    Returns:
        A :class:`ColumnSummary`.
    """
    if control_rows is None:
        control_rows = DEFAULT_CONTROL_ROWS
    if experiment_rows is None:
        experiment_rows = DEFAULT_EXPERIMENT_ROWS

    # Control wells (raw)
    ctrl: Dict[str, np.ndarray] = {}
    for row_letter in control_rows:
        label = f"{row_letter}{col}"
        if label in plate.rgb:
            ctrl[label] = plate.rgb[label]

    # Experiment wells (optionally calibrated)
    if calibration is not None:
        exp = apply_calibration(plate, calibration, col, experiment_rows)
    else:
        exp = {}
        for row_letter in experiment_rows:
            label = f"{row_letter}{col}"
            if label in plate.rgb:
                exp[label] = plate.rgb[label]

    if exp:
        values = np.array(list(exp.values()))
        mean_rgb = values.mean(axis=0)
        std_rgb = values.std(axis=0)
    else:
        mean_rgb = np.zeros(3)
        std_rgb = np.zeros(3)

    return ColumnSummary(
        col=col,
        experiment_rgb=exp,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        control_rgb=ctrl,
    )


# ------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------

def write_measurements_csv(
    path: str,
    summaries: List[ColumnSummary],
) -> str:
    """Write per-well measurements to a CSV file.

    Each row is one well with columns: well, col, row_type, R, G, B.

    Returns the path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "column", "row_type", "R", "G", "B"])

        for summary in summaries:
            for label, rgb in summary.control_rgb.items():
                writer.writerow([
                    label, summary.col, "control",
                    round(float(rgb[0]), 1),
                    round(float(rgb[1]), 1),
                    round(float(rgb[2]), 1),
                ])
            for label, rgb in summary.experiment_rgb.items():
                writer.writerow([
                    label, summary.col, "experiment",
                    round(float(rgb[0]), 1),
                    round(float(rgb[1]), 1),
                    round(float(rgb[2]), 1),
                ])

    return path


# ------------------------------------------------------------------
# Runtime control-driven calibration (no fixed row assumptions)
# ------------------------------------------------------------------

#: Channel index constants for clarity.
_CH_RED, _CH_GREEN, _CH_BLUE = 0, 1, 2
_CHANNEL_NAMES = {_CH_RED: "red", _CH_GREEN: "green", _CH_BLUE: "blue"}


def _neutrality(rgb: np.ndarray) -> float:
    """Score how neutral (grey/white) an RGB value is.

    Returns a value in [0, 1] where 1 is perfectly neutral.
    Computed as ``1 - coefficient_of_variation`` of the three channels,
    scaled so that equal channels give 1.0 and maximally skewed gives ~0.
    """
    mean = rgb.mean()
    if mean < 1.0:
        return 0.0
    cv = rgb.std() / mean  # coefficient of variation
    # cv=0 is perfectly neutral; cv~0.8+ is very coloured
    return float(max(0.0, 1.0 - cv))


def _brightness(rgb: np.ndarray) -> float:
    """Simple brightness as mean of RGB channels."""
    return float(rgb.mean())


def _dominant_channel(rgb: np.ndarray) -> int:
    """Return the index (0=R, 1=G, 2=B) of the dominant channel."""
    return int(np.argmax(rgb))


def infer_control_roles(
    control_wells: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """Infer semantic roles from observed control well RGB values.

    Given a dict of ``{well_label: rgb_array}`` for the control wells of a
    single column, identify which well is the water/reference and which are
    the red, green, and blue dye controls.

    Selection rules (deterministic, no ML):
      * **Water/reference**: the well with the highest combined brightness
        and neutrality score (``brightness * neutrality``).
      * **Dye controls**: from the remaining wells, each is assigned the
        role corresponding to its dominant RGB channel.  Ties are broken
        by the magnitude of the dominant channel.

    Args:
        control_wells: ``{label: (3,) RGB array}`` for 4 control wells.

    Returns:
        ``{well_label: role}`` where role is one of
        ``"water"``, ``"red"``, ``"green"``, ``"blue"``.

    Raises:
        ValueError: If fewer than 4 wells are provided, or if the three
            dye wells do not cover all three channels uniquely.
    """
    if len(control_wells) < 4:
        raise ValueError(
            f"Need at least 4 control wells, got {len(control_wells)}"
        )

    # --- Step 1: identify water well ---
    # Score each well: brightness * neutrality.  Water is bright + neutral.
    scores = {
        label: _brightness(rgb) * _neutrality(rgb)
        for label, rgb in control_wells.items()
    }
    water_label = max(scores, key=scores.get)

    # --- Step 2: assign dye roles to remaining wells ---
    remaining = {
        label: rgb
        for label, rgb in control_wells.items()
        if label != water_label
    }

    # Group candidates by their dominant channel; pick the strongest per channel.
    # "strongest" = highest value in the dominant channel.
    channel_candidates: Dict[int, List[Tuple[str, float]]] = {
        _CH_RED: [], _CH_GREEN: [], _CH_BLUE: [],
    }
    for label, rgb in remaining.items():
        ch = _dominant_channel(rgb)
        channel_candidates[ch].append((label, float(rgb[ch])))

    assigned: Dict[str, str] = {water_label: "water"}
    used_labels: set = set()

    # Assign each channel's best candidate
    for ch in (_CH_RED, _CH_GREEN, _CH_BLUE):
        candidates = [
            (lbl, val) for lbl, val in channel_candidates[ch]
            if lbl not in used_labels
        ]
        if not candidates:
            raise ValueError(
                f"No control well has dominant {_CHANNEL_NAMES[ch]} channel. "
                f"Cannot assign all three dye roles from the observed controls."
            )
        best_label = max(candidates, key=lambda x: x[1])[0]
        assigned[best_label] = _CHANNEL_NAMES[ch]
        used_labels.add(best_label)

    return assigned


@dataclass
class RuntimeReference:
    """Reusable runtime calibration reference built from observed controls.

    This replaces fixed row-role assumptions with roles inferred at runtime
    from actual control well readings.

    Attributes:
        roles: ``{well_label: role}`` — inferred control roles.
        calibration: The :class:`PlateCalibration` derived from these controls.
        source_column: The 1-based column number the reference was built from.
        water_rgb: Raw RGB of the inferred water well (for diagnostics).
    """
    roles: Dict[str, str]
    calibration: PlateCalibration
    source_column: int
    water_rgb: np.ndarray


def build_runtime_reference(
    plate: LabeledPlate,
    col: int,
    control_labels: Optional[List[str]] = None,
    white_reference: float = 255.0,
    use_channel_scaling: bool = False,
) -> RuntimeReference:
    """Build a runtime calibration reference from one column's controls.

    This is the runtime-driven replacement for :func:`build_plate_calibration`
    when the physical row-to-role mapping is not known ahead of time.

    Args:
        plate: Labeled plate with extracted RGB.
        col: 1-based column number to read controls from.
        control_labels: Explicit list of well labels for the control wells
            (e.g. ``["A1", "B1", "C1", "D1"]``).  If *None*, uses the
            default control rows A-D for the given column.
        white_reference: Target white level (default 255).
        use_channel_scaling: Compute per-channel dye scaling.

    Returns:
        A :class:`RuntimeReference` containing the inferred roles and
        calibration, ready for reuse on any column.
    """
    # Collect control well RGB values
    if control_labels is None:
        control_labels = [
            f"{row}{col}" for row in DEFAULT_CONTROL_ROWS
        ]

    control_wells: Dict[str, np.ndarray] = {}
    for label in control_labels:
        rgb = plate.rgb.get(label)
        if rgb is None:
            raise KeyError(f"Control well {label} not found in plate")
        control_wells[label] = rgb

    # Infer roles
    roles = infer_control_roles(control_wells)

    # Find water well and build white balance
    water_label = [lbl for lbl, role in roles.items() if role == "water"][0]
    water_rgb = control_wells[water_label].copy()
    safe_water = np.maximum(water_rgb, 1.0)
    white_scale = white_reference / safe_water

    # Optional channel scaling from inferred dye wells
    channel_scales = None
    if use_channel_scaling:
        dye_channel_map = {"red": _CH_RED, "green": _CH_GREEN, "blue": _CH_BLUE}
        scales = np.ones(3)
        for label, role in roles.items():
            if role in dye_channel_map:
                ch = dye_channel_map[role]
                raw = control_wells[label]
                wb_value = raw[ch] * white_scale[ch]
                if wb_value > 1.0:
                    scales[ch] = white_reference / wb_value
        channel_scales = scales

    calibration = PlateCalibration(
        white_scale=white_scale,
        channel_scales=channel_scales,
    )

    return RuntimeReference(
        roles=roles,
        calibration=calibration,
        source_column=col,
        water_rgb=water_rgb,
    )


def apply_reference_to_column(
    plate: LabeledPlate,
    reference: RuntimeReference,
    col: int,
    experiment_rows: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Apply a :class:`RuntimeReference` calibration to experiment wells.

    The reference may come from a different column (quick-mode reuse).

    Args:
        plate: Labeled plate with extracted RGB.
        col: 1-based column number to calibrate.
        experiment_rows: Row letters for experiment wells
            (default: E-H).

    Returns:
        ``{label: calibrated_rgb}`` for each experiment well.
    """
    return apply_calibration(
        plate, reference.calibration, col, experiment_rows,
    )
