"""Custom OpenOT2 step handlers for color mixing active learning.

Each handler is a plain callable ``(step, context) -> dict`` registered
with the :class:`TaskRunner`.  The runner resolves ``$ref`` bindings in
``step.params`` *before* calling the handler, so ``image_path`` and
``observed_rgb`` arrive pre-populated from prior step outputs.

All assay-specific work is delegated to :mod:`src.tasks.color_mixing.api`.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

import numpy as np

from src.tasks.color_mixing.api import (
    analyze_capture,
    CaptureResult,
    fit_observation,
)
from src.tasks.color_mixing.config import load_task_config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: safe serialisation for numpy types
# ----------------------------------------------------------------------

def _to_list(obj):
    """Convert numpy arrays / scalars to plain Python lists / floats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _serialize_reference(ref) -> Optional[dict]:
    """Serialize a RuntimeReference to a JSON-safe dict."""
    if ref is None:
        return None
    return {
        "roles": ref.roles,
        "white_scale": ref.calibration.white_scale.tolist(),
        "channel_scales": (
            ref.calibration.channel_scales.tolist()
            if ref.calibration.channel_scales is not None
            else None
        ),
        "source_column": ref.source_column,
        "water_rgb": ref.water_rgb.tolist(),
    }


def _deserialize_reference(data: dict):
    """Reconstruct a RuntimeReference from a serialized dict."""
    from src.color.calibration import PlateCalibration, RuntimeReference

    white_scale = np.array(data["white_scale"], dtype=float)
    channel_scales = (
        np.array(data["channel_scales"], dtype=float)
        if data.get("channel_scales") is not None
        else None
    )
    calibration = PlateCalibration(
        white_scale=white_scale,
        channel_scales=channel_scales,
    )
    return RuntimeReference(
        roles=data["roles"],
        calibration=calibration,
        source_column=data["source_column"],
        water_rgb=np.array(data["water_rgb"], dtype=float),
    )


# ======================================================================
# handle_extract_rgb
# ======================================================================

def handle_extract_rgb(step, context=None) -> Dict[str, Any]:
    """Extract RGB from a plate image after capture.

    The runner resolves ``image_path`` via ``$ref`` from the capture step
    before this handler is called, so it arrives ready to use.

    Delegates all image analysis to :func:`analyze_capture` from the
    task API.

    Params expected in ``step.params``:
        image_path : str
            Path to the captured plate image (resolved via $ref).
        col : int
            0-based column index.
        grid_path : str
            Path to grid calibration JSON.
        config_path : str
            Path to experiment YAML (for loading ColorMixingConfig).
        cached_reference : dict or None
            Serialized :class:`RuntimeReference` from column 1.
        skip_controls : bool
            Whether controls were skipped for this column.
        is_quick : bool
            Whether running in quick mode.

    Returns a dict with:
        mean_rgb, std_rgb, raw_mean_rgb : list[float]
        used_calibration : bool
        runtime_reference : dict or None
    """
    params = step.params if hasattr(step, "params") else step
    image_path: str = params.get("image_path", "")
    col: int = int(params["col"])
    config_path: str = params.get("config_path", "")
    grid_path: str = params.get("grid_path", "")
    cached_ref_data: Optional[dict] = params.get("cached_reference")
    skip_controls: bool = bool(params.get("skip_controls", False))
    is_quick: bool = bool(params.get("is_quick", False))

    # Also accept config_path from context if not in params
    if not config_path and context and isinstance(context, dict):
        config_path = context.get("config_path", "")

    if not image_path:
        return {
            "deferred": True,
            "reason": "waiting_for_capture_output",
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "raw_mean_rgb": [0.0, 0.0, 0.0],
            "used_calibration": False,
            "runtime_reference": None,
        }

    # Load grid calibration if provided
    grid = None
    if grid_path:
        try:
            from src.vision.geometry import load_grid_calibration
            grid = load_grid_calibration(grid_path)
        except Exception as exc:
            logger.warning("Could not load grid calibration from %s: %s", grid_path, exc)

    # Deserialize cached reference
    cached_ref = None
    if cached_ref_data is not None:
        try:
            cached_ref = _deserialize_reference(cached_ref_data)
        except Exception as exc:
            logger.warning("Could not deserialize cached reference: %s", exc)

    # Load task config for analyze_capture
    cfg = None
    if config_path:
        try:
            cfg = load_task_config(config_path)
        except Exception as exc:
            logger.warning("Could not load task config from %s: %s", config_path, exc)

    # Delegate to the task API
    if cfg is not None:
        try:
            result: CaptureResult = analyze_capture(
                image_path=image_path,
                cfg=cfg,
                col_idx=col,
                grid=grid,
                skip_controls=skip_controls,
                cached_reference=cached_ref,
                is_quick=is_quick,
            )
            return {
                "mean_rgb": _to_list(result.mean_rgb),
                "std_rgb": _to_list(result.std_rgb),
                "raw_mean_rgb": _to_list(result.raw_mean_rgb),
                "used_calibration": result.used_calibration,
                "runtime_reference": _serialize_reference(result.runtime_reference),
            }
        except Exception as exc:
            logger.error("analyze_capture failed: %s", exc)
            return {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "mean_rgb": [0.0, 0.0, 0.0],
                "std_rgb": [0.0, 0.0, 0.0],
                "raw_mean_rgb": [0.0, 0.0, 0.0],
                "used_calibration": False,
                "runtime_reference": None,
            }

    # Fallback: no config available — use raw extraction only
    try:
        from src.vision.extraction import extract_experiment_rgb

        raw_result = extract_experiment_rgb(
            image_path=image_path,
            col=col,
            grid=grid,
            apply_white_balance=False,
        )
        return {
            "mean_rgb": _to_list(raw_result["experiment_mean"]),
            "std_rgb": _to_list(raw_result["experiment_std"]),
            "raw_mean_rgb": _to_list(raw_result["experiment_mean"]),
            "used_calibration": False,
            "runtime_reference": None,
        }
    except Exception as exc:
        logger.error("Raw extraction fallback failed: %s", exc)
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "raw_mean_rgb": [0.0, 0.0, 0.0],
            "used_calibration": False,
            "runtime_reference": None,
        }


# ======================================================================
# handle_suggest_volumes
# ======================================================================

def handle_suggest_volumes(step, context=None) -> Dict[str, Any]:
    """Use the GP surrogate to suggest the next dye volumes.

    Params expected in ``step.params``:
        target_rgb : list[float]
        total_volume : float
        all_X : list[list[float]]
        all_Y : list[list[float]]
        distance_metric : str
        acquisition : str  (``"EI"`` or ``"UCB"``)
        model_type : str   (``"correlated_gp"`` or ``"independent_gp"``)
        bounds_min : list[float]
        bounds_max : list[float]
        phase : str        (``"random"`` or ``"bo"``)
        seed : int
        iteration : int    (current 0-based iteration index)

    Returns a dict with:
        volumes : list[float]
        phase : str
    """
    from src.ml import AcquisitionFunction, create_surrogate, sample_simplex

    params = step.params if hasattr(step, "params") else step
    target_rgb = np.array(params["target_rgb"], dtype=float)
    total_volume = float(params["total_volume"])
    phase = str(params.get("phase", "random"))
    seed = int(params.get("seed", 42))
    iteration = int(params.get("iteration", 0))

    # ------------------------------------------------------------------
    # Random phase — sample from the simplex
    # ------------------------------------------------------------------
    if phase == "random":
        rng = np.random.default_rng(seed=seed + iteration)
        volumes = sample_simplex(1, total_volume, d=3, rng=rng)[0]
        return {
            "volumes": _to_list(volumes),
            "phase": "random",
        }

    # ------------------------------------------------------------------
    # Bayesian optimization phase
    # ------------------------------------------------------------------
    all_X = np.array(params.get("all_X", []), dtype=float)
    all_Y = np.array(params.get("all_Y", []), dtype=float)
    distance_metric = str(params.get("distance_metric", "rgb_euclidean"))
    acquisition = str(params.get("acquisition", "EI"))
    model_type = str(params.get("model_type", "correlated_gp"))
    bounds_min = np.array(params.get("bounds_min", [0, 0, 0]), dtype=float)
    bounds_max = np.array(params.get("bounds_max", [200, 200, 200]), dtype=float)
    bounds = np.array([bounds_min, bounds_max], dtype=float)

    if len(all_X) < 2:
        # Not enough data for GP — fall back to random
        logger.warning(
            "Only %d observations — falling back to random sampling", len(all_X)
        )
        rng = np.random.default_rng(seed=seed + iteration)
        volumes = sample_simplex(1, total_volume, d=3, rng=rng)[0]
        return {
            "volumes": _to_list(volumes),
            "phase": "random",
            "note": "Fell back to random — insufficient data for GP",
        }

    try:
        surrogate = create_surrogate(
            model_type=model_type,
            bounds=bounds,
            total_volume=total_volume,
        )
        surrogate.fit(all_X, all_Y)

        acq = AcquisitionFunction(
            kind=acquisition,
            target_rgb=target_rgb,
            total_volume=total_volume,
            distance_metric=distance_metric,
        )
        candidates = acq.suggest(surrogate, n_candidates=1)
        volumes = candidates[0]

        return {
            "volumes": _to_list(volumes),
            "phase": "bo",
        }

    except Exception as exc:
        logger.error("GP suggestion failed: %s", exc)
        # Graceful fallback to random sampling
        rng = np.random.default_rng(seed=seed + iteration)
        volumes = sample_simplex(1, total_volume, d=3, rng=rng)[0]
        return {
            "volumes": _to_list(volumes),
            "phase": "random",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "note": "GP failed — fell back to random sampling",
        }


# ======================================================================
# handle_fit_gp
# ======================================================================

def handle_fit_gp(step, context=None) -> Dict[str, Any]:
    """Update GP history with a new observation, compute distance, check convergence.

    The runner resolves ``observed_rgb`` via ``$ref`` from the extract_rgb
    step before this handler is called.

    Params expected in ``step.params``:
        volumes : list[float]
        observed_rgb : list[float]   (resolved via $ref)
        target_rgb : list[float]
        distance_metric : str
        convergence_threshold : float
        iteration : int
        all_X : list[list[float]]
        all_Y : list[list[float]]
        all_dist : list[float]

    Returns a dict with:
        distance : float
        converged : bool
        best_distance : float
        best_iteration : int   (1-based)
        all_X : updated list
        all_Y : updated list
        all_dist : updated list
    """
    from src.color.metrics import color_distance

    params = step.params if hasattr(step, "params") else step

    volumes_raw = params.get("volumes", [])
    observed_rgb_raw = params.get("observed_rgb", [])
    target_rgb_raw = params.get("target_rgb", [])
    if not volumes_raw or not observed_rgb_raw or not target_rgb_raw:
        return {
            "deferred": True,
            "reason": "waiting_for_post_hoc_binding",
            "distance": None,
            "converged": False,
            "best_distance": None,
            "best_iteration": None,
            "all_X": list(params.get("all_X", [])),
            "all_Y": list(params.get("all_Y", [])),
            "all_dist": list(params.get("all_dist", [])),
        }

    volumes = np.array(volumes_raw, dtype=float)
    observed_rgb = np.array(observed_rgb_raw, dtype=float)
    target_rgb = np.array(target_rgb_raw, dtype=float)
    distance_metric = str(params.get("distance_metric", "rgb_euclidean"))
    convergence_threshold = float(params.get("convergence_threshold", 50.0))
    iteration = int(params.get("iteration", 0))

    # Reconstruct history lists
    all_X = [list(x) for x in params.get("all_X", [])]
    all_Y = [list(y) for y in params.get("all_Y", [])]
    all_dist = list(params.get("all_dist", []))

    # Compute distance for this observation
    try:
        dist = float(color_distance(observed_rgb, target_rgb, distance_metric))
    except Exception as exc:
        logger.error("Distance computation failed: %s", exc)
        dist = float("inf")

    # Append new observation
    all_X.append(_to_list(volumes))
    all_Y.append(_to_list(observed_rgb))
    all_dist.append(dist)

    # Best so far
    best_idx = int(np.argmin(all_dist))
    best_distance = all_dist[best_idx]
    converged = dist < convergence_threshold

    logger.info(
        "Iteration %d: dist=%.2f  best=%.2f (iter %d)  converged=%s",
        iteration, dist, best_distance, best_idx + 1, converged,
    )

    return {
        "distance": dist,
        "converged": converged,
        "best_distance": best_distance,
        "best_iteration": best_idx + 1,
        "all_X": all_X,
        "all_Y": all_Y,
        "all_dist": all_dist,
    }
