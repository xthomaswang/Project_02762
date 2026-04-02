"""
End-to-end active learning loop for colorimetric assay optimization.

Orchestrates: ML suggests volumes → robot dispenses (controls + experiments)
→ camera captures → color extracted → ML updates → repeat.

Per-column protocol (8 steps):
  Phase 1 (controls, left single-channel):
    1. Red   → row A
    2. Green → row B
    3. Blue  → row C
    4. Water → row D
  Phase 2 (experiment, right 8-channel with 4 tips E-H):
    5. Red   (Vr µL) → rows E-H
    6. Green (Vg µL) → rows E-H
    7. Blue  (Vb µL) → rows E-H
    8. Mix   → rows E-H
  Phase 3: settle → image → extract RGB → update GPs
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from openot2 import OT2Client, OT2Operations

from src.ml import (
    AcquisitionFunction,
    create_surrogate,
    sample_simplex,
)
from src.vision.extraction import extract_experiment_rgb
from src.color.metrics import color_distance
from src.color.calibration import (
    RuntimeReference,
    build_runtime_reference,
    apply_reference_to_column,
    extract_labeled_plate,
    summarize_column,
    write_measurements_csv,
    ColumnSummary,
)
from src.vision.geometry import AnyGrid, load_grid_calibration
from src.robot import ParkPosition, resolve_park_position, park_for_imaging


# ======================================================================
# Main active learning loop
# ======================================================================

def run_active_learning_loop(
    config_path: str,
    robot: Optional[OT2Client] = None,
    n_initial: Optional[int] = None,
    n_iterations: Optional[int] = None,
    acquisition_kind: Optional[str] = None,
    start_column: int = 0,
    seed_X: Optional[np.ndarray] = None,
    seed_Y: Optional[np.ndarray] = None,
    mode: str = "full",
    grid_calibration: Optional[AnyGrid] = None,
):
    """
    Main active learning loop.

    Args:
        config_path: Path to experiment YAML config.
        robot: OT2Client instance (created from config if None).
        n_initial: Override config's experiment.n_initial.
        n_iterations: Override config's experiment.n_optimization.
        acquisition_kind: Override config's ml.acquisition.
        start_column: Column index to start from (0-based). Use this to
            resume after manually running some columns.
        seed_X: (n, 3) array of prior volume observations to seed the GP.
        seed_Y: (n, 3) array of prior RGB observations to seed the GP.
        mode: "full" (default) — use all config values as-is.
            "quick" — fewer rinses, fewer randoms, skip repeat controls.
            Quick mode is opt-in only.
        grid_calibration: Optional PlateGrid for well extraction. If None,
            attempts to load from config calibration.grid_path, then
            falls back to DEFAULT_GRID.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Parse config ---
    target_rgb = np.array(config["target_color"], dtype=float)
    total_volume = float(config["total_volume_ul"])
    convergence_threshold = float(
        config.get("convergence_threshold")
        or config.get("convergence", {}).get("rgb_distance_threshold", 50)
    )
    seed = config.get("seed", 42)

    bounds_cfg = config["volume_bounds"]
    bounds = np.array([bounds_cfg["min"], bounds_cfg["max"]], dtype=float)

    exp_cfg = config.get("experiment", {})
    n_initial_explicit = n_initial  # Save whether user passed it explicitly
    n_initial = n_initial or exp_cfg.get("n_initial", 5)
    n_opt = n_iterations or exp_cfg.get("n_optimization", 7)
    max_per_plate = exp_cfg.get("max_per_plate", 12)
    batch_mode = exp_cfg.get("batch_mode", True)

    ml_cfg = config.get("ml", {})
    model_type = ml_cfg.get("model", "correlated_gp")
    acq_kind = acquisition_kind or ml_cfg.get("acquisition", "EI")
    distance_metric = ml_cfg.get("distance_metric", "rgb_euclidean")

    output_cfg = config.get("output", {})
    base_dir = output_cfg.get("base_dir", "data")

    # --- Quick mode overrides ---
    quick_overrides = {}
    if mode == "quick":
        if n_initial_explicit is None:
            n_initial = 5 # i random exploration experiments
        quick_overrides["rinse_cycles"] = 1
        quick_overrides["mix_cycles"] = 2
        quick_overrides["skip_controls_after_first"] = True
        print(f"  [QUICK MODE] n_initial={n_initial}, rinse_cycles=1, "
              f"mix_cycles=2, skip controls after col 1")
    else:
        print(f"  [FULL MODE] Using config values as-is")

    # --- Resolve grid calibration ---
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if grid_calibration is None:
        cal_cfg = config.get("calibration", {})
        grid_path = cal_cfg.get("grid_path")
        if grid_path:
            if not os.path.isabs(grid_path):
                grid_path = os.path.join(config_dir, grid_path)
            if os.path.exists(grid_path):
                grid_calibration = load_grid_calibration(grid_path)

    # --- Initialize robot ---
    if robot is None:
        robot_cfg = config.get("robot", {})
        robot = OT2Client(
            robot_ip=robot_cfg.get("ip", "169.254.8.56"),
            port=robot_cfg.get("port", 31950),
        )

    # --- Initialize ML ---
    surrogate = create_surrogate(model_type=model_type, bounds=bounds, total_volume=total_volume)
    acq = AcquisitionFunction(
        kind=acq_kind,
        target_rgb=target_rgb,
        total_volume=total_volume,
        distance_metric=distance_metric,
    )

    # --- Setup robot run ---
    clean_cfg = config.get("cleaning", {})
    rinse_cycles = quick_overrides.get("rinse_cycles", clean_cfg.get("rinse_cycles", 3))
    ops = OT2Operations(
        client=robot,
        rinse_cycles=rinse_cycles,
        rinse_volume=clean_cfg.get("rinse_volume_ul", 250),
    )

    robot_run_id = robot.create_run()
    labware_ids = _load_all_labware(robot, config)

    # Human-readable local folder name: <experiment_name>_YYYYMMDD_HHMMSS
    exp_name = config.get("metadata", {}).get("name", "run")
    run_id = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- Resolve imaging park position ---
    try:
        park_pos = resolve_park_position(config, robot)
    except ValueError:
        park_pos = None  # fall back to home() if park slot not loaded

    # --- Data storage ---
    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []
    all_dist: List[float] = []
    log_rows: List[Dict] = []
    all_summaries: List[ColumnSummary] = []
    cached_reference: Optional[RuntimeReference] = None  # quick-mode reuse

    # --- Load seed data (from prior manual experiments) ---
    if seed_X is not None and seed_Y is not None:
        seed_X = np.atleast_2d(seed_X)
        seed_Y = np.atleast_2d(seed_Y)
        for j in range(len(seed_X)):
            all_X.append(seed_X[j])
            all_Y.append(seed_Y[j])
            all_dist.append(float(color_distance(seed_Y[j], target_rgb, distance_metric)))
        print(f"  Loaded {len(seed_X)} seed observation(s) from prior experiments")
        for j in range(len(seed_X)):
            print(f"    Seed {j+1}: V=({seed_X[j][0]:.1f},{seed_X[j][1]:.1f},{seed_X[j][2]:.1f}) "
                  f"-> RGB=({seed_Y[j][0]:.0f},{seed_Y[j][1]:.0f},{seed_Y[j][2]:.0f}) "
                  f"dist={all_dist[j]:.1f}")

    rng = np.random.default_rng(seed=seed)
    col_idx = start_column

    # ==========================================
    # Phase 1: Random exploration
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: {n_initial} random exploration experiments")
    if start_column > 0:
        print(f"  (resuming from column {start_column + 1})")
    print(f"{'='*60}\n")

    X_init = sample_simplex(n_initial, total_volume, d=3, rng=rng)

    for i, volumes in enumerate(X_init):
        col_idx = start_column + i
        if col_idx >= max_per_plate:
            if not batch_mode:
                print(f"\n*** Plate full (column {max_per_plate}) and batch_mode=False — stopping. ***")
                break
            _pause_for_plate_swap(robot)
            col_idx = 0

        print(f"\n--- Random experiment {i+1}/{n_initial} (column {col_idx+1}) ---")
        print(f"    Volumes: Vr={volumes[0]:.1f}, Vg={volumes[1]:.1f}, Vb={volumes[2]:.1f}")

        skip_controls = (quick_overrides.get("skip_controls_after_first", False)
                         and col_idx > start_column)
        exp_result = _run_single_experiment(
            ops, config, labware_ids, volumes, col_idx, run_id, base_dir,
            skip_controls=skip_controls,
            mix_cycles_override=quick_overrides.get("mix_cycles"),
            grid=grid_calibration,
            park_pos=park_pos,
            cached_reference=cached_reference,
            is_quick=(mode == "quick"),
        )
        observed_rgb = exp_result["mean_rgb"]
        dist = color_distance(observed_rgb, target_rgb, distance_metric)

        all_X.append(volumes)
        all_Y.append(observed_rgb)
        all_dist.append(dist)
        if "summary" in exp_result:
            all_summaries.append(exp_result["summary"])
        if "runtime_reference" in exp_result and cached_reference is None:
            cached_reference = exp_result["runtime_reference"]
        _log_iteration(log_rows, i, col_idx, volumes, observed_rgb, dist, "random")

        _display_iteration(
            iteration=i, phase="random", col_idx=col_idx,
            volumes=volumes, exp_result=exp_result, target_rgb=target_rgb,
            all_dist=all_dist, distance_metric=distance_metric,
        )

    # Fit GPs on initial data
    surrogate.fit(np.array(all_X), np.array(all_Y))

    # ==========================================
    # Phase 2: Bayesian optimization
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: {n_opt} Bayesian optimization iterations")
    print(f"{'='*60}\n")

    for i in range(n_opt):
        col_idx = start_column + n_initial + i
        if col_idx >= max_per_plate:
            if not batch_mode:
                print(f"\n*** Plate full (column {max_per_plate}) and batch_mode=False — stopping. ***")
                break
            _pause_for_plate_swap(robot)
            col_idx = 0

        candidates = acq.suggest(surrogate, n_candidates=1)
        volumes = candidates[0]

        print(f"\n--- BO iteration {i+1}/{n_opt} (column {col_idx+1}) ---")
        print(f"    Suggested: Vr={volumes[0]:.1f}, Vg={volumes[1]:.1f}, Vb={volumes[2]:.1f}")

        skip_controls = quick_overrides.get("skip_controls_after_first", False)
        exp_result = _run_single_experiment(
            ops, config, labware_ids, volumes, col_idx, run_id, base_dir,
            skip_controls=skip_controls,
            mix_cycles_override=quick_overrides.get("mix_cycles"),
            grid=grid_calibration,
            park_pos=park_pos,
            cached_reference=cached_reference,
            is_quick=(mode == "quick"),
        )
        observed_rgb = exp_result["mean_rgb"]
        dist = color_distance(observed_rgb, target_rgb, distance_metric)

        all_X.append(volumes)
        all_Y.append(observed_rgb)
        all_dist.append(dist)
        if "summary" in exp_result:
            all_summaries.append(exp_result["summary"])
        if "runtime_reference" in exp_result and cached_reference is None:
            cached_reference = exp_result["runtime_reference"]
        _log_iteration(log_rows, n_initial + i, col_idx, volumes, observed_rgb, dist, "bo")

        surrogate.update(volumes.reshape(1, 3), observed_rgb.reshape(1, 3))

        # Preview next suggestion for display (if not last iteration)
        next_volumes = None
        if i < n_opt - 1 and dist >= convergence_threshold:
            try:
                next_candidates = acq.suggest(surrogate, n_candidates=1)
                next_volumes = next_candidates[0]
            except Exception:
                pass

        _display_iteration(
            iteration=n_initial + i, phase="bo", col_idx=col_idx,
            volumes=volumes, exp_result=exp_result, target_rgb=target_rgb,
            all_dist=all_dist, surrogate=surrogate, next_volumes=next_volumes,
            distance_metric=distance_metric,
        )

        if dist < convergence_threshold:
            print(f"\n*** CONVERGED! Distance {dist:.1f} < threshold {convergence_threshold} ***")
            break

    # ==========================================
    # Summary
    # ==========================================
    robot.home()
    best_idx = int(np.argmin(all_dist))
    print(f"\n{'='*60}")
    print(f"  ACTIVE LEARNING COMPLETE")
    print(f"  Total experiments: {len(all_X)}")
    print(f"  Best distance: {all_dist[best_idx]:.1f} (iteration {best_idx+1})")
    print(f"  Best volumes: Vr={all_X[best_idx][0]:.1f}, Vg={all_X[best_idx][1]:.1f}, Vb={all_X[best_idx][2]:.1f}")
    print(f"  Best RGB: ({all_Y[best_idx][0]:.0f}, {all_Y[best_idx][1]:.0f}, {all_Y[best_idx][2]:.0f})")
    print(f"  Target RGB: ({target_rgb[0]:.0f}, {target_rgb[1]:.0f}, {target_rgb[2]:.0f})")
    print(f"{'='*60}\n")

    run_folder = os.path.join(base_dir, "runs", run_id)
    _save_results_csv(log_rows, run_folder)

    # Save color calibration measurements if available
    if all_summaries:
        results_dir = os.path.join(run_folder, "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meas_path = os.path.join(results_dir, f"measurements_{ts}.csv")
        write_measurements_csv(meas_path, all_summaries)
        print(f"[PIPELINE] Measurements saved: {meas_path}")

    return np.array(all_X), np.array(all_Y), np.array(all_dist)


# ======================================================================
# Single experiment execution
# ======================================================================

def _run_single_experiment(
    ops: OT2Operations,
    config: Dict[str, Any],
    labware_ids: Dict[str, Any],
    volumes: np.ndarray,
    col_idx: int,
    run_id: str,
    base_dir: str,
    skip_controls: bool = False,
    mix_cycles_override: Optional[int] = None,
    grid=None,
    park_pos: Optional[ParkPosition] = None,
    cached_reference: Optional[RuntimeReference] = None,
    is_quick: bool = False,
) -> Dict[str, Any]:
    """
    Execute one complete experiment for a single column.

    Args:
        skip_controls: If True, skip Phase 1 (control dispensing) to save time.
        mix_cycles_override: Override mix cycles from config (e.g. 2 for quick mode).
        park_pos: Park position for imaging (uses home() if None).
        cached_reference: Cached first-column RuntimeReference for quick-mode reuse.
        is_quick: True when running in quick mode.

    Returns:
        Dict with keys:
            "mean_rgb": (3,) mean RGB across experiment replicates
            "std_rgb": (3,) std RGB across experiment replicates
            "experiment_rgb": (4, 3) raw per-replicate RGB
            "calibrated_experiment_rgb": (4, 3) RGB used for optimization
            "control_rgb": dict mapping row label -> (3,) RGB
            "image_path": path to captured plate image
            "summary": ColumnSummary (if a summary could be built)
            "runtime_reference": RuntimeReference (if built from this column)
            "used_calibration": whether calibrated_experiment_rgb differs from raw
            "reference_source_column": source column for the active reference
            "calibration_warning": explicit fallback reason, if any
    """
    exp_cfg = config.get("experiment", {})
    settle_time = exp_cfg.get("settle_time_seconds", 45)
    mix_cycles = mix_cycles_override or exp_cfg.get("mix_cycles", 3)
    mix_volume = exp_cfg.get("mix_volume_ul", 200)

    well_col = col_idx + 1  # 1-based for well naming
    Vr, Vg, Vb = float(volumes[0]), float(volumes[1]), float(volumes[2])

    cleaning_id = labware_ids["cleaning_id"]
    robot = ops.client

    # ---- Phase 1: Control dispensing (left single-channel) ----
    if skip_controls:
        print(f"  [Phase 1] Controls SKIPPED (quick mode)")
    else:
        print(f"  [Phase 1] Controls (column {well_col})")
        robot.use_pipette("left")

        controls = [
            ("red",   labware_ids["red_source_id"],   "A1", f"A{well_col}", "A1"),
            ("green", labware_ids["green_source_id"], "A1", f"B{well_col}", "B1"),
            ("blue",  labware_ids["blue_source_id"],  "A1", f"C{well_col}", "C1"),
            ("water", labware_ids["water_source_id"], "A1", f"D{well_col}", "D1"),
        ]

        control_volume = float(config.get("total_volume_ul", 200))

        for dye_name, source_id, src_well, dest_well, tip_well in controls:
            ops.transfer(
                tiprack_id=labware_ids["tiprack_left_id"],
                source_id=source_id,
                dest_id=labware_ids["dispense_id"],
                tip_well=tip_well,
                source_well=src_well,
                dest_well=dest_well,
                volume=control_volume,
                cleaning_id=cleaning_id,
                rinse_col=_get_control_rinse_col(dye_name, config),
            )

    # ---- Phase 2: Experiment dispensing (right 8-channel) ----
    print(f"  [Phase 2] Experiments (column {well_col})")
    robot.use_pipette("right")

    dye_transfers = [
        ("red",   labware_ids["red_source_id"],   "A1", 1, Vr, "A1"),
        ("green", labware_ids["green_source_id"], "A1", 2, Vg, "A2"),
        ("blue",  labware_ids["blue_source_id"],  "A1", 3, Vb, "A3"),
    ]

    for dye_name, source_id, src_well, tip_col, vol, rinse_col in dye_transfers:
        if vol < 1.0:
            print(f"    Skipping {dye_name} (volume {vol:.1f} µL < 1.0)")
            continue
        ops.transfer(
            tiprack_id=labware_ids["tiprack_right_id"],
            source_id=source_id,
            dest_id=labware_ids["dispense_id"],
            tip_well=f"A{tip_col}",
            source_well=src_well,
            dest_well=f"A{well_col}",
            volume=vol,
            cleaning_id=cleaning_id,
            rinse_col=rinse_col,
        )

    # Mixing step
    ops.mix(
        tiprack_id=labware_ids["tiprack_right_id"],
        labware_id=labware_ids["dispense_id"],
        tip_well="A4",
        mix_well=f"A{well_col}",
        cycles=mix_cycles,
        volume=mix_volume,
        cleaning_id=cleaning_id,
        rinse_col="A4",
    )

    # ---- Phase 3: Settle + Park + Image + Extract RGB ----
    # Start settle clock immediately after final liquid step.
    # Park motion counts toward settle time.
    settle_start = time.monotonic()

    if park_pos is not None:
        print(f"  [Phase 3] Parking + settling ({settle_time}s) + imaging")
        park_for_imaging(robot, park_pos)
    else:
        print(f"  [Phase 3] Homing + settling ({settle_time}s) + imaging")
        robot.home()

    # Sleep only the remaining settle time after the park/home motion
    elapsed = time.monotonic() - settle_start
    remaining = max(0, settle_time - elapsed)
    if remaining > 0:
        time.sleep(remaining)

    image_path = _capture_plate_image(config, run_id, col_idx, base_dir)

    # Always extract raw (no white balance) to avoid D-row dependency
    result = extract_experiment_rgb(
        image_path=image_path,
        col=col_idx,
        experiment_rows=config.get("well_layout", {}).get("experiment_rows", ["E", "F", "G", "H"]),
        control_rows=config.get("well_layout", {}).get("control_rows", ["A", "B", "C", "D"]),
        grid=grid,
        apply_white_balance=False,
    )

    raw_mean = result["experiment_mean"]
    raw_std = result["experiment_std"]

    # ---- Color calibration (runtime, control-driven) ----
    # Use calibrated values for optimization when available so that all
    # columns are in the same color coordinate system.
    col_summary = None
    runtime_ref = None
    mean_rgb = raw_mean
    std_rgb = raw_std
    calibrated_exp_rgb = result["experiment_rgb"]
    used_calibration = False
    reference_source_column = None
    calibration_warning: Optional[str] = None

    plate = None
    try:
        import cv2 as _cv2
        _img = _cv2.imread(image_path)
        if _img is None:
            calibration_warning = (
                f"Could not reload {image_path} for calibration; using raw RGB."
            )
        else:
            plate = extract_labeled_plate(_img, grid=grid, cols=[well_col])
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
                print(
                    f"    [QUICK] Using cached reference from column "
                    f"{cached_reference.source_column}"
                )
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

    if calibration_warning is not None:
        print(f"    [WARNING] {calibration_warning}")

    print(f"    Mean RGB: ({mean_rgb[0]:.0f}, {mean_rgb[1]:.0f}, {mean_rgb[2]:.0f}) "
          f"± ({std_rgb[0]:.1f}, {std_rgb[1]:.1f}, {std_rgb[2]:.1f})")

    out: Dict[str, Any] = {
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb,
        "raw_mean_rgb": raw_mean,
        "raw_std_rgb": raw_std,
        "experiment_rgb": result["experiment_rgb"],
        "calibrated_experiment_rgb": calibrated_exp_rgb,
        "control_rgb": result["control_rgb"],
        "image_path": image_path,
        "used_calibration": used_calibration,
        "reference_source_column": reference_source_column,
        "calibration_warning": calibration_warning,
    }
    if col_summary is not None:
        out["summary"] = col_summary
    if runtime_ref is not None:
        out["runtime_reference"] = runtime_ref
    return out


# ======================================================================
# Setup helpers (experiment-specific config parsing)
# ======================================================================

def _load_all_labware(robot: OT2Client, config: Dict[str, Any]) -> Dict[str, Any]:
    """Load all pipettes and labware from experiment config."""
    ids: Dict[str, Any] = {}

    pip_cfg = config["pipettes"]
    ids["right_pipette_id"] = robot.load_pipette(pip_cfg["right"]["name"], "right")
    ids["left_pipette_id"] = robot.load_pipette(pip_cfg["left"]["name"], "left")

    lab_cfg = config["labware"]
    for tr in lab_cfg["tipracks"]:
        lw_id = robot.load_labware(tr["name"], tr["slot"])
        key = "tiprack_right_id" if tr["for"] == "right" else "tiprack_left_id"
        ids[key] = lw_id

    # Separate dye reservoirs (slots 7, 8, 9)
    src_cfg = lab_cfg["sources"]
    ids["red_source_id"] = robot.load_labware(src_cfg["red"]["name"], src_cfg["red"]["slot"])
    ids["green_source_id"] = robot.load_labware(src_cfg["green"]["name"], src_cfg["green"]["slot"])
    ids["blue_source_id"] = robot.load_labware(src_cfg["blue"]["name"], src_cfg["blue"]["slot"])

    # Pure water reservoir (slot 5)
    water_cfg = lab_cfg["water"]
    ids["water_source_id"] = robot.load_labware(water_cfg["name"], water_cfg["slot"])

    # Cleaning reservoir (slot 4)
    ids["cleaning_id"] = robot.load_labware(
        lab_cfg["cleaning"]["name"], lab_cfg["cleaning"]["slot"],
    )

    # Dispense plate (slot 1)
    ids["dispense_id"] = robot.load_labware(
        lab_cfg["dispense"]["name"], lab_cfg["dispense"]["slot"],
    )

    return ids


def _get_control_rinse_col(dye_name: str, config: Dict) -> str:
    """Get the rinse column for a control dye from config."""
    cleaning_cols = config.get("labware", {}).get("cleaning", {}).get("columns", {})
    mapping = {
        "red": cleaning_cols.get("control_rinse", "A6"),
        "green": cleaning_cols.get("control_rinse", "A6"),
        "blue": cleaning_cols.get("control_rinse", "A6"),
        "water": cleaning_cols.get("water_rinse", "A5"),
    }
    return mapping.get(dye_name, "A6")


# ======================================================================
# Imaging
# ======================================================================

def _capture_plate_image(
    config: Dict[str, Any],
    run_id: str,
    col_idx: int,
    base_dir: str,
) -> str:
    """Capture a plate image using the overhead camera."""
    from vision import USBCamera

    cam_cfg = config.get("camera", {})
    camera = USBCamera(
        camera_id=cam_cfg.get("device_id", 0),
        width=cam_cfg.get("width", 1920),
        height=cam_cfg.get("height", 1080),
        warmup_frames=cam_cfg.get("warmup_frames", 10),
    )

    run_folder = os.path.join(base_dir, "runs", run_id)
    os.makedirs(run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(run_folder, f"col{col_idx+1}_{timestamp}.jpg")

    with camera:
        frame = camera.capture()
        if frame is None:
            raise RuntimeError("Camera capture returned None — check camera connection")
        import cv2
        cv2.imwrite(image_path, frame)

    print(f"    Image saved: {image_path}")
    return image_path


# ======================================================================
# Plate swap & logging
# ======================================================================

def _pause_for_plate_swap(robot: OT2Client):
    """Pause robot for plate and tip swap."""
    print("\n*** PLATE FULL — pausing for plate swap ***")
    robot.home()
    robot.pause("Plate full. Replace plate, refill cleaning water, then resume.")


def _log_iteration(
    log_rows: List[Dict],
    iteration: int,
    col_idx: int,
    volumes: np.ndarray,
    observed_rgb: np.ndarray,
    distance: float,
    phase: str,
):
    """Append one row to the experiment log."""
    log_rows.append({
        "iteration": iteration,
        "column": col_idx + 1,
        "phase": phase,
        "Vr": round(float(volumes[0]), 2),
        "Vg": round(float(volumes[1]), 2),
        "Vb": round(float(volumes[2]), 2),
        "obs_R": round(float(observed_rgb[0]), 1),
        "obs_G": round(float(observed_rgb[1]), 1),
        "obs_B": round(float(observed_rgb[2]), 1),
        "distance": round(distance, 2),
        "timestamp": datetime.now().isoformat(),
    })


def _save_results_csv(log_rows: List[Dict], run_folder: str):
    """Save experiment results to ``<run_folder>/results/``."""
    import pandas as pd

    if not log_rows:
        return
    results_dir = os.path.join(run_folder, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"experiment_{timestamp}.csv")
    pd.DataFrame(log_rows).to_csv(csv_path, index=False)
    print(f"[PIPELINE] Results saved: {csv_path}")


# ======================================================================
# Iteration display (photo + prediction + adjustment plan)
# ======================================================================

def _display_iteration(
    iteration: int,
    phase: str,
    col_idx: int,
    volumes: np.ndarray,
    exp_result: Dict[str, Any],
    target_rgb: np.ndarray,
    all_dist: List[float],
    surrogate=None,
    next_volumes: Optional[np.ndarray] = None,
    distance_metric: str = "rgb_euclidean",
):
    """
    Display a visual summary of one experiment iteration.

    Shows: captured plate photo (thumbnail), extracted colors for the column,
    GP prediction vs observation, and the adjustment plan for next iteration.
    """
    import matplotlib.pyplot as plt
    import cv2

    mean_rgb = exp_result["mean_rgb"]
    std_rgb = exp_result["std_rgb"]
    exp_rgb = exp_result.get("calibrated_experiment_rgb", exp_result["experiment_rgb"])
    ctrl_rgb = exp_result["control_rgb"]
    image_path = exp_result["image_path"]
    dist = color_distance(mean_rgb, target_rgb, distance_metric)

    # --- Build figure: [photo] [column colors] [prediction comparison] ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5),
                             gridspec_kw={"width_ratios": [3, 1, 2, 2]})
    fig.suptitle(
        f"{'Random' if phase == 'random' else 'BO'} #{iteration+1} — "
        f"Column {col_idx+1} — Distance: {dist:.1f}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Captured plate photo (thumbnail)
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img)
        axes[0].set_title("Captured plate", fontsize=9)
    axes[0].axis("off")

    # Panel 2: Column color swatches (A-H)
    row_labels = list("ABCDEFGH")
    row_colors = []
    for label in row_labels[:4]:
        rgb = ctrl_rgb.get(label, np.array([200, 200, 200]))
        row_colors.append(np.clip(rgb / 255.0, 0, 1))
    for rgb in exp_rgb:
        row_colors.append(np.clip(rgb / 255.0, 0, 1))

    for r, (label, color) in enumerate(zip(row_labels, row_colors)):
        axes[1].barh(7 - r, 1, color=color, edgecolor="gray", linewidth=0.5)
        axes[1].text(-0.1, 7 - r, label, ha="right", va="center", fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.5, 7.5)
    axes[1].set_title(f"Col {col_idx+1}", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel 3: Observed vs Target color comparison
    obs_norm = np.clip(mean_rgb / 255.0, 0, 1)
    tgt_norm = np.clip(target_rgb / 255.0, 0, 1)
    axes[2].barh(1, 1, color=obs_norm, edgecolor="black", linewidth=1)
    axes[2].barh(0, 1, color=tgt_norm, edgecolor="black", linewidth=1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Target", "Observed"], fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_title("Color comparison", fontsize=9)
    # Add RGB text
    axes[2].text(0.5, 1, f"({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})",
                 ha="center", va="center", fontsize=8,
                 color="white" if np.mean(obs_norm) < 0.5 else "black")
    axes[2].text(0.5, 0, f"({target_rgb[0]:.0f},{target_rgb[1]:.0f},{target_rgb[2]:.0f})",
                 ha="center", va="center", fontsize=8,
                 color="white" if np.mean(tgt_norm) < 0.5 else "black")

    # Panel 4: Prediction report text
    axes[3].axis("off")
    report_lines = [
        f"Volumes: R={volumes[0]:.1f} G={volumes[1]:.1f} B={volumes[2]:.1f}",
        f"Observed: ({mean_rgb[0]:.0f}, {mean_rgb[1]:.0f}, {mean_rgb[2]:.0f})",
        f"Std dev:  ({std_rgb[0]:.1f}, {std_rgb[1]:.1f}, {std_rgb[2]:.1f})",
        f"Target:   ({target_rgb[0]:.0f}, {target_rgb[1]:.0f}, {target_rgb[2]:.0f})",
        f"Distance: {dist:.1f}",
    ]
    if exp_result.get("used_calibration"):
        ref_col = exp_result.get("reference_source_column")
        report_lines.append(f"Cal:      ref col {ref_col}")
    elif exp_result.get("calibration_warning"):
        report_lines.append("Cal:      raw fallback")

    # GP prediction comparison (BO phase only)
    if surrogate is not None:
        try:
            pred_mean, pred_std = surrogate.predict(volumes.reshape(1, 3))
            pred = pred_mean[0]
            report_lines.append("")
            report_lines.append(f"GP predicted: ({pred[0]:.0f}, {pred[1]:.0f}, {pred[2]:.0f})")
            err = mean_rgb - pred
            report_lines.append(f"Pred error:   ({err[0]:+.0f}, {err[1]:+.0f}, {err[2]:+.0f})")
        except Exception:
            pass

    # Next step plan
    report_lines.append("")
    if len(all_dist) > 0:
        best_idx = int(np.argmin(all_dist))
        report_lines.append(f"Best so far: dist={all_dist[best_idx]:.1f} (iter {best_idx+1})")

    if next_volumes is not None:
        report_lines.append(f"Next: R={next_volumes[0]:.1f} G={next_volumes[1]:.1f} B={next_volumes[2]:.1f}")

    for i, line in enumerate(report_lines):
        axes[3].text(0, 0.95 - i * 0.08, line, transform=axes[3].transAxes,
                     fontsize=8, fontfamily="monospace", va="top")
    axes[3].set_title("Results & Plan", fontsize=9)

    plt.tight_layout()
    plt.show()
