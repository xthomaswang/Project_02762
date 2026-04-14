"""
Thin orchestration layer for active learning experiments.

All color-mixing-specific logic (deck layout, control/experiment well
mapping, calibration) lives in :mod:`src.tasks.color_mixing`.  This
module only wires together: config → robot → task API → ML loop.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from openot2 import OT2Client, OT2Operations

from src.ml import (
    AcquisitionFunction,
    create_surrogate,
    sample_simplex,
)
from src.color.metrics import color_distance
from src.color.calibration import (
    ColumnSummary,
    RuntimeReference,
    write_measurements_csv,
)
from src.vision.geometry import AnyGrid, load_grid_calibration
from src.robot import ParkPosition, resolve_park_position, park_for_imaging

from src.tasks.color_mixing.config import ColorMixingConfig
from src.tasks.color_mixing.plugin import ColorMixingPlugin
from src.tasks.color_mixing.api import (
    build_iteration_steps,
    execute_steps,
    analyze_capture,
    fit_observation,
    build_robot_calibration_profile,
    CaptureResult,
    DeckProfile,
)


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

    Delegates all assay-specific logic to src.tasks.color_mixing.

    Args:
        config_path: Path to color mixing YAML config.
        robot: OT2Client instance (created from config if None).
        n_initial: Override config's experiment.n_initial.
        n_iterations: Override config's experiment.n_optimization.
        acquisition_kind: Override config's ml.acquisition.
        start_column: Column index to start from (0-based).
        seed_X: (n, 3) prior volume observations.
        seed_Y: (n, 3) prior RGB observations.
        mode: "full" or "quick".
        grid_calibration: Optional PlateGrid for well extraction.
    """
    plugin = ColorMixingPlugin()
    cfg = plugin.load_config(config_path)

    # --- Parse overrides ---
    target_rgb = np.array(cfg.target_color, dtype=float)
    total_volume = cfg.total_volume_ul
    convergence_threshold = cfg.convergence_threshold
    seed = cfg.seed

    bounds = np.array([cfg.volume_bounds_min, cfg.volume_bounds_max], dtype=float)

    n_initial_explicit = n_initial
    n_initial = n_initial or cfg.experiment.n_initial
    n_opt = n_iterations or cfg.experiment.n_optimization
    max_per_plate = cfg.experiment.max_per_plate
    batch_mode = cfg.experiment.batch_mode

    acq_kind = acquisition_kind or cfg.ml.acquisition
    distance_metric = cfg.ml.distance_metric
    base_dir = cfg.output_base_dir

    # --- Quick mode overrides ---
    quick_overrides: Dict[str, Any] = {}
    if mode == "quick":
        if n_initial_explicit is None:
            n_initial = 5
        quick_overrides["rinse_cycles"] = 1
        quick_overrides["mix_cycles"] = 2
        quick_overrides["skip_controls_after_first"] = True
        print(f"  [QUICK MODE] n_initial={n_initial}, rinse_cycles=1, "
              f"mix_cycles=2, skip controls after col 1")
    else:
        print(f"  [FULL MODE] Using config values as-is")

    # --- Resolve grid calibration ---
    if grid_calibration is None:
        grid_path = cfg.resolve_path(cfg.calibration.grid_path)
        if os.path.exists(grid_path):
            grid_calibration = load_grid_calibration(grid_path)

    # --- Initialize robot ---
    if robot is None:
        robot = OT2Client(robot_ip=cfg.robot_ip, port=cfg.robot_port)

    # --- Initialize ML ---
    surrogate = create_surrogate(
        model_type=cfg.ml.model, bounds=bounds, total_volume=total_volume,
    )
    acq = AcquisitionFunction(
        kind=acq_kind,
        target_rgb=target_rgb,
        total_volume=total_volume,
        distance_metric=distance_metric,
    )

    # --- Setup robot run ---
    rinse_cycles = quick_overrides.get("rinse_cycles", cfg.cleaning_protocol.rinse_cycles)
    ops = OT2Operations(
        client=robot,
        rinse_cycles=rinse_cycles,
        rinse_volume=cfg.cleaning_protocol.rinse_volume_ul,
    )

    robot.create_run()
    profile = build_robot_calibration_profile(cfg, robot)

    exp_name = "color_mixing"
    run_id = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- Resolve imaging park position ---
    try:
        park_pos = resolve_park_position(
            {"imaging": {"park": {
                "slot": cfg.park.slot,
                "well": cfg.park.well,
                "offset": list(cfg.park.offset),
            }}},
            robot,
        )
    except ValueError:
        park_pos = None

    # --- Data storage ---
    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []
    all_dist: List[float] = []
    log_rows: List[Dict] = []
    all_summaries: List[ColumnSummary] = []
    cached_reference: Optional[RuntimeReference] = None

    # --- Load seed data ---
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

        result = _run_single_iteration(
            cfg, ops, profile, volumes, col_idx, run_id, base_dir,
            skip_controls=skip_controls,
            mix_cycles_override=quick_overrides.get("mix_cycles"),
            grid=grid_calibration,
            park_pos=park_pos,
            cached_reference=cached_reference,
            is_quick=(mode == "quick"),
        )
        observed_rgb = result.mean_rgb
        dist = color_distance(observed_rgb, target_rgb, distance_metric)

        all_X.append(volumes)
        all_Y.append(observed_rgb)
        all_dist.append(dist)
        if result.summary is not None:
            all_summaries.append(result.summary)
        if result.runtime_reference is not None and cached_reference is None:
            cached_reference = result.runtime_reference
        _log_iteration(log_rows, i, col_idx, volumes, observed_rgb, dist, "random")

        _display_iteration(
            iteration=i, phase="random", col_idx=col_idx,
            volumes=volumes, result=result, target_rgb=target_rgb,
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

        result = _run_single_iteration(
            cfg, ops, profile, volumes, col_idx, run_id, base_dir,
            skip_controls=skip_controls,
            mix_cycles_override=quick_overrides.get("mix_cycles"),
            grid=grid_calibration,
            park_pos=park_pos,
            cached_reference=cached_reference,
            is_quick=(mode == "quick"),
        )
        observed_rgb = result.mean_rgb
        dist = color_distance(observed_rgb, target_rgb, distance_metric)

        all_X.append(volumes)
        all_Y.append(observed_rgb)
        all_dist.append(dist)
        if result.summary is not None:
            all_summaries.append(result.summary)
        if result.runtime_reference is not None and cached_reference is None:
            cached_reference = result.runtime_reference
        _log_iteration(log_rows, n_initial + i, col_idx, volumes, observed_rgb, dist, "bo")

        fit_observation(surrogate, volumes, observed_rgb)

        # Preview next suggestion
        next_volumes = None
        if i < n_opt - 1 and dist >= convergence_threshold:
            try:
                next_candidates = acq.suggest(surrogate, n_candidates=1)
                next_volumes = next_candidates[0]
            except Exception:
                pass

        _display_iteration(
            iteration=n_initial + i, phase="bo", col_idx=col_idx,
            volumes=volumes, result=result, target_rgb=target_rgb,
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

    if all_summaries:
        results_dir = os.path.join(run_folder, "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meas_path = os.path.join(results_dir, f"measurements_{ts}.csv")
        write_measurements_csv(meas_path, all_summaries)
        print(f"[PIPELINE] Measurements saved: {meas_path}")

    return np.array(all_X), np.array(all_Y), np.array(all_dist)


# ======================================================================
# Single iteration (thin wrapper over task API)
# ======================================================================

def _run_single_iteration(
    cfg: ColorMixingConfig,
    ops: OT2Operations,
    profile: DeckProfile,
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
) -> CaptureResult:
    """Execute one column: dispense → settle → image → extract."""
    settle_time = cfg.experiment.settle_time_seconds
    robot = ops.client

    well_col = col_idx + 1

    # ---- Dispense (steps from task API) ----
    if skip_controls:
        print(f"  [Phase 1] Controls SKIPPED (quick mode)")
    else:
        print(f"  [Phase 1] Controls (column {well_col})")

    print(f"  [Phase 2] Experiments (column {well_col})")

    steps = build_iteration_steps(
        cfg, volumes, col_idx,
        skip_controls=skip_controls,
        mix_cycles_override=mix_cycles_override,
    )
    execute_steps(steps, profile, ops)

    # ---- Settle + Park + Image ----
    settle_start = time.monotonic()

    if park_pos is not None:
        print(f"  [Phase 3] Parking + settling ({settle_time}s) + imaging")
        park_for_imaging(robot, park_pos)
    else:
        print(f"  [Phase 3] Homing + settling ({settle_time}s) + imaging")
        robot.home()

    elapsed = time.monotonic() - settle_start
    remaining = max(0, settle_time - elapsed)
    if remaining > 0:
        time.sleep(remaining)

    image_path = _capture_plate_image(cfg, run_id, col_idx, base_dir)

    # ---- Analyze capture (task API) ----
    result = analyze_capture(
        image_path=image_path,
        cfg=cfg,
        col_idx=col_idx,
        grid=grid,
        skip_controls=skip_controls,
        cached_reference=cached_reference,
        is_quick=is_quick,
    )

    if result.calibration_warning is not None:
        print(f"    [WARNING] {result.calibration_warning}")

    print(f"    Mean RGB: ({result.mean_rgb[0]:.0f}, {result.mean_rgb[1]:.0f}, {result.mean_rgb[2]:.0f}) "
          f"± ({result.std_rgb[0]:.1f}, {result.std_rgb[1]:.1f}, {result.std_rgb[2]:.1f})")

    return result


# ======================================================================
# Imaging
# ======================================================================

def _capture_plate_image(
    cfg: ColorMixingConfig,
    run_id: str,
    col_idx: int,
    base_dir: str,
) -> str:
    """Capture a plate image using the overhead camera."""
    from openot2.vision import USBCamera

    camera = USBCamera(
        camera_id=cfg.camera.device_id,
        width=cfg.camera.width,
        height=cfg.camera.height,
        warmup_frames=cfg.camera.warmup_frames,
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
# Iteration display
# ======================================================================

def _display_iteration(
    iteration: int,
    phase: str,
    col_idx: int,
    volumes: np.ndarray,
    result: CaptureResult,
    target_rgb: np.ndarray,
    all_dist: List[float],
    surrogate=None,
    next_volumes: Optional[np.ndarray] = None,
    distance_metric: str = "rgb_euclidean",
):
    import matplotlib.pyplot as plt
    import cv2

    mean_rgb = result.mean_rgb
    std_rgb = result.std_rgb
    exp_rgb = result.calibrated_experiment_rgb
    ctrl_rgb = result.control_rgb
    image_path = result.image_path
    dist = color_distance(mean_rgb, target_rgb, distance_metric)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5),
                             gridspec_kw={"width_ratios": [3, 1, 2, 2]})
    fig.suptitle(
        f"{'Random' if phase == 'random' else 'BO'} #{iteration+1} — "
        f"Column {col_idx+1} — Distance: {dist:.1f}",
        fontsize=12, fontweight="bold",
    )

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img)
        axes[0].set_title("Captured plate", fontsize=9)
    axes[0].axis("off")

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

    obs_norm = np.clip(mean_rgb / 255.0, 0, 1)
    tgt_norm = np.clip(target_rgb / 255.0, 0, 1)
    axes[2].barh(1, 1, color=obs_norm, edgecolor="black", linewidth=1)
    axes[2].barh(0, 1, color=tgt_norm, edgecolor="black", linewidth=1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Target", "Observed"], fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_title("Color comparison", fontsize=9)
    axes[2].text(0.5, 1, f"({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})",
                 ha="center", va="center", fontsize=8,
                 color="white" if np.mean(obs_norm) < 0.5 else "black")
    axes[2].text(0.5, 0, f"({target_rgb[0]:.0f},{target_rgb[1]:.0f},{target_rgb[2]:.0f})",
                 ha="center", va="center", fontsize=8,
                 color="white" if np.mean(tgt_norm) < 0.5 else "black")

    axes[3].axis("off")
    report_lines = [
        f"Volumes: R={volumes[0]:.1f} G={volumes[1]:.1f} B={volumes[2]:.1f}",
        f"Observed: ({mean_rgb[0]:.0f}, {mean_rgb[1]:.0f}, {mean_rgb[2]:.0f})",
        f"Std dev:  ({std_rgb[0]:.1f}, {std_rgb[1]:.1f}, {std_rgb[2]:.1f})",
        f"Target:   ({target_rgb[0]:.0f}, {target_rgb[1]:.0f}, {target_rgb[2]:.0f})",
        f"Distance: {dist:.1f}",
    ]
    if result.used_calibration:
        ref_col = result.reference_source_column
        report_lines.append(f"Cal:      ref col {ref_col}")
    elif result.calibration_warning:
        report_lines.append("Cal:      raw fallback")

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
