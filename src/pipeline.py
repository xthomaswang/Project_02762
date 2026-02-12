"""
End-to-end active learning loop for colorimetric assay optimization.

Orchestrates: ML suggests volumes → robot dispenses → camera captures →
color extracted → ML updates → repeat.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.robot import OT2Robot
from src.ml import SurrogateModel, AcquisitionFunction, color_distance


def run_active_learning_loop(
    config_path: str,
    robot: Optional[OT2Robot] = None,
    n_initial: int = 5,
    n_iterations: int = 20,
    acquisition_kind: str = "EI",
):
    """
    Main active learning loop.

    Args:
        config_path: Path to experiment YAML config.
        robot: OT2Robot instance (created from config if None).
        n_initial: Number of random initial experiments.
        n_iterations: Number of BO iterations after initial batch.
        acquisition_kind: 'EI' or 'UCB'.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    target_rgb = np.array(config["target_color"])  # e.g. [128, 0, 255]
    bounds = np.array(config.get("volume_bounds", [[0, 0, 0], [300, 300, 300]]),
                      dtype=float)

    if robot is None:
        robot_cfg = config.get("robot", {})
        robot = OT2Robot(
            ip=robot_cfg.get("ip", "169.254.8.56"),
            port=robot_cfg.get("port", 31950),
        )

    surrogate = SurrogateModel(bounds=bounds)
    acq = AcquisitionFunction(kind=acquisition_kind)

    # -- Phase 1: Initial random experiments --
    print(f"\n[PIPELINE] Phase 1: {n_initial} random experiments")
    rng = np.random.default_rng(seed=config.get("seed", 42))
    X_init = rng.uniform(bounds[0], bounds[1], size=(n_initial, bounds.shape[1]))
    Y_init = np.array([_run_single_experiment(robot, config, volumes, target_rgb)
                       for volumes in X_init])

    surrogate.fit(X_init, Y_init)
    all_X = X_init.copy()
    all_Y = Y_init.copy()

    # -- Phase 2: Bayesian optimization iterations --
    print(f"\n[PIPELINE] Phase 2: {n_iterations} BO iterations")
    for i in range(n_iterations):
        print(f"\n--- Iteration {i + 1}/{n_iterations} ---")
        candidates = acq.suggest(surrogate, n_candidates=1)
        y_new = np.array([_run_single_experiment(robot, config, candidates[0], target_rgb)])
        surrogate.update(candidates, y_new)
        all_X = np.vstack([all_X, candidates])
        all_Y = np.append(all_Y, y_new)

        best_idx = int(np.argmin(all_Y))
        print(f"[PIPELINE] Best so far: distance={all_Y[best_idx]:.4f}, "
              f"volumes={all_X[best_idx]}")

    print("\n[PIPELINE] Active learning complete.")
    print(f"[PIPELINE] Best result: distance={all_Y[np.argmin(all_Y)]:.4f}")
    return all_X, all_Y


def _run_single_experiment(robot: OT2Robot, config: Dict[str, Any],
                           volumes: np.ndarray,
                           target_rgb: np.ndarray) -> float:
    """
    Execute one experiment: dispense volumes, capture image, extract color,
    return distance to target.

    This is a skeleton — the actual dispensing + camera + color extraction
    should be filled in when the full hardware pipeline is integrated.
    """
    print(f"[PIPELINE] Running experiment with volumes: {volumes}")

    # TODO: Integrate with protocol.execute_protocol or direct robot commands
    #   1. robot.aspirate / robot.dispense for each dye channel
    #   2. Wait for mixing
    #   3. Capture image with vision.Camera
    #   4. Extract RGB from well region
    #   5. Compute distance

    # Placeholder: return random distance (replace with real measurement)
    observed_rgb = np.clip(volumes[:3], 0, 255)  # placeholder
    dist = color_distance(observed_rgb, target_rgb)
    print(f"[PIPELINE] Observed distance: {dist:.4f}")
    return dist
