#!/usr/bin/env python3
"""CLI entry point for running an active learning experiment."""

import argparse
import sys
import os

# Allow running from project root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preflight import load_config, run_device_precheck_from_config, load_or_create_grid_calibration
from src.pipeline import run_active_learning_loop


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning colorimetric assay optimization.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--n-initial", type=int, default=None,
        help="Number of random initial experiments (overrides config).",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=None,
        help="Number of Bayesian optimization iterations (overrides config).",
    )
    parser.add_argument(
        "--acquisition", type=str, default="EI", choices=["EI", "UCB"],
        help="Acquisition function kind.",
    )
    parser.add_argument(
        "--mode", type=str, default="full", choices=["full", "quick"],
        help="Run mode: 'full' (default, safe) or 'quick' (fewer rinses, skip controls after col 1).",
    )
    parser.add_argument(
        "--skip-precheck", action="store_true",
        help="Skip device precheck.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Device precheck
    if not args.skip_precheck:
        report = run_device_precheck_from_config(config)
        if not report.all_ok:
            print("Device precheck FAILED — aborting.")
            print(f"  Robot: {'OK' if report.robot.reachable else 'UNREACHABLE'}")
            print(f"  Cameras: {len(report.cameras)} found")
            sys.exit(1)
        print("Device precheck OK")

    # Load grid calibration if available
    grid = None
    try:
        grid = load_or_create_grid_calibration(config)
    except ValueError:
        print("No grid calibration found — using default grid")

    run_active_learning_loop(
        config_path=args.config,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acquisition_kind=args.acquisition,
        grid_calibration=grid,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
