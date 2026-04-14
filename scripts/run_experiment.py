#!/usr/bin/env python3
"""CLI entry point for running an active learning experiment.

Uses the plugin architecture as the canonical path: loads the task
plugin, derives config from it, and runs the experiment pipeline.

Usage::

    python -m scripts.run_experiment --config configs/color_mixing.yaml
    python -m scripts.run_experiment --config configs/color_mixing.yaml --skip-precheck
"""

import argparse
import sys

from src.tasks.color_mixing.plugin import ColorMixingPlugin
from src.preflight import run_device_precheck_from_config, load_or_create_grid_calibration
from src.pipeline import run_active_learning_loop


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning colorimetric assay optimization.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/color_mixing.yaml",
        help="Path to experiment YAML config (default: configs/color_mixing.yaml).",
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

    # ---- Load plugin and config through plugin ----
    plugin = ColorMixingPlugin()
    config = plugin.load_config(args.config)

    # Device precheck (uses raw YAML for preflight compatibility)
    if not args.skip_precheck:
        from src.preflight import load_config as load_raw_config

        raw_config = load_raw_config(args.config)
        report = run_device_precheck_from_config(raw_config)
        if not report.all_ok:
            print("Device precheck FAILED — aborting.")
            print(f"  Robot: {'OK' if report.robot.reachable else 'UNREACHABLE'}")
            print(f"  Cameras: {len(report.cameras)} found")
            sys.exit(1)
        print("Device precheck OK")

    # Load grid calibration if available
    grid = None
    try:
        from src.preflight import load_config as load_raw_config

        raw_config = load_raw_config(args.config)
        grid = load_or_create_grid_calibration(raw_config)
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
