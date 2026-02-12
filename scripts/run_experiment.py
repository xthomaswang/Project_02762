#!/usr/bin/env python3
"""CLI entry point for running an active learning experiment."""

import argparse
import sys
import os

# Allow running from project root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "--n-initial", type=int, default=5,
        help="Number of random initial experiments.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=20,
        help="Number of Bayesian optimization iterations.",
    )
    parser.add_argument(
        "--acquisition", type=str, default="EI", choices=["EI", "UCB"],
        help="Acquisition function kind.",
    )
    args = parser.parse_args()

    run_active_learning_loop(
        config_path=args.config,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acquisition_kind=args.acquisition,
    )


if __name__ == "__main__":
    main()
