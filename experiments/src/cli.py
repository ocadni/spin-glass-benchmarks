#!/usr/bin/env python3
"""Command-line interface for running experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.src.config import ExperimentConfig
from experiments.src.runner import ExperimentRunner


def main():
    """Run experiment from command line."""
    parser = argparse.ArgumentParser(description="Run spin glass solver experiment")
    parser.add_argument(
        "experiment_meta",
        type=Path,
        help="Path to experiment_meta.json",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Root directory for experiments (default: repository root)",
    )

    args = parser.parse_args()

    config = ExperimentConfig.from_file(args.experiment_meta)
    runner = ExperimentRunner(args.experiment_root)
    runner.run_experiment(config)


if __name__ == "__main__":
    main()
