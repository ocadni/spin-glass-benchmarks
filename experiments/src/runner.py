"""Experiment runner for executing solver experiments."""

from __future__ import annotations

import hashlib
import re
import sys
import threading
import time
from pathlib import Path

from experiments.src.config import AlgorithmConfig, ExperimentConfig
from experiments.src.machine_specs import MachineSpecsCollector
from experiments.src.results import CommonResult, DiagnosticsResult, MachineSpecs


class ExperimentRunner:
    """Orchestrates experiment execution across algorithms and instances."""

    def __init__(self, experiment_root: Path | None = None):
        self.repo_root = self._find_repo_root()
        self.experiment_root = Path(experiment_root) if experiment_root else (self.repo_root / "experiments")

    def _find_repo_root(self) -> Path:
        """Find repository root (contains solvers_v2/)."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "solvers_v2").exists():
                return current
            current = current.parent
        raise RuntimeError("Could not find repository root (looking for solvers_v2/)")

    def run_experiment(self, config: ExperimentConfig) -> None:
        """Run full experiment across all algorithms and instances."""
        print(f"Starting experiment: {config.experiment_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Family: {config.family}")
        print(f"Algorithms: {', '.join(config.algorithms)}")
        print(f"Instances: {len(config.instances)}")

        # Collect machine specs once
        machine_specs = MachineSpecsCollector.collect()
        print(f"Machine: {machine_specs.hostname}")
        print(f"GPU: {machine_specs.gpu['name'] if machine_specs.gpu['available'] else 'None'}")

        for algorithm in config.algorithms:
            self._run_algorithm(config, algorithm, machine_specs)

    def _run_algorithm(
        self,
        experiment_config: ExperimentConfig,
        algorithm: str,
        machine_specs: MachineSpecs,
    ) -> None:
        """Run one algorithm across all instances."""
        algo_dir = (
            self.experiment_root
            / experiment_config.researcher
            / experiment_config.experiment_name
            / algorithm
        )
        config_path = algo_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config not found: {config_path}")

        algo_config = AlgorithmConfig.from_file(config_path)

        # Generate parameter combinations
        param_combinations = algo_config.generate_parameter_combinations()
        if not param_combinations:
            param_combinations = [algo_config.solver_parameters]

        total_runs = (
            len(experiment_config.instances)
            * len(algo_config.seeds)
            * len(param_combinations)
        )

        print(f"\n{algorithm}: {total_runs} runs")
        run_count = 0
        start_time = time.time()
        run_times = []  # Track individual run times for ETA

        for instance_path in experiment_config.instances:
            for seed in algo_config.seeds:
                for params in param_combinations:
                    run_count += 1
                    instance_name = Path(instance_path).name

                    # Progress bar
                    self._print_progress(
                        algorithm=algorithm,
                        run_count=run_count,
                        total_runs=total_runs,
                        instance_name=instance_name,
                        seed=seed,
                        elapsed_time=time.time() - start_time,
                        run_times=run_times,
                    )

                    try:
                        run_start = time.time()

                        # Create shared state for solver progress
                        solver_progress = {"step": 0, "total": 0, "label": ""}

                        # Start progress updater thread
                        stop_progress = threading.Event()
                        progress_thread = threading.Thread(
                            target=self._update_progress_during_run,
                            args=(
                                algorithm,
                                run_count,
                                total_runs,
                                instance_name,
                                seed,
                                start_time,
                                run_start,
                                run_times,
                                stop_progress,
                                solver_progress,
                            ),
                            daemon=True,
                        )
                        progress_thread.start()

                        # Create progress callback for solver
                        def progress_callback(step: int, total: int, label: str):
                            solver_progress["step"] = step
                            solver_progress["total"] = total
                            solver_progress["label"] = label

                        self._run_single(
                            family=experiment_config.family,
                            algorithm=algorithm,
                            instance_path=instance_path,
                            parameters=params,
                            seed=seed,
                            machine_specs=machine_specs,
                            output_diagnostics=algo_config.output_diagnostics,
                            run_dir=algo_dir / "runs",
                            progress_callback=progress_callback,
                        )

                        # Stop progress updater
                        stop_progress.set()
                        progress_thread.join(timeout=0.5)

                        run_duration = time.time() - run_start
                        run_times.append(run_duration)

                        # Clear line and show completion
                        print(f"\r  [{run_count}/{total_runs}] ✓ {instance_name}, seed={seed} ({run_duration:.1f}s)" + " " * 20)
                    except Exception as e:
                        stop_progress.set()
                        print(f"\r  [{run_count}/{total_runs}] ✗ {instance_name}, seed={seed} - ERROR: {e}" + " " * 20)

        # Final summary
        total_time = time.time() - start_time
        avg_time = sum(run_times) / len(run_times) if run_times else 0
        print(f"\n  Algorithm completed in {self._format_time(total_time)} (avg {avg_time:.1f}s per run)")

    def _update_progress_during_run(
        self,
        algorithm: str,
        run_count: int,
        total_runs: int,
        instance_name: str,
        seed: int,
        start_time: float,
        run_start: float,
        run_times: list,
        stop_event: threading.Event,
        solver_progress: dict,
    ) -> None:
        """Update progress display while instance is running."""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        while not stop_event.is_set():
            spinner = spinners[spinner_idx % len(spinners)]
            spinner_idx += 1

            elapsed_time = time.time() - start_time
            run_elapsed = time.time() - run_start

            # Calculate ETA
            if run_times:
                avg_time = sum(run_times) / len(run_times)
                remaining_runs = total_runs - run_count + 1
                eta_seconds = avg_time * remaining_runs
                eta_str = self._format_time(eta_seconds)
            else:
                eta_str = "calculating..."

            elapsed_str = self._format_time(elapsed_time)
            run_elapsed_str = self._format_time(run_elapsed)
            remaining = total_runs - run_count + 1

            # Build progress line with solver progress if available
            if solver_progress["total"] > 0:
                solver_info = f"Step {solver_progress['step']}/{solver_progress['total']} ({solver_progress['label']})"
            else:
                solver_info = "initializing"

            progress = (
                f"\r  [{run_count}/{total_runs}] {spinner} {algorithm} | "
                f"{instance_name}, seed={seed} | "
                f"{solver_info} | "
                f"Running: {run_elapsed_str} | Elapsed: {elapsed_str} | ETA: {eta_str} | "
                f"Remaining: {remaining}"
            )

            # Print without newline, flush immediately
            print(progress, end="", flush=True)

            # Update every 0.1 seconds
            stop_event.wait(0.1)

    def _print_progress(
        self,
        algorithm: str,
        run_count: int,
        total_runs: int,
        instance_name: str,
        seed: int,
        elapsed_time: float,
        run_times: list,
    ) -> None:
        """Display initial progress line (used before run starts)."""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinners[run_count % len(spinners)]

        # Calculate ETA
        if run_times:
            avg_time = sum(run_times) / len(run_times)
            remaining_runs = total_runs - run_count + 1
            eta_seconds = avg_time * remaining_runs
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        elapsed_str = self._format_time(elapsed_time)
        remaining = total_runs - run_count + 1

        # Build progress line
        progress = (
            f"\r  [{run_count}/{total_runs}] {spinner} {algorithm} | "
            f"{instance_name}, seed={seed} | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"Remaining: {remaining}"
        )

        # Print without newline, flush immediately
        print(progress, end="", flush=True)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _run_single(
        self,
        family: str,
        algorithm: str,
        instance_path: str,
        parameters: dict,
        seed: int,
        machine_specs: MachineSpecs,
        output_diagnostics: bool,
        run_dir: Path,
        progress_callback=None,
    ) -> None:
        """Run single instance and save results."""
        # Get family adapter
        runner = self._get_family_runner(family)

        # Resolve instance path
        full_instance_path = self.repo_root / instance_path

        # Compute instance hash and identifier
        instance_sha256 = self._compute_sha256(full_instance_path)
        num_spins = self._extract_num_spins(full_instance_path)
        instance_id = self._make_instance_id(full_instance_path)

        # Create run directory
        run_folder = run_dir / f"{instance_id}_seed{seed}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Run solver with timing
        start_time = time.time()

        if family == "rrg":
            result = runner.run_pairwise_case(
                algorithm, full_instance_path, parameters, seed, progress_callback
            )
        else:
            result = runner.run_baseline_case(
                algorithm, full_instance_path, parameters, seed, progress_callback
            )

        runtime_seconds = time.time() - start_time

        # Save common results
        common_result = CommonResult(
            instance_path=instance_path,
            instance_sha256=instance_sha256,
            num_spins=num_spins,
            family=family,
            algorithm=algorithm,
            seed=seed,
            runtime_seconds=runtime_seconds,
            metrics=result.metrics,
            machine_specs=machine_specs,
            solver_parameters=parameters,
        )
        common_result.to_file(run_folder / "common.json")

        # Save diagnostics if available and requested
        if output_diagnostics and hasattr(result, "diagnostics") and result.diagnostics:
            diagnostics_result = DiagnosticsResult(
                algorithm=algorithm,
                diagnostics=result.diagnostics,
            )
            diagnostics_result.to_file(run_folder / "diagnostics.json")

    def _get_family_runner(self, family: str):
        """Get family adapter module."""
        if family == "sk":
            from solvers_v2.families import sk

            return sk
        elif family == "ea2d":
            from solvers_v2.families.ea import ea2d

            return ea2d
        elif family == "ea3d":
            from solvers_v2.families.ea import ea3d

            return ea3d
        elif family == "rrg":
            from solvers_v2.families import rrg

            return rrg
        else:
            raise ValueError(f"Unknown family: {family}")

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_num_spins(self, path: Path) -> int:
        """Extract number of spins from instance filename."""
        match = re.search(r"_N(\d+)_", path.name)
        if match:
            return int(match.group(1))
        # Fallback: read from file header
        with open(path) as f:
            first_line = f.readline()
            if first_line.startswith("#"):
                match = re.search(r"N=(\d+)", first_line)
                if match:
                    return int(match.group(1))
        raise ValueError(f"Cannot extract num_spins from {path}")

    def _make_instance_id(self, path: Path) -> str:
        """Create short identifier from instance path."""
        # e.g., "instances/sk/N50/sk_couplings_N50_J0_seed1051730.txt"
        # -> "N50_seed1051730"
        stem = path.stem
        n_match = re.search(r"N(\d+)", stem)
        seed_match = re.search(r"seed(\d+)", stem)
        if n_match and seed_match:
            return f"N{n_match.group(1)}_seed{seed_match.group(1)}"
        return stem[:50]  # Fallback
