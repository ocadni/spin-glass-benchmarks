"""Configuration schemas for experiments."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ParameterSweep:
    """Parameter sweep configuration."""

    sweep_type: str  # "grid" or "list"
    parameters: dict[str, list]  # param_name -> values to sweep

    def __post_init__(self):
        if self.sweep_type not in ("grid", "list"):
            raise ValueError(f"sweep_type must be 'grid' or 'list', got {self.sweep_type!r}")


@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm in an experiment."""

    algorithm: str
    seeds: list[int]
    solver_parameters: dict
    parameter_sweep: ParameterSweep | None = None
    output_diagnostics: bool = True

    @classmethod
    def from_file(cls, path: Path) -> AlgorithmConfig:
        """Load from config.json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        parameter_sweep = None
        if data.get("parameter_sweep"):
            parameter_sweep = ParameterSweep(**data["parameter_sweep"])

        return cls(
            algorithm=data["algorithm"],
            seeds=data["seeds"],
            solver_parameters=data["solver_parameters"],
            parameter_sweep=parameter_sweep,
            output_diagnostics=data.get("output_diagnostics", True),
        )

    def to_file(self, path: Path) -> None:
        """Save to config.json."""
        data = {
            "schema_version": 1,
            "algorithm": self.algorithm,
            "seeds": self.seeds,
            "solver_parameters": self.solver_parameters,
            "parameter_sweep": None,
            "output_diagnostics": self.output_diagnostics,
        }

        if self.parameter_sweep:
            data["parameter_sweep"] = {
                "sweep_type": self.parameter_sweep.sweep_type,
                "parameters": self.parameter_sweep.parameters,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate_parameter_combinations(self) -> list[dict]:
        """Generate all parameter combinations for sweeps."""
        if not self.parameter_sweep:
            return []

        base_params = self.solver_parameters.copy()
        sweep_params = self.parameter_sweep.parameters

        if self.parameter_sweep.sweep_type == "grid":
            # Cartesian product of all parameter values
            keys = list(sweep_params.keys())
            values = [sweep_params[k] for k in keys]
            combinations = []

            for combo in itertools.product(*values):
                params = base_params.copy()
                for key, value in zip(keys, combo):
                    params[key] = value
                combinations.append(params)

            return combinations

        elif self.parameter_sweep.sweep_type == "list":
            # Parallel iteration (all lists must be same length)
            lengths = {len(v) for v in sweep_params.values()}
            if len(lengths) != 1:
                raise ValueError(
                    "list sweep requires all parameter lists to have same length"
                )

            combinations = []
            num_combinations = lengths.pop()

            for i in range(num_combinations):
                params = base_params.copy()
                for key, values in sweep_params.items():
                    params[key] = values[i]
                combinations.append(params)

            return combinations

        return []


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    researcher: str
    experiment_name: str
    description: str
    family: str  # "sk", "ea2d", "ea3d", "rrg"
    instances: list[str]  # Paths relative to repo root
    algorithms: list[str]
    created_at: str  # ISO timestamp
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if self.family not in ("sk", "ea2d", "ea3d", "rrg"):
            raise ValueError(f"family must be one of sk/ea2d/ea3d/rrg, got {self.family!r}")

    @classmethod
    def from_file(cls, path: Path) -> ExperimentConfig:
        """Load from experiment_meta.json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            researcher=data["researcher"],
            experiment_name=data["experiment_name"],
            description=data["description"],
            family=data["family"],
            instances=data["instances"],
            algorithms=data["algorithms"],
            created_at=data["created_at"],
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )

    def to_file(self, path: Path) -> None:
        """Save to experiment_meta.json."""
        data = {
            "schema_version": 1,
            "researcher": self.researcher,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "created_at": self.created_at,
            "family": self.family,
            "instances": self.instances,
            "algorithms": self.algorithms,
            "tags": self.tags,
            "notes": self.notes,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def create_timestamp() -> str:
        """Create ISO timestamp for current time."""
        return datetime.utcnow().isoformat() + "Z"
