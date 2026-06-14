"""Result schemas for experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class MachineSpecs:
    """Hardware and software specifications."""

    hostname: str
    cpu: dict  # model, physical_cores, total_cores, frequency_mhz
    memory_gb: float
    gpu: dict  # available, name, compute_capability, memory_gb
    os: dict  # system, release, version
    python: str
    torch: str
    numpy: str
    cuda_version: str | None


@dataclass
class CommonResult:
    """Standard results collected from every solver run."""

    schema_version: int = 1
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Instance info
    instance_path: str = ""
    instance_sha256: str = ""
    num_spins: int = 0
    family: str = ""

    # Execution info
    algorithm: str = ""
    seed: int = 0
    runtime_seconds: float = 0.0

    # Standard metrics (from SolverResult.metrics)
    metrics: dict = field(default_factory=dict)

    # Machine info
    machine_specs: MachineSpecs | None = None

    # Solver parameters used
    solver_parameters: dict = field(default_factory=dict)

    def to_file(self, path: Path) -> None:
        """Save to common.json."""
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: Path) -> CommonResult:
        """Load from common.json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct MachineSpecs
        machine_specs = None
        if data.get("machine_specs"):
            machine_specs = MachineSpecs(**data["machine_specs"])

        return cls(
            schema_version=data.get("schema_version", 1),
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            instance_path=data["instance_path"],
            instance_sha256=data["instance_sha256"],
            num_spins=data["num_spins"],
            family=data["family"],
            algorithm=data["algorithm"],
            seed=data["seed"],
            runtime_seconds=data["runtime_seconds"],
            metrics=data["metrics"],
            machine_specs=machine_specs,
            solver_parameters=data["solver_parameters"],
        )


@dataclass
class DiagnosticsResult:
    """Algorithm-specific diagnostic data (optional)."""

    schema_version: int = 1
    algorithm: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_file(self, path: Path) -> None:
        """Save to diagnostics.json."""
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: Path) -> DiagnosticsResult:
        """Load from diagnostics.json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            schema_version=data.get("schema_version", 1),
            algorithm=data["algorithm"],
            diagnostics=data["diagnostics"],
        )
