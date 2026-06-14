from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "tests_data" / "solver_baselines"
GENERATORS_DIR = ROOT / "generators"

if str(GENERATORS_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATORS_DIR))

from pairwise_io import load_pairwise_instance  # noqa: E402


IMPORT_MODULES_TO_CLEAR = (
    "geometry",
    "utilities",
    "data_loads",
    "monte_carlo",
    "made",
    "global_steps",
)

DEFAULT_SEED = 1729
TOLERANCE = 1e-7


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    family: str
    algorithm: str
    solver_file: Path
    fixture: Path
    seed: int
    parameters: dict[str, Any]
    runner: Callable[[Any, "BaselineSpec"], tuple[Any, Any]]
    matrix_mode: str

    @property
    def baseline_path(self) -> Path:
        return BASELINE_DIR / f"{self.baseline_id}.json"


def _path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def baseline_specs() -> list[BaselineSpec]:
    sk_fixture = _path(
        "tests_data",
        "instances",
        "sk",
        "N50",
        "sk_couplings_N50_J0_seed1051730.txt",
    )
    ea2d_fixture = _path(
        "tests_data",
        "instances",
        "ea2d",
        "N100",
        "ea2d_couplings_N100_J0_seed2011730.txt",
    )
    ea3d_fixture = _path(
        "tests_data",
        "instances",
        "ea3d",
        "N512",
        "ea3d_couplings_N512_J0_seed3009730.txt",
    )

    sk_dir = _path("solvers", "System specific solvers", "SK", "Modern", "optimization")
    ea_dir = _path("solvers", "System specific solvers", "EA3D", "Modern", "optimization")

    annealing_params = {
        "pop_size": 8,
        "MCsteps": 2,
        "Tstart": 1.92,
        "Tend": 0.1,
        "num_temps": 4,
        "schedule": "linearT",
        "high_temp_thermalization_steps": 1,
    }
    greedy_params = {
        "pop_size": 8,
    }
    pt_params = {
        "MCsteps": 2,
        "swap_interval": 1,
        "Tstart": 1.92,
        "Tend": 0.1,
        "num_temps": 4,
        "schedule": "linearT",
        "high_temp_thermalization_steps": 1,
    }

    return [
        BaselineSpec(
            baseline_id="sk_simulated_annealing",
            family="sk",
            algorithm="simulated_annealing",
            solver_file=sk_dir / "simulated_annealing_SK.py",
            fixture=sk_fixture,
            seed=DEFAULT_SEED,
            parameters=dict(annealing_params),
            runner=_run_sk_simulated_annealing,
            matrix_mode="sk_legacy_upper",
        ),
        BaselineSpec(
            baseline_id="sk_population_annealing",
            family="sk",
            algorithm="population_annealing",
            solver_file=sk_dir / "population_annealing_SK.py",
            fixture=sk_fixture,
            seed=DEFAULT_SEED,
            parameters=dict(annealing_params),
            runner=_run_sk_population_annealing,
            matrix_mode="sk_legacy_upper",
        ),
        BaselineSpec(
            baseline_id="sk_parallel_tempering",
            family="sk",
            algorithm="parallel_tempering",
            solver_file=sk_dir / "PT_SK.py",
            fixture=sk_fixture,
            seed=DEFAULT_SEED,
            parameters=dict(pt_params),
            runner=_run_sk_parallel_tempering,
            matrix_mode="sk_legacy_upper",
        ),
        BaselineSpec(
            baseline_id="sk_greedy",
            family="sk",
            algorithm="greedy",
            solver_file=sk_dir / "greedy_SK.py",
            fixture=sk_fixture,
            seed=DEFAULT_SEED,
            parameters=dict(greedy_params),
            runner=_run_sk_greedy,
            matrix_mode="sk_legacy_upper",
        ),
        BaselineSpec(
            baseline_id="ea2d_simulated_annealing",
            family="ea2d",
            algorithm="simulated_annealing",
            solver_file=ea_dir / "simulated_annealing.py",
            fixture=ea2d_fixture,
            seed=DEFAULT_SEED,
            parameters={**annealing_params, "L": 10, "dimension": "2d"},
            runner=_run_ea_simulated_annealing,
            matrix_mode="symmetric",
        ),
        BaselineSpec(
            baseline_id="ea3d_simulated_annealing",
            family="ea3d",
            algorithm="simulated_annealing",
            solver_file=ea_dir / "simulated_annealing.py",
            fixture=ea3d_fixture,
            seed=DEFAULT_SEED,
            parameters={**annealing_params, "L": 8, "dimension": "3d"},
            runner=_run_ea_simulated_annealing,
            matrix_mode="symmetric",
        ),
        BaselineSpec(
            baseline_id="ea3d_population_annealing",
            family="ea3d",
            algorithm="population_annealing",
            solver_file=ea_dir / "population_annealing.py",
            fixture=ea3d_fixture,
            seed=DEFAULT_SEED,
            parameters={**annealing_params, "L": 8, "dimension": "3d"},
            runner=_run_ea_population_annealing,
            matrix_mode="symmetric",
        ),
    ]


def collect_baseline(spec: BaselineSpec) -> dict[str, Any]:
    module = load_old_solver_module(spec)
    set_reproducible_seed(spec.seed)

    with force_cpu_torch(), redirect_stdout(io.StringIO()):
        schedule, observ = spec.runner(module, spec)

    min_energy = _float_history(observ.get_observable_history("min_energy"))
    mean_energy = _float_history(observ.get_observable_history("mean_energy"))

    return {
        "schema_version": 1,
        "baseline_id": spec.baseline_id,
        "family": spec.family,
        "algorithm": spec.algorithm,
        "solver_file": _relative(spec.solver_file),
        "fixture": _relative(spec.fixture),
        "fixture_sha256": _file_sha256(spec.fixture),
        "matrix_mode": spec.matrix_mode,
        "environment": environment_metadata(),
        "seed_values": {
            "python_random": spec.seed,
            "numpy": spec.seed,
            "torch": spec.seed,
        },
        "parameters": dict(spec.parameters),
        "schedule_length": len(schedule) if schedule is not None else None,
        "metrics": {
            "min_energy": min_energy,
            "mean_energy": mean_energy,
            "final_min_energy": min_energy[-1],
            "final_mean_energy": mean_energy[-1],
            "best_min_energy": min(min_energy),
        },
    }


def write_all_baselines() -> list[Path]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in baseline_specs():
        data = collect_baseline(spec)
        spec.baseline_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(spec.baseline_path)
    return written


def compare_with_baseline(spec: BaselineSpec) -> None:
    if not spec.baseline_path.exists():
        raise AssertionError(
            f"missing solver baseline fixture: {_relative(spec.baseline_path)}; "
            "run scripts/generate_solver_baselines.py"
        )

    expected = json.loads(spec.baseline_path.read_text(encoding="utf-8"))
    observed = collect_baseline(spec)

    assert observed["fixture_sha256"] == expected["fixture_sha256"]
    assert observed["schedule_length"] == expected["schedule_length"]
    assert observed["parameters"] == expected["parameters"]
    assert observed["seed_values"] == expected["seed_values"]

    for metric_name in ("min_energy", "mean_energy"):
        expected_values = expected["metrics"][metric_name]
        observed_values = observed["metrics"][metric_name]
        np.testing.assert_allclose(
            observed_values,
            expected_values,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{spec.baseline_id} {metric_name}",
        )

    for metric_name in ("final_min_energy", "final_mean_energy", "best_min_energy"):
        np.testing.assert_allclose(
            observed["metrics"][metric_name],
            expected["metrics"][metric_name],
            rtol=0.0,
            atol=TOLERANCE,
        )


def load_old_solver_module(spec: BaselineSpec) -> Any:
    for name in IMPORT_MODULES_TO_CLEAR:
        sys.modules.pop(name, None)

    old_path = list(sys.path)
    family_root = spec.solver_file.parents[2]
    legacy_dir = family_root / "Legacy" / "packages"
    optimization_dir = family_root / "Modern" / "optimization"

    sys.path.insert(0, str(legacy_dir))
    sys.path.insert(0, str(optimization_dir))
    try:
        module_name = f"_old_solver_{spec.baseline_id}"
        spec_obj = importlib.util.spec_from_file_location(module_name, spec.solver_file)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"cannot import {spec.solver_file}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


@contextmanager
def force_cpu_torch():
    original_randint = torch.randint
    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda
    original_synchronize = torch.cuda.synchronize
    original_empty_cache = torch.cuda.empty_cache

    def randint_cpu(*args, **kwargs):
        if str(kwargs.get("device")) == "cuda":
            kwargs["device"] = "cpu"
        return original_randint(*args, **kwargs)

    def tensor_cuda_noop(self, *args, **kwargs):
        return self

    def module_cuda_noop(self, *args, **kwargs):
        return self

    torch.randint = randint_cpu
    torch.Tensor.cuda = tensor_cuda_noop
    torch.nn.Module.cuda = module_cuda_noop
    torch.cuda.synchronize = lambda *args, **kwargs: None
    torch.cuda.empty_cache = lambda *args, **kwargs: None
    try:
        yield
    finally:
        torch.randint = original_randint
        torch.Tensor.cuda = original_tensor_cuda
        torch.nn.Module.cuda = original_module_cuda
        torch.cuda.synchronize = original_synchronize
        torch.cuda.empty_cache = original_empty_cache


def load_coupling_matrix(spec: BaselineSpec) -> torch.Tensor:
    instance = load_pairwise_instance(spec.fixture)
    matrix = torch.zeros(instance.num_spins, instance.num_spins)
    for interaction in instance.interactions:
        matrix[interaction.i, interaction.j] = interaction.coupling
        if spec.matrix_mode == "symmetric":
            matrix[interaction.j, interaction.i] = interaction.coupling
    return matrix


def _run_sk_simulated_annealing(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    temperatures, observ, _elapsed = module.simulated_annealing(
        load_coupling_matrix(spec),
        pop_size=params["pop_size"],
        num_steps_MC=params["MCsteps"],
        N=instance.num_spins,
        Tstart=params["Tstart"],
        Tend=params["Tend"],
        Observables=module.Observables,
        schedule=params["schedule"],
        num_temps_determiner=params["num_temps"],
        high_temp_thermalization_steps=params["high_temp_thermalization_steps"],
        device="cpu",
    )
    return temperatures, observ


def _run_sk_population_annealing(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    temperatures, observ, _elapsed = module.population_annealing(
        instance.num_spins,
        load_coupling_matrix(spec),
        pop_size=params["pop_size"],
        num_steps_MC=params["MCsteps"],
        Tstart=params["Tstart"],
        Tend=params["Tend"],
        Observables=module.Observables,
        schedule=params["schedule"],
        num_temps_determiner=params["num_temps"],
        high_temp_thermalization_steps=params["high_temp_thermalization_steps"],
    )
    return temperatures, observ


def _run_sk_parallel_tempering(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    temperatures, observ, _elapsed = module.parallel_tempering(
        instance.num_spins,
        load_coupling_matrix(spec),
        num_replicas=params["num_temps"],
        num_steps_MC=params["MCsteps"],
        swap_interval=params["swap_interval"],
        Tstart=params["Tstart"],
        Tend=params["Tend"],
        Observables=module.Observables,
        schedule=params["schedule"],
        num_temps_determiner=params["num_temps"],
        high_temp_thermalization_steps=params["high_temp_thermalization_steps"],
    )
    return temperatures, observ


def _run_sk_greedy(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    observ, _elapsed = module.greedy(
        load_coupling_matrix(spec),
        pop_size=params["pop_size"],
        N=instance.num_spins,
        Observables=module.Observables,
    )
    return None, observ


def _run_ea_simulated_annealing(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    temperatures, observ, _elapsed = module.simulated_annealing(
        params["L"],
        load_coupling_matrix(spec),
        pop_size=params["pop_size"],
        num_steps_MC=params["MCsteps"],
        N=instance.num_spins,
        Tstart=params["Tstart"],
        Tend=params["Tend"],
        Observables=module.Observables,
        schedule=params["schedule"],
        num_temps_determiner=params["num_temps"],
        high_temp_thermalization_steps=params["high_temp_thermalization_steps"],
        dimension=params["dimension"],
    )
    return temperatures, observ


def _run_ea_population_annealing(module: Any, spec: BaselineSpec) -> tuple[Any, Any]:
    params = spec.parameters
    instance = load_pairwise_instance(spec.fixture)
    temperatures, observ, _elapsed = module.population_annealing(
        params["L"],
        load_coupling_matrix(spec),
        pop_size=params["pop_size"],
        num_steps_MC=params["MCsteps"],
        Tstart=params["Tstart"],
        Tend=params["Tend"],
        Observables=module.Observables,
        schedule=params["schedule"],
        num_temps_determiner=params["num_temps"],
        high_temp_thermalization_steps=params["high_temp_thermalization_steps"],
        reweight_mode="multinomial",
    )
    return temperatures, observ


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _float_history(values: list[Any]) -> list[float]:
    history = []
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        history.append(float(value))
    return history


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))
