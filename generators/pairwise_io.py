"""Text I/O for V0 pairwise Ising benchmark instances."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pairwise_instance import PairwiseInstance


SUPPORTED_FAMILIES = {"sk", "ea2d", "ea3d", "rrg"}
HEADER_ORDER = (
    "model",
    "N",
    "meanJ",
    "seed",
    "distribution",
    "field",
    "num_fields",
    "num_couplings",
    "graph",
    "degree",
    "graph_seed",
)


def parse_metadata_header(line: str) -> dict[str, str]:
    if not line.startswith("#"):
        return {}
    metadata: dict[str, str] = {}
    for token in line[1:].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metadata[key] = value
    return metadata


def infer_n_from_filename(path: Path) -> int:
    match = re.search(r"_N(\d+)_", path.name)
    if match is None:
        raise ValueError(f"cannot infer N from filename: {path}")
    return int(match.group(1))


def infer_family_from_filename(path: Path) -> str:
    name = path.name
    marker = "_couplings_"
    if marker not in name:
        raise ValueError(f"cannot infer family from filename: {path}")
    return name.split(marker, 1)[0]


def load_pairwise_instance(path: str | Path) -> PairwiseInstance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty instance file: {path}")

    metadata = parse_metadata_header(lines[0])
    data_lines = lines[1:] if metadata else lines
    family = metadata.get("model", infer_family_from_filename(path))
    num_spins = int(metadata.get("N", infer_n_from_filename(path)))
    num_fields = int(metadata.get("num_fields", num_spins))
    expected_couplings = metadata.get("num_couplings")

    if len(data_lines) < num_fields:
        raise ValueError(f"file has fewer field rows than expected: {path}")

    field_rows = [_parse_field_row(line, path) for line in data_lines[:num_fields]]
    coupling_rows = [_parse_coupling_row(line, path) for line in data_lines[num_fields:]]

    if expected_couplings is not None and len(coupling_rows) != int(expected_couplings):
        raise ValueError(
            f"expected {expected_couplings} couplings, found {len(coupling_rows)} in {path}"
        )

    metadata = _metadata_with_inferred_graph_params(
        family=family,
        num_spins=num_spins,
        num_couplings=len(coupling_rows),
        metadata=metadata,
    )

    return PairwiseInstance.from_edges(
        family=family,
        num_spins=num_spins,
        couplings=coupling_rows,
        fields=field_rows,
        metadata=metadata,
    )


def write_pairwise_instance(instance: PairwiseInstance, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = canonical_metadata_for_write(instance)
    header = "# " + " ".join(f"{key}={metadata[key]}" for key in _ordered_keys(metadata))
    lines = [header]
    lines.extend(f"{index} {format_instance_number(value)}" for index, value in enumerate(instance.fields))
    lines.extend(
        f"{interaction.i} {interaction.j} {format_instance_number(interaction.coupling)}"
        for interaction in instance.interactions
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def canonical_metadata_for_write(instance: PairwiseInstance) -> dict[str, str]:
    metadata = {str(key): str(value) for key, value in instance.metadata.items()}
    metadata["model"] = instance.family
    metadata["N"] = str(instance.num_spins)
    metadata["num_fields"] = str(instance.num_spins)
    metadata["num_couplings"] = str(len(instance.interactions))
    metadata = _metadata_with_inferred_graph_params(
        family=instance.family,
        num_spins=instance.num_spins,
        num_couplings=len(instance.interactions),
        metadata=metadata,
    )
    return metadata


def format_instance_number(value: float) -> str:
    """Format instance-file numbers like the original generator: five decimals."""
    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def default_instance_path(instance: PairwiseInstance, root: str | Path) -> Path:
    root = Path(root)
    metadata = canonical_metadata_for_write(instance)
    mean_j = metadata.get("meanJ", "0")
    seed = metadata.get("seed", "none")
    filename = f"{instance.family}_couplings_N{instance.num_spins}_J{mean_j}_seed{seed}.txt"
    return root / instance.family / f"N{instance.num_spins}" / filename


def benchmark_files(root: str | Path, families: Iterable[str] = SUPPORTED_FAMILIES) -> list[Path]:
    root = Path(root)
    allowed = set(families)
    files: list[Path] = []
    for path in sorted(root.rglob("*_couplings_*.txt")):
        try:
            family = infer_family_from_filename(path)
        except ValueError:
            continue
        if family in allowed:
            files.append(path)
    return files


def _ordered_keys(metadata: dict[str, str]) -> list[str]:
    ordered = [key for key in HEADER_ORDER if key in metadata]
    ordered.extend(sorted(key for key in metadata if key not in HEADER_ORDER))
    return ordered


def _metadata_with_inferred_graph_params(
    family: str,
    num_spins: int,
    num_couplings: int,
    metadata: dict[str, str],
) -> dict[str, str]:
    metadata = dict(metadata)
    if family != "rrg":
        return metadata

    metadata.setdefault("graph", "random_regular")
    if "degree" not in metadata and num_spins > 0:
        numerator = 2 * num_couplings
        if numerator % num_spins == 0:
            metadata["degree"] = str(numerator // num_spins)
    if "graph_seed" not in metadata and "seed" in metadata:
        metadata["graph_seed"] = metadata["seed"]
    return metadata


def _parse_field_row(line: str, path: Path) -> tuple[int, float]:
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"invalid field row in {path}: {line!r}")
    return int(parts[0]), float(parts[1])


def _parse_coupling_row(line: str, path: Path) -> tuple[int, int, float]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"invalid coupling row in {path}: {line!r}")
    return int(parts[0]), int(parts[1]), float(parts[2])
