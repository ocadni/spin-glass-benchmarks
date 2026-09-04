"""Fast text writers for generated benchmark instances."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


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
    "dim",
    "K",
)
DEFAULT_CHUNK_SIZE = 100_000


def format_instance_number(value: float) -> str:
    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def normalize_distribution(distribution: str) -> str:
    return "rademacher" if distribution == "radamacher" else distribution


def instance_file_path(
    *,
    root: str | Path,
    family: str,
    num_spins: int,
    mean_j: float,
    seed: int | None,
    include_family_dir: bool = True,
) -> Path:
    root = Path(root)
    if include_family_dir:
        root = root / family
    filename = (
        f"{family}_couplings_N{num_spins}_"
        f"J{format_instance_number(mean_j)}_seed{'none' if seed is None else seed}.txt"
    )
    size_dir = f"N{num_spins}"
    if not include_family_dir and root.name == size_dir:
        return root / filename
    return root / size_dir / filename


def write_pairwise_text_file(
    *,
    path: str | Path,
    family: str,
    num_spins: int,
    mean_j: float,
    seed: int | None,
    distribution: str,
    field: float,
    num_couplings: int,
    field_values: float | np.ndarray | list[float] | tuple[float, ...] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        family=family,
        num_spins=num_spins,
        mean_j=mean_j,
        seed=seed,
        distribution=distribution,
        field=field,
        num_couplings=num_couplings,
        extra_metadata=extra_metadata,
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(f"{key}={metadata[key]}" for key in _ordered_keys(metadata)) + "\n")
        write_fields(handle, num_spins, field if field_values is None else field_values, chunk_size=chunk_size)
    return path


def append_couplings(
    path: str | Path,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        write_three_column_rows(handle, i_indices, j_indices, values, chunk_size=chunk_size)


def write_fields(handle, num_spins: int, values, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    indices = np.arange(num_spins, dtype=np.int64)
    if np.isscalar(values):
        fields = np.full(num_spins, float(values), dtype=np.float64)
    else:
        fields = np.asarray(values, dtype=np.float64)
        if fields.shape != (num_spins,):
            raise ValueError("field values must have length N")
    write_two_column_rows(handle, indices, fields, chunk_size=chunk_size)


def write_two_column_rows(
    handle,
    left: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    total = len(values)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        left_text = np.char.mod("%d", left[start:stop])
        value_text = format_number_array(values[start:stop])
        lines = np.char.add(np.char.add(left_text, " "), np.char.add(value_text, "\n"))
        handle.write("".join(lines.tolist()))


def write_three_column_rows(
    handle,
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    total = len(values)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        left_text = np.char.mod("%d", left[start:stop])
        right_text = np.char.mod("%d", right[start:stop])
        value_text = format_number_array(values[start:stop])
        prefix = np.char.add(np.char.add(left_text, " "), np.char.add(right_text, " "))
        lines = np.char.add(np.char.add(prefix, value_text), "\n")
        handle.write("".join(lines.tolist()))


def format_number_array(values: np.ndarray) -> np.ndarray:
    text = np.char.mod("%.5f", values.astype(np.float64, copy=False))
    text = np.char.rstrip(np.char.rstrip(text, "0"), ".")
    return np.where((text == "") | (text == "-0"), "0", text)


def _metadata(
    *,
    family: str,
    num_spins: int,
    mean_j: float,
    seed: int | None,
    distribution: str,
    field: float,
    num_couplings: int,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, str]:
    metadata = {
        "model": family,
        "N": str(num_spins),
        "meanJ": format_instance_number(mean_j),
        "seed": "none" if seed is None else str(seed),
        "distribution": distribution,
        "field": format_instance_number(field),
        "num_fields": str(num_spins),
        "num_couplings": str(num_couplings),
    }
    if extra_metadata:
        metadata.update({str(key): str(value) for key, value in extra_metadata.items()})
    return metadata


def _ordered_keys(metadata: Mapping[str, str]) -> list[str]:
    ordered = [key for key in HEADER_ORDER if key in metadata]
    ordered.extend(sorted(key for key in metadata if key not in HEADER_ORDER))
    return ordered
