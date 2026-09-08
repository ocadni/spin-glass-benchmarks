#!/usr/bin/env python3
"""Download spin-glass benchmark instances from the Hugging Face dataset."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Laplaxe/spin-glass-benchmarks"  # TODO: set to your HF dataset repo id

ALL_TYPES = {"sk", "ea2d", "ea3d"}
IMPLEMENTED_TYPES = {"sk", "ea3d"}

INSTANCES_DIR = Path(__file__).resolve().parent

FILENAME_PATTERN = re.compile(r"^(?P<family>[a-z0-9]+)/N(?P<N>\d+)/.*_seed(?P<seed>\d+)\.txt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download spin-glass benchmark instances from the Hugging Face dataset."
    )
    parser.add_argument(
        "--type",
        choices=sorted(ALL_TYPES | {"all"}),
        help="Instance family to download, or 'all' for everything available.",
    )
    parser.add_argument("--N", type=int, help="Number of spins. Requires --type.")
    parser.add_argument("--seed", type=int, help="Instance seed. Requires --type and --N.")
    parser.add_argument(
        "--info",
        action="store_true",
        help="List what's available on the Hub instead of downloading anything.",
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face dataset repo id.")
    args = parser.parse_args()

    if args.type is None and not args.info:
        parser.error("nothing to do: specify --type (sk/ea2d/ea3d/all) or --info")
    if args.type == "all" and (args.N is not None or args.seed is not None):
        parser.error("--N/--seed cannot be combined with --type all")
    if args.seed is not None and args.N is None:
        parser.error("--seed requires --N to also be specified")
    if args.N is not None and args.type is None:
        parser.error("--N requires --type to also be specified")
    return args


def build_patterns(args: argparse.Namespace) -> list[str] | None:
    if args.type == "all":
        return None  # no filter: download everything in the repo
    if args.N is None:
        return [f"{args.type}/**"]
    if args.seed is None:
        return [f"{args.type}/N{args.N}/**"]
    return [f"{args.type}/N{args.N}/{args.type}_couplings_N{args.N}_J*_seed{args.seed}.txt"]


def show_info(repo_id: str, type_filter: str | None, n_filter: int | None) -> None:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")

    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for path in files:
        match = FILENAME_PATTERN.match(path)
        if not match:
            continue
        family = match.group("family")
        n = int(match.group("N"))
        if type_filter and type_filter != "all" and family != type_filter:
            continue
        if n_filter is not None and n != n_filter:
            continue
        counts[family][n] += 1

    if not counts:
        print("No matching instances found on the Hub.")
        return

    for family in sorted(counts):
        total = sum(counts[family].values())
        print(f"{family}: {total} instance(s)")
        for n in sorted(counts[family]):
            print(f"  N{n}: {counts[family][n]} seed(s)")


def main() -> None:
    args = parse_args()

    if args.info:
        show_info(args.repo_id, args.type, args.N)
        return

    if args.type in ALL_TYPES - IMPLEMENTED_TYPES:
        print(f"'{args.type}' instances are not available for download yet.")
        sys.exit(1)

    patterns = build_patterns(args)
    downloaded_to = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=INSTANCES_DIR,
        allow_patterns=patterns,
    )
    print(f"Downloaded to {downloaded_to}")


if __name__ == "__main__":
    main()
