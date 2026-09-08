#!/usr/bin/env python3
"""Upload spin-glass benchmark instances to the Hugging Face dataset.

Companion to download_instances.py. Pushes every instances/<type>/N*/*.txt
file to the same relative path in the dataset repo (instances/ea3d/N1000/foo.txt
-> ea3d/N1000/foo.txt on the Hub), so download_instances.py's --type/--N/--seed
filters keep working against whatever gets uploaded here.

Requires a Hugging Face account with write access to the dataset repo and
being logged in (`huggingface-cli login`) or an HF_TOKEN in the environment.

Usage:
    pip install huggingface_hub
    python instances/upload_instances.py --type ea3d
    python instances/upload_instances.py --type ea3d --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Laplaxe/spin-glass-benchmarks"

ALL_TYPES = {"sk", "ea2d", "ea3d"}

INSTANCES_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload spin-glass benchmark instances to the Hugging Face dataset."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(ALL_TYPES),
        help="Instance family to upload (its instances/<type>/ directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be uploaded without uploading them.",
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face dataset repo id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_dir = INSTANCES_DIR / args.type

    if not family_dir.is_dir():
        raise SystemExit(f"{family_dir} does not exist locally, nothing to upload.")

    files = sorted(family_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt instance files found under {family_dir}.")

    print(f"{len(files)} file(s) under {family_dir} to upload as '{args.type}/**' on {args.repo_id}:")
    for f in files[:5]:
        print(f"  {f.relative_to(INSTANCES_DIR)}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return

    api = HfApi()
    api.upload_folder(
        folder_path=str(family_dir),
        path_in_repo=args.type,
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["*.txt"],
    )
    print(f"Uploaded {len(files)} file(s) to {args.repo_id}:{args.type}/")


if __name__ == "__main__":
    main()
