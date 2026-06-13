#!/usr/bin/env bash
set -euo pipefail

# This script shows example commands for generating spin glass instances.
#
# Launch it from the repository root with either:
#   bash scripts/generate_instance_examples.sh
# or, if executable:
#   ./scripts/generate_instance_examples.sh
#
# Generated files are written to benchmarks/examples by default.

OUTDIR="benchmarks/examples"
mkdir -p "$OUTDIR"

python generators/generator.py sk 100 --seed 1 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python generators/generator.py ea --L 8 --dim 2 --seed 2 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python generators/generator.py ea --L 5 --dim 3 --seed 3 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python generators/generator.py rrg 100 --degree 3 --seed 4 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0
