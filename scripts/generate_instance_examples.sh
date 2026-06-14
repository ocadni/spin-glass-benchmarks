#!/usr/bin/env bash
set -euo pipefail

# This script shows example commands for generating spin glass instances.
#
# Launch it from the repository root with either:
#   bash scripts/generate_instance_examples.sh
# or, if executable:
#   ./scripts/generate_instance_examples.sh
#
# Generated files are written to instances/examples by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATOR="$REPO_ROOT/scripts/generate_pairwise_instance.py"
OUTDIR="${OUTDIR:-$REPO_ROOT/instances/examples}"
mkdir -p "$OUTDIR"

python "$GENERATOR" sk 100 --seed 1 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python "$GENERATOR" ea2d --L 8 --seed 2 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python "$GENERATOR" ea3d --L 5 --seed 3 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0

python "$GENERATOR" rrg 100 --degree 3 --seed 4 --outdir "$OUTDIR" --meanJ 0 --distribution gaussian --field 0
