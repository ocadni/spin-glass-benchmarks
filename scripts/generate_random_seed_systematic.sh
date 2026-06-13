#!/usr/bin/env bash
set -euo pipefail

# Batch-generate instances with random seeds.
#
# Launch from anywhere with either:
#   bash scripts/generate_random_seed_systematic.sh
# or, if executable:
#   ./scripts/generate_random_seed_systematic.sh
#
# Edit the N/L lists below with the system sizes you want.
# SK and RRG use N. EA 2D and 3D use the lattice linear size L.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATOR="$REPO_ROOT/generators/generator.py"
BENCHMARKS_DIR="$REPO_ROOT/benchmarks"

SK_OUTDIR="$BENCHMARKS_DIR/sk"
EA2D_OUTDIR="$BENCHMARKS_DIR/ea2d"
EA3D_OUTDIR="$BENCHMARKS_DIR/ea3d"
RRG_OUTDIR="$BENCHMARKS_DIR/rrg"

REPEATS=10

SK_N_VALUES=(50 100 150 200 300)
EA2D_L_VALUES=(10 16 32 48)
EA3D_L_VALUES=(8 10 12 14)
RRG_N_VALUES=(50 100 150 200 300)

mkdir -p "$SK_OUTDIR" "$EA2D_OUTDIR" "$EA3D_OUTDIR" "$RRG_OUTDIR"

random_seed() {
  echo "$RANDOM$RANDOM"
}

for N in "${SK_N_VALUES[@]}"; do
  for _ in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" sk "$N" --seed "$(random_seed)" --outdir "$SK_OUTDIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for L in "${EA2D_L_VALUES[@]}"; do
  for _ in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" ea --L "$L" --dim 2 --seed "$(random_seed)" --outdir "$EA2D_OUTDIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for L in "${EA3D_L_VALUES[@]}"; do
  for _ in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" ea --L "$L" --dim 3 --seed "$(random_seed)" --outdir "$EA3D_OUTDIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for N in "${RRG_N_VALUES[@]}"; do
  for _ in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" rrg "$N" --degree 3 --seed "$(random_seed)" --outdir "$RRG_OUTDIR" --meanJ 0 --distribution gaussian --field 0
  done
done
