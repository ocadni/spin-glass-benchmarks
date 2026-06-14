#!/usr/bin/env bash
set -euo pipefail

# Batch-generate instances with deterministic seeds.
#
# Launch from anywhere with either:
#   bash scripts/generate_random_seed_systematic.sh
# or, if executable:
#   ./scripts/generate_random_seed_systematic.sh
#
# Edit the N/L lists below with the system sizes you want.
# SK and RRG use N. EA 2D and 3D use the lattice linear size L.
# RRG graph generation uses the explicit degree in RRG_DEGREE.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATOR="$REPO_ROOT/generators/generate_pairwise_instance.py"
INSTANCES_DIR="${INSTANCES_DIR:-$REPO_ROOT/instances}"

REPEATS=10
BASE_SEED="${BASE_SEED:-1729}"

SK_N_VALUES=(50 100 150 200 300)
EA2D_L_VALUES=(10 16 32 48)
EA3D_L_VALUES=(8 10 12 14)
RRG_N_VALUES=(50 100 150 200 300)
RRG_DEGREE=3

mkdir -p "$INSTANCES_DIR"

deterministic_seed() {
  local family_offset="$1"
  local size="$2"
  local repeat="$3"
  echo $((BASE_SEED + family_offset * 1000000 + size * 1000 + repeat))
}

for N in "${SK_N_VALUES[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" sk "$N" --seed "$(deterministic_seed 1 "$N" "$repeat")" --outdir "$INSTANCES_DIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for L in "${EA2D_L_VALUES[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" ea2d --L "$L" --seed "$(deterministic_seed 2 "$L" "$repeat")" --outdir "$INSTANCES_DIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for L in "${EA3D_L_VALUES[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" ea3d --L "$L" --seed "$(deterministic_seed 3 "$L" "$repeat")" --outdir "$INSTANCES_DIR" --meanJ 0 --distribution gaussian --field 0
  done
done

for N in "${RRG_N_VALUES[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    python "$GENERATOR" rrg "$N" --degree "$RRG_DEGREE" --seed "$(deterministic_seed 4 "$N" "$repeat")" --outdir "$INSTANCES_DIR" --meanJ 0 --distribution gaussian --field 0
  done
done
