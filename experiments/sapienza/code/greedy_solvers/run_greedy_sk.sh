#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

SOURCE="$SCRIPT_DIR/greedy.c"
BINARY="${GREEDY_BIN:-$SCRIPT_DIR/greedy}"
INSTANCES_ROOT="${INSTANCES_ROOT:-$REPO_ROOT/instances/sk}"
RESULTS_FILE="${RESULTS_FILE:-$SCRIPT_DIR/results.txt}"

POP_SIZE="${POP_SIZE:-1}"
SWEEPS="${SWEEPS:-100000}"
MODE="${MODE:-random}"
ZERO_FIELDS="${ZERO_FIELDS:-false}"
MAX_INSTANCES_PER_N="${MAX_INSTANCES_PER_N:-100}"
RUN_SEED=""
MAX_RUN_SEED=2147483647

random_run_seed() {
  local raw
  raw="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"
  echo $((raw % (MAX_RUN_SEED + 1)))
}

validate_run_seed() {
  local seed="$1"
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "error: --seed must be an integer between 0 and $MAX_RUN_SEED" >&2
    exit 1
  fi
  if [ "$seed" -gt "$MAX_RUN_SEED" ]; then
    echo "error: --seed must be between 0 and $MAX_RUN_SEED" >&2
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --seed)
      if [ "$#" -lt 2 ]; then
        echo "error: --seed requires a value" >&2
        exit 1
      fi
      RUN_SEED="$2"
      validate_run_seed "$RUN_SEED"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -lt 1 ]; then
  echo "usage: $0 [--seed RUN_SEED] N [N ...]" >&2
  echo "example: POP_SIZE=10000 SWEEPS=1 MODE=random $0 --seed 1 50 100 2000" >&2
  echo "example: POP_SIZE=10000 SWEEPS=1 MODE=random $0 50 100 2000" >&2
  exit 1
fi

if [ ! -x "$BINARY" ] || [ "$SOURCE" -nt "$BINARY" ]; then
  cc -O3 -std=c11 "$SOURCE" -lm -o "$BINARY"
fi

if [ ! -f "$RESULTS_FILE" ]; then
  printf "average_steps N instance_seed run_seed min_energy_perspin elapsed_time\n" > "$RESULTS_FILE"
fi

for N in "$@"; do
  instance_dir="$INSTANCES_ROOT/N$N"
  if [ ! -d "$instance_dir" ]; then
    echo "warning: missing instance directory: $instance_dir" >&2
    continue
  fi

  counter=0
  for instance in "$instance_dir"/sk_couplings_N"${N}"_J0_seed*.txt; do
    if [ ! -f "$instance" ]; then
      continue
    fi

    filename="$(basename "$instance")"
    instance_seed="${filename##*seed}"
    instance_seed="${instance_seed%.txt}"
    current_run_seed="$RUN_SEED"
    if [ -z "$current_run_seed" ]; then
      current_run_seed="$(random_run_seed)"
    fi

    "$BINARY" "$instance" \
      --num-spins "$N" \
      --pop-size "$POP_SIZE" \
      --sweeps "$SWEEPS" \
      --mode "$MODE" \
      --zero_fields "$ZERO_FIELDS" \
      --seed "$current_run_seed" \
      --instance-seed "$instance_seed" >> "$RESULTS_FILE"
    counter=$((counter + 1))

    if [ "$counter" -ge "$MAX_INSTANCES_PER_N" ]; then
      break
    fi
  done

  if [ "$counter" -eq 0 ]; then
    echo "warning: no matching SK instances found for N=$N in $instance_dir" >&2
  fi
done

echo "wrote results to $RESULTS_FILE"
