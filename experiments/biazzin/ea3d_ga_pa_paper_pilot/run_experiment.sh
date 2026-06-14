#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "$REPO_ROOT"

resolve_conda_env_path() {
  local env_name="$1"
  conda env list | awk -v env="$env_name" '$1 == env {print $NF; exit}'
}

if command -v conda >/dev/null 2>&1; then
  if [ -n "$CONDA_ENV_NAME" ]; then
    ENV_PATH="$(resolve_conda_env_path "$CONDA_ENV_NAME")"
  else
    ENV_PATH="$(resolve_conda_env_path sgbench-solvers-gpu)"
    if [ -z "$ENV_PATH" ]; then
      ENV_PATH="$(resolve_conda_env_path sgbench-solvers-core)"
    fi
  fi

  if [ -n "$ENV_PATH" ] && [ -x "$ENV_PATH/bin/python" ]; then
    RUNNER=("$ENV_PATH/bin/python")
  else
    echo "ERROR: No usable solver conda environment found."
    echo "Expected one of: sgbench-solvers-gpu, sgbench-solvers-core"
    echo "To bypass this check and use current Python, set: ALLOW_BASE_PYTHON=1"
    if [ "${ALLOW_BASE_PYTHON:-0}" != "1" ]; then
      exit 1
    fi
    echo "WARNING: Using current Python (ALLOW_BASE_PYTHON=1)"
    RUNNER=(python)
  fi
else
  echo "ERROR: Conda not found."
  echo "To bypass this check and use current Python, set: ALLOW_BASE_PYTHON=1"
  if [ "${ALLOW_BASE_PYTHON:-0}" != "1" ]; then
    exit 1
  fi
  echo "WARNING: Using current Python (ALLOW_BASE_PYTHON=1)"
  RUNNER=(python)
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Running ea3d_ga_pa_paper_pilot"
echo "Python runner: ${RUNNER[*]}"
echo "Experiment: $SCRIPT_DIR/experiment_meta.json"

"${RUNNER[@]}" -m experiments.src.cli "$SCRIPT_DIR/experiment_meta.json"
