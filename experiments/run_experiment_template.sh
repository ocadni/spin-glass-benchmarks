#!/bin/bash
# Template script for running experiments
# Copy this to your experiment directory and modify as needed
#
# Usage:
#   1. Copy this file to your experiment directory:
#      cp experiments/run_experiment_template.sh experiments/yourname/your_experiment/run_experiment.sh
#   2. Make it executable:
#      chmod +x experiments/yourname/your_experiment/run_experiment.sh
#   3. Run it:
#      ./experiments/yourname/your_experiment/run_experiment.sh

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration - modify these as needed
ENVIRONMENT_NAME="sgbench-solvers-gpu"  # or sgbench-solvers-core for CPU
USE_GPU=true  # Set to false for CPU-only

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Running Experiment${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if environment exists
if ! conda env list | grep -q "$ENVIRONMENT_NAME"; then
    echo -e "${RED}Error: $ENVIRONMENT_NAME environment not found${NC}"
    echo "Please install it first:"
    if [ "$USE_GPU" = true ]; then
        echo "  conda env create -f $REPO_ROOT/environments/solvers-core-gpu.yml"
    else
        echo "  conda env create -f $REPO_ROOT/environments/solvers-core-cpu.yml"
    fi
    exit 1
fi

# Get the python path from the environment
PYTHON_PATH="$(conda env list | grep $ENVIRONMENT_NAME | awk '{print $NF}')/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}Error: Python not found in $ENVIRONMENT_NAME environment${NC}"
    exit 1
fi

# Check GPU if enabled
if [ "$USE_GPU" = true ]; then
    echo -e "${GREEN}Checking GPU availability...${NC}"
    if nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        echo -e "  GPU detected: $GPU_INFO"

        # Set environment variables for deterministic GPU operations
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        echo -e "  Using deterministic GPU operations"
    else
        echo -e "${YELLOW}Warning: nvidia-smi not available. GPU may not be accessible.${NC}"
    fi
else
    echo -e "${GREEN}Running on CPU${NC}"
fi
echo ""

# Find experiment_meta.json
EXPERIMENT_META="$SCRIPT_DIR/experiment_meta.json"

if [ ! -f "$EXPERIMENT_META" ]; then
    echo -e "${RED}Error: experiment_meta.json not found in $SCRIPT_DIR${NC}"
    echo "Expected: $EXPERIMENT_META"
    exit 1
fi

# Display experiment info
echo -e "${GREEN}Experiment configuration:${NC}"
RESEARCHER=$(grep -o '"researcher": *"[^"]*"' "$EXPERIMENT_META" | cut -d'"' -f4)
EXPERIMENT_NAME=$(grep -o '"experiment_name": *"[^"]*"' "$EXPERIMENT_META" | cut -d'"' -f4)
FAMILY=$(grep -o '"family": *"[^"]*"' "$EXPERIMENT_META" | cut -d'"' -f4)
echo "  Researcher: $RESEARCHER"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Family: $FAMILY"
echo "  Config: $EXPERIMENT_META"
echo ""

# Run the experiment
echo -e "${GREEN}Starting experiment...${NC}"
echo ""

cd "$REPO_ROOT"

$PYTHON_PATH -m experiments.src.cli "$EXPERIMENT_META"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "Results saved in:"
    echo "  $SCRIPT_DIR/*/runs/"
    echo ""
    echo "Analyze results:"
    echo "  ls $SCRIPT_DIR/*/runs/"
    echo ""
    echo "Or use Python:"
    echo "  from pathlib import Path"
    echo "  from experiments.src.analysis import load_experiment"
    echo "  exp = load_experiment(Path('$SCRIPT_DIR'))"
    echo "  results = exp.get_algorithm_results('your_algorithm')"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Experiment failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}================================${NC}"
    exit $EXIT_CODE
fi
