#!/bin/bash
# Run sk_ga_initial experiment
# This script runs the experiment with GPU support and deterministic CUDA operations

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Running sk_ga_initial Experiment${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if GPU environment exists
if ! conda env list | grep -q "sgbench-solvers-gpu"; then
    echo -e "${RED}Error: sgbench-solvers-gpu environment not found${NC}"
    echo "Please install it first:"
    echo "  conda env create -f $REPO_ROOT/environments/solvers-core-gpu.yml"
    exit 1
fi

# Get the python path from the environment
PYTHON_PATH="$(conda env list | grep sgbench-solvers-gpu | awk '{print $NF}')/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}Error: Python not found in sgbench-solvers-gpu environment${NC}"
    exit 1
fi

# Verify GPU is available
echo -e "${GREEN}Checking GPU availability...${NC}"
if nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "  GPU detected: $GPU_INFO"
else
    echo -e "${RED}Warning: nvidia-smi not available. Will try to run anyway.${NC}"
fi
echo ""

# Set environment variables for deterministic GPU operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Run the experiment
echo -e "${GREEN}Starting experiment...${NC}"
echo "  Experiment: sk_ga_initial"
echo "  Algorithms: global_annealing, simulated_annealing"
echo "  Instances: 2 SK instances (N=50)"
echo "  Seeds: 1729, 4242"
echo "  Total runs: 8 (2 instances × 2 seeds × 2 algorithms)"
echo ""

cd "$REPO_ROOT"

$PYTHON_PATH -m experiments.src.cli "$SCRIPT_DIR/experiment_meta.json"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "Results saved in:"
    echo "  $SCRIPT_DIR/global_annealing/runs/"
    echo "  $SCRIPT_DIR/simulated_annealing/runs/"
    echo ""
    echo "View results:"
    echo "  ls $SCRIPT_DIR/global_annealing/runs/"
    echo "  cat $SCRIPT_DIR/global_annealing/runs/N50_seed1051730_seed1729/common.json"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Experiment failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}================================${NC}"
    exit $EXIT_CODE
fi
