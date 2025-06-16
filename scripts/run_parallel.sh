#!/bin/bash
# run_parallel.sh
# This script launches multiple training runs in parallel with different seeds.
# Usage example:
# ./run_parallel.sh --dataset Phenotype --model thomas-sounack/BioClinical-ModernBERT-base --lr 2e-5 --wd 0.01 --epochs 3 --batch_size 16 --accumulation_steps 1

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$0")
# Compute the project root directory (assuming script.sh is in scripts/ at the root level)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bert24

# Default list of seeds
seeds=(42 43 44 45 46)

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --model) model="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --wd) wd="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --accumulation_steps) accumulation_steps="$2"; shift ;;
        --seed) seeds=("$2"); shift ;;  # overwrite default seed list with a single seed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Check for required parameters
if [[ -z "$dataset" || -z "$model" || -z "$lr" || -z "$wd" || -z "$epochs" || -z "$batch_size" || -z "$accumulation_steps" ]]; then
    echo "Usage: $0 --dataset <dataset> --model <model> --lr <learning_rate> --wd <weight_decay> --epochs <epochs> --batch_size <batch_size> --accumulation_steps <accumulation_steps>"
    exit 1
fi

# Launch experiments with each seed in parallel
for seed in "${seeds[@]}"; do
    echo "Launching run with seed $seed"
    python "$PROJECT_ROOT/main.py" \
        --dataset "$dataset" \
        --model "$model" \
        --lr "$lr" \
        --wd "$wd" \
        --epochs "$epochs" \
        --seed "$seed" \
        --batch_size "$batch_size" \
        --accumulation_steps "$accumulation_steps" &
done

# Wait for all background processes to complete
wait
echo "All experiments finished."