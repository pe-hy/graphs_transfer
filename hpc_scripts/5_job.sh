#!/bin/bash
#SBATCH --job-name=mixed_train_24                     # Job name
#SBATCH --output=logs/mixed_train/mixed_train_%A_%a.out  # Standard output (%A = array job ID, %a = task ID)
#SBATCH --error=logs/mixed_train/mixed_train_%A_%a.err   # Error log
#SBATCH --time=24:00:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                     # Number of nodes requested
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --gpus=1                                      # Number of GPUs requested
#SBATCH --cpus-per-task=8                            # Number of CPU cores per task
#SBATCH --mem=64GB                                    # Memory limit
#SBATCH --partition=small-g                           # Partition name
#SBATCH --array=1-24                                  # Array of 24 jobs

# Create logs directory if it doesn't exist
mkdir -p logs/mixed_train

# Define base format combinations (12 combinations)
declare -a BASE_COMBINATIONS=(
    "standard:indices"
    "standard:grammar" 
    "standard:grammarindices"
    "indices:standard"
    "indices:grammar"
    "indices:grammarindices"
    "grammar:standard"
    "grammar:indices"
    "grammar:grammarindices"
    "grammarindices:standard"
    "grammarindices:indices"
    "grammarindices:grammar"
)

# Define two different data configurations
# Config 1: Original (2048:256 ratio) - tasks 1-12
# Config 2: Balanced (1152:1152 ratio) - tasks 13-24

if [[ $SLURM_ARRAY_TASK_ID -le 12 ]]; then
    # First configuration (original ratio)
    CONFIG="original"
    COMBINATION_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    CONFIG_NAME="2048-256"
else
    # Second configuration (balanced ratio)
    CONFIG="512" 
    COMBINATION_IDX=$((SLURM_ARRAY_TASK_ID - 13))
    CONFIG_NAME="2048-512"
fi

# Get the combination for this task
COMBINATION=${BASE_COMBINATIONS[$COMBINATION_IDX]}

# Extract format1 and format2 from combination
IFS=':' read -r FORMAT1 FORMAT2 <<< "$COMBINATION"

echo "Starting training for combination: $FORMAT1 + $FORMAT2 (${CONFIG} config)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG ($CONFIG_NAME)"

# Set data paths based on configuration
if [[ "$CONFIG" == "original" ]]; then
    DATA_DIR="data/256/mixed_${FORMAT1}_${FORMAT2}"
else
    DATA_DIR="data/512/mixed_${FORMAT1}_${FORMAT2}"
fi

TRAIN_FILE="${DATA_DIR}/train.json"
TEST_MAIN_FILE="${DATA_DIR}/test_main.json"
TEST_SECOND_FILE="${DATA_DIR}/test_second.json"

# Check if data files exist
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [[ ! -f "$TEST_MAIN_FILE" ]]; then
    echo "Error: Test main file not found: $TEST_MAIN_FILE"
    exit 1
fi

if [[ ! -f "$TEST_SECOND_FILE" ]]; then
    echo "Error: Test second file not found: $TEST_SECOND_FILE"
    exit 1
fi

echo "Using files:"
echo "  Train: $TRAIN_FILE"
echo "  Test main: $TEST_MAIN_FILE"
echo "  Test second: $TEST_SECOND_FILE"

# Create a unique model name for this combination and config
MODEL_NAME="${CONFIG_NAME}_${FORMAT1}-${FORMAT2}-6l-4h-64d-cos"
WANDB_PROJECT="graph_transfer_mixed"

echo "Model name: $MODEL_NAME"
echo "Wandb project: $WANDB_PROJECT"

# Run the training with hydra overrides
singularity exec \
    $SIF \
    python train.py \
    data.train_file="$TRAIN_FILE" \
    data.test_file="$TEST_MAIN_FILE" \
    data.format="mixed_${FORMAT1}_${FORMAT2}_${CONFIG}" \
    model.name="$MODEL_NAME" \
    wandb.model_name="$MODEL_NAME" \
    wandb.proj_name="$WANDB_PROJECT" \
    model.n_layer=6 \
    model.n_head=4 \
    model.n_embd=64 \
    model.batch_size=64 \
    model.block_size=128 \
    model.epochs=200 \
    optim.lr_type=None \
    optim.lr=0.003 \
    optim.warmup_steps=1560 \
    optim.weight_decay=0.01 \
    optim.max_lr=0.002

echo "Training completed for combination: $FORMAT1 + $FORMAT2 (${CONFIG} config)"