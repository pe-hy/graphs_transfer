#!/bin/bash
#SBATCH --job-name=mixed_train_24
#SBATCH --output=logs/mixed_train/mixed_train_%A_%a.out
#SBATCH --error=logs/mixed_train/mixed_train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=project_465001424
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=small-g
#SBATCH --array=1-7

# Create logs directory if it doesn't exist
mkdir -p logs/mixed_train

# Define only the 7 required combinations in correct order
declare -a COMBINATIONS=(
    "standard:grammar"            # 1 - 256
    "standard:grammarindices"     # 2 - 256
    "grammar:standard"            # 3 - 256
    "indices:standard"            # 4 - 256
    "standard:indices"            # 5 - 256
    "standard:grammarindices"     # 6 - 512
    "grammarindices:grammar"      # 7 - 512
)

# Determine configuration based on task ID
if [[ $SLURM_ARRAY_TASK_ID -le 5 ]]; then
    CONFIG="original"
    CONFIG_NAME="2048-256"
    DATA_PREFIX="data/256"
else
    CONFIG="512"
    CONFIG_NAME="2048-512"
    DATA_PREFIX="data/512"
fi

# Get the correct combination
COMBINATION_IDX=$((SLURM_ARRAY_TASK_ID - 1))
COMBINATION=${COMBINATIONS[$COMBINATION_IDX]}

# Extract format1 and format2
IFS=':' read -r FORMAT1 FORMAT2 <<< "$COMBINATION"

echo "Starting training for combination: $FORMAT1 + $FORMAT2 (${CONFIG} config)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG ($CONFIG_NAME)"

# Define data file paths
DATA_DIR="${DATA_PREFIX}/mixed_${FORMAT1}_${FORMAT2}"
TRAIN_FILE="${DATA_DIR}/train.json"
TEST_MAIN_FILE="${DATA_DIR}/test_main.json"
TEST_SECOND_FILE="${DATA_DIR}/test_second.json"

# Check data files exist
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

# Construct model and wandb names
MODEL_NAME="${CONFIG_NAME}_${FORMAT1}-${FORMAT2}-6l-4h-64d-cos"
WANDB_PROJECT="graph_transfer_mixed"

echo "Model name: $MODEL_NAME"
echo "Wandb project: $WANDB_PROJECT"

# Run training
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
