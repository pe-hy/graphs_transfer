#!/bin/bash
#SBATCH --job-name=gen_mixed                          # Job name
#SBATCH --output=logs/gen_mixed/gen_mixed_%j.out      # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/gen_mixed/gen_mixed_%j.err       # Error log
#SBATCH --time=02:00:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                     # Number of nodes requested
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --gpus=1                                      # Number of GPUs requested
#SBATCH --cpus-per-task=16                            # Number of CPU cores per task
#SBATCH --mem=64GB                                    # Memory limit
#SBATCH --partition=small-g                           # Partition name

# Common parameters for all format combinations
TRAIN_FORMAT1=2048
TRAIN_FORMAT2=512
TEST_FORMAT1=1024
TEST_FORMAT2=1024
GRAPH_PATH="data/graph.pkl"
SEED=42

# Format combinations (format1 format2)
COMBINATIONS=(
    "standard indices"
    "standard grammar"
    "standard grammarindices"
    "indices standard"
    "indices grammar"
    "indices grammarindices"
    "grammar standard"
    "grammar indices"
    "grammar grammarindices"
    "grammarindices standard"
    "grammarindices indices"
    "grammarindices grammar"
)

echo "Starting mixed format data generation..."
echo "Common parameters:"
echo "  Train format1: $TRAIN_FORMAT1"
echo "  Train format2: $TRAIN_FORMAT2"
echo "  Test format1: $TEST_FORMAT1"
echo "  Test format2: $TEST_FORMAT2"
echo "  Graph path: $GRAPH_PATH"
echo "  Seed: $SEED"
echo ""

# Loop through all format combinations
for combination in "${COMBINATIONS[@]}"; do
    # Split the combination into format1 and format2
    read -r format1 format2 <<< "$combination"
    
    echo "Processing combination: $format1 + $format2"
    
    # Create output directory name
    output_dir="data/512/mixed_${format1}_${format2}"
    
    singularity exec \
        $SIF \
        python data_generation/gen_mix.py \
        --format1 "$format1" \
        --format2 "$format2" \
        --train_format1 $TRAIN_FORMAT1 \
        --train_format2 $TRAIN_FORMAT2 \
        --test_format1 $TEST_FORMAT1 \
        --test_format2 $TEST_FORMAT2 \
        --graph_path "$GRAPH_PATH" \
        --output_dir "$output_dir" \
        --seed $SEED
    
    echo "Completed: $format1 + $format2"
    echo ""
done

echo "All mixed format combinations generated!"
echo ""
echo "Generated directories:"
for combination in "${COMBINATIONS[@]}"; do
    read -r format1 format2 <<< "$combination"
    echo "  data/mixed_${format1}_${format2}/"
done