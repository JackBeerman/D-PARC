#!/usr/bin/env bash
# ==============================================================================
# Training Script: BURGERS - Small Deform
# ==============================================================================
# Dataset: burgers
# Model: small_deform
# ==============================================================================

#SBATCH -A sds_baek_energetic
#SBATCH -J burgers_small_deform
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a40
#SBATCH -t 72:00:00
#SBATCH -c 1
#SBATCH --mem=80G

# Setup
module purge
module load miniforge cuda/11.8.0
source activate dparc

PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"
export PYTHONPATH="/home/jtb3sud/.conda/envs/dparc/lib/python3.11/site-packages"

# Data paths
TRAIN_DIRS="/project/vil_baek/data/physics/PARCTorch/Burgers/train_interpolation"
VAL_DIRS="/project/vil_baek/data/physics/PARCTorch/Burgers/val_interpolation"

# Output paths - organized by dataset and model
SAVE_DIR="/scratch/jtb3sud/Jesus/burgers/weights/small_deform"
LOSS_DIR="/scratch/jtb3sud/Jesus/burgers/loss/small_deform"
mkdir -p $SAVE_DIR $LOSS_DIR logs

# Training parameters
BATCH_SIZE=4
NUM_EPOCHS=500
SAVE_FREQUENCY=25
DATASET="burgers"
MODEL_CONFIG="small_deform"
FUTURE_STEPS=3  # ← ADD THIS: Number of future timesteps to predict

echo "=========================================="
echo "Training: BURGERS - Small Deform"
echo "Dataset: $DATASET"
echo "Model Config: $MODEL_CONFIG"
echo "Future Steps: $FUTURE_STEPS"  # ← ADD THIS
echo "Start Time: $(date)"
echo "=========================================="

$PYTHON_PATH jesus.py \
    --train_dirs $TRAIN_DIRS \
    --val_dirs $VAL_DIRS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --load_mod "/scratch/jtb3sud/Jesus/burgers/weights/small_deform/latest_checkpoint.pth" \
    --save_frequency $SAVE_FREQUENCY \
    --save_dir $SAVE_DIR \
    --loss_dir $LOSS_DIR \
    --dataset $DATASET \
    --start_epoch 1001 \
    --resume_training \
    --model_config $MODEL_CONFIG \
    --future_steps $FUTURE_STEPS \
    --device cuda

echo "Training completed: $(date)"

# Use the updated status checker
$PYTHON_PATH check_training_status_v2.py \  # ← CHANGED: Use v2
    --loss_dir $LOSS_DIR \
    --weights_dir $SAVE_DIR