#!/usr/bin/env bash
# ==============================================================================
# Training Script: HMX - Small No Deform
# ==============================================================================
# Dataset: hmx
# Model: small_no_deform
# ==============================================================================

#SBATCH -A sds_baek_energetic
#SBATCH -J hmx_small_no_deform
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
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
TRAIN_DIRS="/project/vil_baek/data/physics/PARCTorch/HMX/train"
VAL_DIRS="/project/vil_baek/data/physics/PARCTorch/HMX/val"

# Output paths - organized by dataset and model
SAVE_DIR="/scratch/jtb3sud/Jesus/em/weights/small_no_deform"
LOSS_DIR="/scratch/jtb3sud/Jesus/em/loss/small_no_deform"
mkdir -p $SAVE_DIR $LOSS_DIR logs

# Training parameters
BATCH_SIZE=1
NUM_EPOCHS=425
SAVE_FREQUENCY=25
DATASET="hmx"
MODEL_CONFIG="small_no_deform"
FUTURE_STEPS=2  # ← ADD THIS: Number of future timesteps to predict

echo "=========================================="
echo "Training: HMX - Small No Deform"
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
    --load_mod "/scratch/jtb3sud/Jesus/em/weights/small_no_deform/checkpoint_epoch_1575.pth" \
    --save_frequency $SAVE_FREQUENCY \
    --save_dir $SAVE_DIR \
    --loss_dir $LOSS_DIR \
    --dataset $DATASET \
    --start_epoch 1576 \
    --resume_training \
    --model_config $MODEL_CONFIG \
    --future_steps $FUTURE_STEPS \
    --device cuda

echo "Training completed: $(date)"

# Use the updated status checker
$PYTHON_PATH check_training_status_v2.py \
    --loss_dir $LOSS_DIR \
    --weights_dir $SAVE_DIR