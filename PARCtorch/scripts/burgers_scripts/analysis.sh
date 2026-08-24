#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic      # Account name
#SBATCH -J burgers_pixel_analysis  # Job name: UPDATED for per-pixel analysis
#SBATCH -o %x.out                   # Standard output file
#SBATCH -e %x.err                   # Standard error file
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1
#SBATCH -t 05:55:00                 # Time limit (HH:MM:SS) - Increased slightly for analysis


# ---------------------------
# Environment Setup
# ---------------------------
echo "Loading modules and activating Conda environment..."
module purge
module load miniforge cuda/11.4.2
module load gcc/12.4.0
source activate dparc

# Use the same Python as in JupyterLab
PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"
# Ensure access to required Python packages
export PYTHONPATH="/home/jtb3sud/.conda/envs/dparc/lib/python3.11/site-packages"

# ---------------------------
# Define Paths and Parameters
# ---------------------------
# --- USER: Please modify these parameters as needed ---
LAYER_TO_ANALYZE="upBlock_1" # Options: "initial_doubleConv", "upBlock_1"
FUTURE_STEPS=100             # Number of future steps to predict for the model
BATCH_SIZE=4                # Batch size for processing the data
# ---

BASE_DIR="/home/jtb3sud/PARCtorch/PARCtorch"
MODEL_CHECKPOINT="${BASE_DIR}/burgers/1431_1800/weights/burgers_model_epoch_10.pth"
EVAL_DATA="/project/vil_baek/data/physics/PARCTorch/Burgers/test"
MIN_MAX_PATH="${BASE_DIR}/data/bur_min_max.json"
# UPDATED: Output directory for the analysis results
OUTPUT_DIR="${BASE_DIR}/scripts/eval_results/burgers/per_pixel_analysis_${LAYER_TO_ANALYZE}"

echo "--------------------------------"
echo "Job Configuration:"
echo "Model: ${MODEL_CHECKPOINT}"
echo "Data: ${EVAL_DATA}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Layer: ${LAYER_TO_ANALYZE}"
echo "Future Steps: ${FUTURE_STEPS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "--------------------------------"

# ---------------------------
# Run the Analysis Script
# ---------------------------
# UPDATED: This command runs the new per-pixel analysis script.
# IMPORTANT: Make sure your new Python script is saved as 'burgers_per_pixel_analysis.py'
echo "Starting Python script for per-pixel analysis..."
$PYTHON_PATH per_pixel_analysis.py \
  --model_checkpoint "${MODEL_CHECKPOINT}" \
  --eval_data "${EVAL_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --min_max_path "${MIN_MAX_PATH}" \
  --layer_name ${LAYER_TO_ANALYZE} \
  --future_steps ${FUTURE_STEPS} \
  --batch_size ${BATCH_SIZE}

echo "Script finished successfully."
#SBATCH -c 4                        # CPU cores
#SBATCH --mem=40G                   # Memory allocation
